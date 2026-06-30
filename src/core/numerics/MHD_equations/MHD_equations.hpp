#ifndef CORE_NUMERICS_MHD_EQUATIONS_HPP
#define CORE_NUMERICS_MHD_EQUATIONS_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"

// the magnetic fluxes computations should be removed from here
namespace PHARE::core
{
template<bool Hall>
class MHDEquations
{
public:
    constexpr static bool hall = Hall;

    MHDEquations(double const gamma, double const eta, double const nu)
        : gamma_{gamma}
        , eta_{eta}
        , nu_{nu}
    {
    }

    template<auto direction>
    auto compute(auto const& u) const
    {
        auto const rho = u.rho;
        auto const V   = u.V;
        auto const B   = u.totalB(); // total field, used by the induction flux
        auto const B1  = u.B1;       // perturbation field
        auto const B0  = u.B0;       // static background field
        auto const P   = u.P;

        // Well-balanced B-split Maxwell stress: the magnetic part of the momentum flux is the
        // total-field stress (½|B|² Id − BB) minus the static background (B0) self-stress
        // (½|B0|² Id − B0B0). Developing B = B1 + B0 this leaves only the B1 and cross terms:
        //   Id (½|B1|² + B1·B0) − B1B1 − B1B0 − B0B1.
        // For a curl-free B0 the physical force (∇×B0)×B0 is zero, but the discrete divergence of
        // the B0-only stress is not (B0 reconstructed independently to L/R faces), bending the
        // upstream field. Removing it makes a B1 = 0 / V = 0 / uniform-P state a steady state.
        auto const MagPressure = 0.5 * (B1.x * B1.x + B1.y * B1.y + B1.z * B1.z)
                                 + (B1.x * B0.x + B1.y * B0.y + B1.z * B0.z);
        auto const GeneralisedPressure = P + MagPressure;

        // HD-only energy: kinetic + thermal. Magnetic energy transport (E × B1 for
        // perturbation energy Etot1 in B-split formulation) is provided by the CT
        // Poynting correction in apply_poynting_correction() using B1 edge fields.
        auto const Ehd = 0.5 * rho * (V.x * V.x + V.y * V.y + V.z * V.z)
                         + P / (gamma_ - 1.0);

        if constexpr (direction == Direction::X)
        {
            auto F_rho   = rho * V.x;
            auto F_rhoVx = rho * V.x * V.x + GeneralisedPressure
                           - (B1.x * B1.x + B1.x * B0.x + B0.x * B1.x);
            auto F_rhoVy = rho * V.x * V.y - (B1.x * B1.y + B1.x * B0.y + B0.x * B1.y);
            auto F_rhoVz = rho * V.x * V.z - (B1.x * B1.z + B1.x * B0.z + B0.x * B1.z);
            auto F_Bx    = 0.0;
            auto F_By    = B.y * V.x - V.y * B.x;
            auto F_Bz    = B.z * V.x - V.z * B.x;
            auto F_Etot  = (Ehd + P) * V.x;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
        if constexpr (direction == Direction::Y)
        {
            auto F_rho   = rho * V.y;
            auto F_rhoVx = rho * V.y * V.x - (B1.y * B1.x + B1.y * B0.x + B0.y * B1.x);
            auto F_rhoVy = rho * V.y * V.y + GeneralisedPressure
                           - (B1.y * B1.y + B1.y * B0.y + B0.y * B1.y);
            auto F_rhoVz = rho * V.y * V.z - (B1.y * B1.z + B1.y * B0.z + B0.y * B1.z);
            auto F_Bx    = B.x * V.y - V.x * B.y;
            auto F_By    = 0.0;
            auto F_Bz    = B.z * V.y - V.z * B.y;
            auto F_Etot  = (Ehd + P) * V.y;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
        if constexpr (direction == Direction::Z)
        {
            auto F_rho   = rho * V.z;
            auto F_rhoVx = rho * V.z * V.x - (B1.z * B1.x + B1.z * B0.x + B0.z * B1.x);
            auto F_rhoVy = rho * V.z * V.y - (B1.z * B1.y + B1.z * B0.y + B0.z * B1.y);
            auto F_rhoVz = rho * V.z * V.z + GeneralisedPressure
                           - (B1.z * B1.z + B1.z * B0.z + B0.z * B1.z);
            auto F_Bx    = B.x * V.z - V.x * B.z;
            auto F_By    = B.y * V.z - V.y * B.z;
            auto F_Bz    = 0.0;
            auto F_Etot  = (Ehd + P) * V.z;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
    }

    template<auto direction>
    auto compute(auto const& u, auto const& J) const
    {
        PerIndex f = compute<direction>(u);

        if constexpr (Hall)
            hall_contribution_<direction>(u.rho, u.totalB(), J, f.B1, f.P);
        // if constexpr (Resistivity)
        //     resistive_contributions_<direction>(eta_, u.totalB(), J, f.B1, f.P);

        return f;
    }

    // template<auto direction>
    // auto compute(auto const& u, auto const& J, auto const& LaplJ) const
    // {
    //     PerIndex f = compute<direction>(u);
    //
    //     if constexpr (Hall)
    //         hall_contribution_<direction>(u.rho, u.B, J, f.B, f.P);
    //     if constexpr (Resistivity)
    //         resistive_contributions_<direction>(eta_, u.B, J, f.B, f.P);
    //
    //     resistive_contributions_<direction>(nu_, u.B, -LaplJ, f.B, f.P);
    //
    //     return f;
    // }

    template<auto direction>
    void resistive_contributions(auto const& coef, auto const& Bt, auto const& Jt, auto& F_B,
                                 auto& F_Etot) const
    // Can be used for both resistivity with J and eta and hyper resistivity with laplJ and nu. The
    // work is done on the tranverse riemann averaged components avoid extra reconstructions. This
    // optimisation is possible since these operations are linear.
    {
        if constexpr (direction == Direction::X)
        {
            F_B.y += -Jt.z * coef;
            F_B.z += Jt.y * coef;
            F_Etot += (Jt.y * Bt.z - Jt.z * Bt.y) * coef;
        }
        if constexpr (direction == Direction::Y)
        {
            F_B.x += Jt.z * coef;
            F_B.z += -Jt.x * coef;
            F_Etot += (Jt.z * Bt.x - Jt.x * Bt.z) * coef;
        }
        if constexpr (direction == Direction::Z)
        {
            F_B.x += -Jt.y * coef;
            F_B.y += Jt.x * coef;
            F_Etot += (Jt.x * Bt.y - Jt.y * Bt.x) * coef;
        }
    }

private:
    double const gamma_;
    double const eta_;
    double const nu_;

    template<auto direction>
    void hall_contribution_(auto const& rho, auto const& B, auto const& J, auto& F_B,
                            auto& F_Etot) const
    {
        auto const invRho = 1.0 / rho;

        auto const JxB_x = J.y * B.z - J.z * B.y;
        auto const JxB_y = J.z * B.x - J.x * B.z;
        auto const JxB_z = J.x * B.y - J.y * B.x;

        if constexpr (direction == Direction::X)
        {
            F_B.y += -JxB_z * invRho;
            F_B.z += JxB_y * invRho;
            // Hall energy flux already captured by CT Poynting correction via E×B1
            // where E includes the Hall term (J×B)/rho.
        }
        if constexpr (direction == Direction::Y)
        {
            F_B.x += JxB_z * invRho;
            F_B.z += -JxB_x * invRho;
        }
        if constexpr (direction == Direction::Z)
        {
            F_B.x += -JxB_y * invRho;
            F_B.y += JxB_x * invRho;
        }
    }
};

} // namespace PHARE::core

#endif
