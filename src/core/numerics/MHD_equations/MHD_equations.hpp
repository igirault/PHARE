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

    MHDEquations(double const gamma)
        : gamma_{gamma}
    {
    }

    template<auto direction>
    auto compute(auto const& u) const
    {
        auto const rho = u.rho;
        auto const V   = u.V;
        auto const B   = u.B;
        auto const P   = u.P;

        auto const GeneralisedPressure = P + 0.5 * (B.x * B.x + B.y * B.y + B.z * B.z);
        auto const HydroEnergy         = eosPToEtot(gamma_, rho, V.x, V.y, V.z, 0., 0., 0., P);

        if constexpr (direction == Direction::X)
        {
            auto F_rho   = rho * V.x;
            auto F_rhoVx = rho * V.x * V.x + GeneralisedPressure - B.x * B.x;
            auto F_rhoVy = rho * V.x * V.y - B.x * B.y;
            auto F_rhoVz = rho * V.x * V.z - B.x * B.z;
            auto F_Bx    = 0.0;
            auto F_By    = B.y * V.x - V.y * B.x;
            auto F_Bz    = B.z * V.x - V.z * B.x;
            auto F_Etot  = (HydroEnergy + P) * V.x;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
        if constexpr (direction == Direction::Y)
        {
            auto F_rho   = rho * V.y;
            auto F_rhoVx = rho * V.y * V.x - B.y * B.x;
            auto F_rhoVy = rho * V.y * V.y + GeneralisedPressure - B.y * B.y;
            auto F_rhoVz = rho * V.y * V.z - B.y * B.z;
            auto F_Bx    = B.x * V.y - V.x * B.y;
            auto F_By    = 0.0;
            auto F_Bz    = B.z * V.y - V.z * B.y;
            auto F_Etot  = (HydroEnergy + P) * V.y;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
        if constexpr (direction == Direction::Z)
        {
            auto F_rho   = rho * V.z;
            auto F_rhoVx = rho * V.z * V.x - B.z * B.x;
            auto F_rhoVy = rho * V.z * V.y - B.z * B.y;
            auto F_rhoVz = rho * V.z * V.z + GeneralisedPressure - B.z * B.z;
            auto F_Bx    = B.x * V.z - V.x * B.z;
            auto F_By    = B.y * V.z - V.y * B.z;
            auto F_Bz    = 0.0;
            auto F_Etot  = (HydroEnergy + P) * V.z;

            return PerIndex{F_rho, {F_rhoVx, F_rhoVy, F_rhoVz}, {F_Bx, F_By, F_Bz}, F_Etot};
        }
    }

private:
    double const gamma_;
};

} // namespace PHARE::core

#endif
