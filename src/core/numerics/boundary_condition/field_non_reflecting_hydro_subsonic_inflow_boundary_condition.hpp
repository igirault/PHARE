#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/thermo/thermo.hpp"
#include "core/utilities/point/point.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>

namespace PHARE::core
{

/**
 * @brief Hydrodynamic LODI characteristic non-reflecting subsonic inflow.
 *
 * Counterpart to FieldNonReflectingHydroSubsonicOutflowBoundaryCondition. Prescribes a
 * target inflow state (ρ*, V*) — pressure stays free, its value at the boundary is set
 * entirely by the outgoing acoustic wave coming from the interior.
 *
 * Single BC class, registered for ρ, ρv, and Etot1; each apply() dispatches on the
 * field's physical quantity.
 *
 * Discretisation
 * ──────────────
 * Same boundary-face evaluation as the outflow BC: per tangential slice, the LAST
 * physical (interior) cell and the FIRST ghost cell flank the boundary face at distance
 * dn/2 each. Their mean gives the cell-centered primitive at the boundary; the centered
 * difference `(outward - inward) / dn` is a 2nd-order ∂/∂n AT the boundary.
 *
 * Eigenvalues in the outward-normal frame at a subsonic inflow (u_n_out < 0, |u_n|<c):
 *     λ_+ = u_n + c  > 0 → L_4 outgoing (leaves the domain)
 *     λ_0 = u_n      < 0 → L_2 (entropy), L_3 (shear) incoming
 *     λ_- = u_n - c  < 0 → L_1 (acoustic) incoming
 *
 * Outgoing L_4 from the same one-sided diff as the outflow BC:
 *     L_4 = (u_n + c)(∂p/∂n + ρ c ∂u_n/∂n)
 *
 * The three incoming amplitudes are soft-relaxed (Poinsot-Lele 1992 soft inflow) toward
 * the user-supplied targets, with relaxation rates set by the Rudy-Strikwerda formula
 * K = σ (1 − M²) c / L_box:
 *     L_1 = K · ρ c   (u_n   − u_n*)        # acoustic, drives normal velocity
 *     L_2 = K · ρ c²  (ρ     − ρ*)          # entropy,  drives density
 *     L_3 = K         (u_t   − u_t*)        # shear,    drives tangential velocity
 *
 * Same LODI ODE (forward-Euler over ctx.dt) as the outflow:
 *     ρ_bdr_new   = ρ_b   − dt (L_2 + ½(L_1 + L_4)) / c²
 *     u_n_bdr_new = u_n   − dt (L_4 − L_1) / (2 ρ_b c)
 *     u_t_bdr_new = u_t   − dt L_3
 *     p_bdr_new   = p_b   − dt ½(L_1 + L_4)
 *
 * Ghost cells are filled by the same 2nd-order Dirichlet-style extrapolation:
 *     φ_ghost_new = 2 φ_bdr_new − φ_mirror_new
 *
 * For Etot1, the boundary value is reconstructed from the LODI-derived
 * (ρ_bdr, V_bdr, p_bdr) and B projected at cell centers, then extrapolated.
 *
 * Hydrodynamic only — the LODI relations carry no magnetic eigenmodes. B at the inlet
 * is handled separately via a DivergenceFreeTransverseDirichlet BC (target B supplied
 * through the user dict, same recipe as the existing free-pressure-inflow).
 *
 * @tparam ScalarOrTensorFieldT Scalar field (ρ, Etot1) or vector field (ρv).
 * @tparam GridLayoutT          Grid layout type.
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldNonReflectingHydroSubsonicInflowBoundaryCondition
    : public IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>
{
public:
    using Super                  = IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>;
    using field_type             = Super::field_type;
    using physical_quantity_type = typename GridLayoutT::Quantity;
    using scalar_quantity_type   = typename physical_quantity_type::Scalar;
    using vector_quantity_type   = typename physical_quantity_type::Vector;
    using vectorfield_type       = VecField<field_type, physical_quantity_type>;

    static constexpr std::size_t dimension = Super::dimension;
    static constexpr bool is_scalar        = Super::is_scalar;

    FieldNonReflectingHydroSubsonicInflowBoundaryCondition(double rho_target,
                                                           std::array<double, 3> V_target,
                                                           double sigma, double length_scale,
                                                           std::shared_ptr<Thermo> thermo)
        : rho_target_{rho_target}
        , V_target_{V_target}
        , sigma_{sigma}
        , length_scale_{length_scale}
        , thermo_{std::move(thermo)}
    {
    }

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::NonReflectingHydroSubsonicInflow;
    }

    void apply(ScalarOrTensorFieldT& field, BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        std::size_t const dir_n   = static_cast<std::size_t>(direction);
        double const sign_n       = (side == Side::Upper) ? +1.0 : -1.0;
        double const dn           = gridLayout.meshSize()[dir_n];

        std::uint32_t const interior_n
            = (side == Side::Upper) ? gridLayout.physicalEndIndex(QtyCentering::dual, direction)
                                    : gridLayout.physicalStartIndex(QtyCentering::dual, direction);
        std::uint32_t const ghost_n = (side == Side::Upper) ? interior_n + 1 : interior_n - 1;

        std::map<TangentialKey, LODIResult> cache;

        auto fillScalarGhosts = [&](field_type& comp, auto pickBdr) {
            auto fieldBox = gridLayout.toFieldBox(localGhostBox, comp.physicalQuantity());
            QtyCentering const centering
                = GridLayoutT::centering(comp.physicalQuantity())[dir_n];
            for (auto const& ghostIdx : fieldBox)
            {
                auto const& lodi = lookupOrCompute_(cache, ghostIdx, dir_n, sign_n, dn, interior_n,
                                                    ghost_n, ctx);
                auto const mirror
                    = gridLayout.boundaryMirrored(direction, side, centering, ghostIdx);
                comp(ghostIdx) = 2.0 * pickBdr(lodi, ghostIdx) - comp(mirror);
            }
        };

        if constexpr (is_scalar)
        {
            if (field.physicalQuantity() == scalar_quantity_type::rho)
            {
                fillScalarGhosts(field, [](LODIResult const& l, GhostIdx const&) {
                    return l.rho_bdr;
                });
            }
            else if (field.physicalQuantity() == scalar_quantity_type::Etot1)
            {
                fillScalarGhosts(field, [&](LODIResult const& l, GhostIdx const& g) {
                    return etotAtBoundary_(g, dir_n, interior_n, ghost_n, l, ctx);
                });
            }
        }
        else
        {
            if (field.physicalQuantity() == vector_quantity_type::rhoV)
            {
                auto comps = field.components();
                fillScalarGhosts(std::get<0>(comps),
                                 [](LODIResult const& l, GhostIdx const&) { return l.rhoVx_bdr; });
                fillScalarGhosts(std::get<1>(comps),
                                 [](LODIResult const& l, GhostIdx const&) { return l.rhoVy_bdr; });
                fillScalarGhosts(std::get<2>(comps),
                                 [](LODIResult const& l, GhostIdx const&) { return l.rhoVz_bdr; });
            }
        }
    }

private:
    using GhostIdx      = Point<std::uint32_t, dimension>;
    using TangentialKey = std::array<std::uint32_t, (dimension == 0 ? 1 : dimension)>;

    /// Conservative boundary-face state at the new time.
    struct LODIResult
    {
        double rho_bdr;
        double rhoVx_bdr;
        double rhoVy_bdr;
        double rhoVz_bdr;
        double p_bdr;
    };

    TangentialKey tangKey_(GhostIdx const& idx, std::size_t dir_n) const
    {
        TangentialKey k{};
        std::size_t out = 0;
        for (std::size_t i = 0; i < dimension; ++i)
            if (i != dir_n)
                k[out++] = idx[i];
        return k;
    }

    std::pair<GhostIdx, GhostIdx> pairAt_(GhostIdx const& ghostIdx, std::size_t dir_n,
                                          std::uint32_t interior_n, std::uint32_t ghost_n) const
    {
        GhostIdx interior = ghostIdx;
        GhostIdx ghost    = ghostIdx;
        interior[dir_n]   = interior_n;
        ghost[dir_n]      = ghost_n;
        return {interior, ghost};
    }

    LODIResult const& lookupOrCompute_(std::map<TangentialKey, LODIResult>& cache,
                                       GhostIdx const& ghostIdx, std::size_t dir_n, double sign_n,
                                       double dn, std::uint32_t interior_n, std::uint32_t ghost_n,
                                       Super::boundary_condition_context_type const& ctx)
    {
        auto const key = tangKey_(ghostIdx, dir_n);
        auto it        = cache.find(key);
        if (it != cache.end())
            return it->second;

        auto const [interiorIdx, ghostIdxBdr] = pairAt_(ghostIdx, dir_n, interior_n, ghost_n);

        auto& rho_old  = ctx.accessor_old.getField(scalar_quantity_type::rho);
        auto& Etot_old = ctx.accessor_old.getField(scalar_quantity_type::Etot1);
        auto rhoV_old  = ctx.accessor_old.getVecField(vector_quantity_type::rhoV);
        auto B_new     = ctx.accessor_new.getVecField(vector_quantity_type::B1);
        auto rhoVc     = rhoV_old.components();
        auto Bc        = B_new.components();

        auto projectB = [&](GhostIdx const& idx) {
            double const bx = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(
                std::get<0>(Bc), idx);
            double const by = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(
                std::get<1>(Bc), idx);
            double const bz = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(
                std::get<2>(Bc), idx);
            return std::array<double, 3>{bx, by, bz};
        };

        // Pressure reconstructed cell-by-cell from Etot1 + ρ + ρv via EOS.
        // P_old is not guaranteed filled in the ghost layer; Etot1_old IS.
        auto pressureAt = [&](GhostIdx const& idx) {
            double const rho_c = rho_old(idx);
            double const Vx_c  = std::get<0>(rhoVc)(idx) / rho_c;
            double const Vy_c  = std::get<1>(rhoVc)(idx) / rho_c;
            double const Vz_c  = std::get<2>(rhoVc)(idx) / rho_c;
            auto const Bv      = projectB(idx);
            double const e_int = internalEnergyFromTotalEnergy(Etot_old(idx), rho_c, Vx_c, Vy_c,
                                                               Vz_c, Bv[0], Bv[1], Bv[2]);
            thermo_->setState_DU(rho_c, e_int / rho_c);
            return thermo_->pressure();
        };

        double const rho_g = rho_old(ghostIdxBdr);
        double const rho_i = rho_old(interiorIdx);
        double const Vx_g  = std::get<0>(rhoVc)(ghostIdxBdr) / rho_g;
        double const Vy_g  = std::get<1>(rhoVc)(ghostIdxBdr) / rho_g;
        double const Vz_g  = std::get<2>(rhoVc)(ghostIdxBdr) / rho_g;
        double const Vx_i  = std::get<0>(rhoVc)(interiorIdx) / rho_i;
        double const Vy_i  = std::get<1>(rhoVc)(interiorIdx) / rho_i;
        double const Vz_i  = std::get<2>(rhoVc)(interiorIdx) / rho_i;
        double const p_g   = pressureAt(ghostIdxBdr);
        double const p_i   = pressureAt(interiorIdx);

        // boundary face = arithmetic mean of flanking cells
        double const rho_b = 0.5 * (rho_g + rho_i);
        double const Vx_b  = 0.5 * (Vx_g + Vx_i);
        double const Vy_b  = 0.5 * (Vy_g + Vy_i);
        double const Vz_b  = 0.5 * (Vz_g + Vz_i);
        double const p_b   = 0.5 * (p_g + p_i);

        // outward normal velocity / single tangential velocity at the boundary face
        double const u_n = sign_n * ((dir_n == 0) ? Vx_b : Vy_b);
        double const u_t = (dir_n == 0) ? Vy_b : Vx_b;

        // target outward-normal / tangential velocities (from user-supplied cartesian V*)
        double const u_n_target = sign_n * ((dir_n == 0) ? V_target_[0] : V_target_[1]);
        double const u_t_target = (dir_n == 0) ? V_target_[1] : V_target_[0];

        thermo_->setState_DP(rho_b, p_b);
        double const c = thermo_->soundSpeed();

        // centered ∂/∂n at the boundary face: outward-side minus inward-side, divided by dn
        auto diff_n = [&](double f_g, double f_i) { return sign_n * (f_g - f_i) / dn; };
        double const dP_dn  = diff_n(p_g, p_i);
        double const duN_dn = diff_n((dir_n == 0 ? Vx_g : Vy_g) * sign_n,
                                     (dir_n == 0 ? Vx_i : Vy_i) * sign_n);

        // Outgoing acoustic wave (away from interior — only one outgoing at subsonic inflow)
        double const L4 = (u_n + c) * (dP_dn + rho_b * c * duN_dn);

        // Incoming amplitudes soft-relaxed toward the target inflow state.
        // Sign/factor convention chosen so that — when L_i are plugged into the LODI ODE
        // — each primitive decays toward its target with rate K (in the limit L_4 → 0):
        //     dρ/dt   = -(L_2 + ½(L_1+L_4)) / c²   →   L_2 = +c² K (ρ-ρ*)        gives dρ/dt   ≃ -K δρ
        //     du_n/dt = -(L_4 - L_1) / (2 ρ c)     →   L_1 = -2 ρ c K (u_n-u_n*) gives du_n/dt ≃ -K δu_n
        //     du_t/dt = -L_3                         →   L_3 = +K (u_t-u_t*)       gives du_t/dt ≃ -K δu_t
        // K = σ (1 - M²) c / L_box  (Rudy-Strikwerda / Yoo-Im rate).
        double const M  = std::abs(u_n) / c;
        double const K  = sigma_ * (1.0 - M * M) * c / length_scale_;
        double const L1 = -2.0 * K * rho_b * c * (u_n - u_n_target);
        double const L2 = K * c * c * (rho_b - rho_target_);
        double const L3 = K * (u_t - u_t_target);

        double const dt      = ctx.dt;
        double const rho_bdr = rho_b - dt * (L2 + 0.5 * (L1 + L4)) / (c * c);
        double const u_n_bdr = u_n - dt * (L4 - L1) / (2.0 * rho_b * c);
        double const u_t_bdr = u_t - dt * L3;
        double const p_bdr   = p_b - dt * 0.5 * (L1 + L4);

        double const Vx_bdr = (dir_n == 0) ? sign_n * u_n_bdr : u_t_bdr;
        double const Vy_bdr = (dir_n == 0) ? u_t_bdr : sign_n * u_n_bdr;
        double const Vz_bdr = Vz_b;

        LODIResult r{rho_bdr, rho_bdr * Vx_bdr, rho_bdr * Vy_bdr, rho_bdr * Vz_bdr, p_bdr};
        return cache.emplace(key, r).first->second;
    }

    /// Reconstruct Etot at the boundary face from LODI-derived primitives + B at the boundary.
    /// To stay consistent even with a non-zero magnetic field, but the BC shouldn't be use with
    /// non-zero magnetic field.
    double etotAtBoundary_(GhostIdx const& ghostIdx, std::size_t dir_n, std::uint32_t interior_n,
                           std::uint32_t ghost_n, LODIResult const& lodi,
                           Super::boundary_condition_context_type const& ctx)
    {
        auto B_new = ctx.accessor_new.getVecField(vector_quantity_type::B1);
        auto Bc    = B_new.components();

        auto projectB = [&](GhostIdx const& idx) {
            double const bx = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(
                std::get<0>(Bc), idx);
            double const by = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(
                std::get<1>(Bc), idx);
            double const bz = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(
                std::get<2>(Bc), idx);
            return std::array<double, 3>{bx, by, bz};
        };

        auto const [interiorIdx, ghostIdxBdr] = pairAt_(ghostIdx, dir_n, interior_n, ghost_n);
        auto const B_i                        = projectB(interiorIdx);
        auto const B_g                        = projectB(ghostIdxBdr);
        std::array<double, 3> const B_bdr{0.5 * (B_g[0] + B_i[0]), 0.5 * (B_g[1] + B_i[1]),
                                          0.5 * (B_g[2] + B_i[2])};

        double const Vx = lodi.rhoVx_bdr / lodi.rho_bdr;
        double const Vy = lodi.rhoVy_bdr / lodi.rho_bdr;
        double const Vz = lodi.rhoVz_bdr / lodi.rho_bdr;

        thermo_->setState_DP(lodi.rho_bdr, lodi.p_bdr);
        double const e_int = thermo_->internalEnergy() * lodi.rho_bdr;
        return totalEnergyFromInternalEnergy(e_int, lodi.rho_bdr, Vx, Vy, Vz, B_bdr[0], B_bdr[1],
                                             B_bdr[2]);
    }

    double rho_target_;
    std::array<double, 3> V_target_;
    double sigma_;
    double length_scale_;
    std::shared_ptr<Thermo> thermo_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP
