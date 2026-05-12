#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_CHARACTERISTIC_FIXED_PRESSURE_OUTFLOW_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_CHARACTERISTIC_FIXED_PRESSURE_OUTFLOW_BOUNDARY_CONDITION_HPP

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
 * @brief Hydrodynamic LODI characteristic outflow with target pressure relaxation.
 *
 * Single BC class, registered separately for ρ, ρv, and Etot1 on the outflow side; each
 * apply() call dispatches on the physical quantity of the field and writes only the
 * ghost cells of that field.
 *
 * Discretisation
 * ──────────────
 * ρ, ρv, Etot1, P are cell-centered (dual along every direction). Per tangential slice
 * of the boundary face, gather
 *   • the LAST physical (interior) cell along the normal direction,
 *   • the FIRST ghost cell on the other side of the boundary face.
 * Both are at distance dn/2 from the boundary; their arithmetic mean is the boundary
 * face value of any cell-centered primitive, and `(outward - inward) / dn` is the
 * centered 2nd-order normal derivative AT the boundary.
 *
 * Wave amplitudes and forward-Euler LODI update at the boundary face (Poinsot-Lele):
 *
 *     L1 = (u_n - c)(∂p/∂n - ρ c ∂u_n/∂n)    # incoming, soft-relaxed
 *     L2 =   u_n   (∂ρ/∂n - ∂p/∂n / c²)       # outgoing
 *     L3 =   u_n   ∂u_t/∂n                    # outgoing
 *     L4 = (u_n + c)(∂p/∂n + ρ c ∂u_n/∂n)    # outgoing
 *     L1 = sigma (1 - M²) c / L_box · (p_b - p_target)
 *
 *     ρ_bdr_new   = ρ_b - dt/c² (L2 + ½(L1 + L4))
 *     u_n_bdr_new = u_n - dt/(2 ρ c)(L4 - L1)
 *     u_t_bdr_new = u_t - dt L3
 *     p_bdr_new   = p_b - dt ½(L1 + L4)
 *
 * Once the new boundary values are known, ghost cells are filled by the same 2nd-order
 * extrapolation used by Dirichlet:
 *
 *     φ_ghost_new = 2 φ_bdr_new - φ_mirror_new
 *
 * For Etot, the boundary value is reconstructed from (ρ_bdr_new, V_bdr_new, p_bdr_new,
 * B_bdr_new) via Thermo, and the same Dirichlet-style extrapolation is applied. B is
 * sampled at cell centers by projecting from face-centered storage (HD: B = 0).
 *
 * Hydrodynamic only — the LODI relations carry no magnetic eigenmodes.
 *
 * @tparam ScalarOrTensorFieldT Scalar field (ρ, Etot1) or vector field (ρv).
 * @tparam GridLayoutT          Grid layout type.
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldCharacteristicFixedPressureOutflowBoundaryCondition
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

    FieldCharacteristicFixedPressureOutflowBoundaryCondition(double p_target, double sigma,
                                                             double length_scale,
                                                             std::shared_ptr<Thermo> thermo)
        : p_target_{p_target}
        , sigma_{sigma}
        , length_scale_{length_scale}
        , thermo_{std::move(thermo)}
    {
    }

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::CharacteristicFixedPressureOutflow;
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

        // Last interior cell and first ghost cell flanking the boundary face,
        // along the normal direction. Indices on the tangential axis are filled per ghost
        // cell during the loop.
        std::uint32_t const interior_n
            = (side == Side::Upper) ? gridLayout.physicalEndIndex(QtyCentering::dual, direction)
                                    : gridLayout.physicalStartIndex(QtyCentering::dual, direction);
        std::uint32_t const ghost_n = (side == Side::Upper) ? interior_n + 1 : interior_n - 1;

        std::map<TangentialKey, LODIResult> cache;

        // helper to fill ghost cells of a scalar component using a Dirichlet-style
        // 2nd-order extrapolation with a per-LODI boundary value picked by `pickBdr`.
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
            // other scalars (e.g. P) fall through to whatever earlier fill already wrote.
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

    /// To group together ghost cells that have same tangential coordinates (aligned along the
    /// normal)
    TangentialKey tangKey_(GhostIdx const& idx, std::size_t dir_n) const
    {
        TangentialKey k{};
        std::size_t out = 0;
        for (std::size_t i = 0; i < dimension; ++i)
            if (i != dir_n)
                k[out++] = idx[i];
        return k;
    }

    /// Build interior- and ghost-cell index points from the tangential coordinates of @p ghostIdx
    /// and the normal indices given by @p interior_n / @p ghost_n.
    std::pair<GhostIdx, GhostIdx> pairAt_(GhostIdx const& ghostIdx, std::size_t dir_n,
                                          std::uint32_t interior_n, std::uint32_t ghost_n) const
    {
        GhostIdx interior = ghostIdx;
        GhostIdx ghost    = ghostIdx;
        interior[dir_n]   = interior_n;
        ghost[dir_n]      = ghost_n;
        return {interior, ghost};
    }

    /// Retrieve LODI results if ghost cell belongs to a group of cell already encountered, or
    /// compute LODI quantities if not
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

        // Pressure is reconstructed cell-by-cell from Etot1 + ρ + ρv via the EOS.
        // P_old is not guaranteed filled in the ghost layer; Etot1_old IS (registered as a
        // moment in the messenger). HD assumption: B contribution is negligible / zero.
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

        // boundary face = arithmetic mean of flanking cells (2nd-order)
        double const rho_b = 0.5 * (rho_g + rho_i);
        double const Vx_b  = 0.5 * (Vx_g + Vx_i);
        double const Vy_b  = 0.5 * (Vy_g + Vy_i);
        double const Vz_b  = 0.5 * (Vz_g + Vz_i);
        double const p_b   = 0.5 * (p_g + p_i);

        // outward normal velocity / single tangential velocity at the boundary face
        double const u_n = sign_n * ((dir_n == 0) ? Vx_b : Vy_b);
        double const u_t = (dir_n == 0) ? Vy_b : Vx_b;

        thermo_->setState_DP(rho_b, p_b);
        double const c = thermo_->soundSpeed();

        // centered ∂/∂n at the boundary face: outward-side minus inward-side, divided by dn
        auto diff_n          = [&](double f_g, double f_i) { return sign_n * (f_g - f_i) / dn; };
        double const drho_dn = diff_n(rho_g, rho_i);
        double const dP_dn   = diff_n(p_g, p_i);
        double const duN_dn
            = diff_n((dir_n == 0 ? Vx_g : Vy_g) * sign_n, (dir_n == 0 ? Vx_i : Vy_i) * sign_n);
        double const duT_dn = diff_n((dir_n == 0 ? Vy_g : Vx_g), (dir_n == 0 ? Vy_i : Vx_i));

        double const M  = std::abs(u_n) / c;
        double const K  = sigma_ * (1.0 - M * M) * c / length_scale_;
        double const L1 = K * (p_b - p_target_);
        double const L2 = u_n * (drho_dn - dP_dn / (c * c));
        double const L3 = u_n * duT_dn;
        double const L4 = (u_n + c) * (dP_dn + rho_b * c * duN_dn);

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
    /// non-zero magnetic field
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

    double p_target_;
    double sigma_;
    double length_scale_;
    std::shared_ptr<Thermo> thermo_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_CHARACTERISTIC_FIXED_PRESSURE_OUTFLOW_BOUNDARY_CONDITION_HPP
