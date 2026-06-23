#ifndef PHARE_CORE_INNER_BOUNDARY_FIELD_IONOSPHERIC_CONVECTION_MOMENTUM_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_INNER_BOUNDARY_FIELD_IONOSPHERIC_CONVECTION_MOMENTUM_INNER_BOUNDARY_CONDITION_HPP

#include "core/inner_boundary/field_inner_boundary_condition.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"

#include <cmath>
#include <stdexcept>

namespace PHARE::core
{
/**
 * @brief Tanaka momentum inner-boundary condition for the ionospheric-convection body BC.
 *
 * Replaces a plain Neumann momentum (which lets plasma penetrate the body and drives a
 * pressure sink at the surface). It enforces, per ghost element, the two Tanaka conditions
 * on the conserved momentum rhoV (no rho needed, since condition 2 is on r^2*rho*u_n):
 *
 *   1. rho u x B = 0 written at the boundary point -> the surface momentum (½(ghost+mirror))
 *      has no component perpendicular to b̂.
 *   2. d(r^2 rho u_n)/dr = 0  -> r^2 rhoV_n = const.
 *
 * Combined:  rhoV_ghost = (1 + (r_m/r_g)^2) (rhoV_m·n / (b̂·n)) b̂ - rhoV_m.
 *
 * b̂ is the *total* field (B1+B0) interpolated at the BOUNDARY point (the formula is invariant to
 * the magnitude of b̂). When B is ~tangent to the surface (|b̂·n| < eps, division ill-posed) the BC
 * falls back to a symmetric condition (reverse normal component, keep tangential) -> zero normal
 * momentum at the surface, Neumann tangential. Sphere-only: r is the distance to the sphere centre.
 *
 * @note Uses only conserved quantities (rhoV, B1, B0), all up to date when the momentum BC runs
 *       first in applyToMoments — no dependency on a reconstructed velocity or ghost rho.
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
class FieldIonosphericConvectionMomentumInnerBoundaryCondition
    : public FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>
{
public:
    using Super = FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    using value_type                    = typename Super::field_type::value_type;
    using field_type                    = Super::field_type;
    using inner_boundary_mesh_data_type = Super::inner_boundary_mesh_data_type;
    using ghost_elem_data_type          = Super::ghost_elem_data_type;
    using interpolator_type             = Super::interpolator_type;
    using context_type                  = Super::context_type;
    using inner_boundary_type           = Super::inner_boundary_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    explicit FieldIonosphericConvectionMomentumInnerBoundaryCondition(
        inner_boundary_type const* geometry)
        : geometry_{geometry}
    {
    }

    FieldInnerBoundaryConditionType getType() const override
    {
        return FieldInnerBoundaryConditionType::IonosphericConvectionMomentum;
    }

    void apply(ScalarOrTensorFieldT& scalarOrTensorField, GridLayoutT const& layout,
               inner_boundary_mesh_data_type const& boundaryMeshData,
               context_type const& ctx) override
    {
        // momentum is a vector quantity; the scalar instantiation is a meaningless no-op.
        if constexpr (is_scalar)
        {
            return;
        }
        else
        {
            static_assert(N == 3, "ionospheric-convection momentum BC requires 3 components");

            auto rhoVfields = scalarOrTensorField.components();
            auto B1fields   = ctx.statenew.B1.components();
            auto B0fields   = ctx.statenew.B0.components();

            auto const* sphere = dynamic_cast<SphereInnerBoundary<dimension> const*>(geometry_);
            if (sphere == nullptr)
                throw std::runtime_error(
                    "ionospheric-convection momentum BC requires a sphere inner boundary");
            auto const& center = sphere->center();

            // Floor on the total-field magnitude below which the field direction b̂ is
            // meaningless and we fall back to the bounded symmetric condition.
            constexpr double eps = 1e-3;
            // Floor on |b̂·n| below which the perpendicular-to-b condition is ill-posed (B
            // near-tangent to the surface, e.g. the magnetic equator of a dipole). Below it we
            // fall back to the bounded symmetric condition. This also caps the 1/(b̂·n) gain of
            // the Tanaka formula to ~1/eps_bn: too small (1e-3) and the equatorial band amplifies
            // the ghost momentum ~1000x, feeding a runaway -> NaN. 0.1 caps it to ~10x.
            constexpr double eps_bn = 0.1;

            for_N<N>([&](auto ic) {
                constexpr auto i       = ic();
                auto& currentField     = std::get<i>(rhoVfields);
                auto const centering   = GridLayoutT::centering(currentField);
                auto const& ghostElems = boundaryMeshData.getGhostDataFromCentering(centering);

                for (ghost_elem_data_type const& ghostElem : ghostElems)
                {
                    if (!ghostElem.interpValid)
                        continue;

                    // ghost-node physical coords and the boundary (surface) point, midway between
                    // the ghost and its true mirror: project = 0.5*(ghost + mirror). The surface
                    // point (and the field direction b̂ read there) stays exact; only the momentum
                    // sample moves to the farthest interpolable point Q when the mirror is off-patch.
                    auto const amrIdxU = layout.localToAMR(ghostElem.index);
                    Point<int, dimension> amrIdx;
                    for_N<dimension>(
                        [&](auto kc) { amrIdx[kc()] = static_cast<int>(amrIdxU[kc()]); });
                    auto const ghostCoord = layout.fieldNodeCoordinates(currentField, amrIdx);
                    Point<double, dimension> boundaryPoint;
                    for_N<dimension>([&](auto kc) {
                        constexpr auto k = kc();
                        boundaryPoint[k] = 0.5 * (ghostCoord[k] + ghostElem.mirrorPoint[k]);
                    });

                    // momentum is read at the sample point Q; the field direction b̂ is read at the
                    // boundary point (where the perpendicular-to-b condition is imposed).
                    bool allInterpolable = true;
                    for_N<N>([&](auto jc) {
                        constexpr auto j = jc();
                        auto const cV    = GridLayoutT::centering(std::get<j>(rhoVfields));
                        auto const cB1   = GridLayoutT::centering(std::get<j>(B1fields));
                        auto const cB0   = GridLayoutT::centering(std::get<j>(B0fields));
                        if (!interpolator_type::pointIsInterpolable(layout, ghostElem.interpPoint, cV)
                            || !interpolator_type::pointIsInterpolable(layout, boundaryPoint, cB1)
                            || !interpolator_type::pointIsInterpolable(layout, boundaryPoint, cB0))
                            allInterpolable = false;
                    });
                    if (!allInterpolable)
                        continue;

                    Point<value_type, N> rhoV_q; // momentum at the sample point Q
                    Point<value_type, N> B_b;    // total field B1+B0 at the boundary point
                    for_N<N>([&](auto jc) {
                        constexpr auto j = jc();
                        rhoV_q[j]        = this->interpolator_(layout, std::get<j>(rhoVfields),
                                                               ghostElem.interpPoint);
                        B_b[j]           = this->interpolator_(layout, std::get<j>(B1fields),
                                                               boundaryPoint)
                                 + this->interpolator_(layout, std::get<j>(B0fields), boundaryPoint);
                    });

                    // outward boundary normal, padded to N components (z = 0 in 2D)
                    Point<value_type, N> n{};
                    for_N<dimension>([&](auto kc) {
                        constexpr auto k = kc();
                        n[k]             = static_cast<value_type>(ghostElem.normal[k]);
                    });

                    // radial distances to the sphere centre (ghost inside, sample Q outside)
                    double rg2 = 0., rq2 = 0.;
                    for_N<dimension>([&](auto kc) {
                        constexpr auto k = kc();
                        double const dg  = ghostCoord[k] - center[k];
                        double const dq  = ghostElem.interpPoint[k] - center[k];
                        rg2 += dg * dg;
                        rq2 += dq * dq;
                    });
                    double const r_g = std::sqrt(rg2);
                    double const r_q = std::sqrt(rq2);
                    if (r_g <= 0.)
                        continue;

                    // surface weight of the linear-along-normal model: the surface momentum is
                    // (1-t)*rhoV_ghost + t*rhoV_q, with t -> 1/2 when Q is the true mirror.
                    double const t   = -ghostElem.phiGhost / (ghostElem.phiInterp - ghostElem.phiGhost);
                    double const inv = 1.0 / (1.0 - t);

                    auto const rhoVn   = dot_product(rhoV_q, n); // sample normal momentum
                    double const Bnorm = std::sqrt(dot_product(B_b, B_b));

                    Point<value_type, N> rhoV_ghost;
                    if (Bnorm > eps)
                    {
                        auto const bhat = B_b * (static_cast<value_type>(1.) / Bnorm);
                        double const bn = dot_product(bhat, n);
                        if (std::abs(bn) >= eps_bn)
                        {
                            // null perpendicular-to-b surface momentum + r^2 rhoV_n conservation,
                            // with the surface taken from the linear model at weight t:
                            //   rhoV_g = ( [t + kappa(1-t)] (rhoV_q·n / b̂·n) b̂ - t rhoV_q ) / (1-t)
                            // kappa = (r_q/r_g)^2. Reduces to (1+(r_m/r_g)^2)(rhoV_m·n/b̂·n)b̂ - rhoV_m
                            // when Q is the true mirror (t = 1/2, r_q = r_m).
                            double const kappa = (r_q / r_g) * (r_q / r_g);
                            value_type const c
                                = static_cast<value_type>((t + kappa * (1.0 - t)) * inv)
                                  * (rhoVn / static_cast<value_type>(bn));
                            rhoV_ghost = bhat * c - rhoV_q * static_cast<value_type>(t * inv);
                        }
                        else
                        {
                            // B ~tangent to the surface: symmetric (zero surface-normal momentum,
                            // Neumann tangential) under the same linear model.
                            rhoV_ghost = rhoV_q - n * (rhoVn * static_cast<value_type>(inv));
                        }
                    }
                    else
                    {
                        // negligible field: fall back to symmetric
                        rhoV_ghost = rhoV_q - n * (rhoVn * static_cast<value_type>(inv));
                    }

                    currentField(ghostElem.index) = rhoV_ghost[i];
                }
            });
        }
    }

private:
    inner_boundary_type const* geometry_{nullptr};
};

} // namespace PHARE::core
#endif // PHARE_CORE_INNER_BOUNDARY_FIELD_IONOSPHERIC_CONVECTION_MOMENTUM_INNER_BOUNDARY_CONDITION_HPP
