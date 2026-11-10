#ifndef PHARE_CORE_INNER_BOUNDARY_FIELD_TOTAL_ENERGY_FROM_PRESSURE_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_INNER_BOUNDARY_FIELD_TOTAL_ENERGY_FROM_PRESSURE_INNER_BOUNDARY_CONDITION_HPP

#include "core/models/quantities/mhd_quantities.hpp"
#include "core/inner_boundary/field_inner_boundary_condition.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/utilities/index/index.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <tuple>

namespace PHARE::core
{
template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
class FieldTotalEnergyFromPressureInnerBoundaryCondition
    : public FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>
{
public:
    using Super = FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    using field_type                    = Super::field_type;
    using inner_boundary_mesh_data_type = Super::inner_boundary_mesh_data_type;
    using ghost_elem_data_type          = Super::ghost_elem_data_type;
    using interpolator_type             = Super::interpolator_type;
    using context_type                  = Super::context_type;

    using scalar_inner_bc_type
        = FieldInnerBoundaryCondition<field_type, GridLayoutT, PhysicalStateT>;

    static constexpr size_t dimension = Super::dimension;
    static constexpr bool is_scalar   = Super::is_scalar;

    static constexpr int late_priority = 10;

    FieldTotalEnergyFromPressureInnerBoundaryCondition(
        std::unique_ptr<scalar_inner_bc_type> pressureBC, double gamma)
        : pressureBC_{std::move(pressureBC)}
        , gamma_{gamma}
    {
    }

    FieldInnerBoundaryConditionType getType() const override
    {
        return FieldInnerBoundaryConditionType::TotalEnergyFromPressure;
    }

    int priority() const override { return late_priority; }

    bool fillsNonInterpolableGhosts() const override
    {
        return pressureBC_->fillsNonInterpolableGhosts();
    }

    void apply(ScalarOrTensorFieldT& scalarOrTensorField, GridLayoutT const& layout,
               inner_boundary_mesh_data_type const& boundaryMeshData,
               context_type const& ctx) override
    {
        if constexpr (!is_scalar)
        {
            return;
        }
        else
        {
            auto& Etot1Field = scalarOrTensorField;
            auto& P          = ctx.statenew.P;
            auto& rho        = ctx.statenew.rho;

            auto rhoVc  = ctx.statenew.rhoV.components();
            auto& rhoVx = std::get<0>(rhoVc);
            auto& rhoVy = std::get<1>(rhoVc);
            auto& rhoVz = std::get<2>(rhoVc);

            auto B1c  = ctx.statenew.B1.components();
            auto& B1x = std::get<0>(B1c);
            auto& B1y = std::get<1>(B1c);
            auto& B1z = std::get<2>(B1c);

            auto const& cellStatus = boundaryMeshData.cellStatusField();

            layout.evalOnGhostBox(P, [&](auto&... args) {
                auto const idx = core::MeshIndex<dimension>{args...};
                auto const st  = cellStatus(idx);
                if (st != toDouble(ElemStatus::Fluid) && st != toDouble(ElemStatus::Cut))
                    return;

                auto const rho_m = rho(idx);
                if (rho_m <= min_density_)
                    return;

                auto const vx = rhoVx(idx) / rho_m;
                auto const vy = rhoVy(idx) / rho_m;
                auto const vz = rhoVz(idx) / rho_m;

                auto const bx = GridLayoutT::template project<MHDQuantity::Scalar::Bx,
                                                              MHDQuantity::Scalar::rho>(B1x, idx);
                auto const by = GridLayoutT::template project<MHDQuantity::Scalar::By,
                                                              MHDQuantity::Scalar::rho>(B1y, idx);
                auto const bz = GridLayoutT::template project<MHDQuantity::Scalar::Bz,
                                                              MHDQuantity::Scalar::rho>(B1z, idx);

                P(idx) = eosEtot1ToP(gamma_, rho_m, vx, vy, vz, bx, by, bz, Etot1Field(idx));
            });

            pressureBC_->apply(P, layout, boundaryMeshData, ctx);

            auto const centering   = GridLayoutT::centering(Etot1Field);
            auto const& ghostElems = boundaryMeshData.getGhostDataFromCentering(centering);

            bool const fillAll = pressureBC_->fillsNonInterpolableGhosts();

            for (ghost_elem_data_type const& ghostElem : ghostElems)
            {
                if (!fillAll && !ghostElem.mirrorIsInterpolable)
                    continue;

                auto const idx = toMeshIndex_(ghostElem.index);

                auto const rho_g = rho(idx);
                if (rho_g <= min_density_)
                    continue;

                auto const vx = rhoVx(idx) / rho_g;
                auto const vy = rhoVy(idx) / rho_g;
                auto const vz = rhoVz(idx) / rho_g;

                auto const bx = GridLayoutT::template project<MHDQuantity::Scalar::Bx,
                                                              MHDQuantity::Scalar::rho>(B1x, idx);
                auto const by = GridLayoutT::template project<MHDQuantity::Scalar::By,
                                                              MHDQuantity::Scalar::rho>(B1y, idx);
                auto const bz = GridLayoutT::template project<MHDQuantity::Scalar::Bz,
                                                              MHDQuantity::Scalar::rho>(B1z, idx);

                Etot1Field(idx) = eosPToEtot1(gamma_, rho_g, vx, vy, vz, bx, by, bz, P(idx));
            }
        }
    }

private:
    static auto toMeshIndex_(Point<std::uint32_t, dimension> const& p)
    {
        if constexpr (dimension == 1)
            return core::MeshIndex<1>{p[0]};
        else if constexpr (dimension == 2)
            return core::MeshIndex<2>{p[0], p[1]};
        else
            return core::MeshIndex<3>{p[0], p[1], p[2]};
    }

    static double const min_density_;

    std::unique_ptr<scalar_inner_bc_type> pressureBC_;
    double gamma_;
};

template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
double const FieldTotalEnergyFromPressureInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT,
                                                                PhysicalStateT>::min_density_
    = std::sqrt(1024 * std::numeric_limits<double>::min());

} // namespace PHARE::core
#endif
