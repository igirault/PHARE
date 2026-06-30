#ifndef PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/logger.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"

#include <cmath>

namespace PHARE::core
{
static auto const min_value = std::sqrt(1024 * std::numeric_limits<double>::min());

auto rhoVToV(auto& rho, auto const& rhoVx, auto const& rhoVy, auto const& rhoVz)
{
    auto const vx = rhoVx / rho;
    auto const vy = rhoVy / rho;
    auto const vz = rhoVz / rho;

    return std::array{vx, vy, vz};
}

auto rhoVToV(auto& rho, auto const& rhoV)
{
    return rhoVToV(rho, rhoV[0], rhoV[1], rhoV[2]);
}

template<typename GridLayout>
class ToPrimitiveConverter_ref;

template<typename GridLayout>
class ToPrimitiveConverter : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    ToPrimitiveConverter(PHARE::initializer::PHAREDict const& dict)
        : gamma_{dict["heat_capacity_ratio"].template to<double>()}
        , pressure_floor_{dict.contains("pressure_floor")
                              ? dict["pressure_floor"].template to<double>()
                              : 0.0}
        , density_floor_{dict.contains("density_floor")
                             ? dict["density_floor"].template to<double>()
                             : 0.0}
    {
    }

    template<typename Field, typename VecField>
    void operator()(Field& rho, VecField const& rhoV, VecField const& B1, VecField const& B0,
                    Field& Etot1, VecField& V, Field& P) const
    {
        ToPrimitiveConverter_ref<GridLayout>{*this->layout_}(gamma_, pressure_floor_, density_floor_,
                                                            rho, rhoV, B1, B0, Etot1, V, P);
    }

private:
    double const gamma_;
    double const pressure_floor_;
    double const density_floor_;
};

template<typename GridLayout>
class ToPrimitiveConverter_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    ToPrimitiveConverter_ref(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename Field, typename VecField>
    void operator()(double const gamma, double const pressure_floor, double const density_floor,
                    Field& rho, VecField const& rhoV, VecField const& B1, VecField const& B0,
                    Field& Etot1, VecField& V, Field& P) const
    {
        // Floor the conserved mass density first (keeping momentum, like Athena++'s
        // ConservedToPrimitive): an under-resolved coarse shock can drive rho < 0, which then
        // makes V = rhoV/rho and the Riemann fast speed sqrt(gamma P / rho) blow up. Flooring
        // rho here (before V-recovery) bounds V and keeps the reconstruction input positive.
        floorDensityOnGhostBox(density_floor, rho);

        rhoVToVOnGhostBox(rho, rhoV, V);

        eosEtot1ToPWithFloorOnGhostBox(gamma, pressure_floor, rho, rhoV, B1, B0, Etot1, P);
    }

    // Clamp conserved mass density up to density_floor in place. Momentum (rhoV) and energy
    // (Etot1) are left unchanged; the subsequent V-recovery and pressure floor then use the
    // floored rho, so velocity stays bounded and the pressure stays consistent.
    template<typename Field>
    void floorDensityOnGhostBox(double const density_floor, Field& rho) const
    {
        if (density_floor <= 0.0)
            return;
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            MeshIndex<Field::dimension> const index{args...};
            if (rho(index) < density_floor)
            {
                PHARE_LOG_LINE_SS("density floored: " << rho(index) << " -> " << density_floor);
                rho(index) = density_floor;
            }
        });
    }

    // used for diagnostics
    template<typename Field, typename VecField>
    void rhoVToVOnGhostBox(Field& rho, VecField const& rhoV, VecField& V) const
    {
        layout_.evalOnGhostBox(rho,
                               [&](auto&... args) mutable { rhoVToV_(rho, rhoV, V, {args...}); });
    }

    // Read-only pressure recovery (diagnostics): never mutates Etot1, never floors.
    template<typename Field, typename VecField>
    void eosEtot1ToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                               VecField const& B1, VecField const& B0, Field const& Etot1,
                               Field& P) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            MeshIndex<Field::dimension> const index{args...};
            P(index) = recoverP_(gamma, rho, rhoV, B1, Etot1, index);
        });
    }

    template<typename Field, typename VecField>
    void eosEtotToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                              VecField const& B1, VecField const& B0, Field const& Etot1,
                              Field& P) const
    {
        eosEtot1ToPOnGhostBox(gamma, rho, rhoV, B1, B0, Etot1, P);
    }

    // Solve path: recover P with a positivity floor. When the floor triggers we clamp P AND
    // rewrite the conserved perturbation energy Etot1 consistently, so the conserved state stays
    // in sync and the next conversion does not re-derive the bad value.
    template<typename Field, typename VecField>
    void eosEtot1ToPWithFloorOnGhostBox(double const gamma, double const pressure_floor,
                                        Field const& rho, VecField const& rhoV, VecField const& B1,
                                        VecField const& B0, Field& Etot1, Field& P) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            MeshIndex<Field::dimension> const index{args...};
            auto p = recoverP_(gamma, rho, rhoV, B1, Etot1, index);

            // an RK substage can momentarily drive the recovered pressure below a physical floor
            // (or negative -> NaN sound speed), notably at cut-cell ghosts of the inner boundary.
            // Catch non-finite p too: `p < floor` is false for NaN, so an already-NaN pressure
            // would otherwise slip through unfloored.
            if (pressure_floor > 0.0 && (!std::isfinite(p) || p < pressure_floor))
            {
                Point<int, dimension> localIdx;
                for (std::size_t d = 0; d < dimension; ++d)
                    localIdx[d] = static_cast<int>(index[d]);
                auto const amrIdx = layout_.localToAMR(localIdx);
                auto const pos    = layout_.fieldNodeCoordinates(rho, amrIdx);
                PHARE_LOG_LINE_SS("pressure floored: " << p << " -> " << pressure_floor
                                                       << " at local index " << localIdx << " AMR "
                                                       << amrIdx << " position " << pos);
                auto const [vx, vy, vz, b1x, b1y, b1z] = cellPrimitives_(rho, rhoV, B1, index);
                p             = pressure_floor;
                Etot1(index)  = eosPToEtot1(gamma, rho(index), vx, vy, vz, b1x, b1y, b1z,
                                            pressure_floor);
            }
            P(index) = p;
        });
    }

private:
    template<typename Field, typename VecField>
    static void rhoVToV_(Field& rho, VecField const& rhoV, VecField& V,
                         MeshIndex<Field::dimension> index)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);

        auto& Vx = V(Component::X);
        auto& Vy = V(Component::Y);
        auto& Vz = V(Component::Z);

        auto&& [x, y, z] = rhoVToV(rho(index), rhoVx(index), rhoVy(index), rhoVz(index));
        Vx(index)        = x;
        Vy(index)        = y;
        Vz(index)        = z;
    }

    // cell-centered (vx, vy, vz, b1x, b1y, b1z) at index, shared by the read-only and floored
    // pressure recoveries.
    template<typename Field, typename VecField>
    static auto cellPrimitives_(Field const& rho, VecField const& rhoV, VecField const& B1,
                                MeshIndex<Field::dimension> index)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);
        auto const& B1x   = B1(Component::X);
        auto const& B1y   = B1(Component::Y);
        auto const& B1z   = B1(Component::Z);
        auto const vx  = rhoVx(index) / rho(index);
        auto const vy  = rhoVy(index) / rho(index);
        auto const vz  = rhoVz(index) / rho(index);
        auto const b1x = GridLayout::template project<GridLayout::faceXToCellCenter>(B1x, index);
        auto const b1y = GridLayout::template project<GridLayout::faceYToCellCenter>(B1y, index);
        auto const b1z = GridLayout::template project<GridLayout::faceZToCellCenter>(B1z, index);
        return std::array{vx, vy, vz, b1x, b1y, b1z};
    }

    template<typename Field, typename VecField>
    static auto recoverP_(double const gamma, Field const& rho, VecField const& rhoV,
                          VecField const& B1, Field const& Etot1, MeshIndex<Field::dimension> index)
    {
        auto const [vx, vy, vz, b1x, b1y, b1z] = cellPrimitives_(rho, rhoV, B1, index);
        return eosEtot1ToP(gamma, rho(index), vx, vy, vz, b1x, b1y, b1z, Etot1(index));
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
