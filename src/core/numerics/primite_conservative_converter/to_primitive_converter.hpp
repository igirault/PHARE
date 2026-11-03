#ifndef PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP


#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/index/index.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/numerics/primite_conservative_converter/positivity_floors.hpp"

#include <cmath>

namespace PHARE::core
{
static auto const min_value = std::sqrt(1024 * std::numeric_limits<double>::min());

auto rhoVToV(auto& rho, auto const& rhoVx, auto const& rhoVy, auto const& rhoVz)
{
    auto const vx = rhoVx / rho;
    auto const vy = rhoVy / rho;
    auto const vz = rhoVz / rho;

    return std::make_tuple(vx, vy, vz);
}


template<typename GridLayout>
class ToPrimitiveConverter
{
    constexpr static auto dimension = GridLayout::dimension;
    using Scalar                    = GridLayout::Scalar;

public:
    ToPrimitiveConverter(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename Field, typename VecField>
    ClipCounts operator()(double const gamma, Field& rho, VecField const& rhoV, VecField const& B1,
                          Field& Etot1, VecField& V, Field& P,
                          PositivityFloors const& floors = {}) const
    {
        ClipCounts counts;

        if (floors.density > 0.)
            layout_.evalOnGhostBox(rho, [&](auto&... args) {
                MeshIndex<Field::dimension> const index{args...};
                if (!(rho(index) >= floors.density))
                {
                    rho(index) = floors.density;
                    counts.density += isInPhysicalBox_(rho, index);
                }
            });

        rhoVToVOnGhostBox(rho, rhoV, V);

        eosEtot1ToPOnGhostBox(gamma, rho, rhoV, B1, Etot1, P);

        if (floors.pressure > 0.)
            layout_.evalOnGhostBox(P, [&](auto&... args) {
                MeshIndex<Field::dimension> const index{args...};
                if (std::isfinite(P(index)) && P(index) >= floors.pressure)
                    return;
                P(index) = floors.pressure;
                rewriteEtot1_(gamma, rho, V, B1, P, Etot1, index);
                counts.pressure += isInPhysicalBox_(P, index);
            });

        return counts;
    }

    // used for diagnostics
    template<typename Field, typename VecField>
    void rhoVToVOnGhostBox(Field& rho, VecField const& rhoV, VecField& V) const
    {
        layout_.evalOnGhostBox(rho,
                               [&](auto&... args) mutable { rhoVToV_(rho, rhoV, V, {args...}); });
    }

    template<typename Field, typename VecField>
    void eosEtot1ToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                               VecField const& B1, Field const& Etot1, Field& P) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            eosEtot1ToP_(gamma, rho, rhoV, B1, Etot1, P, {args...});
        });
    }

private:
    template<typename Field>
    bool isInPhysicalBox_(Field const& field, MeshIndex<Field::dimension> const& index) const
    {
        for (std::size_t d = 0; d < Field::dimension; ++d)
        {
            auto const dir = static_cast<Direction>(d);
            if (index[d] < layout_.physicalStartIndex(field, dir)
                or index[d] > layout_.physicalEndIndex(field, dir))
                return false;
        }
        return true;
    }

    template<typename Field, typename VecField>
    static void rewriteEtot1_(double const gamma, Field const& rho, VecField const& V,
                              VecField const& B1, Field const& P, Field& Etot1,
                              MeshIndex<Field::dimension> index)
    {
        auto const b1x
            = GridLayout::template project<MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::rho>(
                B1(Component::X), index);
        auto const b1y
            = GridLayout::template project<MHDQuantity::Scalar::By, MHDQuantity::Scalar::rho>(
                B1(Component::Y), index);
        auto const b1z
            = GridLayout::template project<MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::rho>(
                B1(Component::Z), index);
        Etot1(index)
            = eosPToEtot1(gamma, rho(index), V(Component::X)(index), V(Component::Y)(index),
                          V(Component::Z)(index), b1x, b1y, b1z, P(index));
    }

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

    template<typename Field, typename VecField>
    static void eosEtot1ToP_(double const gamma, Field const& rho, VecField const& rhoV,
                             VecField const& B1, Field const& Etot1, Field& P,
                             MeshIndex<Field::dimension> index)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);

        auto const& B1x = B1(Component::X);
        auto const& B1y = B1(Component::Y);
        auto const& B1z = B1(Component::Z);

        auto const vx  = rhoVx(index) / rho(index);
        auto const vy  = rhoVy(index) / rho(index);
        auto const vz  = rhoVz(index) / rho(index);
        auto const b1x = GridLayout::template project<Scalar::Bx, Scalar::rho>(B1x, index);
        auto const b1y = GridLayout::template project<Scalar::By, Scalar::rho>(B1y, index);
        auto const b1z = GridLayout::template project<Scalar::Bz, Scalar::rho>(B1z, index);
        P(index)       = eosEtot1ToP(gamma, rho(index), vx, vy, vz, b1x, b1y, b1z, Etot1(index));
    }


private:
    GridLayout layout_;
};

} // namespace PHARE::core

#endif
