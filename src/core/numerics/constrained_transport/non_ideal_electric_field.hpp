#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_ELECTRIC_FIELD_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_ELECTRIC_FIELD_HPP

#include "core/def.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/numerics/differential_operators.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/meta/meta_utilities.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <cmath>
#include <type_traits>
#include <vector>
#include <algorithm>

namespace PHARE::core
{

template<typename VecField>
class NonIdealElectricFieldState
{
public:
    NonIdealElectricFieldState() = default;
    explicit NonIdealElectricFieldState(bool const isNonIdeal)
    {
        if (isNonIdeal)
            E_.emplace_back("E_nonideal", MHDQuantity::Vector::E);
    }

    NO_DISCARD std::vector<VecField>& getRunTimeResourcesViewList() { return E_; }
    NO_DISCARD std::vector<VecField> const& getRunTimeResourcesViewList() const { return E_; }

    NO_DISCARD bool active() const { return !E_.empty(); }

    NO_DISCARD auto& E() { return E_[0]; }
    NO_DISCARD auto const& E() const { return E_[0]; }

private:
    std::vector<VecField> E_;
};


template<typename GridLayout, bool Hall>
class NonIdealElectricField : public OhmInfo
{
    using Super                     = OhmInfo;
    constexpr static auto dimension = GridLayout::dimension;

public:
    using Info_t = Super;

    NonIdealElectricField(OhmInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
    {
    }

    void operator()(auto& non_ideal_electric_state, auto const& mhd_state, auto const& B0) const
    {
        auto& E         = non_ideal_electric_state.E();
        auto const& J   = mhd_state.J;
        auto const& B1  = mhd_state.B1;
        auto const& rho = mhd_state.rho;

        Constexprifier{isResistive(), isHyperResistive(),
                       hyper_mode}([&]<bool isResistive, bool isHyperResistive, HyperMode hyper>() {
            for_N<3>([&](auto i) {
                constexpr auto component = static_cast<Component>(i());
                auto& Ec                 = E(component);
                auto const& Jc           = J(component);
                layout_.evalOnBox(Ec, [&](auto&... args) {
                    MeshIndex<dimension> idx{args...};
                    double e = 0.;
                    if constexpr (isResistive)
                        e += eta * Jc(idx);
                    if constexpr (isHyperResistive)
                        e -= hyper_coef_<component, hyper>(B1, B0, rho, idx)
                             * laplacian(layout_, Jc, idx);
                    if constexpr (Hall)
                        e += hall_<component>(J, B1, B0, rho, idx);
                    Ec(idx) = e;
                });
            });
        });
    }

private:
    template<auto srcJ, auto face, typename JField, typename B1Field, typename B0Field,
             typename RhoField>
    struct FaceProduct
    {
        using type                      = double;
        constexpr static auto dimension = GridLayout::dimension;

        NO_DISCARD double operator()(MeshIndex<dimension> const& p) const
        {
            return GridLayout::template project<srcJ, face>(j, p) * (b1(p) + b0(p))
                   / GridLayout::template project<MHDQuantity::Scalar::rho, face>(rho, p);
        }

        JField const& j;
        B1Field const& b1;
        B0Field const& b0;
        RhoField const& rho;
    };

    template<auto dst, auto srcJ, auto face>
    static double pair_(auto const& Ja, auto const& B1b, auto const& B0b, auto const& rho,
                        MeshIndex<dimension> idx)
    {
        FaceProduct<srcJ, face, std::decay_t<decltype(Ja)>, std::decay_t<decltype(B1b)>,
                    std::decay_t<decltype(B0b)>, std::decay_t<decltype(rho)>> const product{
            Ja, B1b, B0b, rho};
        return core::project<GridLayout::centering(face), GridLayout::centering(dst)>(product, idx);
    }

    template<auto component>
    double hall_(auto const& J, auto const& B1, auto const& B0, auto const& rho,
                 MeshIndex<dimension> idx) const
    {
        using Scalar = MHDQuantity::Scalar;

        auto const& Jx  = J(Component::X);
        auto const& Jy  = J(Component::Y);
        auto const& Jz  = J(Component::Z);
        auto const& B1x = B1(Component::X);
        auto const& B1y = B1(Component::Y);
        auto const& B1z = B1(Component::Z);
        auto const& B0x = B0(Component::X);
        auto const& B0y = B0(Component::Y);
        auto const& B0z = B0(Component::Z);

        if constexpr (component == Component::X)
            return pair_<Scalar::Ex, Scalar::Jy, Scalar::Bz>(Jy, B1z, B0z, rho, idx)
                   - pair_<Scalar::Ex, Scalar::Jz, Scalar::By>(Jz, B1y, B0y, rho, idx);
        else if constexpr (component == Component::Y)
            return pair_<Scalar::Ey, Scalar::Jz, Scalar::Bx>(Jz, B1x, B0x, rho, idx)
                   - pair_<Scalar::Ey, Scalar::Jx, Scalar::Bz>(Jx, B1z, B0z, rho, idx);
        else
            return pair_<Scalar::Ez, Scalar::Jx, Scalar::By>(Jx, B1y, B0y, rho, idx)
                   - pair_<Scalar::Ez, Scalar::Jy, Scalar::Bx>(Jy, B1x, B0x, rho, idx);
    }

    template<auto component, HyperMode hyper>
    double hyper_coef_(auto const& B1, auto const& B0, auto const& rho,
                       MeshIndex<dimension> idx) const
    {
        if constexpr (hyper == HyperMode::constant)
            return nu;
        else
        {
            auto const& meshSize = layout_.meshSize();
            auto const dx        = *std::min_element(meshSize.begin(), meshSize.end());

            using Scalar = MHDQuantity::Scalar;

            auto coef = [&]<auto dst>() {
                auto const bx
                    = GridLayout::template project<Scalar::Bx, dst>(B1(Component::X), idx)
                      + GridLayout::template project<Scalar::Bx, dst>(B0(Component::X), idx);
                auto const by
                    = GridLayout::template project<Scalar::By, dst>(B1(Component::Y), idx)
                      + GridLayout::template project<Scalar::By, dst>(B0(Component::Y), idx);
                auto const bz
                    = GridLayout::template project<Scalar::Bz, dst>(B1(Component::Z), idx)
                      + GridLayout::template project<Scalar::Bz, dst>(B0(Component::Z), idx);
                auto const n = GridLayout::template project<Scalar::rho, dst>(rho, idx);
                auto const b = std::sqrt(bx * bx + by * by + bz * bz);
                return nu * dx * dx * (b / n + 1);
            };

            if constexpr (component == Component::X)
                return coef.template operator()<MHDQuantity::Scalar::Ex>();
            else if constexpr (component == Component::Y)
                return coef.template operator()<MHDQuantity::Scalar::Ey>();
            else
                return coef.template operator()<MHDQuantity::Scalar::Ez>();
        }
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
