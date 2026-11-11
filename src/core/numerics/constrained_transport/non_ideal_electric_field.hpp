#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_ELECTRIC_FIELD_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_ELECTRIC_FIELD_HPP

#include "core/def.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/numerics/differential_operators.hpp"
#include "core/numerics/hyper_resistivity.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/field/field_scratch.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/meta/meta_utilities.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <array>
#include <cmath>
#include <cstddef>
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

    void operator()(auto& non_ideal_electric_state, auto const& mhd_state) const
    {
        auto& E         = non_ideal_electric_state.E();
        auto const& J   = mhd_state.J;
        auto const& B   = mhd_state.B;
        auto const& rho = mhd_state.rho;

        Constexprifier{isResistive(), isHyperResistive(),
                       hyper_mode}([&]<bool isResistive, bool isHyperResistive, HyperMode hyper>() {
            auto compute = [&](auto const& kappa) {
                for_N<3>([&](auto i) {
                    constexpr auto component = static_cast<Component>(i());
                    auto& Ec                 = E(component);
                    auto const& Jc           = J(component);
                    layout_.evalOnBox(Ec, [&](auto&... args) {
                        MeshIndex<dimension> idx{args...};
                        double e = 0.;
                        if constexpr (isResistive)
                            e += eta * Jc(idx);
                        if constexpr (isHyperResistive and hyper == HyperMode::constant)
                            e -= nu * laplacian(layout_, Jc, idx);
                        if constexpr (isHyperResistive and hyper == HyperMode::spatial)
                            e += hyperResistiveDivergence<component,
                                                          MHDQuantity::Scalar::ScalarAllPrimal>(
                                layout_, Jc, kappa, idx);
                        if constexpr (Hall)
                            e += hall_<component>(J, B, rho, idx);
                        Ec(idx) = e;
                    });
                });
            };

            if constexpr (isHyperResistive and hyper == HyperMode::spatial)
            {
                ScratchWrite scratch;
                compute(cachedKappa_(scratch, B, rho));
            }
            else
                compute(0);
        });
    }

private:
    template<typename BField, typename RhoField>
    struct HyperKappa
    {
        template<auto dst>
        NO_DISCARD PHARE_NOINLINE double operator()(MeshIndex<dimension> const& p) const
        {
            using Scalar  = MHDQuantity::Scalar;
            auto const bx = GridLayout::template project<Scalar::Bx, dst>(B(Component::X), p);
            auto const by = GridLayout::template project<Scalar::By, dst>(B(Component::Y), p);
            auto const bz = GridLayout::template project<Scalar::Bz, dst>(B(Component::Z), p);
            auto const n  = GridLayout::template project<Scalar::rho, dst>(rho, p);
            return coef * (std::sqrt(bx * bx + by * by + bz * bz) / n + 1);
        }

        double coef;
        BField const& B;
        RhoField const& rho;
    };

    struct CachedKappa
    {
        template<auto dst>
        NO_DISCARD double operator()(MeshIndex<dimension> const& p) const
        {
            using Scalar = MHDQuantity::Scalar;
            if constexpr (dst == Scalar::ScalarAllPrimal)
                return node(p);
            else if constexpr (dst == Scalar::Bx)
                return bx(p);
            else if constexpr (dst == Scalar::By)
                return by(p);
            else
            {
                static_assert(dst == Scalar::Bz);
                return bz(p);
            }
        }

        NdArrayView<dimension, double> node, bx, by, bz;
    };

    template<auto location>
    void fillKappa_(NdArrayView<dimension, double>& view, auto const& kappa) const
    {
        constexpr auto centering = GridLayout::centering(location);
        Box<std::size_t, dimension> box;
        for (std::size_t d = 0; d < dimension; ++d)
        {
            box.lower[d] = layout_.ghostStartIndex(centering[d], static_cast<Direction>(d)) + 1;
            box.upper[d] = layout_.ghostEndIndex(centering[d], static_cast<Direction>(d)) - 1;
        }
        for_each_point(box,
                       [&](auto const& p) { view(p) = kappa.template operator()<location>(p); });
    }

    CachedKappa cachedKappa_(ScratchWrite& scratch, auto const& B, auto const& rho) const
    {
        using Scalar = MHDQuantity::Scalar;

        auto const& meshSize = layout_.meshSize();
        auto const dx        = *std::min_element(meshSize.begin(), meshSize.end());
        HyperKappa<std::decay_t<decltype(B)>, std::decay_t<decltype(rho)>> const kappa{nu * dx * dx,
                                                                                       B, rho};

        auto const nodeShape = layout_.allocSize(Scalar::ScalarAllPrimal);
        auto const bxShape   = layout_.allocSize(Scalar::Bx);
        auto const byShape   = layout_.allocSize(Scalar::By);
        auto const bzShape   = layout_.allocSize(Scalar::Bz);

        std::size_t const nodeSize = product(nodeShape, std::size_t{1});
        std::size_t const bxSize   = product(bxShape, std::size_t{1});
        std::size_t const bySize   = product(byShape, std::size_t{1});
        std::size_t const bzSize   = product(bzShape, std::size_t{1});

        auto* data = scratch.get().get_no_copy(nodeSize + bxSize + bySize + bzSize).data();

        CachedKappa cache{
            NdArrayView<dimension, double>{data, nodeShape},
            NdArrayView<dimension, double>{data + nodeSize, bxShape},
            NdArrayView<dimension, double>{data + nodeSize + bxSize, byShape},
            NdArrayView<dimension, double>{data + nodeSize + bxSize + bySize, bzShape}};

        fillKappa_<Scalar::ScalarAllPrimal>(cache.node, kappa);
        fillKappa_<Scalar::Bx>(cache.bx, kappa);
        fillKappa_<Scalar::By>(cache.by, kappa);
        fillKappa_<Scalar::Bz>(cache.bz, kappa);
        return cache;
    }

    template<auto srcJ, auto face, typename JField, typename BField, typename RhoField>
    struct FaceProduct
    {
        using type                      = double;
        constexpr static auto dimension = GridLayout::dimension;

        NO_DISCARD double operator()(MeshIndex<dimension> const& p) const
        {
            return GridLayout::template project<srcJ, face>(j, p) * b(p)
                   / GridLayout::template project<MHDQuantity::Scalar::rho, face>(rho, p);
        }

        JField const& j;
        BField const& b;
        RhoField const& rho;
    };

    template<auto dst, auto srcJ, auto face>
    static double pair_(auto const& Ja, auto const& Bb, auto const& rho, MeshIndex<dimension> idx)
    {
        FaceProduct<srcJ, face, std::decay_t<decltype(Ja)>, std::decay_t<decltype(Bb)>,
                    std::decay_t<decltype(rho)>> const product{Ja, Bb, rho};
        return core::project<GridLayout::centering(face), GridLayout::centering(dst)>(product, idx);
    }

    template<auto component>
    double hall_(auto const& J, auto const& B, auto const& rho, MeshIndex<dimension> idx) const
    {
        using Scalar = MHDQuantity::Scalar;

        auto const& Jx = J(Component::X);
        auto const& Jy = J(Component::Y);
        auto const& Jz = J(Component::Z);
        auto const& Bx = B(Component::X);
        auto const& By = B(Component::Y);
        auto const& Bz = B(Component::Z);

        if constexpr (component == Component::X)
            return pair_<Scalar::Ex, Scalar::Jy, Scalar::Bz>(Jy, Bz, rho, idx)
                   - pair_<Scalar::Ex, Scalar::Jz, Scalar::By>(Jz, By, rho, idx);
        else if constexpr (component == Component::Y)
            return pair_<Scalar::Ey, Scalar::Jz, Scalar::Bx>(Jz, Bx, rho, idx)
                   - pair_<Scalar::Ey, Scalar::Jx, Scalar::Bz>(Jx, Bz, rho, idx);
        else
            return pair_<Scalar::Ez, Scalar::Jx, Scalar::By>(Jx, By, rho, idx)
                   - pair_<Scalar::Ez, Scalar::Jy, Scalar::Bx>(Jy, Bx, rho, idx);
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
