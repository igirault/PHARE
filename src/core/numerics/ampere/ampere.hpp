#ifndef PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"


namespace PHARE::core
{
template<typename GridLayout>
class Ampere_ref;

template<typename GridLayout>
class Ampere : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename VecField>
    void operator()(VecField const& B, VecField& J)
    {
        if (!this->hasLayout())
            throw std::runtime_error(
                "Error - Ampere - GridLayout not set, cannot proceed to calculate ampere()");

        Ampere_ref{*this->layout_}(B, J);
    }

    // total-field overload: J = curl(B1 + B0). Uses linearity of the curl so no
    // temporary total field is needed.
    template<typename VecField>
    void operator()(VecField const& B1, VecField const& B0, VecField& J)
    {
        if (!this->hasLayout())
            throw std::runtime_error(
                "Error - Ampere - GridLayout not set, cannot proceed to calculate ampere()");

        Ampere_ref{*this->layout_}(B1, B0, J);
    }
};

template<typename GridLayout>
class Ampere_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    Ampere_ref(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename VecField>
    void operator()(VecField const& B, VecField& J) const
    {
        // can't use structured bindings because
        //   "reference to local binding declared in enclosing function"
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        Point<std::uint32_t, dimension> shrink;

        for (size_t i = 0; i < dimension; ++i)
        {
            shrink[i] = 1;
        }

        layout_.evalOnShrinkedGhostBox(Jx, shrink,
                                       [&](auto&... args) mutable { JxEq_(Jx, B, args...); });
        layout_.evalOnShrinkedGhostBox(Jy, shrink,
                                       [&](auto&... args) mutable { JyEq_(Jy, B, args...); });
        layout_.evalOnShrinkedGhostBox(Jz, shrink,
                                       [&](auto&... args) mutable { JzEq_(Jz, B, args...); });
    }

    // total-field curl: J = curl(B1 + B0)
    template<typename VecField>
    void operator()(VecField const& B1, VecField const& B0, VecField& J) const
    {
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        Point<std::uint32_t, dimension> shrink;

        for (size_t i = 0; i < dimension; ++i)
        {
            shrink[i] = 1;
        }

        layout_.evalOnShrinkedGhostBox(
            Jx, shrink, [&](auto&... args) mutable { JxEq_(Jx, B1, B0, args...); });
        layout_.evalOnShrinkedGhostBox(
            Jy, shrink, [&](auto&... args) mutable { JyEq_(Jy, B1, B0, args...); });
        layout_.evalOnShrinkedGhostBox(
            Jz, shrink, [&](auto&... args) mutable { JzEq_(Jz, B1, B0, args...); });
    }


private:
    GridLayout layout_;

    template<typename VecField, typename Field, typename... Indexes>
    void JxEq_(Field& Jx, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [_, By, Bz] = B();

        if constexpr (dimension == 1)
            Jx(ijk...) = 0.0;

        if constexpr (dimension == 2)
            Jx(ijk...) = layout_.template deriv<Direction::Y>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jx(ijk...) = layout_.template deriv<Direction::Y>(Bz, {ijk...})
                         - layout_.template deriv<Direction::Z>(By, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JyEq_(Field& Jy, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1 || dimension == 2)
            Jy(ijk...) = -layout_.template deriv<Direction::X>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jy(ijk...) = layout_.template deriv<Direction::Z>(Bx, {ijk...})
                         - layout_.template deriv<Direction::X>(Bz, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JzEq_(Field& Jz, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1)
            Jz(ijk...) = layout_.template deriv<Direction::X>(By, {ijk...});

        else
            Jz(ijk...) = layout_.template deriv<Direction::X>(By, {ijk...})
                         - layout_.template deriv<Direction::Y>(Bx, {ijk...});
    }

    // ---- total-field variants: deriv of (B1 + B0) via curl linearity ----

    template<typename VecField, typename Field, typename... Indexes>
    void JxEq_(Field& Jx, VecField const& B1, VecField const& B0, Indexes const&... ijk) const
    {
        auto const& [b1x_, B1y, B1z] = B1();
        auto const& [b0x_, B0y, B0z] = B0();

        if constexpr (dimension == 1)
            Jx(ijk...) = 0.0;

        if constexpr (dimension == 2)
            Jx(ijk...) = layout_.template deriv<Direction::Y>(B1z, {ijk...})
                         + layout_.template deriv<Direction::Y>(B0z, {ijk...});

        if constexpr (dimension == 3)
            Jx(ijk...) = layout_.template deriv<Direction::Y>(B1z, {ijk...})
                         + layout_.template deriv<Direction::Y>(B0z, {ijk...})
                         - layout_.template deriv<Direction::Z>(B1y, {ijk...})
                         - layout_.template deriv<Direction::Z>(B0y, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JyEq_(Field& Jy, VecField const& B1, VecField const& B0, Indexes const&... ijk) const
    {
        auto const& [B1x, B1y, B1z] = B1();
        auto const& [B0x, B0y, B0z] = B0();

        if constexpr (dimension == 1 || dimension == 2)
            Jy(ijk...) = -layout_.template deriv<Direction::X>(B1z, {ijk...})
                         - layout_.template deriv<Direction::X>(B0z, {ijk...});

        if constexpr (dimension == 3)
            Jy(ijk...) = layout_.template deriv<Direction::Z>(B1x, {ijk...})
                         + layout_.template deriv<Direction::Z>(B0x, {ijk...})
                         - layout_.template deriv<Direction::X>(B1z, {ijk...})
                         - layout_.template deriv<Direction::X>(B0z, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JzEq_(Field& Jz, VecField const& B1, VecField const& B0, Indexes const&... ijk) const
    {
        auto const& [B1x, B1y, B1z] = B1();
        auto const& [B0x, B0y, B0z] = B0();

        if constexpr (dimension == 1)
            Jz(ijk...) = layout_.template deriv<Direction::X>(B1y, {ijk...})
                         + layout_.template deriv<Direction::X>(B0y, {ijk...});

        else
            Jz(ijk...) = layout_.template deriv<Direction::X>(B1y, {ijk...})
                         + layout_.template deriv<Direction::X>(B0y, {ijk...})
                         - layout_.template deriv<Direction::Y>(B1x, {ijk...})
                         - layout_.template deriv<Direction::Y>(B0x, {ijk...});
    }
};

} // namespace PHARE::core
#endif
