#ifndef PHARE_CORE_MODELS_EXTERNAL_FIELD_HPP
#define PHARE_CORE_MODELS_EXTERNAL_FIELD_HPP

#include "core/def.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <span>
#include <vector>

namespace PHARE::core
{
/**
 * @brief The external magnetic field and its time derivative.
 *
 * @tparam VecFieldT vecfield implementation
 *
 */
template<typename VecFieldT>
class ExternalField
{
public:
    using vecfield_type = VecFieldT;
    using tensor_type   = VecFieldT::tensor_t;

    static constexpr std::size_t dimension = VecFieldT::dimension;

    /**
     * @param name prefix of the resource names
     * @param withCoordinates allocate the node coordinates cache, needed by updaters evaluating
     * user functions on whole arrays of coordinates
     */
    explicit ExternalField(std::string const& name, bool withCoordinates)
        : B0{name + "_B0", tensor_type::B}
        , dB0dt{name + "_dB0dt", tensor_type::B}
        , scratch{name + "_scratch", tensor_type::E}
    {
        if (withCoordinates)
        {
            static constexpr std::array<char const*, 3> dirNames{"x", "y", "z"};
            for (std::size_t i = 0; i < dimension; ++i)
                coordinates_.emplace_back(name + "_coords_cache_" + dirNames[i], tensor_type::E);
        }
    }

    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const
    {
        return core::isUsable(B0, dB0dt, scratch)
               and std::ranges::all_of(coordinates_, [](auto const& c) { return c.isUsable(); });
    }

    NO_DISCARD bool isSettable() const
    {
        return core::isSettable(B0, dB0dt, scratch)
               and std::ranges::all_of(coordinates_, [](auto const& c) { return c.isSettable(); });
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(B0, dB0dt, scratch);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(B0, dB0dt, scratch);
    }

    // for the ResourcesManager only, which keeps references to the elements: never resize
    NO_DISCARD std::vector<VecFieldT> const& getRunTimeResourcesViewList() const
    {
        return coordinates_;
    }

    NO_DISCARD std::vector<VecFieldT>& getRunTimeResourcesViewList() { return coordinates_; }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------

    /**
     * @brief node coordinates cache, empty unless constructed withCoordinates: component c of
     * coordinates()[d] holds coordinate d at the nodes of the E-centered component c
     */
    NO_DISCARD std::span<VecFieldT const> coordinates() const { return coordinates_; }
    NO_DISCARD std::span<VecFieldT> coordinates() { return coordinates_; }

    VecFieldT B0;
    VecFieldT dB0dt;
    VecFieldT scratch; //<! E-centered work vecfield, holding the vector potential while B0 and
                       // dB0dt are computed

private:
    std::vector<VecFieldT> coordinates_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_HPP
