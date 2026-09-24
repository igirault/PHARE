#ifndef PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_USER_DEFINED_HPP
#define PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_USER_DEFINED_HPP

#include "core/models/external_field_updater.hpp"
#include "core/data/user/user_field_updater.hpp"
#include "core/utilities/space_time_function.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace PHARE::core
{


/**
 * @brief External field updater  with the vector potential and its derivative with respect to time
 * specified by python user-defined functions.
 *
 * @tparam VecFieldT vecfield implementation
 * @tparam GridLayoutT grid layout implementation
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterUserDefined : public IExternalFieldUpdater<VecFieldT, GridLayoutT>
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    static constexpr std::size_t N         = VecFieldT::size();

    using Super                          = IExternalFieldUpdater<VecFieldT, GridLayoutT>;
    using vecfield_type                  = VecFieldT;
    using space_time_function_type       = SpaceTimeFunction<dimension>;
    using space_time_function_array_type = std::array<space_time_function_type, N>;

    ExternalFieldUpdaterUserDefined(space_time_function_array_type potential,
                                    std::optional<space_time_function_array_type> derivative
                                    = std::nullopt)
        : potential_{std::move(potential)}
        , potential_time_derivative_{std::move(derivative)} {};

    virtual ~ExternalFieldUpdaterUserDefined() = default;

    NO_DISCARD bool isTimeDependent() const final { return potential_time_derivative_.has_value(); }

    NO_DISCARD bool needsCoordinates() const final { return isTimeDependent(); }

    void virtual computePotential(vecfield_type& a0, double time, GridLayoutT const& layout,
                                  std::span<vecfield_type const> coordinates) final
    {
        evaluate_(a0, potential_, time, layout, coordinates);
    };

    void virtual computePotentialTimeDerivative(vecfield_type& da0_dt, double time,
                                                GridLayoutT const& layout,
                                                std::span<vecfield_type const> coordinates) final
    {
        if (!potential_time_derivative_)
            throw std::runtime_error(
                "computePotentialTimeDerivative called on a constant external field.");
        evaluate_(da0_dt, *potential_time_derivative_, time, layout, coordinates);
    };

private:
    space_time_function_array_type potential_;
    std::optional<space_time_function_array_type> potential_time_derivative_;

    /**
     * @brief evaluate the user functions from the coordinates cache when there is one, computing
     * the coordinates on the fly otherwise: a static field is evaluated once per patch lifetime,
     * so it gets no cache
     */
    void evaluate_(vecfield_type& vecfield, space_time_function_array_type const& funcs,
                   double time, GridLayoutT const& layout,
                   std::span<vecfield_type const> coordinates) const
    {
        // a time dependent field without its cache is a wiring bug: correct, but slow
        assert(!isTimeDependent() or !coordinates.empty());

        if (coordinates.empty())
            UserFieldUpdater::update(vecfield, layout, funcs, time);
        else
            UserFieldUpdater::update(vecfield, funcs, time, coordinates);
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_USER_DEFINED_HPP
