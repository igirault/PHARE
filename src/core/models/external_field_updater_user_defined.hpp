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
        : Super{derivative.has_value()}
        , potential_{std::move(potential)}
        , potential_time_derivative_{std::move(derivative)} {};

    virtual ~ExternalFieldUpdaterUserDefined() = default;

    void virtual computePotential(vecfield_type& a0, double time, GridLayoutT const& layout) final
    {
        UserFieldUpdater::update(a0, layout, potential_, time);
    };
    void virtual computePotentialTimeDerivative(vecfield_type& da0_dt, double time,
                                                GridLayoutT const& layout) final
    {
        if (potential_time_derivative_)
            UserFieldUpdater::update(da0_dt, layout, potential_time_derivative_.value(), time);
        else
            throw std::runtime_error(
                "computePotentialTimeDerivative called on a constant external field.");
    };

private:
    space_time_function_array_type potential_;
    std::optional<space_time_function_array_type> potential_time_derivative_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_USER_DEFINED_HPP
