#ifndef PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
#define PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP

#include "core/models/external_field_updater.hpp"
#include "core/models/external_field_updater_defs.hpp"
#include "core/models/external_field_updater_dipole.hpp"
#include "core/models/external_field_updater_zero.hpp"
#include "core/utilities/space_time_function.hpp"

#include "external_field_updater_user_defined.hpp"
#include "initializer/data_provider.hpp"
#include "initializer/dict_utils.hpp"

#include <memory>
#include <string>
#include <optional>
#include <stdexcept>

namespace PHARE::core
{
/**
 * @brief Factory for the external field updater.
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterFactory
{
private:
    using Interface   = IExternalFieldUpdater<VecFieldT, GridLayoutT>;
    using Dipole      = ExternalFieldUpdaterDipole<VecFieldT, GridLayoutT>;
    using Zero        = ExternalFieldUpdaterZero<VecFieldT, GridLayoutT>;
    using UserDefined = ExternalFieldUpdaterUserDefined<VecFieldT, GridLayoutT>;

public:
    using point_type               = Interface::point_type;
    using value_type               = Interface::value_type;
    using space_time_function_type = SpaceTimeFunction<GridLayoutT::dimension>;

    static constexpr std::size_t dimension = GridLayoutT::dimension;
    static constexpr std::size_t N         = VecFieldT::size();

    ExternalFieldUpdaterFactory() = delete;

    static std::unique_ptr<Interface> createZero() { return std::make_unique<Zero>(); }

    static std::unique_ptr<Interface> create(initializer::PHAREDict const& dict)
    {
        auto const type = cppdict::get_value(dict, "type", ExternalFieldUpdaterType::Zero);

        switch (type)
        {
            case ExternalFieldUpdaterType::Zero: return std::make_unique<Zero>();

            case ExternalFieldUpdaterType::Dipole: {
                auto position
                    = point_type{initializer::parseDimXYZType<double, dimension>(dict, "position")};
                auto moment = typename Dipole::vector_type{
                    initializer::parseDimXYZType<value_type, dimension>(dict, "moment")};
                return std::make_unique<Dipole>(position, moment);
            }

            case ExternalFieldUpdaterType::UserDefined: {
                auto potential
                    = initializer::parseDimXYZType<space_time_function_type, N>(dict, "potential");

                std::optional<std::array<space_time_function_type, N>> derivative;
                if (dict["is_time_dependent"].template to<bool>())
                    derivative = initializer::parseDimXYZType<space_time_function_type, N>(
                        dict, "potential_time_derivative");

                return std::make_unique<UserDefined>(std::move(potential), std::move(derivative));
            }
        }
        throw std::runtime_error("external field updater: unknown type");
    }
};

} // namespace PHARE::core


#endif // PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
