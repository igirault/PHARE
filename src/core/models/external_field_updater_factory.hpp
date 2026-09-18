#ifndef PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
#define PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP

#include "core/models/external_field_updater.hpp"
#include "core/models/external_field_updater_defs.hpp"
#include "core/models/external_field_updater_dipole.hpp"
#include "core/models/external_field_updater_none.hpp"

#include "initializer/data_provider.hpp"
#include "initializer/dict_utils.hpp"

#include <memory>
#include <stdexcept>

namespace PHARE::core
{
/**
 * @brief Factory for the external field updater.
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterFactory
{
public:
    using Interface  = IExternalFieldUpdater<VecFieldT, GridLayoutT>;
    using Dipole     = ExternalFieldUpdaterDipole<VecFieldT, GridLayoutT>;
    using None       = ExternalFieldUpdaterNone<VecFieldT, GridLayoutT>;
    using point_type = Interface::point_type;
    using value_type = Interface::value_type;

    static constexpr std::size_t dimension = GridLayoutT::dimension;

    ExternalFieldUpdaterFactory() = delete;

    static std::unique_ptr<Interface> create(initializer::PHAREDict const& dict)
    {
        auto const type = cppdict::get_value(dict, "type", ExternalFieldUpdaterType::None);

        switch (type)
        {
            case ExternalFieldUpdaterType::None: return std::make_unique<None>();

            case ExternalFieldUpdaterType::Dipole: {
                auto position
                    = point_type{initializer::parseDimXYZType<double, dimension>(dict, "position")};
                auto moment = typename Dipole::vector_type{
                    initializer::parseDimXYZType<value_type, dimension>(dict, "moment")};
                return std::make_unique<Dipole>(position, moment);
            }

            case ExternalFieldUpdaterType::UserDefined:
                throw std::runtime_error(
                    "external field updater: 'user-defined' is not implemented yet");
        }

        throw std::runtime_error("external field updater: unknown type");
    }
};

} // namespace PHARE::core


#endif // PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
