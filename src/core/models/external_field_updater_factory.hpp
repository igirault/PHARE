#ifndef PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
#define PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP

#include "core/models/external_field_updater.hpp"
#include "core/models/external_field_updater_defs.hpp"
#include "core/models/external_field_updater_dipole.hpp"
#include "core/models/external_field_updater_none.hpp"
#include "core/utilities/point/point.hpp"

#include "initializer/data_provider.hpp"
#include "initializer/dict_utils.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace PHARE::core
{
/**
 * @brief Factory for the external field updater.
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterFactory
{
public:
    using Interface   = IExternalFieldUpdater<VecFieldT, GridLayoutT>;
    using point_type  = Interface::point_type;
    using vector_type = Point<double, 3>;

    static constexpr std::size_t dimension = GridLayoutT::dimension;

    ExternalFieldUpdaterFactory() = delete;

    /**
     * @brief build the updater described by @p parentDict[@p key]
     *
     * A simulation that prescribes no external field has no such key at all, in which case the
     * zero external field is used.
     */
    static std::unique_ptr<Interface> create(initializer::PHAREDict const& parentDict,
                                             std::string const& key)
    {
        if (!parentDict.contains(key))
            return std::make_unique<ExternalFieldUpdaterNone<VecFieldT, GridLayoutT>>();

        return create(parentDict[key]);
    }

    static std::unique_ptr<Interface> create(initializer::PHAREDict const& dict)
    {
        auto const type = cppdict::get_value(dict, "type", ExternalFieldUpdaterType::None);

        switch (type)
        {
            case ExternalFieldUpdaterType::None:
                return std::make_unique<ExternalFieldUpdaterNone<VecFieldT, GridLayoutT>>();

            case ExternalFieldUpdaterType::Dipole:
                return std::make_unique<ExternalFieldUpdaterDipole<VecFieldT, GridLayoutT>>(
                    point_type{initializer::parseDimXYZType<double, dimension>(dict, "position")},
                    vector_type{initializer::parseDimXYZType<double, 3>(dict, "moment")});

            case ExternalFieldUpdaterType::UserDefined:
                throw std::runtime_error(
                    "external field updater: 'user_defined' is not implemented yet");
        }

        throw std::runtime_error("external field updater: unknown type");
    }
};

} // namespace PHARE::core


#endif // PHARE_CORE_EXTERNAL_FIELD_UPDATER_FACTORY_HPP
