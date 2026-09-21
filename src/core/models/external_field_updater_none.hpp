#ifndef PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_NONE_HPP
#define PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_NONE_HPP

#include "core/models/external_field_updater.hpp"

#include <cassert>
#include <cstddef>

namespace PHARE::core
{

/**
 * @brief Zero external field.
 *
 * @tparam VecFieldT vecfield implementation
 * @tparam GridLayoutT grid layout implementation
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterNone : public IExternalFieldUpdater<VecFieldT, GridLayoutT>
{
public:
    using Super               = IExternalFieldUpdater<VecFieldT, GridLayoutT>;
    using vecfield_type       = Super::vecfield_type;
    using value_type          = Super::value_type;
    using point_type          = Super::point_type;
    using component_type      = Super::component_type;
    using external_field_type = Super::external_field_type;

    static constexpr std::size_t dimension = GridLayoutT::dimension;

    ExternalFieldUpdaterNone()
        : Super(false) {};

    virtual ~ExternalFieldUpdaterNone() = default;

    void virtual operator()(external_field_type& externalField, GridLayoutT const& layout,
                            double time) final
    {
        externalField.B0.zero();
        externalField.dB0dt.zero();
    }

    void virtual computePotential(vecfield_type& a0, double time,
                                  GridLayoutT const& layout) final{};
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_NONE_HPP
