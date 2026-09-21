#ifndef PHARE_TEST_CORE_MODELS_TEST_EXTERNAL_FIELD_FIXTURES_HPP
#define PHARE_TEST_CORE_MODELS_TEST_EXTERNAL_FIELD_FIXTURES_HPP

#include "core/models/external_field.hpp"

#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include <string>

namespace PHARE::core
{
/**
 * @brief an ExternalField that owns the memory of its vecfields, for tests.
 *
 * Mirrors what the resources manager does on a patch: the fields of an ExternalField are views,
 * whose buffers are set here from grids owned by this fixture.
 */
template<std::size_t dim>
class UsableExternalField : public ExternalField<VecFieldMHD<dim>>
{
public:
    using Super = ExternalField<VecFieldMHD<dim>>;

    template<typename GridLayout>
    UsableExternalField(std::string const& name, GridLayout const& layout)
        : Super{name}
        , b0_{name + "_B0", layout, MHDQuantity::Vector::B}
        , dB0dt_{name + "_dB0dt", layout, MHDQuantity::Vector::B}
        , scratch_{name + "_scratch", layout, MHDQuantity::Vector::E}
    {
        b0_.set_on(this->B0);
        dB0dt_.set_on(this->dB0dt);
        scratch_.set_on(this->scratch);
    }

    Super& super() { return *this; }

private:
    UsableVecFieldMHD<dim> b0_;
    UsableVecFieldMHD<dim> dB0dt_;
    UsableVecFieldMHD<dim> scratch_;
};

} // namespace PHARE::core

#endif /* PHARE_TEST_CORE_MODELS_TEST_EXTERNAL_FIELD_FIXTURES_HPP */
