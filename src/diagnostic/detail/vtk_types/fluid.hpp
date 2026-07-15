#ifndef PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_FLUID_HPP
#define PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_FLUID_HPP

#include "core/logger.hpp"

#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/hybrid_model.hpp"
#include "diagnostic/detail/vtkh5_type_writer.hpp"

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace PHARE::diagnostic::vtkh5
{

template<typename H5Writer>
class FluidDiagnosticWriter : public H5TypeWriter<H5Writer>
{
    using Super              = H5TypeWriter<H5Writer>;
    using VTKFileWriter      = Super::VTKFileWriter;
    using VTKFileInitializer = Super::VTKFileInitializer;
    using GridLayout         = H5Writer::GridLayout;

public:
    FluidDiagnosticWriter(H5Writer& h5Writer)
        : Super{h5Writer}
    {
    }

    void setup(DiagnosticProperties&) override;
    void write(DiagnosticProperties&) override;
    void compute(DiagnosticProperties&) override;

private:
    struct Info
    {
        std::vector<std::size_t> offset_per_level
            = std::vector<std::size_t>(amr::MAX_LEVEL_IDX + 1);
    };

    struct FluidInitializer
    {
        std::optional<std::size_t> operator()(auto const ilvl);

        FluidDiagnosticWriter* writer;
        DiagnosticProperties& diagnostic;
        VTKFileInitializer& file_initializer;
    };

    struct FluidWriter
    {
        void operator()(auto const& layout);

        FluidDiagnosticWriter* writer;
        DiagnosticProperties& diagnostic;
        VTKFileWriter& file_writer;
    };

    std::unordered_map<std::string, Info> mem;
};



template<typename H5Writer>
std::optional<std::size_t>
FluidDiagnosticWriter<H5Writer>::FluidInitializer::operator()(auto const ilvl)
{
    auto& modelView = writer->h5Writer_.modelView();
    std::optional<std::size_t> ret;

    modelView.forEachFluidQuantity(
        [&](auto const& q) {
            if (!ret and diagnostic.quantity == q.path())
                ret = file_initializer.initFieldFileLevel(ilvl);
        },
        [&](auto const& q) {
            if (!ret and diagnostic.quantity == q.path())
                ret = file_initializer.template initTensorFieldFileLevel<1>(ilvl);
        },
        [&](auto const&) { /* rank-2 (momentum_tensor) not supported by VTKHDF writer */ });

    return ret;
}


template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::setup(DiagnosticProperties& diagnostic)
{
    PHARE_LOG_SCOPE(3, "FluidDiagnosticWriter<H5Writer>::setup");

    auto& modelView = this->h5Writer_.modelView();

    VTKFileInitializer initializer{diagnostic, this};

    if (mem.count(diagnostic.quantity) == 0)
        mem.try_emplace(diagnostic.quantity);
    auto& info = mem[diagnostic.quantity];

    auto const init = [&](auto const ilvl) -> std::optional<std::size_t> {
        return FluidInitializer{this, diagnostic, initializer}(ilvl);
    };

    modelView.onLevels(
        [&](auto const& level) {
            PHARE_LOG_SCOPE(3, "FluidDiagnosticWriter<H5Writer>::setup_level");

            auto const ilvl = level.getLevelNumber();
            if (auto const offset = init(ilvl))
                info.offset_per_level[ilvl] = *offset;
        },
        [&](int const ilvl) {
            PHARE_LOG_SCOPE(3, "FluidDiagnosticWriter<H5Writer>::setup_missing_level");

            init(ilvl);
        },
        this->h5Writer_.minLevel, this->h5Writer_.maxLevel);
}



template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::FluidWriter::operator()(auto const& layout)
{
    auto& modelView = writer->h5Writer_.modelView();

    modelView.visitActiveFluidQuantity(
        diagnostic.quantity, layout, writer->h5Writer_.timestamp(),
        /*compute_derived=*/true, //
        [&](auto const&, auto& field) { file_writer.writeField(field, layout); },
        [&](auto const&, auto& vecF) { file_writer.template writeTensorField<1>(vecF, layout); },
        [&](auto const&, auto&) { /* rank-2 (momentum_tensor) not supported by VTKHDF writer */ });
}



template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    PHARE_LOG_SCOPE(3, "FluidDiagnosticWriter<H5Writer>::write");

    auto& modelView  = this->h5Writer_.modelView();
    auto const& info = mem[diagnostic.quantity];

    modelView.onLevels(
        [&](auto const& level) {
            auto const ilvl = level.getLevelNumber();

            VTKFileWriter writer{diagnostic, this, info.offset_per_level[ilvl]};

            auto const write_quantity = [&](auto& layout, auto const&, auto const) {
                FluidWriter{this, diagnostic, writer}(layout);
            };

            modelView.visitHierarchy(write_quantity, ilvl, ilvl);
        },
        this->h5Writer_.minLevel, this->h5Writer_.maxLevel);
}



template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::compute(DiagnosticProperties&)
{
    // derived quantities are computed per patch during write(), into scratch views
}

} // namespace PHARE::diagnostic::vtkh5



#endif /* PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_FLUID_HPP */
