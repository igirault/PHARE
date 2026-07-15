#ifndef PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_ELECTROMAG_HPP
#define PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_ELECTROMAG_HPP

#include "diagnostic/detail/vtkh5_type_writer.hpp"

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace PHARE::diagnostic::vtkh5
{

template<typename H5Writer>
class ElectromagDiagnosticWriter : public H5TypeWriter<H5Writer>
{
    using Super              = H5TypeWriter<H5Writer>;
    using VTKFileWriter      = Super::VTKFileWriter;
    using VTKFileInitializer = Super::VTKFileInitializer;

public:
    ElectromagDiagnosticWriter(H5Writer& h5Writer)
        : Super{h5Writer}
    {
    }

    void setup(DiagnosticProperties&) override;
    void write(DiagnosticProperties&) override;
    void compute(DiagnosticProperties&) override {}

private:
    struct Info
    {
        std::vector<std::size_t> offset_per_level = std::vector<std::size_t>(amr::MAX_LEVEL_IDX);
    };

    std::unordered_map<std::string, Info> mem;
};


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::setup(DiagnosticProperties& diagnostic)
{
    auto& modelView = this->h5Writer_.modelView();
    VTKFileInitializer initializer{diagnostic, this};

    if (mem.count(diagnostic.quantity) == 0)
        mem.try_emplace(diagnostic.quantity);
    auto& info = mem[diagnostic.quantity];

    auto const init = [&](auto const& level) -> std::optional<std::size_t> {
        std::optional<std::size_t> ret;

        modelView.forEachEmQuantity(
            [&](auto const& q) {
                if (!ret and diagnostic.quantity == q.path())
                    ret = initializer.initFieldFileLevel(level);
            },
            [&](auto const& q) {
                if (!ret and diagnostic.quantity == q.path())
                    ret = initializer.template initTensorFieldFileLevel<1>(level);
            });

        return ret;
    };

    modelView.onLevels(
        [&](auto const& level) {
            auto const ilvl = level.getLevelNumber();
            if (auto const offset = init(ilvl))
                info.offset_per_level[ilvl] = *offset;
        },
        [&](int const ilvl) { // missing level
            init(ilvl);
        },
        this->h5Writer_.minLevel, this->h5Writer_.maxLevel);
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    auto& modelView = this->h5Writer_.modelView();
    auto& info      = mem[diagnostic.quantity];

    modelView.onLevels(
        [&](auto const& level) {
            auto const ilvl = level.getLevelNumber();

            VTKFileWriter writer{diagnostic, this, info.offset_per_level[ilvl]};

            auto const write_quantity = [&](auto& layout, auto const&, auto const) {
                PHARE_LOG_SCOPE(3, "ElectromagDiagnosticWriter<H5Writer>::write_quantity");

                modelView.visitActiveEmQuantity(
                    diagnostic.quantity, layout, this->h5Writer_.timestamp(),
                    /*compute_derived=*/true, //
                    [&](auto const&, auto& field) { writer.writeField(field, layout); },
                    [&](auto const&, auto& vecF) {
                        writer.template writeTensorField<1>(vecF, layout);
                    });
            };

            modelView.visitHierarchy(write_quantity, ilvl, ilvl);
        },
        this->h5Writer_.minLevel, this->h5Writer_.maxLevel);
}


} // namespace PHARE::diagnostic::vtkh5

#endif /* PHARE_DIAGNOSTIC_DETAIL_VTK_TYPES_ELECTROMAG_HPP */
