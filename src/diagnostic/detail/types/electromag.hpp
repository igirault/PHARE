#ifndef PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP
#define PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP

#include "core/data/vecfield/vecfield_component.hpp"

#include "diagnostic/detail/h5typewriter.hpp"

#include <stdexcept>

namespace PHARE::diagnostic::h5
{
/*
 * Possible outputs
 *
 * /t#/pl#/p#/electromag_(B, E)/(x,y,z)
 */
template<typename H5Writer>
class ElectromagDiagnosticWriter : public H5TypeWriter<H5Writer>
{
public:
    using Super = H5TypeWriter<H5Writer>;
    using Super::checkCreateFileFor_;
    using Super::fileData_;
    using Super::h5Writer_;
    using Super::initDataSets_;
    using Super::writeAttributes_;
    using Super::writeGhostsAttr_;
    using Attributes = Super::Attributes;
    using GridLayout = H5Writer::GridLayout;
    using FloatType  = H5Writer::FloatType;

    ElectromagDiagnosticWriter(H5Writer& h5Writer)
        : Super{h5Writer}
    {
    }

    void write(DiagnosticProperties&) override;
    void compute(DiagnosticProperties&) override {}

    void createFiles(DiagnosticProperties& diagnostic) override;

    void getDataSetInfo(DiagnosticProperties& diagnostic, std::size_t iLevel,
                        std::string const& patchID, Attributes& patchAttributes) override;

    void initDataSets(DiagnosticProperties& diagnostic,
                      std::unordered_map<std::size_t, std::vector<std::string>> const& patchIDs,
                      Attributes& patchAttributes, std::size_t maxLevel) override;

    void writeAttributes(
        DiagnosticProperties&, Attributes&,
        std::unordered_map<std::size_t, std::vector<std::pair<std::string, Attributes>>>&,
        std::size_t maxLevel) override;
};


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::createFiles(DiagnosticProperties& diagnostic)
{
    auto const create
        = [&](auto const& q) { checkCreateFileFor_(diagnostic, fileData_, q.tree, q.name); };
    this->h5Writer_.modelView().forEachEmQuantity(create, create);
}


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::getDataSetInfo(DiagnosticProperties& diagnostic,
                                                          std::size_t iLevel,
                                                          std::string const& patchID,
                                                          Attributes& patchAttributes)
{
    auto& h5Writer         = this->h5Writer_;
    std::string lvlPatchID = std::to_string(iLevel) + "_" + patchID;

    auto const infoVF = [&](auto& vecF, std::string name, auto& attr) {
        for (auto& [id, type] : core::Components::componentMap())
        {
            // highfive doesn't accept uint32 which ndarray.shape() is
            auto const& array_shape = vecF.getComponent(type).shape();
            attr[name][id]          = std::vector<std::size_t>(array_shape.data(),
                                                               array_shape.data() + array_shape.size());
            auto ghosts = GridLayout::nDNbrGhosts(vecF.getComponent(type).physicalQuantity());
            for (std::uint8_t i = 1; i < GridLayout::dimension; ++i)
                if (ghosts[i] != ghosts[i - 1])
                    throw std::runtime_error("ghosts per direction must be constant");
            attr[name][id + "_ghosts"] = static_cast<std::size_t>(ghosts[0]);
        }
    };

    auto const infoScalar = [&](auto& field, std::string name, auto& attr) {
        auto const& shape = field.shape();
        attr[name]        = std::vector<std::size_t>(shape.data(), shape.data() + shape.size());
        auto ghosts       = GridLayout::nDNbrGhosts(field.physicalQuantity());
        for (std::uint8_t i = 1; i < GridLayout::dimension; ++i)
            if (ghosts[i] != ghosts[i - 1])
                throw std::runtime_error("ghosts per direction must be constant");
        attr[name + "_ghosts"] = static_cast<std::size_t>(ghosts[0]);
    };

    auto& attr = patchAttributes[lvlPatchID];
    h5Writer.modelView().visitActiveEmQuantity(
        diagnostic.quantity, h5Writer.patchLayout(), h5Writer.timestamp(),
        /*compute_derived=*/false, //
        [&](auto const& q, auto& field) { infoScalar(field, q.name, attr); },
        [&](auto const& q, auto& vecF) { infoVF(vecF, q.name, attr); });
}


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::initDataSets(
    DiagnosticProperties& diagnostic,
    std::unordered_map<std::size_t, std::vector<std::string>> const& patchIDs,
    Attributes& patchAttributes, std::size_t maxLevel)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = *fileData_.at(diagnostic.quantity);

    auto const initVF = [&](auto& path, auto& attr, std::string key, auto null) {
        for (auto& [id, type] : core::Components::componentMap())
        {
            auto vFPath = path + "/" + key + "_" + id;
            h5Writer.template createDataSet<FloatType>(
                h5file, vFPath,
                null ? std::vector<std::size_t>(GridLayout::dimension, 0)
                     : attr[key][id].template to<std::vector<std::size_t>>());

            this->writeGhostsAttr_(h5file, vFPath,
                                   null ? 0 : attr[key][id + "_ghosts"].template to<std::size_t>(),
                                   null);
        }
    };

    auto const initScalar = [&](auto& path, auto& attr, std::string key, auto null) {
        auto dsPath = path + "/" + key;
        h5Writer.template createDataSet<FloatType>(
            h5file, dsPath,
            null ? std::vector<std::size_t>(GridLayout::dimension, 0)
                 : attr[key].template to<std::vector<std::size_t>>());
        this->writeGhostsAttr_(h5file, dsPath,
                               null ? 0 : attr[key + "_ghosts"].template to<std::size_t>(), null);
    };

    auto const initPatch = [&](auto& level, auto& attr, std::string patchID = "") {
        bool null = patchID.empty();
        std::string path{h5Writer.getPatchPathAddTimestamp(level, patchID)};

        h5Writer.modelView().forEachEmQuantity(
            [&](auto const& q) {
                if (diagnostic.quantity == q.path())
                    initScalar(path, attr, q.name, null);
            },
            [&](auto const& q) {
                if (diagnostic.quantity == q.path())
                    initVF(path, attr, q.name, null);
            });
    };

    initDataSets_(patchIDs, patchAttributes, maxLevel, initPatch);
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = *fileData_.at(diagnostic.quantity);

    std::string const path = h5Writer.patchPath() + "/";

    h5Writer.modelView().visitActiveEmQuantity(
        diagnostic.quantity, h5Writer.patchLayout(), h5Writer.timestamp(),
        /*compute_derived=*/true, //
        [&](auto const& q, auto& field) {
            h5file.template write_data_set_flat<GridLayout::dimension>(path + q.name, field.data());
        },
        [&](auto const& q, auto& vecF) {
            h5Writer.writeTensorFieldAsDataset(h5file, path + q.name, vecF);
        });
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::writeAttributes(
    DiagnosticProperties& diagnostic, Attributes& fileAttributes,
    std::unordered_map<std::size_t, std::vector<std::pair<std::string, Attributes>>>&
        patchAttributes,
    std::size_t maxLevel)
{
    writeAttributes_(diagnostic, Super::h5FileForQuantity(diagnostic), fileAttributes,
                     patchAttributes, maxLevel);
}


} // namespace PHARE::diagnostic::h5

#endif /* PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_H */
