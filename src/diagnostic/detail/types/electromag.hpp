#ifndef PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP
#define PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/data/derived_quantity/derived_scratch.hpp"

#include "diagnostic/detail/h5typewriter.hpp"

#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/hybrid_model.hpp"

#include <stdexcept>
#include <type_traits>

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

    using ModelView_t = std::decay_t<decltype(std::declval<H5Writer&>().modelView())>;
    using Model_t     = typename ModelView_t::Model_t;

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

private:
    auto isActiveDiag(DiagnosticProperties const& diagnostic, std::string const& tree,
                      std::string var)
    {
        return diagnostic.quantity == tree + var;
    };
};


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::createFiles(DiagnosticProperties& diagnostic)
{
    std::string tree = "/";
    checkCreateFileFor_(diagnostic, fileData_, tree, "EM_B", "EM_E", "EM_J", "EM_divB");
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
        auto ghosts        = GridLayout::nDNbrGhosts(field.physicalQuantity());
        for (std::uint8_t i = 1; i < GridLayout::dimension; ++i)
            if (ghosts[i] != ghosts[i - 1])
                throw std::runtime_error("ghosts per direction must be constant");
        attr[name + "_ghosts"] = static_cast<std::size_t>(ghosts[0]);
    };

    if (isActiveDiag(diagnostic, "/", "EM_B"))
    {
        auto& B = h5Writer.modelView().getB();
        infoVF(B, "EM_B", patchAttributes[lvlPatchID]);
    }
    if constexpr (solver::is_hybrid_model_v<Model_t>)
    {
        if (isActiveDiag(diagnostic, "/", "EM_E"))
        {
            auto& E = h5Writer.modelView().getE();
            infoVF(E, "EM_E", patchAttributes[lvlPatchID]);
        }
    }

    auto& modelView     = h5Writer.modelView();
    auto const& derived = modelView.derivedQuantities();
    auto const& layout  = h5Writer.patchLayout();

    for (auto const& dq : derived.template quantities<1>())
        if (isActiveDiag(diagnostic, "/", "EM_" + dq->name()))
        {
            auto vecfield = core::derived_vector_view<typename Model_t::physical_quantity_type>(
                modelView.derivedVecScratch(), dq->centering(), layout);
            infoVF(vecfield, "EM_" + dq->name(), patchAttributes[lvlPatchID]);
        }

    if constexpr (solver::is_mhd_model_v<Model_t>)
    {
        for (auto const& dq : derived.template quantities<0>())
            if (isActiveDiag(diagnostic, "/", "EM_" + dq->name()))
            {
                auto field = core::derived_scalar_view<typename Model_t::physical_quantity_type>(
                    modelView.derivedScalarScratch(), dq->centering(), layout);
                infoScalar(field, "EM_" + dq->name(), patchAttributes[lvlPatchID]);
            }
    }
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

    auto const& derived = h5Writer.modelView().derivedQuantities();

    auto const initPatch = [&](auto& level, auto& attr, std::string patchID = "") {
        bool null = patchID.empty();
        std::string path{h5Writer.getPatchPathAddTimestamp(level, patchID)};
        std::string tree = "/";

        if (isActiveDiag(diagnostic, tree, "EM_B"))
            initVF(path, attr, "EM_B", null);
        if constexpr (solver::is_hybrid_model_v<Model_t>)
        {
            if (isActiveDiag(diagnostic, tree, "EM_E"))
                initVF(path, attr, "EM_E", null);
        }

        for (auto const& dq : derived.template quantities<1>())
            if (isActiveDiag(diagnostic, tree, "EM_" + dq->name()))
                initVF(path, attr, "EM_" + dq->name(), null);

        if constexpr (solver::is_mhd_model_v<Model_t>)
        {
            for (auto const& dq : derived.template quantities<0>())
                if (isActiveDiag(diagnostic, tree, "EM_" + dq->name()))
                    initScalar(path, attr, "EM_" + dq->name(), null);
        }
    };

    initDataSets_(patchIDs, patchAttributes, maxLevel, initPatch);
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = *fileData_.at(diagnostic.quantity);

    std::string tree = "/";
    std::string path = h5Writer.patchPath() + "/";

    if (isActiveDiag(diagnostic, tree, "EM_B"))
    {
        auto& B = h5Writer.modelView().getB();
        h5Writer.writeTensorFieldAsDataset(h5file, path + "EM_B", B);
    }
    if constexpr (solver::is_hybrid_model_v<Model_t>)
    {
        if (isActiveDiag(diagnostic, tree, "EM_E"))
        {
            auto& E = h5Writer.modelView().getE();
            h5Writer.writeTensorFieldAsDataset(h5file, path + "EM_E", E);
        }
    }

    auto& modelView     = h5Writer.modelView();
    auto const& derived = modelView.derivedQuantities();
    auto const& layout  = h5Writer.patchLayout();
    auto const time     = h5Writer.timestamp();

    for (auto const& dq : derived.template quantities<1>())
        if (isActiveDiag(diagnostic, tree, "EM_" + dq->name()))
        {
            auto vecfield = core::derived_vector_view<typename Model_t::physical_quantity_type>(
                modelView.derivedVecScratch(), dq->centering(), layout);
            dq->compute(modelView.state(), layout, vecfield, time);
            h5Writer.writeTensorFieldAsDataset(h5file, path + "EM_" + dq->name(), vecfield);
        }

    if constexpr (solver::is_mhd_model_v<Model_t>)
    {
        for (auto const& dq : derived.template quantities<0>())
            if (isActiveDiag(diagnostic, tree, "EM_" + dq->name()))
            {
                auto field = core::derived_scalar_view<typename Model_t::physical_quantity_type>(
                    modelView.derivedScalarScratch(), dq->centering(), layout);
                dq->compute(modelView.state(), layout, field, time);
                h5file.template write_data_set_flat<GridLayout::dimension>(path + "EM_" + dq->name(),
                                                                           field.data());
            }
    }
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
