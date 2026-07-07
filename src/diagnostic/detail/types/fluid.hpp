#ifndef PHARE_DIAGNOSTIC_DETAIL_TYPES_FLUID_HPP
#define PHARE_DIAGNOSTIC_DETAIL_TYPES_FLUID_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/interpolator/interpolator.hpp"

#include "diagnostic/detail/h5typewriter.hpp"

#include <stdexcept>

namespace PHARE::diagnostic::h5
{
/*
 * It is assumed that each patch has equal number of populations
 *
 * Possible outputs
 *
 * /t#/pl#/p#/ions/density
 * /t#/pl#/p#/ions/bulkVelocity/(x,y,z)
 * /t#/pl#/p#/ions/pop_(1,2,...)/density
 * /t#/pl#/p#/ions/pop_(1,2,...)/bulkVelocity/(x,y,z)
 */
template<typename H5Writer>
class FluidDiagnosticWriter : public H5TypeWriter<H5Writer>
{
public:
    using Super = H5TypeWriter<H5Writer>;
    using Super::checkCreateFileFor_;
    using Super::fileData_;
    using Super::h5Writer_;
    using Super::initDataSets_;
    using Super::writeAttributes_;
    using Super::writeGhostsAttr_;
    using Super::writeIonPopAttributes_;
    using Attributes = Super::Attributes;
    using GridLayout = H5Writer::GridLayout;
    using FloatType  = H5Writer::FloatType;

    static constexpr auto dimension    = GridLayout::dimension;
    static constexpr auto interp_order = GridLayout::interp_order;


    FluidDiagnosticWriter(H5Writer& h5Writer)
        : Super{h5Writer}
    {
    }

    void write(DiagnosticProperties&) override;
    void compute(DiagnosticProperties&) override;

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
void FluidDiagnosticWriter<H5Writer>::compute(DiagnosticProperties& diagnostic)
{
    core::MomentumTensorInterpolator<dimension, interp_order> interpolator;

    auto& h5Writer    = this->h5Writer_;
    auto& modelView   = h5Writer.modelView();
    auto& ions        = modelView.getIons();
    auto const minLvl = this->h5Writer_.minLevel;
    auto const maxLvl = this->h5Writer_.maxLevel;
    // compute the momentum tensor for each population that requires it
    // compute for all ions but that requires the computation of all pop

    // dumps occur after the last substep but before the next first substep
    // at this time, levelGhostPartsNew is emptied and not yet filled
    // and the former levelGhostPartsNew has been moved to levelGhostPartsOld

    auto const fill_schedules = [&](auto& lvl) {
        for (std::size_t i = 0; i < ions.size(); ++i)
            modelView.fillPopMomTensor(lvl, h5Writer.timestamp(), i);
    };

    auto const interpolate_pop = [&](auto& pop, auto& layout, auto&&...) {
        auto& pop_momentum_tensor = pop.momentumTensor();
        pop_momentum_tensor.zero();
        interpolator(pop.domainParticles(), pop_momentum_tensor, layout, pop.mass());
        interpolator(pop.levelGhostParticlesOld(), pop_momentum_tensor, layout, pop.mass());
    };

    if (isActiveDiag(diagnostic, "/ions/", "momentum_tensor"))
    {
        auto const interpolate = [&](auto& layout, auto&&...) {
            for (auto& pop : ions)
                interpolate_pop(pop, layout);
        };
        modelView.visitHierarchy(interpolate, minLvl, maxLvl);

        modelView.onLevels(fill_schedules, minLvl, maxLvl);

        modelView.visitHierarchy( //
            [&](auto&&...) { ions.computeFullMomentumTensor(); }, minLvl, maxLvl);
    }
    else // if not computing total momentum tensor, user may want to compute it for some pop
    {
        for (auto& pop : ions)
        {
            std::string const tree{"/ions/pop/" + pop.name() + "/"};

            if (!isActiveDiag(diagnostic, tree, "momentum_tensor"))
                continue;

            auto const interpolate = [&](auto& layout, auto&&...) { interpolate_pop(pop, layout); };

            modelView.visitHierarchy(interpolate, minLvl, maxLvl);

            modelView.onLevels(fill_schedules, minLvl, maxLvl);
        }
    }
}



template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::createFiles(DiagnosticProperties& diagnostic)
{
    auto const create
        = [&](auto const& q) { checkCreateFileFor_(diagnostic, fileData_, q.tree, q.name); };
    this->h5Writer_.modelView().forEachFluidQuantity(create, create, create);
}




template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::getDataSetInfo(DiagnosticProperties& diagnostic,
                                                     std::size_t iLevel, std::string const& patchID,
                                                     Attributes& patchAttributes)
{
    auto& h5Writer = this->h5Writer_;
    std::string lvlPatchID{std::to_string(iLevel) + "_" + patchID};


    auto const setGhostNbr = [](auto const& field, auto& attr, auto const& name) {
        auto ghosts = GridLayout::nDNbrGhosts(field.physicalQuantity());
        for (std::uint8_t i = 1; i < GridLayout::dimension; ++i)
            if (ghosts[i] != ghosts[i - 1])
                throw std::runtime_error("ghosts per direction must be constant");
        attr[name + "_ghosts"] = static_cast<std::size_t>(ghosts[0]);
    };

    auto const infoDS = [&](auto& field, std::string name, auto& attr) {
        // highfive doesn't accept uint32 which ndarray.shape() is
        auto const& shape = field.shape();
        attr[name]        = std::vector<std::size_t>(shape.data(), shape.data() + shape.size());
        setGhostNbr(field, attr, name);
    };

    auto const infoVF = [&](auto& vecF, std::string name, auto& attr) {
        for (auto const& [id, type] : core::VectorComponents::map())
            infoDS(vecF.getComponent(type), name + "_" + id, attr);
    };

    auto const infoTF = [&](auto& tensorF, std::string name, auto& attr) {
        for (auto const& [id, type] : core::TensorComponents::map())
            infoDS(tensorF.getComponent(type), name + "_" + id, attr);
    };

    auto& attr = patchAttributes[lvlPatchID];
    h5Writer.modelView().visitActiveFluidQuantity(
        diagnostic.quantity, //
        [&](auto const& q, auto& field) { infoDS(field, q.name, attr[q.group]); },
        [&](auto const& q, auto& vecF) { infoVF(vecF, q.name, attr[q.group]); },
        [&](auto const& q, auto& tensorF) { infoTF(tensorF, q.name, attr[q.group]); });
}




template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::initDataSets(
    DiagnosticProperties& diagnostic,
    std::unordered_map<std::size_t, std::vector<std::string>> const& patchIDs,
    Attributes& patchAttributes, std::size_t maxLevel)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = Super::h5FileForQuantity(diagnostic);

    auto const writeGhosts = [&](auto& path, auto& attr, std::string key, auto null) {
        this->writeGhostsAttr_(h5file, path,
                               null ? 0 : attr[key + "_ghosts"].template to<std::size_t>(), null);
    };

    auto const initDS = [&](auto& path, auto& attr, std::string key, auto null) {
        auto dsPath = path + key;
        h5Writer.template createDataSet<FloatType>(
            h5file, dsPath,
            null ? std::vector<std::size_t>(GridLayout::dimension, 0)
                 : attr[key].template to<std::vector<std::size_t>>());
        writeGhosts(dsPath, attr, key, null);
    };
    auto const initVF = [&](auto& path, auto& attr, std::string key, auto null) {
        for (auto& [id, type] : core::Components::componentMap())
            initDS(path, attr, key + "_" + id, null);
    };
    auto const initTF = [&](auto& path, auto& attr, std::string key, auto null) {
        for (auto& [id, type] : core::Components::componentMap<2>())
            initDS(path, attr, key + "_" + id, null);
    };

    auto const initPatch = [&](auto& lvl, auto& attr, std::string patchID = "") {
        bool null        = patchID.empty();
        std::string path = h5Writer.getPatchPathAddTimestamp(lvl, patchID) + "/";

        h5Writer.modelView().forEachFluidQuantity(
            [&](auto const& q) {
                if (diagnostic.quantity == q.path())
                    initDS(path, attr[q.group], q.name, null);
            },
            [&](auto const& q) {
                if (diagnostic.quantity == q.path())
                    initVF(path, attr[q.group], q.name, null);
            },
            [&](auto const& q) {
                if (diagnostic.quantity == q.path())
                    initTF(path, attr[q.group], q.name, null);
            });
    };

    initDataSets_(patchIDs, patchAttributes, maxLevel, initPatch);
}


template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = Super::h5FileForQuantity(diagnostic);

    auto writeDS = [&](auto path, auto& field) {
        h5file.template write_data_set_flat<GridLayout::dimension>(path, field.data());
    };
    auto writeTF
        = [&](auto path, auto& vecF) { h5Writer.writeTensorFieldAsDataset(h5file, path, vecF); };

    std::string const path = h5Writer.patchPath() + "/";

    h5Writer.modelView().visitActiveFluidQuantity(
        diagnostic.quantity, //
        [&](auto const& q, auto& field) { writeDS(path + q.name, field); },
        [&](auto const& q, auto& vecF) { writeTF(path + q.name, vecF); },
        [&](auto const& q, auto& tensorF) { writeTF(path + q.name, tensorF); });
}


template<typename H5Writer>
void FluidDiagnosticWriter<H5Writer>::writeAttributes(
    DiagnosticProperties& diagnostic, Attributes& fileAttributes,
    std::unordered_map<std::size_t, std::vector<std::pair<std::string, Attributes>>>&
        patchAttributes,
    std::size_t maxLevel)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = Super::h5FileForQuantity(diagnostic);

    auto checkWrite = [&](auto& tree, std::string qty, auto const& pop) {
        if (diagnostic.quantity == tree + qty)
            this->writeIonPopAttributes_(h5file, pop);
    };

    for (auto& pop : h5Writer.modelView().getIons())
    {
        std::string tree = "/ions/pop/" + pop.name() + "/";
        checkWrite(tree, "density", pop);
        checkWrite(tree, "charge_density", pop);
        checkWrite(tree, "flux", pop);
        checkWrite(tree, "momentum_tensor", pop);
    }

    writeAttributes_(diagnostic, h5file, fileAttributes, patchAttributes, maxLevel);
}

} // namespace PHARE::diagnostic::h5

#endif /* PHARE_DIAGNOSTIC_DETAIL_TYPES_FLUID_H */
