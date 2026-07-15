#ifndef DIAGNOSTIC_MODEL_VIEW_HPP
#define DIAGNOSTIC_MODEL_VIEW_HPP

#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/utilities/mpi_utils.hpp"
#include "core/data/derived_quantity/mhd_derived_quantities.hpp"
#include "core/data/derived_quantity/hybrid_derived_quantities.hpp"
#include "core/data/derived_quantity/derived_scratch.hpp"

#include "amr/amr_constants.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/hybrid_model.hpp"
#include "amr/messengers/field_operate_transaction.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"

#include "dict.hpp"

#include <SAMRAI/xfer/RefineAlgorithm.h>

#include <type_traits>
#include <utility>

namespace PHARE::diagnostic
{
// Generic Template declaration, to override per Concrete model type
class IModelView
{
public:
    inline virtual ~IModelView();
};
IModelView::~IModelView() {}


/** Identifies one fluid-tree diagnostic quantity: `tree + name` is the
 *  diagnostic quantity string, `group` the per-patch attribute subgroup key. */
struct FluidQtyInfo
{
    std::string tree;
    std::string name;
    std::string group;

    std::string path() const { return tree + name; }
};


template<typename Derived, typename Hierarchy, typename Model>
class BaseModelView : public IModelView
{
public:
    using GridLayout        = Model::gridlayout_type;
    using VecField          = Model::vecfield_type;
    using ResMan            = Model::resources_manager_type;
    using Field             = Model::field_type;
    using TensorFieldData_t = ResMan::template UserTensorField_t</*rank=*/2>::patch_data_type;
    static constexpr auto dimension = Model::dimension;

    using PatchProperties
        = cppdict::Dict<float, double, std::size_t, std::vector<int>, std::vector<std::uint32_t>,
                        std::vector<double>, std::vector<std::size_t>, std::string,
                        std::vector<std::string>>;

    BaseModelView(Hierarchy& hierarchy, Model& model)
        : model_{model}
        , hierarchy_{hierarchy}
    {
    }

    template<typename Action>
    void onLevels(Action&& action, std::size_t const minlvl = 0,
                  std::size_t const maxlvl = amr::MAX_LEVEL_IDX)
    {
        amr::onLevels(hierarchy_, std::forward<Action>(action), minlvl, maxlvl);
    }


    template<typename OnLevel, typename OrMissing>
    void onLevels(OnLevel&& onLevel, OrMissing&& orMissing, std::size_t const minlvl,
                  std::size_t const maxlvl)
    {
        amr::onLevels(hierarchy_, std::forward<OnLevel>(onLevel),
                      std::forward<OrMissing>(orMissing), minlvl, maxlvl);
    }


    template<typename Action>
    void visitHierarchy(Action&& action, int minLevel = 0, int maxLevel = 0)
    {
        amr::visitHierarchy<GridLayout>(hierarchy_, *model_.resourcesManager,
                                        std::forward<Action>(action), minLevel, maxLevel, *this,
                                        model_);
    }

    NO_DISCARD auto boundaryConditions() const { return hierarchy_.boundaryConditions(); }
    NO_DISCARD auto domainBox() const { return hierarchy_.domainBox(); }
    NO_DISCARD auto origin() const { return std::vector<double>(dimension, 0); }
    NO_DISCARD auto cellWidth() const { return hierarchy_.cellWidth(); }
    NO_DISCARD auto maxLevel() const { return hierarchy_.maxLevel(); }

    NO_DISCARD std::string getLayoutTypeString() const
    {
        return std::string{GridLayout::implT::type};
    }

    NO_DISCARD auto getPatchProperties(std::string patchID, GridLayout const& grid) const
    {
        PatchProperties dict;
        dict["origin"]   = grid.origin().toVector();
        dict["nbrCells"] = core::Point<std::uint32_t, Model::dimension>{grid.nbrCells()}.toVector();
        dict["lower"]    = grid.AMRBox().lower.toVector();
        dict["upper"]    = grid.AMRBox().upper.toVector();
        dict["mpi_rank"] = static_cast<std::size_t>(core::mpi::rank());
        return dict;
    }

    NO_DISCARD static auto getEmptyPatchProperties(PatchProperties dict = {})
    {
        dict["origin"]   = std::vector<double>{};
        dict["nbrCells"] = std::vector<std::uint32_t>{};
        dict["lower"]    = std::vector<int>{};
        dict["upper"]    = std::vector<int>{};
        dict["mpi_rank"] = std::size_t{0};
        return dict;
    }

    NO_DISCARD bool hasTagsVectorFor(int ilevel, std::string patch_id) const
    {
        auto key = std::to_string(ilevel) + "_" + patch_id;
        return model_.tags.count(key);
    }

    NO_DISCARD auto& getTagsVectorFor(int ilevel, std::string patch_id) const
    {
        auto key = std::to_string(ilevel) + "_" + patch_id;
        return model_.tags.at(key);
    }


    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return derived().getCompileTimeResourcesViewList();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return derived().getCompileTimeResourcesViewList();
    }

protected:
    /** Enumerate the names of the derived quantities of `category`. Names only —
     *  no patch data access, so callable for null/remote patches and per-level
     *  setup. Callers wrap the raw name (e.g. "EM_" prefix, FluidQtyInfo). */
    template<typename OnScalar, typename OnVector>
    void forEachDerivedQuantity(core::DerivedCategory const category, OnScalar&& onScalar,
                                OnVector&& onVector) const
    {
        auto const& registry = derived().derivedQuantities();
        for (auto const& dq : registry.template quantities<0>())
            if (dq->category() == category)
                onScalar(dq->name());
        for (auto const& dq : registry.template quantities<1>())
            if (dq->category() == category)
                onVector(dq->name());
    }

    /** If `isActive(name)` matches a derived quantity of `category`, build its
     *  centering-correct view over the scratch backing, compute it iff
     *  compute_derived (shape-only consumers skip the compute), and dispatch it.
     *  Returns true on match. */
    template<typename IsActive, typename OnScalar, typename OnVector>
    bool visitActiveDerivedQuantity(core::DerivedCategory const category, IsActive const& isActive,
                                    GridLayout const& layout, double const time,
                                    bool const compute_derived, OnScalar&& onScalar,
                                    OnVector&& onVector)
    {
        using PhysicalQuantity = typename Model::physical_quantity_type;

        auto const& registry = derived().derivedQuantities();

        for (auto const& dq : registry.template quantities<0>())
            if (dq->category() == category and isActive(dq->name()))
            {
                auto field = core::derived_scalar_view<PhysicalQuantity>(
                    derived().derivedScalarScratch(), dq->centering(), layout);
                if (compute_derived)
                {
                    core::zero_scalar_view(field);
                    dq->compute(derived().state(), layout, field, time);
                }
                onScalar(dq->name(), field);
                return true;
            }
        for (auto const& dq : registry.template quantities<1>())
            if (dq->category() == category and isActive(dq->name()))
            {
                auto vecfield = core::derived_vector_view<PhysicalQuantity>(
                    derived().derivedVecScratch(), dq->centering(), layout);
                if (compute_derived)
                {
                    core::zero_vector_view(vecfield);
                    dq->compute(derived().state(), layout, vecfield, time);
                }
                onVector(dq->name(), vecfield);
                return true;
            }
        return false;
    }

    Model& model_;
    Hierarchy& hierarchy_;

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    Derived const& derived() const { return static_cast<Derived const&>(*this); }
};


template<typename Hierarchy, typename Model, typename Enable = void>
class ModelView;


template<typename Hierarchy, typename Model>
class ModelView<Hierarchy, Model, std::enable_if_t<solver::is_hybrid_model_v<Model>>>
    : public BaseModelView<ModelView<Hierarchy, Model>, Hierarchy, Model>
{
    using Super        = BaseModelView<ModelView<Hierarchy, Model>, Hierarchy, Model>;
    using Field        = Model::field_type;
    using VecField     = Model::vecfield_type;
    using TensorFieldT = Model::ions_type::tensorfield_type;

public:
    using Model_t                = Model;
    using physical_quantity_type = Model::physical_quantity_type;
    using State_t                = std::decay_t<decltype(std::declval<Model&>().state)>;
    using GridLayout             = typename Super::GridLayout;

    ModelView(Hierarchy& hierarchy, Model& model)
        : Super{hierarchy, model}
        , derived_{core::makeHybridDerivedQuantities<State_t, GridLayout>()}
    {
        declareMomentumTensorAlgos();
    }

    NO_DISCARD auto& state() { return this->model_.state; }
    NO_DISCARD auto const& state() const { return this->model_.state; }

    NO_DISCARD auto const& derivedQuantities() const { return derived_; }

    NO_DISCARD VecField& derivedVecScratch() { return derivedVecScratch_; }

    NO_DISCARD Field& derivedScalarScratch() { return derivedScalarScratch_; }

    NO_DISCARD VecField& getB() const { return this->model_.state.electromag.B; }

    NO_DISCARD VecField& getE() const { return this->model_.state.electromag.E; }

    NO_DISCARD auto& getIons() const { return this->model_.state.ions; }


    /** Catalogue of fluid-tree quantities (ions + per-population). Enumerates
     *  FluidQtyInfo only — no patch data access, so callable for null/remote
     *  patches and per-level setup. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    void forEachFluidQuantity(OnScalar&& onScalar, OnVector&& onVector, OnTensor&& onTensor) const
    {
        for (auto const& pop : this->model_.state.ions)
        {
            std::string const tree{"/ions/pop/" + pop.name() + "/"};
            std::string const group{"fluid_" + pop.name()};
            onScalar(FluidQtyInfo{tree, "density", group});
            onScalar(FluidQtyInfo{tree, "charge_density", group});
            onVector(FluidQtyInfo{tree, "flux", group});
            onTensor(FluidQtyInfo{tree, "momentum_tensor", group});
        }

        std::string const tree{"/ions/"};
        onScalar(FluidQtyInfo{tree, "charge_density", "ion"});
        onScalar(FluidQtyInfo{tree, "mass_density", "ion"});
        onVector(FluidQtyInfo{tree, "bulkVelocity", "ion"});
        onTensor(FluidQtyInfo{tree, "momentum_tensor", "ion"});
    }

    /** Dispatch the active fluid diagnostic to its per-patch data. All hybrid
     *  fluid quantities are primaries today, so layout/time/compute_derived are
     *  unused; they are part of the signature so both models expose the same
     *  fluid-catalogue API (fluid-category derived quantities plug in here).
     *  Returns true if the diagnostic named a fluid-tree quantity. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    bool visitActiveFluidQuantity(std::string const& quantity, GridLayout const& /*layout*/,
                                  double const /*time*/, bool const /*compute_derived*/,
                                  OnScalar&& onScalar, OnVector&& onVector, OnTensor&& onTensor)
    {
        auto& ions = this->model_.state.ions;

        auto tryQty = [&](std::string const& tree, char const* name, std::string const& group,
                          auto&& data, auto& on) {
            if (quantity == tree + name)
            {
                on(FluidQtyInfo{tree, name, group}, data);
                return true;
            }
            return false;
        };

        for (auto& pop : ions)
        {
            std::string const tree{"/ions/pop/" + pop.name() + "/"};
            std::string const group{"fluid_" + pop.name()};
            if (tryQty(tree, "density", group, pop.particleDensity(), onScalar))
                return true;
            if (tryQty(tree, "charge_density", group, pop.chargeDensity(), onScalar))
                return true;
            if (tryQty(tree, "flux", group, pop.flux(), onVector))
                return true;
            if (tryQty(tree, "momentum_tensor", group, pop.momentumTensor(), onTensor))
                return true;
        }

        std::string const tree{"/ions/"};
        if (tryQty(tree, "charge_density", "ion", ions.chargeDensity(), onScalar))
            return true;
        if (tryQty(tree, "mass_density", "ion", ions.massDensity(), onScalar))
            return true;
        if (tryQty(tree, "bulkVelocity", "ion", ions.velocity(), onVector))
            return true;
        if (tryQty(tree, "momentum_tensor", "ion", ions.momentumTensor(), onTensor))
            return true;
        return false;
    }


    /** Catalogue of EM-tree quantities: primaries plus electromag-category
     *  derived quantities (scalar e.g. divB, vector e.g. J) under the "EM_"
     *  prefix. Names only. */
    template<typename OnScalar, typename OnVector>
    void forEachEmQuantity(OnScalar&& onScalar, OnVector&& onVector) const
    {
        onVector("EM_B");
        onVector("EM_E");

        this->forEachDerivedQuantity(
            core::DerivedCategory::electromag,
            [&](std::string const& name) { onScalar("EM_" + name); },
            [&](std::string const& name) { onVector("EM_" + name); });
    }

    /** Dispatch the active EM diagnostic to its per-patch data; derived
     *  quantities are viewed in scratch and computed iff compute_derived. */
    template<typename OnScalar, typename OnVector>
    bool visitActiveEmQuantity(std::string const& quantity, GridLayout const& layout,
                               double const time, bool const compute_derived, OnScalar&& onScalar,
                               OnVector&& onVector)
    {
        auto const isActive = [&](std::string const& name) { return quantity == "/EM_" + name; };

        if (isActive("B"))
        {
            onVector("EM_B", this->model_.state.electromag.B);
            return true;
        }
        if (isActive("E"))
        {
            onVector("EM_E", this->model_.state.electromag.E);
            return true;
        }

        return this->visitActiveDerivedQuantity(
            core::DerivedCategory::electromag, isActive, layout, time, compute_derived,
            [&](std::string const& name, auto& field) { onScalar("EM_" + name, field); },
            [&](std::string const& name, auto& vecF) { onVector("EM_" + name, vecF); });
    }


    auto& tmpField() { return tmpField_; }

    auto& tmpVecField() { return tmpVec_; }

    template<std::size_t rank = 2>
    auto& tmpTensorField()
    {
        static_assert(rank > 0 and rank < 3);
        if constexpr (rank == 1)
            return tmpVec_;
        else
            return tmpTensor_;
    }

    void fillPopMomTensor(auto& lvl, auto const time, auto const popidx)
    {
        using value_type = TensorFieldT::value_type;
        auto constexpr N = core::detail::tensor_field_dim_from_rank<2>();

        auto& rm   = *this->model_.resourcesManager;
        auto& ions = this->model_.state.ions;

        for (auto patch : rm.enumerate(lvl, ions, tmpTensor_))
            for (std::uint8_t c = 0; c < N; ++c)
                std::memcpy(tmpTensor_[c].data(), ions[popidx].momentumTensor()[c].data(),
                            ions[popidx].momentumTensor()[c].size() * sizeof(value_type));

        MTAlgos[popidx].getOrCreateSchedule(this->hierarchy_, lvl.getLevelNumber()).fillData(time);

        for (auto patch : rm.enumerate(lvl, ions, tmpTensor_))
            for (std::uint8_t c = 0; c < N; ++c)
                std::memcpy(ions[popidx].momentumTensor()[c].data(), tmpTensor_[c].data(),
                            ions[popidx].momentumTensor()[c].size() * sizeof(value_type));
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(tmpField_, tmpVec_, tmpTensor_, derivedVecScratch_,
                                     derivedScalarScratch_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(tmpField_, tmpVec_, tmpTensor_, derivedVecScratch_,
                                     derivedScalarScratch_);
    }

protected:
    void declareMomentumTensorAlgos()
    {
        auto& rm = *this->model_.resourcesManager;

        auto const dst_name = tmpTensor_.name();

        for (auto& pop : this->model_.state.ions)
        {
            auto& MTAlgo        = MTAlgos.emplace_back();
            auto const src_name = pop.momentumTensor().name();

            auto&& [idDst, idSrc] = rm.getIDsList(dst_name, src_name);
            MTAlgo.MTalgo->registerRefine(
                idDst, idSrc, idDst, nullptr,
                std::make_shared<
                    amr::TensorFieldGhostInterpOverlapFillPattern<typename Super::GridLayout,
                                                                  /*rank_=*/2>>());
        }

        // can't create schedules here as the hierarchy has no levels yet
    }

    struct MTAlgo
    {
        auto& getOrCreateSchedule(auto& hierarchy, int const ilvl)
        {
            using PlusEqualsOp = core::PlusEquals<typename VecField::value_type>;
            if (not MTschedules.count(ilvl))
                MTschedules.try_emplace(
                    ilvl, MTalgo->createSchedule(
                              hierarchy.getPatchLevel(ilvl), 0,
                              std::make_shared<amr::FieldBorderOpTransactionFactory<
                                  typename Super::TensorFieldData_t, PlusEqualsOp>>()));
            return *MTschedules[ilvl];
        }

        std::unique_ptr<SAMRAI::xfer::RefineAlgorithm> MTalgo
            = std::make_unique<SAMRAI::xfer::RefineAlgorithm>();
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> MTschedules;
    };

    std::vector<MTAlgo> MTAlgos;
    Field tmpField_{"PHARE_sumField", core::HybridQuantity::Scalar::rho};
    VecField tmpVec_{"PHARE_sumVec", core::HybridQuantity::Vector::V};
    TensorFieldT tmpTensor_{"PHARE_sumTensor", core::HybridQuantity::Tensor::M};
    VecField derivedVecScratch_{"PHARE_derived_vec", core::HybridQuantity::Vector::VecElike};
    Field derivedScalarScratch_{"PHARE_derived_scalar",
                                core::HybridQuantity::Scalar::ScalarNodeCentered};
    core::DerivedQuantityRegistry<State_t, GridLayout> derived_;
};


template<typename Hierarchy, typename Model>
class ModelView<Hierarchy, Model, std::enable_if_t<solver::is_mhd_model_v<Model>>>
    : public BaseModelView<ModelView<Hierarchy, Model>, Hierarchy, Model>
{
    using Field    = Model::field_type;
    using VecField = Model::vecfield_type;

public:
    using Model_t                = Model;
    using physical_quantity_type = Model::physical_quantity_type;
    using Super                  = BaseModelView<ModelView<Hierarchy, Model>, Hierarchy, Model>;
    using State_t                = typename Model::state_type;
    using GridLayout             = typename Super::GridLayout;

    ModelView(Hierarchy& hierarchy, Model& model)
        : Super{hierarchy, model}
        , derived_{core::makeMhdDerivedQuantities<State_t, GridLayout>(
              model.state.gamma(), model.state.eta(), model.state.nu(), model.state.hyperMode(),
              model.state.hall())}
    {
    }

    NO_DISCARD auto& state() { return this->model_.state; }
    NO_DISCARD auto const& state() const { return this->model_.state; }

    NO_DISCARD auto const& derivedQuantities() const { return derived_; }

    NO_DISCARD const Field& getRho() const { return this->model_.state.rho; }

    NO_DISCARD const VecField& getRhoV() const { return this->model_.state.rhoV; }

    NO_DISCARD const VecField& getB() const { return this->model_.state.B; }

    NO_DISCARD const Field& getEtot() const { return this->model_.state.Etot; }

    // for setBuffer function in visitHierarchy
    NO_DISCARD Field& getRho() { return this->model_.state.rho; }

    NO_DISCARD VecField& getRhoV() { return this->model_.state.rhoV; }

    NO_DISCARD VecField& getB() { return this->model_.state.B; }

    NO_DISCARD Field& getEtot() { return this->model_.state.Etot; }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(tmpField_, tmpVec_, derivedScalarScratch_, derivedVecScratch_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(tmpField_, tmpVec_, derivedScalarScratch_, derivedVecScratch_);
    }

    auto& tmpField() { return tmpField_; }

    auto& tmpVecField() { return tmpVec_; }

    template<std::size_t rank = 2>
    auto& tmpTensorField()
    {
        static_assert(rank == 1);
        return tmpVec_;
    }

    NO_DISCARD Field& derivedScalarScratch() { return derivedScalarScratch_; }
    NO_DISCARD VecField& derivedVecScratch() { return derivedVecScratch_; }


    /** The catalogue of fluid-tree ("/mhd/") quantities: primaries plus
     *  fluid-category derived quantities. Enumerates FluidQtyInfo only — no
     *  patch data access, so callable for null/remote patches and per-level
     *  setup. MHD has no rank-2 fluid quantities, so onTensor is never called;
     *  it is part of the signature so both models expose the same API. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    void forEachFluidQuantity(OnScalar&& onScalar, OnVector&& onVector, OnTensor&&) const
    {
        std::string const tree{"/mhd/"};
        std::string const group{"mhd"};

        onScalar(FluidQtyInfo{tree, "rho", group});
        onVector(FluidQtyInfo{tree, "rhoV", group});
        onScalar(FluidQtyInfo{tree, "Etot", group});

        this->forEachDerivedQuantity(
            core::DerivedCategory::fluid,
            [&](std::string const& name) { onScalar(FluidQtyInfo{tree, name, group}); },
            [&](std::string const& name) { onVector(FluidQtyInfo{tree, name, group}); });
    }

    /** Dispatch the active fluid diagnostic to its per-patch data: primaries
     *  are passed through, derived quantities are viewed in scratch and
     *  computed iff compute_derived (shape-only consumers skip the compute).
     *  Returns true if the diagnostic named a fluid-tree quantity. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    bool visitActiveFluidQuantity(std::string const& quantity, GridLayout const& layout,
                                  double const time, bool const compute_derived,
                                  OnScalar&& onScalar, OnVector&& onVector, OnTensor&&)
    {
        std::string const tree{"/mhd/"};
        std::string const group{"mhd"};
        auto const isActive = [&](std::string const& name) { return quantity == tree + name; };

        if (isActive("rho"))
        {
            onScalar(FluidQtyInfo{tree, "rho", group}, this->model_.state.rho);
            return true;
        }
        if (isActive("rhoV"))
        {
            onVector(FluidQtyInfo{tree, "rhoV", group}, this->model_.state.rhoV);
            return true;
        }
        if (isActive("Etot"))
        {
            onScalar(FluidQtyInfo{tree, "Etot", group}, this->model_.state.Etot);
            return true;
        }

        return this->visitActiveDerivedQuantity(
            core::DerivedCategory::fluid, isActive, layout, time, compute_derived,
            [&](std::string const& name, auto& field) {
                onScalar(FluidQtyInfo{tree, name, group}, field);
            },
            [&](std::string const& name, auto& vecF) {
                onVector(FluidQtyInfo{tree, name, group}, vecF);
            });
    }


    /** Catalogue of EM-tree quantities: B plus electromag-category derived
     *  quantities (E, J, divB) under the "EM_" prefix. Names only. */
    template<typename OnScalar, typename OnVector>
    void forEachEmQuantity(OnScalar&& onScalar, OnVector&& onVector) const
    {
        onVector("EM_B");

        this->forEachDerivedQuantity(
            core::DerivedCategory::electromag,
            [&](std::string const& name) { onScalar("EM_" + name); },
            [&](std::string const& name) { onVector("EM_" + name); });
    }

    /** Dispatch the active EM diagnostic to its per-patch data; derived
     *  quantities are viewed in scratch and computed iff compute_derived. */
    template<typename OnScalar, typename OnVector>
    bool visitActiveEmQuantity(std::string const& quantity, GridLayout const& layout,
                               double const time, bool const compute_derived, OnScalar&& onScalar,
                               OnVector&& onVector)
    {
        auto const isActive = [&](std::string const& name) { return quantity == "/EM_" + name; };

        if (isActive("B"))
        {
            onVector("EM_B", this->model_.state.B);
            return true;
        }

        return this->visitActiveDerivedQuantity(
            core::DerivedCategory::electromag, isActive, layout, time, compute_derived,
            [&](std::string const& name, auto& field) { onScalar("EM_" + name, field); },
            [&](std::string const& name, auto& vecF) { onVector("EM_" + name, vecF); });
    }

protected:
    Field tmpField_{"PHARE_sumField_MHD", core::MHDQuantity::Scalar::ScalarAllPrimal};
    VecField tmpVec_{"PHARE_sumVec_MHD", core::MHDQuantity::Vector::VecAllPrimal};

    Field derivedScalarScratch_{"PHARE_derived_scalar", core::MHDQuantity::Scalar::ScalarAllPrimal};
    VecField derivedVecScratch_{"PHARE_derived_vec", core::MHDQuantity::Vector::VecAllPrimal};

    core::DerivedQuantityRegistry<State_t, GridLayout> derived_;
};


} // namespace PHARE::diagnostic



#endif // DIAGNOSTIC_MODEL_VIEW_HPP
