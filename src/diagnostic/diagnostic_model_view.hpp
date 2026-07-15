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

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace PHARE::diagnostic
{
// Generic Template declaration, to override per Concrete model type
class IModelView
{
public:
    inline virtual ~IModelView();
};
IModelView::~IModelView() {}


/** Identifies one diagnostic quantity: `tree + name` is the diagnostic
 *  quantity string, `group` the per-patch attribute subgroup key. */
struct DiagQtyInfo
{
    std::string tree;
    std::string name;
    std::string group;

    std::string path() const { return tree + name; }
};


namespace detail
{
    /** Rank-2 data type for a model's diagnostic catalogue. Models without ion
     *  populations have no rank-2 quantities and never register tensor entries,
     *  so any copyable placeholder type works for them. */
    template<typename Model, typename = void>
    struct diag_tensorfield
    {
        using type = typename Model::vecfield_type;
    };

    template<typename Model>
    struct diag_tensorfield<Model, std::void_t<typename Model::ions_type>>
    {
        using type = typename Model::ions_type::tensorfield_type;
    };
} // namespace detail


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


    // ------------------------------------------------------------------------
    // Diagnostic quantity catalogue: every quantity (primary or derived, fluid
    // or EM tree) is registered exactly once at ModelView construction as a
    // (DiagQtyInfo, access) entry. `access` is lazy — enumeration never touches
    // patch data, so forEach* stays safe for null/remote patches — and returns
    // a shallow view: a copy of the state field's view object for primaries, a
    // centering-correct scratch view (computed iff compute_derived) for derived
    // quantities. The active quantity string resolves through one hash lookup
    // instead of per-patch catalogue scans.
    // ------------------------------------------------------------------------

    /** Fluid-tree catalogue, names only. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    void forEachFluidQuantity(OnScalar&& onScalar, OnVector&& onVector, OnTensor&& onTensor) const
    {
        for (auto const& e : fluidScalars_)
            onScalar(e.info);
        for (auto const& e : fluidVectors_)
            onVector(e.info);
        for (auto const& e : fluidTensors_)
            onTensor(e.info);
    }

    /** EM-tree catalogue, names only. */
    template<typename OnScalar, typename OnVector>
    void forEachEmQuantity(OnScalar&& onScalar, OnVector&& onVector) const
    {
        for (auto const& e : emScalars_)
            onScalar(e.info);
        for (auto const& e : emVectors_)
            onVector(e.info);
    }

    /** Dispatch the active fluid diagnostic to its per-patch data.
     *  Returns true if the diagnostic named a fluid-tree quantity. */
    template<typename OnScalar, typename OnVector, typename OnTensor>
    bool visitActiveFluidQuantity(std::string const& quantity, GridLayout const& layout,
                                  double const time, bool const compute_derived,
                                  OnScalar&& onScalar, OnVector&& onVector, OnTensor&& onTensor)
    {
        auto const it = qtyIndex_.find(quantity);
        if (it == qtyIndex_.end())
            return false;

        auto const dispatch = [&](auto const& entries, auto& on) {
            auto const& e = entries[it->second.idx];
            auto data     = e.access(layout, time, compute_derived);
            on(e.info, data);
            return true;
        };

        switch (it->second.kind)
        {
            case Handle::Kind::FluidScalar: return dispatch(fluidScalars_, onScalar);
            case Handle::Kind::FluidVector: return dispatch(fluidVectors_, onVector);
            case Handle::Kind::FluidTensor: return dispatch(fluidTensors_, onTensor);
            default: return false;
        }
    }

    /** Dispatch the active EM diagnostic to its per-patch data.
     *  Returns true if the diagnostic named an EM-tree quantity. */
    template<typename OnScalar, typename OnVector>
    bool visitActiveEmQuantity(std::string const& quantity, GridLayout const& layout,
                               double const time, bool const compute_derived, OnScalar&& onScalar,
                               OnVector&& onVector)
    {
        auto const it = qtyIndex_.find(quantity);
        if (it == qtyIndex_.end())
            return false;

        auto const dispatch = [&](auto const& entries, auto& on) {
            auto const& e = entries[it->second.idx];
            auto data     = e.access(layout, time, compute_derived);
            on(e.info, data);
            return true;
        };

        switch (it->second.kind)
        {
            case Handle::Kind::EmScalar: return dispatch(emScalars_, onScalar);
            case Handle::Kind::EmVector: return dispatch(emVectors_, onVector);
            default: return false;
        }
    }

protected:
    using TensorField_t = typename detail::diag_tensorfield<Model>::type;

    /** One catalogue entry: identity + lazy per-patch data access. */
    template<typename DataT>
    struct QtyEntry
    {
        DiagQtyInfo info;
        std::function<DataT(GridLayout const&, double time, bool compute_derived)> access;
    };

    struct Handle
    {
        enum class Kind { FluidScalar, FluidVector, FluidTensor, EmScalar, EmVector };
        Kind kind;
        std::size_t idx;
    };

    using ScalarAccess = std::function<Field(GridLayout const&, double, bool)>;
    using VectorAccess = std::function<VecField(GridLayout const&, double, bool)>;
    using TensorAccess = std::function<TensorField_t(GridLayout const&, double, bool)>;

    void addFluidScalar_(DiagQtyInfo info, ScalarAccess access)
    {
        addEntry_(fluidScalars_, Handle::Kind::FluidScalar, std::move(info), std::move(access));
    }
    void addFluidVector_(DiagQtyInfo info, VectorAccess access)
    {
        addEntry_(fluidVectors_, Handle::Kind::FluidVector, std::move(info), std::move(access));
    }
    void addFluidTensor_(DiagQtyInfo info, TensorAccess access)
    {
        addEntry_(fluidTensors_, Handle::Kind::FluidTensor, std::move(info), std::move(access));
    }
    void addEmScalar_(DiagQtyInfo info, ScalarAccess access)
    {
        addEntry_(emScalars_, Handle::Kind::EmScalar, std::move(info), std::move(access));
    }
    void addEmVector_(DiagQtyInfo info, VectorAccess access)
    {
        addEntry_(emVectors_, Handle::Kind::EmVector, std::move(info), std::move(access));
    }

    /** Register every quantity of the model's derived-quantity registry:
     *  fluid-category ones under (fluidTree, fluidGroup), electromag-category
     *  ones under the EM tree with the "EM_" prefix. Access builds the
     *  centering-correct scratch view and computes it iff compute_derived
     *  (shape-only consumers skip the compute). */
    template<typename Registry>
    void addDerivedQuantityEntries_(Registry const& registry, std::string const& fluidTree,
                                    std::string const& fluidGroup)
    {
        using PhysicalQuantity = typename Model::physical_quantity_type;
        auto* self             = static_cast<Derived*>(this);

        for (auto const& dqp : registry.template quantities<0>())
        {
            auto const* dq = dqp.get();
            ScalarAccess access
                = [self, dq](GridLayout const& layout, double const time, bool const compute) {
                      auto field = core::derived_scalar_view<PhysicalQuantity>(
                          self->derivedScalarScratch(), dq->centering(), layout);
                      if (compute)
                      {
                          core::zero_scalar_view(field);
                          dq->compute(self->state(), layout, field, time);
                      }
                      return field;
                  };
            if (dq->category() == core::DerivedCategory::fluid)
                addFluidScalar_({fluidTree, dq->name(), fluidGroup}, std::move(access));
            else
                addEmScalar_({"/", "EM_" + dq->name(), "em"}, std::move(access));
        }

        for (auto const& dqp : registry.template quantities<1>())
        {
            auto const* dq = dqp.get();
            VectorAccess access
                = [self, dq](GridLayout const& layout, double const time, bool const compute) {
                      auto vecfield = core::derived_vector_view<PhysicalQuantity>(
                          self->derivedVecScratch(), dq->centering(), layout);
                      if (compute)
                      {
                          core::zero_vector_view(vecfield);
                          dq->compute(self->state(), layout, vecfield, time);
                      }
                      return vecfield;
                  };
            if (dq->category() == core::DerivedCategory::fluid)
                addFluidVector_({fluidTree, dq->name(), fluidGroup}, std::move(access));
            else
                addEmVector_({"/", "EM_" + dq->name(), "em"}, std::move(access));
        }
    }

    Model& model_;
    Hierarchy& hierarchy_;

    std::vector<QtyEntry<Field>> fluidScalars_, emScalars_;
    std::vector<QtyEntry<VecField>> fluidVectors_, emVectors_;
    std::vector<QtyEntry<TensorField_t>> fluidTensors_;
    std::unordered_map<std::string, Handle> qtyIndex_;

private:
    template<typename DataT, typename Access>
    void addEntry_(std::vector<QtyEntry<DataT>>& entries, typename Handle::Kind const kind,
                   DiagQtyInfo info, Access&& access)
    {
        qtyIndex_.emplace(info.path(), Handle{kind, entries.size()});
        entries.push_back(QtyEntry<DataT>{std::move(info), std::forward<Access>(access)});
    }

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

        for (std::size_t i = 0; i < model.state.ions.size(); ++i)
        {
            auto const popName = model.state.ions[i].name();
            std::string const tree{"/ions/pop/" + popName + "/"};
            std::string const group{"fluid_" + popName};

            this->addFluidScalar_({tree, "density", group},
                                  [this, i](GridLayout const&, double, bool) -> Field {
                                      return this->model_.state.ions[i].particleDensity();
                                  });
            this->addFluidScalar_({tree, "charge_density", group},
                                  [this, i](GridLayout const&, double, bool) -> Field {
                                      return this->model_.state.ions[i].chargeDensity();
                                  });
            this->addFluidVector_({tree, "flux", group},
                                  [this, i](GridLayout const&, double, bool) -> VecField {
                                      return this->model_.state.ions[i].flux();
                                  });
            this->addFluidTensor_({tree, "momentum_tensor", group},
                                  [this, i](GridLayout const&, double, bool) -> TensorFieldT {
                                      return this->model_.state.ions[i].momentumTensor();
                                  });
        }

        this->addFluidScalar_({"/ions/", "charge_density", "ion"},
                              [this](GridLayout const&, double, bool) -> Field {
                                  return this->model_.state.ions.chargeDensity();
                              });
        this->addFluidScalar_({"/ions/", "mass_density", "ion"},
                              [this](GridLayout const&, double, bool) -> Field {
                                  return this->model_.state.ions.massDensity();
                              });
        this->addFluidVector_({"/ions/", "bulkVelocity", "ion"},
                              [this](GridLayout const&, double, bool) -> VecField {
                                  return this->model_.state.ions.velocity();
                              });
        this->addFluidTensor_({"/ions/", "momentum_tensor", "ion"},
                              [this](GridLayout const&, double, bool) -> TensorFieldT {
                                  return this->model_.state.ions.momentumTensor();
                              });

        this->addEmVector_({"/", "EM_B", "em"},
                           [this](GridLayout const&, double, bool) -> VecField {
                               return this->model_.state.electromag.B;
                           });
        this->addEmVector_({"/", "EM_E", "em"},
                           [this](GridLayout const&, double, bool) -> VecField {
                               return this->model_.state.electromag.E;
                           });

        this->addDerivedQuantityEntries_(derived_, "/ions/", "ion");
    }

    NO_DISCARD auto& state() { return this->model_.state; }
    NO_DISCARD auto const& state() const { return this->model_.state; }

    NO_DISCARD auto const& derivedQuantities() const { return derived_; }

    NO_DISCARD VecField& derivedVecScratch() { return derivedVecScratch_; }

    NO_DISCARD Field& derivedScalarScratch() { return derivedScalarScratch_; }

    NO_DISCARD VecField& getB() const { return this->model_.state.electromag.B; }

    NO_DISCARD VecField& getE() const { return this->model_.state.electromag.E; }

    NO_DISCARD auto& getIons() const { return this->model_.state.ions; }


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
        std::string const tree{"/mhd/"};
        std::string const group{"mhd"};

        this->addFluidScalar_(
            {tree, "rho", group},
            [this](GridLayout const&, double, bool) -> Field { return this->model_.state.rho; });
        this->addFluidVector_({tree, "rhoV", group},
                              [this](GridLayout const&, double, bool) -> VecField {
                                  return this->model_.state.rhoV;
                              });
        this->addFluidScalar_(
            {tree, "Etot", group},
            [this](GridLayout const&, double, bool) -> Field { return this->model_.state.Etot; });

        this->addEmVector_(
            {"/", "EM_B", "em"},
            [this](GridLayout const&, double, bool) -> VecField { return this->model_.state.B; });

        this->addDerivedQuantityEntries_(derived_, tree, group);
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

protected:
    Field tmpField_{"PHARE_sumField_MHD", core::MHDQuantity::Scalar::ScalarAllPrimal};
    VecField tmpVec_{"PHARE_sumVec_MHD", core::MHDQuantity::Vector::VecAllPrimal};

    Field derivedScalarScratch_{"PHARE_derived_scalar", core::MHDQuantity::Scalar::ScalarAllPrimal};
    VecField derivedVecScratch_{"PHARE_derived_vec", core::MHDQuantity::Vector::VecAllPrimal};

    core::DerivedQuantityRegistry<State_t, GridLayout> derived_;
};


} // namespace PHARE::diagnostic



#endif // DIAGNOSTIC_MODEL_VIEW_HPP
