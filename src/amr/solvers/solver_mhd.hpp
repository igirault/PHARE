#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP


#include "amr/amr_constants.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"
#include "amr/utilities/box/amr_box.hpp"

#include "core/errors.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"
#include "core/logger.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/constrained_transport/poynting_stencil_defect.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"
#include "core/utilities/index/index.hpp"

#include "initializer/data_provider.hpp"

#include <SAMRAI/hier/BoxLevel.h>
#include <SAMRAI/hier/CoarseFineBoundary.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
class SolverMHD : public ISolver<AMR_Types>
{
private:
    static constexpr auto dimension = MHDModel::dimension;

    using patch_t     = AMR_Types::patch_t;
    using level_t     = AMR_Types::level_t;
    using hierarchy_t = AMR_Types::hierarchy_t;

    using FieldT      = MHDModel::field_type;
    using VecFieldT   = MHDModel::vecfield_type;
    using MHDStateT   = MHDModel::state_type;
    using GridLayout  = MHDModel::gridlayout_type;
    using Scalar      = GridLayout::Scalar;
    using MHDQuantity = core::MHDQuantity;
    using Messenger   = TimeIntegratorStrategy::Messenger;

    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;

    core::AllFluxes<FieldT, VecFieldT> fluxes_;
    core::MHDSources<FieldT, VecFieldT> sources_{{"B1_source", MHDQuantity::Vector::B1},
                                                 {"Etot1_source", MHDQuantity::Scalar::Etot1}};

    TimeIntegratorStrategy evolve_;

    // Refluxing
    MHDStateT stateOld_{this->name() + "_stateOld"};

    core::AllFluxes<FieldT, VecFieldT> fluxSum_;
    VecFieldT fluxSumE_{this->name() + "_fluxSumE", MHDQuantity::Vector::E};
    EulerUsingComputedFlux<MHDModel> reflux_euler_;

    std::unordered_map<std::size_t, double> oldTime_;

    struct CoarseFineFaces
    {
        std::weak_ptr<SAMRAI::hier::BoxLevel const> boxLevel;
        std::shared_ptr<SAMRAI::hier::CoarseFineBoundary> boundary;
    };
    std::unordered_map<int, CoarseFineFaces> coarseFineFaces_;

    double const gamma_; // adiabatic index
    double const eta_;   // resistivity
    double const nu_;
    core::HyperMode const hyperMode_;
    bool const isHall_;
    bool const consistentRefluxEnergy_;

public:
    SolverMHD(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"MHDSolver"}
        , fluxes_{{"rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                  {"rhoV_fx", MHDQuantity::Vector::VecFlux_x},
                  {"B_fx", MHDQuantity::Vector::VecFlux_x},
                  {"Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                  {"rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                  {"rhoV_fy", MHDQuantity::Vector::VecFlux_y},
                  {"B_fy", MHDQuantity::Vector::VecFlux_y},
                  {"Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                  {"rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                  {"rhoV_fz", MHDQuantity::Vector::VecFlux_z},
                  {"B_fz", MHDQuantity::Vector::VecFlux_z},
                  {"Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
        , evolve_{dict}
        , fluxSum_{{"sumRho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                   {"sumRhoV_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumB_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumEtot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                   {"sumRho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                   {"sumRhoV_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumB_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumEtot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                   {"sumRho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                   {"sumRhoV_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumB_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumEtot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
        , gamma_{dict["to_primitive"]["heat_capacity_ratio"].template to<double>()}
        , eta_{dict["constrained_transport"]["resistivity"].template to<double>()}
        , nu_{dict["fv_method"]["hyper_resistivity"].template to<double>()}
        , hyperMode_{cppdict::get_value(dict, "fv_method/hyper_mode", core::HyperMode::constant)}
        , isHall_{cppdict::get_value(dict, "fv_method/hall", false)}
        , consistentRefluxEnergy_{
              cppdict::get_value(dict, "reflux_energy", std::string{"conservative"})
              == "consistent"}
    {
        static_assert(amr::refinementRatio == 2,
                      "the consistent reflux energy assumes a refinement ratio of 2");
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;

    void registerResources(IPhysicalModel<AMR_Types>& model) override;

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& model, patch_t& patch,
                  double const allocateTime) const override;

    void prepareStep(IPhysicalModel_t& model, level_t& level, double const currentTime) override;

    void accumulateFluxSum(IPhysicalModel_t& model, level_t& level, double const coef) override;

    void resetFluxSum(IPhysicalModel_t& model, level_t& level) override;

    void reflux(IPhysicalModel_t& model, level_t& level, IMessenger& messenger,
                double const time) override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, IPhysicalModel_t& model,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;

    double computeStableDt(IPhysicalModel_t& model, level_t& level,
                           CFLNumbers const& cflNumbers) override;

    void onRegrid() override {}


    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(fluxes_, sources_, fluxSum_, fluxSumE_, stateOld_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(fluxes_, sources_, fluxSum_, fluxSumE_, stateOld_);
    }

private:
    void mhdNaNCheck_(MHDModel& state, level_t const& level, double time);

    void updateCoarseFineFaces_(hierarchy_t const& hierarchy, int const levelNumber);

    void removePoyntingStencilDefect_(MHDModel& mhdModel, level_t& level, double const coef);
};

// -----------------------------------------------------------------------------

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::registerResources(
    IPhysicalModel_t& model)
{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->registerResources(fluxes_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxes_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxes_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fz);
        }
    }

    mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fz);
        }
    }
    mhdmodel.resourcesManager->registerResources(fluxSumE_);

    mhdmodel.resourcesManager->registerResources(sources_);

    mhdmodel.resourcesManager->registerResources(stateOld_);

    evolve_.registerResources(mhdmodel);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::allocate(
    IPhysicalModel_t& model, patch_t& patch, double const allocateTime) const

{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->allocate(fluxes_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxes_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxes_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.Etot_fz, patch, allocateTime);
        }
    }

    mhdmodel.resourcesManager->allocate(fluxSum_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxSum_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxSum_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fz, patch, allocateTime);
        }
    }
    mhdmodel.resourcesManager->allocate(fluxSumE_, patch, allocateTime);

    mhdmodel.resourcesManager->allocate(sources_, patch, allocateTime);

    mhdmodel.resourcesManager->allocate(stateOld_, patch, allocateTime);

    evolve_.allocate(mhdmodel, patch, allocateTime);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::fillMessengerInfo(
    std::unique_ptr<amr::IMessengerInfo> const& info) const

{
    auto& mhdInfo = dynamic_cast<amr::MHDMessengerInfo&>(*info);

    evolve_.fillMessengerInfo(mhdInfo);

    auto&& [timeFluxes, timeElectric] = evolve_.exposeFluxes();

    mhdInfo.reflux          = core::AllFluxesNames{timeFluxes};
    mhdInfo.refluxElectric  = timeElectric.name();
    mhdInfo.fluxSum         = core::AllFluxesNames{fluxSum_};
    mhdInfo.fluxSumElectric = fluxSumE_.name();
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::prepareStep(IPhysicalModel_t& model,
                                                                         level_t& level,
                                                                         double const currentTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::prepareStep");

    oldTime_[level.getLevelNumber()] = currentTime;

    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    auto& rho   = mhdModel.state.rho;
    auto& rhoV  = mhdModel.state.rhoV;
    auto& B1    = mhdModel.state.B1;
    auto& Etot1 = mhdModel.state.Etot1;

    for (auto& patch : level)
    {
        auto dataOnPatch
            = mhdModel.resourcesManager->setOnPatch(*patch, rho, rhoV, B1, Etot1, stateOld_);

        mhdModel.resourcesManager->setTime(stateOld_.rho, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.rhoV, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.B1, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.Etot1, *patch, currentTime);

        stateOld_.rho.copyData(rho);
        stateOld_.rhoV.copyData(rhoV);
        stateOld_.B1.copyData(B1);
        stateOld_.Etot1.copyData(Etot1);
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::accumulateFluxSum(
    IPhysicalModel_t& model, level_t& level, double const coef)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::accumulateFluxSum");

    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rm       = *mhdModel.resourcesManager;

    // MacOS clang still unhappy with structured bindings captures in lambdas
    auto&& tf          = evolve_.exposeFluxes();
    auto& timeFluxes   = std::get<0>(tf);
    auto& timeElectric = std::get<1>(tf);

    for (auto& _ : rm.enumerate(level, fluxSum_, fluxSumE_, timeFluxes, timeElectric))
    {
        core::operate<core::PlusEqualsProduct>(fluxSum_, timeFluxes, coef);
        core::operate<core::PlusEqualsProduct>(fluxSumE_, timeElectric, coef);
    }

    if (consistentRefluxEnergy_ and level.getLevelNumber() > 0)
        removePoyntingStencilDefect_(mhdModel, level, coef);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::updateCoarseFineFaces_(
    hierarchy_t const& hierarchy, int const levelNumber)
{
    auto const& boxLevel = hierarchy.getPatchLevel(levelNumber)->getBoxLevel();
    auto& entry          = coarseFineFaces_[levelNumber];
    if (entry.boundary and entry.boxLevel.lock() == boxLevel)
        return;

    entry.boxLevel = boxLevel;
    entry.boundary = std::make_shared<SAMRAI::hier::CoarseFineBoundary>(
        hierarchy, levelNumber,
        SAMRAI::hier::IntVector::getZero(SAMRAI::tbox::Dimension{dimension}));
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::removePoyntingStencilDefect_(
    MHDModel& mhdModel, level_t& level, double const coef)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::removePoyntingStencilDefect_");

    if constexpr (dimension > 1)
    {
        auto const entry = coarseFineFaces_.find(level.getLevelNumber());
        if (entry == coarseFineFaces_.end() or !entry->second.boundary)
            throw std::runtime_error("SolverMHD: no coarse-fine boundary for this level");
        auto const& boundary = *entry->second.boundary;

        auto& rm           = *mhdModel.resourcesManager;
        auto&& tf          = evolve_.exposeFluxes();
        auto& timeElectric = std::get<1>(tf);
        auto& B            = mhdModel.state.B1;

        for (auto& patch : level)
        {
            auto const& boxes = boundary.getBoundaries(patch->getGlobalId(), 1);
            if (boxes.empty())
                continue;

            auto const patchBox = amr::phare_box_from<dimension>(patch->getBox());
            std::array<std::vector<core::Box<int, dimension>>, dimension> faces;
            for (auto const& boundaryBox : boxes)
            {
                auto box = amr::phare_box_from<dimension>(boundaryBox.getBox());
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    bool const below = box.upper[d] < patchBox.lower[d];
                    bool const above = box.lower[d] > patchBox.upper[d];
                    if (!below and !above)
                        continue;

                    auto face = box;
                    for (std::size_t k = 0; k < dimension; ++k)
                    {
                        face.lower[k] = std::max(box.lower[k], patchBox.lower[k]);
                        face.upper[k] = std::min(box.upper[k], patchBox.upper[k]);
                    }
                    face.lower[d] = face.upper[d]
                        = below ? patchBox.lower[d] : patchBox.upper[d] + 1;
                    faces[d].push_back(face);
                }
            }

            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            auto _            = rm.setOnPatch(*patch, fluxSum_, timeElectric, B, stateOld_);
            core::PoyntingStencilDefect<GridLayout>{layout}(fluxSum_, timeElectric, stateOld_.B1, B,
                                                            faces, coef);
        }
    }
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::resetFluxSum(IPhysicalModel_t& model,
                                                                          level_t& level)
{
    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rm       = *mhdModel.resourcesManager;

    for (auto& _ : rm.enumerate(level, fluxSum_, fluxSumE_))
    {
        fluxSum_.zero();
        fluxSumE_.zero();
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::reflux(IPhysicalModel_t& model,
                                                                    level_t& level,
                                                                    IMessenger& messenger,
                                                                    double const time)
{
    auto& bc                          = dynamic_cast<Messenger&>(messenger);
    auto& mhdModel                    = dynamic_cast<MHDModel&>(model);
    auto&& [timeFluxes, timeElectric] = evolve_.exposeFluxes();
    auto& timeSources                 = evolve_.exposeSources();

    reflux_euler_(mhdModel, stateOld_, mhdModel.state, timeElectric, timeFluxes, timeSources, bc,
                  level, time, time - oldTime_[level.getLevelNumber()]);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::advanceLevel(
    hierarchy_t const& hierarchy, int const levelNumber, IPhysicalModel_t& model,
    IMessenger& fromCoarserMessenger, double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::advanceLevel");

    auto& mhdModel    = dynamic_cast<MHDModel&>(model);
    auto& fromCoarser = dynamic_cast<Messenger&>(fromCoarserMessenger);
    auto level        = hierarchy.getPatchLevel(levelNumber);

    if (consistentRefluxEnergy_ and levelNumber > 0)
        updateCoarseFineFaces_(hierarchy, levelNumber);

    try
    {
        evolve_(mhdModel, mhdModel.state, fluxes_, sources_, fromCoarser, *level, currentTime,
                newTime);

        PHARE_DEBUG_DO({ mhdNaNCheck_(mhdModel, *level, currentTime); })
    }
    catch (core::DictionaryException& ex)
    {
        PHARE_LOG_ERROR(ex());
    }

    if (mpi::any_errors())
        throw core::DictionaryException{}("ID", "SolverMHD::advanceLevel");
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
double SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::computeStableDt(
    IPhysicalModel_t& model, level_t& level, CFLNumbers const& cflNumbers)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::computeStableDt");

    auto const [cflHyperbolic, cflWhistler, cflDiffusive, cflHyperResistive] = cflNumbers;

    auto const invCflHyperbolic = 1. / cflHyperbolic;
    auto const invCflWhistler   = 1. / cflWhistler;

    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rho      = mhdModel.state.rho;
    auto& rhoV     = mhdModel.state.rhoV;
    auto& B1       = mhdModel.state.B1;
    auto& Etot1    = mhdModel.state.Etot1;
    auto& external = mhdModel.externalFieldManager.state();

    double maxInvDt          = 0;
    bool nonFiniteCellLogged = false;

    auto const& floors = mhdModel.positivityFloors;
    auto floored       = [](double const value, double const floor) {
        return floor > 0. && value < floor ? floor : value;
    };

    auto scanPatch = [&](GridLayout const& layout, auto&& isEvolved) {
        auto const meshSize = layout.meshSize();

        double invdx2 = 0;
        for (std::size_t d = 0; d < dimension; ++d)
            invdx2 += 1.0 / (meshSize[d] * meshSize[d]);
        auto const invDtDiffusive = 4.0 * eta_ * invdx2 / cflDiffusive;
        auto const hyperSymbol    = 4.0 * invdx2 * 4.0 * invdx2 / cflHyperResistive;
        auto const dxMin          = *std::min_element(meshSize.begin(), meshSize.end());

        auto const& rhoVx = rhoV(core::Component::X);
        auto const& rhoVy = rhoV(core::Component::Y);
        auto const& rhoVz = rhoV(core::Component::Z);
        auto const& B1x   = B1(core::Component::X);
        auto const& B1y   = B1(core::Component::Y);
        auto const& B1z   = B1(core::Component::Z);

        // wave cfl constraint
        mhdModel.externalFieldManager.state().visitB0([&](auto const& B0) {
            layout.evalOnBox(rho, [&](auto&... args) mutable {
                core::MeshIndex<dimension> const index{args...};
                if (!isEvolved(index))
                    return;

                auto const r  = floored(rho(index), floors.density);
                auto const vx = rhoVx(index) / r;
                auto const vy = rhoVy(index) / r;
                auto const vz = rhoVz(index) / r;

                auto const b1x = GridLayout::template project<Scalar::Bx, Scalar::rho>(B1x, index);
                auto const b1y = GridLayout::template project<Scalar::By, Scalar::rho>(B1y, index);
                auto const b1z = GridLayout::template project<Scalar::Bz, Scalar::rho>(B1z, index);

                auto const bx = b1x
                                + GridLayout::template project<Scalar::Bx, Scalar::rho>(
                                    B0(core::Component::X), index);
                auto const by = b1y
                                + GridLayout::template project<Scalar::By, Scalar::rho>(
                                    B0(core::Component::Y), index);
                auto const bz = b1z
                                + GridLayout::template project<Scalar::Bz, Scalar::rho>(
                                    B0(core::Component::Z), index);

                auto const P
                    = floored(core::eosEtot1ToP(gamma_, r, vx, vy, vz, b1x, b1y, b1z, Etot1(index)),
                              floors.pressure);
                auto const BdotB = bx * bx + by * by + bz * bz;

                std::array<double, 3> const v{vx, vy, vz};
                std::array<double, 3> const b{bx, by, bz};

                double invDtHyperbolic = 0;
                double invDtWhistler   = 0;
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    auto const cfast = core::compute_fast_magnetosonic_(gamma_, r, b[d], BdotB, P);
                    invDtHyperbolic += (std::abs(v[d]) + cfast) / meshSize[d];
                    if (isHall_)
                    {
                        auto const k = 2.0 / meshSize[d];
                        invDtWhistler
                            += k * (core::compute_whistler_(k, r, BdotB) - std::sqrt(BdotB / r));
                    }
                }
                invDtHyperbolic *= invCflHyperbolic;
                invDtWhistler *= invCflWhistler;
                auto invDt = invDtHyperbolic + invDtWhistler + invDtDiffusive;
                if (nu_ > 0)
                {
                    auto const kappa = hyperMode_ == core::HyperMode::spatial
                                           ? nu_ * dxMin * dxMin * (std::sqrt(BdotB) / r + 1.0)
                                           : nu_;
                    invDt += kappa * hyperSymbol;
                }
                if (!std::isfinite(invDt))
                {
                    if (!nonFiniteCellLogged)
                    {
                        auto const amrIndex
                            = layout.localToAMR(core::Point<std::uint32_t, dimension>{args...});
                        auto const position = layout.fieldNodeCoordinates(rho, amrIndex);
                        std::stringstream ss;
                        ss << "non-finite wave speed on level " << level.getLevelNumber()
                           << ": cell at (";
                        for (std::size_t d = 0; d < dimension; ++d)
                            ss << (d ? ", " : "") << position[d];
                        ss << "): rho = " << r << ", P = " << P << ", |B| = " << std::sqrt(BdotB);
                        PHARE_LOG_ERROR(ss.str());
                        nonFiniteCellLogged = true; // print only the first one
                    }
                    return;
                }
                maxInvDt = std::max(maxInvDt, invDt);
            });
        });
    };

    auto& rm = *mhdModel.resourcesManager;
    if (mhdModel.hasInnerBoundary())
    {
        auto& ibm = *mhdModel.innerBoundaryManager;
        for (auto& patch : rm.enumerate(level, rho, rhoV, B1, Etot1, external, ibm))
        {
            auto const& cellStatus = ibm.getMeshData().cellStatusField();
            scanPatch(amr::layoutFromPatch<GridLayout>(*patch), [&](auto const& index) {
                return cellStatus(index) != core::toDouble(core::ElemStatus::Inactive);
            });
        }
    }
    else
    {
        for (auto& patch : rm.enumerate(level, rho, rhoV, B1, Etot1, external))
            scanPatch(amr::layoutFromPatch<GridLayout>(*patch), [](auto const&) { return true; });
    }

    if (nonFiniteCellLogged)
        return -1.0;

    return maxInvDt > 0 ? 1.0 / maxInvDt : std::numeric_limits<double>::max();
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::mhdNaNCheck_(MHDModel& model,
                                                                          level_t const& level,
                                                                          double time)
{
    auto& rm  = model.resourcesManager;
    auto& rho = model.state.rho;

    auto check_nans = [&](auto const& field, auto const& origin,
                          core::MeshIndex<MHDModel::dimension> const& index) {
        if (std::isnan(field(index)))
        {
            std::stringstream ss;
            ss << "NaN detected in MHD field at index " << index << " on patch of origin " << origin
               << " on level " << level.getLevelNumber() << " at time " << time;
            core::DictionaryException ex{"cause", ss.str()};
            throw ex;
        }
    };

    for (auto const& patch : rm->enumerate(level, rho))
    {
        auto layout = amr::layoutFromPatch<GridLayout>(*patch);
        layout.evalOnGhostBox(
            rho, [&](auto const&... args) { check_nans(rho, layout.origin(), {args...}); });
    }
}

} // namespace PHARE::solver

#endif
