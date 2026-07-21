#ifndef PHARE_RESTART_MANAGER_HPP_
#define PHARE_RESTART_MANAGER_HPP_


#include "core/def.hpp"
#include "core/logger.hpp"
#include "core/utilities/mpi_utils.hpp"
#include "core/utilities/cadence.hpp"

#include "initializer/data_provider.hpp"

#include "restarts_props.hpp"


#include <cmath>
#include <memory>
#include <utility>



namespace PHARE::restarts
{
class IRestartsManager
{
public:
    virtual bool dump(double timeStamp, double timeStep, std::size_t stepIndex) = 0;
    inline virtual ~IRestartsManager();
};
IRestartsManager::~IRestartsManager() {}



template<typename Writer>
class RestartsManager : public IRestartsManager
{
public:
    bool dump(double timeStamp, double timeStep, std::size_t stepIndex) override;



    RestartsManager(std::unique_ptr<Writer>&& writer_ptr)
        : writer_{std::move(writer_ptr)}
    {
        if (!writer_)
            throw std::runtime_error("Error: RestartsManager received null Writer");
    }


    template<typename Hierarchy, typename ResourceManager_t>
    NO_DISCARD static std::unique_ptr<RestartsManager>
    make_unique(Hierarchy& hier, ResourceManager_t& resman, initializer::PHAREDict const& dict)
    {
        auto rMan = std::make_unique<RestartsManager>(Writer::make_unique(hier, resman, dict));
        auto restarts_are_written = core::any(
            core::generate([&](auto const& v) { return dict.contains(v); },
                           std::vector<std::string>{"write_timestamps", "elapsed_timestamps"}));
        // step_period leaves no timestamp arrays; activate on a non-zero iteration cadence too.
        if (dict.contains("write_step_period")
            and dict["write_step_period"].template to<std::size_t>() > 0)
            restarts_are_written = true;
        if (restarts_are_written) // else is only loading not saving restarts
            rMan->addRestartDict(dict);
        return rMan;
    }



    RestartsManager& addRestartDict(initializer::PHAREDict const& dict);
    RestartsManager& addRestartDict(initializer::PHAREDict&& dict) { return addRestartDict(dict); }


    NO_DISCARD Writer& writer() { return *writer_.get(); }


    RestartsManager(RestartsManager const&)            = delete;
    RestartsManager(RestartsManager&&)                 = delete;
    RestartsManager& operator=(RestartsManager const&) = delete;
    RestartsManager& operator=(RestartsManager&&)      = delete;

private:
    bool needsWrite_(RestartsProperties const& rest, double const timeStamp, double const timeStep)
    {
        auto const simUnit
            = core::cadence_catch_up(rest.writeTimestamps, nextWriteSimUnit_, timeStamp, timeStep);

        auto const elapsed = nextWriteElapsed_ < rest.elapsedTimestamps.size()
                             and core::cadence_elapsed(rest.elapsedTimestamps[nextWriteElapsed_]);

        if (elapsed)
            ++nextWriteElapsed_;

        return simUnit || elapsed;
    }


    std::unique_ptr<RestartsProperties> restarts_properties_;
    std::unique_ptr<Writer> writer_;
    std::size_t nextWriteSimUnit_ = 0;
    std::size_t nextWriteElapsed_ = 0;

    std::time_t const start_time_{core::mpi::unix_timestamp_now()};
};



template<typename Writer>
RestartsManager<Writer>&
RestartsManager<Writer>::addRestartDict(initializer::PHAREDict const& params)
{
    if (restarts_properties_)
        throw std::runtime_error("Restarts invalid, properties already set");

    restarts_properties_ = std::make_unique<RestartsProperties>();

    if (params.contains("write_timestamps"))
        restarts_properties_->writeTimestamps
            = params["write_timestamps"].template to<std::vector<double>>();

    if (params.contains("write_step_period"))
        restarts_properties_->writeStepPeriod
            = params["write_step_period"].template to<std::size_t>();

    if (params.contains("elapsed_timestamps"))
    {
        restarts_properties_->elapsedTimestamps
            = params["elapsed_timestamps"].template to<std::vector<double>>();
        for (auto& time : restarts_properties_->elapsedTimestamps)
            time += start_time_; // expected for comparison later
    }

    assert(params.contains("serialized_simulation"));

    restarts_properties_->fileAttributes["serialized_simulation"]
        = params["serialized_simulation"].template to<std::string>();

    return *this;
}




template<typename Writer>
bool RestartsManager<Writer>::dump(double timeStamp, double timeStep,
                                   std::size_t stepIndex)
{
    if (!restarts_properties_)
        return false; // not active

    // iteration-based cadence: write every writeStepPeriod coarse steps (the only timestamp-free
    // option valid under adaptive dt). needsWrite_ is called unconditionally so its timestamp
    // index still advances. stepIndex is the caller's (Simulator's) real coarse-step
    // counter, restart-safe and independent of how many times dump() itself gets called.
    bool const niterNow = restarts_properties_->writeStepPeriod > 0
                          and (stepIndex % restarts_properties_->writeStepPeriod == 0);
    bool const writeNow = needsWrite_(*restarts_properties_, timeStamp, timeStep);

    if (!(writeNow || niterNow))
        return false; // not needed now

    PHARE_LOG_SCOPE(3, "RestartsManager::dump");
    writer_->dump(*restarts_properties_, timeStamp, stepIndex);
    return true;
}


} // namespace PHARE::restarts


#endif /* PHARE_RESTART_MANAGER_HPP_ */
