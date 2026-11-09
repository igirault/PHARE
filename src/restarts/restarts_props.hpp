#ifndef RESTART_DAO_HPP
#define RESTART_DAO_HPP

#include <cstddef>
#include <vector>

#include "dict.hpp"

#include "core/utilities/time_targets.hpp"

#include <optional>

namespace PHARE::restarts
{
struct RestartsProperties
{
    using FileAttributes = cppdict::Dict<std::string>;

    std::vector<double> writeTimestamps, elapsedTimestamps;

    FileAttributes fileAttributes{};

    std::optional<core::TimeRecord> timeRecord;
};

} // namespace PHARE::restarts

#endif // RESTART_DAO_HPP
