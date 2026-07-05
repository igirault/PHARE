#ifndef PHARE_CORE_DATA_DERIVED_QUANTITY_CENTERING_HPP
#define PHARE_CORE_DATA_DERIVED_QUANTITY_CENTERING_HPP

#include "core/mhd/mhd_quantities.hpp"
#include "core/hybrid/hybrid_quantities.hpp"

#include <stdexcept>
#include <type_traits>

namespace PHARE::core
{
enum class ScalarCentering { cell, node };
enum class VectorCentering { cell, Elike, Blike };


template<typename PhysicalQuantity>
auto scalar_qty(ScalarCentering const centering)
{
    if constexpr (std::is_same_v<PhysicalQuantity, MHDQuantity>)
        return centering == ScalarCentering::cell ? MHDQuantity::Scalar::ScalarCellCentered
                                                  : MHDQuantity::Scalar::ScalarNodeCentered;
    else
    {
        if (centering == ScalarCentering::cell)
            throw std::runtime_error("no cell-centered scalar quantity for hybrid");
        return HybridQuantity::Scalar::ScalarNodeCentered;
    }
}


template<typename PhysicalQuantity>
auto vector_qty(VectorCentering const centering)
{
    using Vector = typename PhysicalQuantity::Vector;

    if (centering == VectorCentering::Elike)
        return Vector::VecElike;
    if (centering == VectorCentering::Blike)
        return Vector::VecBlike;

    if constexpr (std::is_same_v<PhysicalQuantity, MHDQuantity>)
        return MHDQuantity::Vector::VecCellCentered;
    else
        throw std::runtime_error("no cell-centered vector quantity for hybrid");
}

} // namespace PHARE::core

#endif
