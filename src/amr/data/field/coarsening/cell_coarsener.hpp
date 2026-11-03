#ifndef PHARE_AMR_DATA_FIELD_COARSENING_CELL_COARSENER_HPP
#define PHARE_AMR_DATA_FIELD_COARSENING_CELL_COARSENER_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"
#include "amr/resources_manager/amr_utils.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

namespace PHARE::amr
{
template<std::size_t dimension, bool maximum>
class CellCoarsener
{
public:
    CellCoarsener(std::array<core::QtyCentering, dimension> const centering,
                  SAMRAI::hier::Box const& sourceBox, SAMRAI::hier::Box const& destinationBox,
                  SAMRAI::hier::IntVector const&)
        : sourceBox_{sourceBox}
        , destinationBox_{destinationBox}
    {
        for ([[maybe_unused]] auto const c : centering)
            assert(c == core::QtyCentering::dual);
    }

    template<typename FieldT>
    void operator()(FieldT const& fineField, FieldT& coarseField,
                    core::Point<int, dimension> coarseIndex)
    {
        core::Point<int, dimension> fineStart;
        for (std::size_t d = 0; d < dimension; ++d)
            fineStart[d] = coarseIndex[d] * ratio_;

        fineStart                = AMRToLocal(fineStart, sourceBox_);
        auto const coarseLocal   = AMRToLocal(coarseIndex, destinationBox_);
        double reduced           = maximum ? std::numeric_limits<double>::lowest() : 0.;
        auto constexpr nChildren = [] {
            int n = 1;
            for (std::size_t d = 0; d < dimension; ++d)
                n *= ratio_;
            return n;
        }();

        for (int child = 0; child < nChildren; ++child)
        {
            core::Point<int, dimension> fine = fineStart;
            int rest                         = child;
            for (std::size_t d = 0; d < dimension; ++d)
            {
                fine[d] += rest % ratio_;
                rest /= ratio_;
            }
            double const value = fineField(fine.template as<std::size_t>());
            reduced            = maximum ? std::max(reduced, value) : reduced + value;
        }

        coarseField(coarseLocal.template as<std::size_t>())
            = maximum ? reduced : reduced / static_cast<double>(nChildren);
    }

private:
    SAMRAI::hier::Box const sourceBox_;
    SAMRAI::hier::Box const destinationBox_;
    static int constexpr ratio_ = 2;
};

template<std::size_t dimension>
using ConservativeCellCoarsener = CellCoarsener<dimension, false>;

template<std::size_t dimension>
using MaxCellCoarsener = CellCoarsener<dimension, true>;

} // namespace PHARE::amr

#endif
