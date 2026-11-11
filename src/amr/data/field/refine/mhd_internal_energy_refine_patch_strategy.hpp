#ifndef PHARE_AMR_MHD_INTERNAL_ENERGY_REFINE_PATCH_STRATEGY_HPP
#define PHARE_AMR_MHD_INTERNAL_ENERGY_REFINE_PATCH_STRATEGY_HPP

#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/utilities/box/amr_box.hpp"

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/utilities/box/box_spans.hpp"
#include "core/utilities/point/point.hpp"

#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/tbox/Dimension.h"
#include "SAMRAI/xfer/RefinePatchStrategy.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace PHARE::amr
{

template<typename FieldDataT, typename VecFieldDataT>
class MHDInternalEnergyRefinePatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    using gridlayout_type = FieldDataT::gridlayout_type;
    using grid_type       = FieldDataT::grid_type;
    using Scalar          = core::MHDQuantity::Scalar;
    using Fill            = detail::CompositeFill<gridlayout_type, grid_type, 2>;

    static constexpr std::size_t dimension = FieldDataT::dimension;

public:
    enum class Rebuild { everywhere, nonPositivePressure };

    explicit MHDInternalEnergyRefinePatchStrategy(Rebuild const rebuild)
        : rebuild_{rebuild}
    {
    }

    void registerIDs(int const rho, int const rhoV, int const B, int const Etot)
    {
        rhoId_  = rho;
        rhoVId_ = rhoV;
        BId_    = B;
        EtotId_ = Etot;
    }

    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch&, double const,
                                       SAMRAI::hier::IntVector const&) override
    {
    }

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 1};
    }

    void preprocessRefine(SAMRAI::hier::Patch&, SAMRAI::hier::Patch const&,
                          SAMRAI::hier::Box const&, SAMRAI::hier::IntVector const&) override
    {
    }

    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                           SAMRAI::hier::Box const& fine_box,
                           SAMRAI::hier::IntVector const& ratio) override
    {
        assert(rhoId_ >= 0 and rhoVId_ >= 0 and BId_ >= 0 and EtotId_ >= 0);

        auto const fineGhostBox = fine.getPatchData(EtotId_)->getGhostBox();
        auto const fillBox      = fine_box * fineGhostBox;
        if (fillBox.empty())
            return;

        auto const& fineLayout    = FieldDataT::getLayout(fine, EtotId_);
        auto& fineEtot            = FieldDataT::getField(fine, EtotId_);
        auto const fineNonThermal = nonThermalEnergy_(fine);

        std::vector<core::Point<int, dimension>> rebuilt;
        core::for_each_point(phare_box_from<dimension>(fillBox), [&](auto const& amr) {
            auto const cell = fineLayout.AMRToLocal(amr);
            if (rebuild_ == Rebuild::everywhere or fineEtot(cell) - fineNonThermal(cell) <= 0.)
            {
                fineEtot(cell) = nan_;
                rebuilt.push_back(amr);
            }
        });
        if (rebuilt.empty())
            return;

        auto const coarseGhostBox = coarse.getPatchData(EtotId_)->getGhostBox();
        auto coarseCells          = fillBox;
        coarseCells.coarsen(ratio);
        coarseCells.grow(SAMRAI::hier::IntVector::getOne(fillBox.getDim()));
        coarseCells = coarseCells * coarseGhostBox;

        auto const& coarseLayout    = FieldDataT::getLayout(coarse, EtotId_);
        auto const& coarseEtot      = FieldDataT::getField(coarse, EtotId_);
        auto const coarseNonThermal = nonThermalEnergy_(coarse);

        grid_type rhoEps{"rhoEps", Scalar::Etot, coarseEtot.shape(), nan_};
        core::for_each_point(phare_box_from<dimension>(coarseCells), [&](auto const& amr) {
            auto const cell = coarseLayout.AMRToLocal(amr);
            rhoEps(cell)    = coarseEtot(cell) - coarseNonThermal(cell);
        });

        std::array<core::QtyCentering, dimension> centering;
        centering.fill(core::QtyCentering::dual);
        Fill::fillLimited(rhoEps, fineEtot, fillBox, centering, fineGhostBox, coarseGhostBox,
                          ratio);

        for (auto const& amr : rebuilt)
        {
            auto const cell = fineLayout.AMRToLocal(amr);
            fineEtot(cell) += fineNonThermal(cell);
        }
    }

private:
    static constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();

    auto nonThermalEnergy_(SAMRAI::hier::Patch const& patch) const
    {
        auto const& rho  = FieldDataT::getField(patch, rhoId_);
        auto const& rhoV = VecFieldDataT::getFields(patch, rhoVId_);
        auto const& B    = VecFieldDataT::getFields(patch, BId_);

        return [&](core::Point<std::uint32_t, dimension> const& cell) {
            auto const index = cell.template as<std::size_t>();

            auto const m2 = rhoV[0](cell) * rhoV[0](cell) + rhoV[1](cell) * rhoV[1](cell)
                            + rhoV[2](cell) * rhoV[2](cell);

            auto const bx
                = gridlayout_type::template project<Scalar::Bx, Scalar::Etot>(B[0], index);
            auto const by
                = gridlayout_type::template project<Scalar::By, Scalar::Etot>(B[1], index);
            auto const bz
                = gridlayout_type::template project<Scalar::Bz, Scalar::Etot>(B[2], index);

            return 0.5 * m2 / rho(cell) + 0.5 * (bx * bx + by * by + bz * bz);
        };
    }

    Rebuild rebuild_;
    int rhoId_  = -1;
    int rhoVId_ = -1;
    int BId_    = -1;
    int EtotId_ = -1;
};

} // namespace PHARE::amr

#endif
