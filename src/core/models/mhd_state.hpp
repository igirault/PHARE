#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/data/vecfield/vecfield_initializer.hpp"
#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/models/physical_state.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
#include "core/utilities/index/index.hpp"

#include "initializer/data_provider.hpp"

namespace PHARE
{
namespace core
{

    template<typename VecFieldT>
    class MHDState : public IPhysicalState
    {
    public:
        using vecfield_type = VecFieldT;
        using field_type    = typename VecFieldT::field_type;

        static constexpr auto dimension = VecFieldT::dimension;

        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------

        NO_DISCARD bool isUsable() const
        {
            return rho.isUsable() and V.isUsable() and B1.isUsable() and B0.isUsable()
                   and P.isUsable() and rhoV.isUsable() and Etot1.isUsable() and J.isUsable()
                   and E.isUsable() and B0x_Ez.isUsable() and B0y_Ez.isUsable();
        }

        NO_DISCARD bool isSettable() const
        {
            return rho.isSettable() and V.isSettable() and B1.isSettable() and B0.isSettable()
                   and P.isSettable() and rhoV.isSettable() and Etot1.isSettable()
                   and J.isSettable() and E.isSettable() and B0x_Ez.isSettable()
                   and B0y_Ez.isSettable();
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, E, B0x_Ez, B0y_Ez);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, E, B0x_Ez, B0y_Ez);
        }

        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------

        MHDState(PHARE::initializer::PHAREDict const& dict)
            : rho{dict["name"].template to<std::string>() + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{dict["name"].template to<std::string>() + "_" + "V", MHDQuantity::Vector::V}
            , B1{dict["name"].template to<std::string>() + "_" + "B1", MHDQuantity::Vector::B1}
            , B0{dict["name"].template to<std::string>() + "_" + "B0", MHDQuantity::Vector::B0}
            , P{dict["name"].template to<std::string>() + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{dict["name"].template to<std::string>() + "_" + "rhoV",
                   MHDQuantity::Vector::rhoV}
            , Etot1{dict["name"].template to<std::string>() + "_" + "Etot1",
                    MHDQuantity::Scalar::Etot1}


            , E{dict["name"].template to<std::string>() + "_" + "E", MHDQuantity::Vector::E}
            , J{dict["name"].template to<std::string>() + "_" + "J", MHDQuantity::Vector::J}

            // B0 sampled analytically at the Ez edge (ppd corner): used by the constrained
            // transport so the motional EMF is well-balanced w.r.t. grad B0 (no 2-pt averaging).
            , B0x_Ez{dict["name"].template to<std::string>() + "_" + "B0x_Ez",
                     MHDQuantity::Scalar::Ez}
            , B0y_Ez{dict["name"].template to<std::string>() + "_" + "B0y_Ez",
                     MHDQuantity::Scalar::Ez}


            , rhoinit_{dict["density"]["initializer"]
                           .template to<initializer::InitFunction<dimension>>()}
            , Vinit_{dict["velocity"]["initializer"]}
            , totalBInit_{dict["magnetic"]["initializer"]}
            , B0init_{dict["external_magnetic"]["initializer"]}
            , b0xinit_{dict["external_magnetic"]["initializer"]["x_component"]
                           .template to<initializer::InitFunction<dimension>>()}
            , b0yinit_{dict["external_magnetic"]["initializer"]["y_component"]
                           .template to<initializer::InitFunction<dimension>>()}
            , Pinit_{dict["pressure"]["initializer"]
                         .template to<initializer::InitFunction<dimension>>()}
            , gamma_{dict["to_conservative_init"]["heat_capacity_ratio"].template to<double>()}
        {
        }

        MHDState(std::string name)
            : rho{name + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{name + "_" + "V", MHDQuantity::Vector::V}
            , B1{name + "_" + "B1", MHDQuantity::Vector::B1}
            , B0{name + "_" + "B0", MHDQuantity::Vector::B0}
            , P{name + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{name + "_" + "rhoV", MHDQuantity::Vector::rhoV}
            , Etot1{name + "_" + "Etot1", MHDQuantity::Scalar::Etot1}


            , E{name + "_" + "E", MHDQuantity::Vector::E}
            , J{name + "_" + "J", MHDQuantity::Vector::J}

            , B0x_Ez{name + "_" + "B0x_Ez", MHDQuantity::Scalar::Ez}
            , B0y_Ez{name + "_" + "B0y_Ez", MHDQuantity::Scalar::Ez}

            // mirror B0init_ (VecFieldInitializer) which zero-defaults here: keep the edge-B0
            // init functions callable so updateExternalMagneticField never throws on states
            // built without a dict (e.g. RK intermediate / temporary states).
            , b0xinit_{zeroInit_()}
            , b0yinit_{zeroInit_()}

            , gamma_{}
        {
        }

        template<typename GridLayout>
        void updateExternalMagneticField(GridLayout const& layout, double /*time*/ = 0.)
        {
            B0init_.initialize(B0, layout);
            // resample B0x, B0y directly at the Ez edge (ppd) for the well-balanced CT EMF
            FieldUserFunctionInitializer::initialize(B0x_Ez, layout, b0xinit_);
            FieldUserFunctionInitializer::initialize(B0y_Ez, layout, b0yinit_);
        }

        /**
         * @brief Reset a single inactive/ghost cell to a physically safe state.
         *
         * Inactive cells sit deep inside an embedded body and play no role in the fluid
         * solve, but their conservative values still flow through to-primitive conversion,
         * mixing steps, and diagnostics. Pin them to (rho=1, P=1, V=0) and recompute Etot
         * from the current face-centered B at the cell centre.
         *
         * Caller is responsible for checking cellStatus(idx) before invoking.
         */
        template<typename GridLayout, typename Thermo>
        void safeResetInactiveCell(MeshIndex<dimension> const& idx, GridLayout const& /*layout*/,
                                   Thermo& thermo)
        {
            constexpr double safeRho = 1.0;
            constexpr double safeP   = 1.0;

            rho(idx) = safeRho;
            P(idx)   = safeP;

            V(Component::X)(idx) = 0.0;
            V(Component::Y)(idx) = 0.0;
            V(Component::Z)(idx) = 0.0;

            rhoV(Component::X)(idx) = 0.0;
            rhoV(Component::Y)(idx) = 0.0;
            rhoV(Component::Z)(idx) = 0.0;

            auto const bx = GridLayout::template project<GridLayout::faceXToCellCenter>(
                B1(Component::X), idx);
            auto const by = GridLayout::template project<GridLayout::faceYToCellCenter>(
                B1(Component::Y), idx);
            auto const bz = GridLayout::template project<GridLayout::faceZToCellCenter>(
                B1(Component::Z), idx);

            thermo.setState_DP(safeRho, safeP);
            auto const e_int = safeRho * thermo.internalEnergy();
            Etot1(idx) = totalEnergyFromInternalEnergy(e_int, safeRho, 0., 0., 0., bx, by, bz);
        }


        template<typename GridLayout>
        void initialize(GridLayout const& layout)
        {
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            totalBInit_.initialize(B1, layout);
            B0init_.initialize(B0, layout);
            // B0x, B0y sampled at the Ez edge (ppd) for the well-balanced CT EMF
            FieldUserFunctionInitializer::initialize(B0x_Ez, layout, b0xinit_);
            FieldUserFunctionInitializer::initialize(B0y_Ez, layout, b0yinit_);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            for (auto const& component : {Component::X, Component::Y, Component::Z})
            {
                auto& B1c       = B1(component);
                auto const& B0c = B0(component);
                layout.evalOnGhostBox(B1c,
                                      [&](auto&... args) mutable { B1c(args...) -= B0c(args...); });
            }

            ToConservativeConverter_ref{layout, gamma_}(
                rho, V, B1, B0, P, rhoV, Etot1); // initial to conservative conversion because we
                                                 // store conservative quantities on the grid
        }

        field_type rho;
        VecFieldT V;
        VecFieldT B1;
        VecFieldT B0;
        field_type P;

        VecFieldT rhoV;
        field_type Etot1;

        VecFieldT E;
        VecFieldT J;

        // B0 components sampled analytically at the Ez edge (ppd), for well-balanced CT EMF
        field_type B0x_Ez;
        field_type B0y_Ez;

    private:
        static initializer::InitFunction<dimension> zeroInit_()
        {
            if constexpr (dimension == 1)
                return [](auto const& x) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
            else if constexpr (dimension == 2)
                return [](auto const& x, auto const&) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
            else
                return [](auto const& x, auto const&, auto const&) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
        }

        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> totalBInit_;
        VecFieldInitializer<dimension> B0init_;
        initializer::InitFunction<dimension> b0xinit_;
        initializer::InitFunction<dimension> b0yinit_;
        initializer::InitFunction<dimension> Pinit_;

        double const gamma_;
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
