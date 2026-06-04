#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/data/vecfield/vecfield_initializer.hpp"
#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/models/physical_state.hpp"
#include "core/numerics/ampere/ampere.hpp"
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
                   and J0.isUsable() and E.isUsable();
        }

        NO_DISCARD bool isSettable() const
        {
            return rho.isSettable() and V.isSettable() and B1.isSettable() and B0.isSettable()
                   and P.isSettable() and rhoV.isSettable() and Etot1.isSettable()
                   and J.isSettable() and J0.isSettable() and E.isSettable();
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, J0, E);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, J0, E);
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
            , J0{dict["name"].template to<std::string>() + "_" + "J0", MHDQuantity::Vector::J}


            , rhoinit_{dict["density"]["initializer"]
                           .template to<initializer::InitFunction<dimension>>()}
            , Vinit_{dict["velocity"]["initializer"]}
            , totalBInit_{dict["magnetic"]["initializer"]}
            , B0init_{dict["external_magnetic"]["initializer"]}
            , Pinit_{dict["pressure"]["initializer"]
                         .template to<initializer::InitFunction<dimension>>()}
            , gamma_{dict["to_conservative_init"]["heat_capacity_ratio"].template to<double>()}
            , b0FromPotential_{cppdict::get_value(dict, "b0_init_mode", std::string{"components"})
                               == "potential"}
            , b1FromPotential_{cppdict::get_value(dict, "b1_init_mode", std::string{"components"})
                               == "potential"}
        {
            // Vector-potential init (2D): the potential functions are read only when the
            // corresponding field is in "potential" mode, so dicts that predate this feature
            // (e.g. C++ unit-test dicts) need not provide the potential_z keys.
            if (b0FromPotential_)
                a0zInit_ = dict["external_magnetic"]["initializer"]["potential_z"]
                               .template to<initializer::InitFunction<dimension>>();
            if (b1FromPotential_)
                a1zInit_ = dict["perturbed_magnetic"]["initializer"]["potential_z"]
                               .template to<initializer::InitFunction<dimension>>();
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
            , J0{name + "_" + "J0", MHDQuantity::Vector::J}

            , gamma_{}
        {
        }

        template<typename GridLayout>
        void updateExternalMagneticField(GridLayout const& layout, double /*time*/ = 0.)
        {
            if (b0FromPotential_)
                initBFromPotential_(a0zInit_, B0, layout);
            else
                B0init_.initialize(B0, layout);
            Ampere_ref<GridLayout>{layout}(B0, J0); // background current j0 = curl(B0)
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
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            // B0 (analytic background field)
            if (b0FromPotential_)
                initBFromPotential_(a0zInit_, B0, layout);
            else
                B0init_.initialize(B0, layout);
            Ampere_ref<GridLayout>{layout}(B0, J0); // background current j0 = curl(B0)

            // B1 (evolved perturbation)
            if (b1FromPotential_)
            {
                initBFromPotential_(a1zInit_, B1, layout);
            }
            else
            {
                totalBInit_.initialize(B1, layout);
                // In component mode dict["magnetic"] holds the total field B0 + B1, so subtract
                // the analytic B0 to recover B1. When B0 itself comes from a vector potential the
                // Python side could not fold it into the total, so dict["magnetic"] already equals
                // the perturbation and no subtraction is needed.
                if (!b0FromPotential_)
                    for (auto const& component : {Component::X, Component::Y, Component::Z})
                    {
                        auto& B1c       = B1(component);
                        auto const& B0c = B0(component);
                        layout.evalOnGhostBox(
                            B1c, [&](auto&... args) mutable { B1c(args...) -= B0c(args...); });
                    }
            }

            // The potential init used E as an A_z scratch buffer; clear it so t=0 diagnostics and
            // the first read see 0 (the constrained transport recomputes E before its real use).
            if (b0FromPotential_ || b1FromPotential_)
                for (auto const& component : {Component::X, Component::Y, Component::Z})
                {
                    auto& Ec = E(component);
                    layout.evalOnGhostBox(Ec, [&](auto&... args) mutable { Ec(args...) = 0.0; });
                }

            ToConservativeConverter_ref{layout, gamma_}(
                rho, V, B1, B0, P, rhoV, Etot1); // initial to conservative conversion because we
                                                 // store conservative quantities on the grid
        }

        /**
         * @brief Initialise a face-centred magnetic field as the discrete curl of an out-of-plane
         * vector potential A_z: B = curl(A_z z_hat) = (dA_z/dy, -dA_z/dx, 0) (2D only).
         *
         * A_z is sampled at E_z (edge) centring into the E_z scratch buffer over its full ghost
         * box, then B is filled on the domain with the same discrete `deriv` used by Faraday, so
         * the discrete div B = 0 to machine precision. B ghosts are filled by the runtime boundary
         * / messenger machinery (and, for B0, by `updateExternalMagneticField`).
         */
        template<typename GridLayout>
        void initBFromPotential_(initializer::InitFunction<dimension> const& aInit, VecFieldT& Bout,
                                 GridLayout const& layout)
        {
            if constexpr (dimension == 2)
            {
                auto& Az = E(Component::Z); // E_z-centred scratch (overwritten before real use)
                FieldUserFunctionInitializer::initialize(Az, layout, aInit);

                auto& Bx = Bout(Component::X);
                auto& By = Bout(Component::Y);
                auto& Bz = Bout(Component::Z);

                // Fill the full ghost box so B0 ghosts (read by the flux reconstruction and
                // refreshed each substep by updateExternalMagneticField) and B1 ghosts are not
                // left NaN. A_z is filled over its own ghost box, so the discrete deriv is exact
                // on the domain; the outermost ghost layer is approximate but finite.
                layout.evalOnGhostBox(Bx, [&](auto&... args) {
                    Bx(args...) = layout.template deriv<Direction::Y>(Az, {args...});
                });
                layout.evalOnGhostBox(By, [&](auto&... args) {
                    By(args...) = -layout.template deriv<Direction::X>(Az, {args...});
                });
                layout.evalOnGhostBox(Bz, [&](auto&... args) { Bz(args...) = 0.0; });
            }
            else
            {
                (void)aInit;
                (void)Bout;
                (void)layout;
                throw std::runtime_error(
                    "MHDState: vector-potential init (a0z/a1z) is only supported in 2D");
            }
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
        VecFieldT J0; // background current = curl(B0), recomputed whenever B0 is set

    private:
        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> totalBInit_;
        VecFieldInitializer<dimension> B0init_;
        initializer::InitFunction<dimension> Pinit_;

        double const gamma_;

        // Vector-potential init (2D): build B0/B1 from an out-of-plane potential A_z so that
        // div B = 0 discretely. Defaults keep the legacy component-wise init.
        bool b0FromPotential_ = false;
        bool b1FromPotential_ = false;
        initializer::InitFunction<dimension> a0zInit_;
        initializer::InitFunction<dimension> a1zInit_;
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
