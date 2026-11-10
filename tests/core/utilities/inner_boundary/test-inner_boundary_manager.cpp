#include "gtest/gtest.h"

#include <cmath>

#include "core/data/field/field.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_manager.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "initializer/data_provider.hpp"

#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include "tests/core/utilities/inner_boundary/inner_boundary_test_fixtures.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace PHARE::core;

constexpr double eps            = 1e-12;
constexpr double adiabaticIndex = 5.0 / 3.0;

using GridLayout  = IBTestGridLayout2D;
using ScalarField = Field<2, MHDQuantity::Scalar, double>;
using MeshData    = InnerBoundaryMeshData<2, MHDQuantity>;

struct StoredScalar
{
    StoredScalar(std::string const& name, MHDQuantity::Scalar qty, GridLayout const& layout)
        : storage{layout.allocSize(qty)}
        , field{name, qty, storage.data(), storage.shape()}
    {
    }

    void fill(double v)
    {
        for (auto i = 0u; i < field.shape()[0]; ++i)
            for (auto j = 0u; j < field.shape()[1]; ++j)
                field(i, j) = v;
    }

    NdArrayVector<2, double> storage;
    ScalarField field;
};

void fillVec(UsableVecFieldMHD<2>& v, std::array<double, 3> const& values)
{
    for (auto c : {Component::X, Component::Y, Component::Z})
    {
        auto& f = v.getComponent(c);
        for (auto i = 0u; i < f.shape()[0]; ++i)
            for (auto j = 0u; j < f.shape()[1]; ++j)
                f(i, j) = values[static_cast<std::size_t>(c)];
    }
}

struct TestStateStorage
{
    explicit TestStateStorage(GridLayout const& layout)
        : rho_s{"rho", MHDQuantity::Scalar::rho, layout}
        , P_s{"P", MHDQuantity::Scalar::P, layout}
        , Etot1_s{"Etot1", MHDQuantity::Scalar::Etot1, layout}
        , rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV}
        , B1{"B1", layout, MHDQuantity::Vector::B1}
        , E{"E", layout, MHDQuantity::Vector::E}
    {
    }

    StoredScalar rho_s;
    StoredScalar P_s;
    StoredScalar Etot1_s;
    ScalarField& rho   = rho_s.field;
    ScalarField& P     = P_s.field;
    ScalarField& Etot1 = Etot1_s.field;
    UsableVecFieldMHD<2> rhoV;
    UsableVecFieldMHD<2> B1;
    UsableVecFieldMHD<2> E;
};

using Manager = InnerBoundaryManager<MHDQuantity, ScalarField, GridLayout, TestStateStorage>;

struct MeshDataStorage
{
    explicit MeshDataStorage(GridLayout const& layout)
        : phi_storage{layout.allocSize(MHDQuantity::Scalar::NodeCentered)}
    {
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
            elem_storages.emplace_back(
                layout.allocSize(MeshData::scalarFromCentering(MeshData::idxToCentering(i))));
    }

    void bind(MeshData& md)
    {
        ScalarField phi_field{md.signedDistanceAtNodes.name(), MHDQuantity::Scalar::NodeCentered,
                              phi_storage.data(), phi_storage.shape()};
        md.signedDistanceAtNodes.setBuffer(&phi_field);
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
        {
            auto const qty = MeshData::scalarFromCentering(MeshData::idxToCentering(i));
            ScalarField tmp{md.elemStatus[i].name(), qty, elem_storages[i].data(),
                            elem_storages[i].shape()};
            md.elemStatus[i].setBuffer(&tmp);
        }
        md.ghostElemsData._data    = &ghost_array;
        md.degradedElemsData._data = &degraded_array;
    }

    NdArrayVector<2, double> phi_storage;
    std::vector<NdArrayVector<2, double>> elem_storages;
    GhostElemPack<2>::ghost_elem_array_type ghost_array{};
    GhostElemPack<2>::ghost_elem_array_type degraded_array{};
};

struct ManagerFixture
{
    static constexpr char const* BOUNDARY_NAME = "test";
    static constexpr double density            = 2.0;
    static constexpr double pressure           = 0.25;

    Box<int, 2> amr_box{{-10, -10}, {9, 9}};
    IBTestGridLayout2D layout{{1.0, 1.0}, {20u, 20u}, {-10.0, -10.0}, amr_box};
    MeshDataStorage storage{layout};

    std::vector<MHDQuantity::Scalar> scalarQtys{MHDQuantity::Scalar::rho,
                                                MHDQuantity::Scalar::Etot1};
    std::vector<MHDQuantity::Vector> vectorQtys{MHDQuantity::Vector::B1, MHDQuantity::Vector::rhoV,
                                                MHDQuantity::Vector::E};

    Manager manager;

    explicit ManagerFixture(Manager::SafeStateValues safe = {},
                            InnerBoundaryConditionType type
                            = InnerBoundaryConditionType::FixedState,
                            Point<double, 2> center = {0.05, 0.05})
        : manager{std::make_unique<SphereInnerBoundary<2>>(BOUNDARY_NAME, center, 4.0),
                  type,
                  scalarQtys,
                  vectorQtys,
                  adiabaticIndex,
                  FixedStateValues{density, pressure},
                  safe}
    {
        storage.bind(manager.getMeshData());
        manager.classify(layout);
    }

    template<typename FieldT>
    std::size_t count(FieldT const& f, ElemStatus status) const
    {
        auto const& st = manager.getMeshData().getStatusFieldFromCentering(layout.centering(f));
        std::size_t n  = 0;
        for (auto i = 0u; i < f.shape()[0]; ++i)
            for (auto j = 0u; j < f.shape()[1]; ++j)
                if (st(i, j) == toDouble(status))
                    ++n;
        return n;
    }
};

PHARE::initializer::PHAREDict sphereDict()
{
    PHARE::initializer::PHAREDict dict;
    dict["inner_boundary"]["name"]           = std::string{"body"};
    dict["inner_boundary"]["shape"]          = std::string{"sphere"};
    dict["inner_boundary"]["center"]         = std::vector<double>{0.05, 0.05};
    dict["inner_boundary"]["radius"]         = 4.0;
    dict["inner_boundary"]["condition_type"] = std::string{"fixed-state"};
    return dict;
}

auto cellCenter(GridLayout const& layout, auto const& vec, Component c, MeshIndex<2> idx)
{
    if (c == Component::X)
        return GridLayout::project<MHDQuantity::Scalar::Bx, MHDQuantity::Scalar::rho>(
            vec(Component::X), idx);
    if (c == Component::Y)
        return GridLayout::project<MHDQuantity::Scalar::By, MHDQuantity::Scalar::rho>(
            vec(Component::Y), idx);
    return GridLayout::project<MHDQuantity::Scalar::Bz, MHDQuantity::Scalar::rho>(vec(Component::Z),
                                                                                  idx);
}

} // namespace

TEST(InnerBoundaryManager, createReturnsNullWithoutInnerBoundary)
{
    PHARE::initializer::PHAREDict dict;
    EXPECT_EQ(Manager::create(dict, {}, {}, adiabaticIndex), nullptr);
}

TEST(InnerBoundaryManager, fixedStateRequiresDensityAndPressure)
{
    auto dict                         = sphereDict();
    dict["inner_boundary"]["density"] = 1.0;
    EXPECT_THROW(Manager::create(dict, {}, {}, adiabaticIndex), std::runtime_error);
}

TEST(InnerBoundaryManager, fixedDensityDoesNotNeedPressure)
{
    auto dict                                = sphereDict();
    dict["inner_boundary"]["condition_type"] = std::string{"fixed-density"};
    dict["inner_boundary"]["density"]        = 1.0;
    EXPECT_NE(Manager::create(dict, {}, {}, adiabaticIndex), nullptr);
}

TEST(InnerBoundaryManager, fixedDensityRejectsPressure)
{
    auto dict                                = sphereDict();
    dict["inner_boundary"]["condition_type"] = std::string{"fixed-density"};
    dict["inner_boundary"]["density"]        = 1.0;
    dict["inner_boundary"]["pressure"]       = 1.0;
    EXPECT_THROW(Manager::create(dict, {}, {}, adiabaticIndex), std::runtime_error);
}

TEST(InnerBoundaryManager, fixedDensityNeedsMomentExchange)
{
    auto dict                                = sphereDict();
    dict["inner_boundary"]["condition_type"] = std::string{"fixed-density"};
    dict["inner_boundary"]["density"]        = 1.0;
    auto mgr = Manager::create(dict, {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                               {MHDQuantity::Vector::rhoV}, adiabaticIndex);
    ASSERT_NE(mgr, nullptr);
    EXPECT_TRUE(mgr->needsMomentExchange());
}

TEST(InnerBoundaryManager, fixedDensityExchangesOnlyEnergy)
{
    auto dict                                = sphereDict();
    dict["inner_boundary"]["condition_type"] = std::string{"fixed-density"};
    dict["inner_boundary"]["density"]        = 1.0;
    auto mgr = Manager::create(dict, {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                               {MHDQuantity::Vector::rhoV}, adiabaticIndex);
    ASSERT_NE(mgr, nullptr);
    EXPECT_TRUE(mgr->needsMomentExchange(MHDQuantity::Scalar::Etot1));
    EXPECT_FALSE(mgr->needsMomentExchange(MHDQuantity::Scalar::rho));
    EXPECT_FALSE(mgr->needsMomentExchange(MHDQuantity::Vector::rhoV));
}

TEST(InnerBoundaryManager, fixedStateDoesNotNeedMomentExchange)
{
    auto dict                          = sphereDict();
    dict["inner_boundary"]["density"]  = 1.0;
    dict["inner_boundary"]["pressure"] = 1.0;
    auto mgr = Manager::create(dict, {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                               {MHDQuantity::Vector::rhoV}, adiabaticIndex);
    ASSERT_NE(mgr, nullptr);
    EXPECT_FALSE(mgr->needsMomentExchange());
}

TEST(InnerBoundaryManager, rejectsUnknownConditionType)
{
    auto dict                                = sphereDict();
    dict["inner_boundary"]["condition_type"] = std::string{"reflective"};
    dict["inner_boundary"]["density"]        = 1.0;
    dict["inner_boundary"]["pressure"]       = 1.0;
    EXPECT_THROW(Manager::create(dict, {}, {}, adiabaticIndex), std::runtime_error);
}

TEST(InnerBoundaryManager, createReadsSafeStateB1)
{
    auto dict                                                 = sphereDict();
    dict["inner_boundary"]["density"]                         = 1.0;
    dict["inner_boundary"]["pressure"]                        = 1.0;
    dict["inner_boundary"]["inactive_safe_state"]["B1"]["x"]  = 0.5;
    dict["inner_boundary"]["inactive_safe_state"]["B1"]["y"]  = -0.25;
    dict["inner_boundary"]["inactive_safe_state"]["B1"]["z"]  = 0.125;
    dict["inner_boundary"]["inactive_safe_state"]["density"]  = 3.0;
    dict["inner_boundary"]["inactive_safe_state"]["pressure"] = 0.5;

    auto mgr = Manager::create(dict, {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Etot1},
                               {MHDQuantity::Vector::B1}, adiabaticIndex);
    ASSERT_NE(mgr, nullptr);

    Box<int, 2> amr_box{{-10, -10}, {9, 9}};
    IBTestGridLayout2D layout{{1.0, 1.0}, {20u, 20u}, {-10.0, -10.0}, amr_box};
    MeshDataStorage storage{layout};
    storage.bind(mgr->getMeshData());
    mgr->classify(layout);

    TestStateStorage st{layout};
    fillVec(st.B1, {9.0, 9.0, 9.0});

    mgr->setSafeState(st.B1, layout);

    auto const& sx
        = mgr->getMeshData().getStatusFieldFromCentering(layout.centering(st.B1(Component::X)));
    std::size_t n = 0;
    for (auto i = 0u; i < st.B1(Component::X).shape()[0]; ++i)
        for (auto j = 0u; j < st.B1(Component::X).shape()[1]; ++j)
            if (sx(i, j) == toDouble(ElemStatus::Inactive))
            {
                ++n;
                EXPECT_NEAR(st.B1(Component::X)(i, j), 0.5, eps);
            }
    EXPECT_GT(n, 0u);
}

TEST(InnerBoundaryManager, fixedStateGhostsHoldPrescribedState)
{
    ManagerFixture fix;
    auto const& layout = fix.layout;
    TestStateStorage st{layout};

    constexpr double rho = 1.0, vx = 0.3, P = 0.5, b1y = 1.0;
    st.rho_s.fill(rho);
    st.P_s.fill(P);
    st.Etot1_s.fill(eosPToEtot1(adiabaticIndex, rho, vx, 0., 0., 0., b1y, 0., P));
    fillVec(st.rhoV, {rho * vx, 0., 0.});
    fillVec(st.B1, {0., b1y, 0.});

    fix.manager.applyToMoments(layout, InnerBCContext<TestStateStorage>{st, st, 0.0, 0.0});

    auto const& ghosts = fix.manager.getMeshData().getGhostDataFromCentering(
        layout.centering(MHDQuantity::Scalar::rho));
    ASSERT_FALSE(ghosts.empty());

    for (auto const& g : ghosts)
    {
        MeshIndex<2> const idx{g.index[0], g.index[1]};
        EXPECT_NEAR(st.rho(idx), ManagerFixture::density, eps);
        for (auto c : {Component::X, Component::Y, Component::Z})
            EXPECT_NEAR(st.rhoV(c)(idx), 0.0, eps);

        auto const b1x_c = cellCenter(layout, st.B1, Component::X, idx);
        auto const b1y_c = cellCenter(layout, st.B1, Component::Y, idx);
        auto const b1z_c = cellCenter(layout, st.B1, Component::Z, idx);
        auto const P_g   = eosEtot1ToP(adiabaticIndex, st.rho(idx), 0., 0., 0., b1x_c, b1y_c, b1z_c,
                                       st.Etot1(idx));
        EXPECT_NEAR(P_g, ManagerFixture::pressure, eps);
    }
}

TEST(InnerBoundaryManager, fixedDensityGhostPressureMirrorsInterior)
{
    ManagerFixture fix{{}, InnerBoundaryConditionType::FixedDensity};
    auto const& layout = fix.layout;
    TestStateStorage st{layout};

    constexpr double rho = 1.0, vx = 0.3, P = 0.5, b1y = 1.0;
    st.rho_s.fill(rho);
    st.P_s.fill(P);
    st.Etot1_s.fill(eosPToEtot1(adiabaticIndex, rho, vx, 0., 0., 0., b1y, 0., P));
    fillVec(st.rhoV, {rho * vx, 0., 0.});
    fillVec(st.B1, {0., b1y, 0.});

    fix.manager.applyToMoments(layout, InnerBCContext<TestStateStorage>{st, st, 0.0, 0.0});

    auto const& ghosts = fix.manager.getMeshData().getGhostDataFromCentering(
        layout.centering(MHDQuantity::Scalar::rho));
    std::size_t checked = 0;
    for (auto const& g : ghosts)
    {
        MeshIndex<2> const idx{g.index[0], g.index[1]};
        EXPECT_NEAR(st.rho(idx), ManagerFixture::density, eps);
        if (!g.mirrorIsInterpolable)
            continue;
        ++checked;
        auto const b1x_c = cellCenter(layout, st.B1, Component::X, idx);
        auto const b1y_c = cellCenter(layout, st.B1, Component::Y, idx);
        auto const b1z_c = cellCenter(layout, st.B1, Component::Z, idx);
        auto const P_g   = eosEtot1ToP(adiabaticIndex, st.rho(idx), 0., 0., 0., b1x_c, b1y_c, b1z_c,
                                       st.Etot1(idx));
        EXPECT_NEAR(P_g, P, 1e-10);
    }
    EXPECT_GT(checked, 0u);
}

struct EdgeSphereFixture
{
    static constexpr double rho = 1.0, vx = 0.3, P = 0.5, b1y = 1.0;

    explicit EdgeSphereFixture(InnerBoundaryConditionType type)
        : fix{{}, type, Point<double, 2>{8.05, 0.05}}
        , st{fix.layout}
    {
        st.rho_s.fill(rho);
        st.P_s.fill(P);
        st.Etot1_s.fill(etot());
        fillVec(st.rhoV, {rho * vx, 0., 0.});
        fillVec(st.B1, {0., b1y, 0.});
    }

    static double etot() { return eosPToEtot1(adiabaticIndex, rho, vx, 0., 0., 0., b1y, 0., P); }

    auto& ghosts()
    {
        return fix.manager.getMeshData().getGhostDataFromCentering(
            fix.layout.centering(MHDQuantity::Scalar::rho));
    }

    std::size_t nonInterpolable()
    {
        std::size_t n = 0;
        for (auto const& g : ghosts())
            n += g.mirrorIsInterpolable ? 0 : 1;
        return n;
    }

    ManagerFixture fix;
    TestStateStorage st;
};

TEST(InnerBoundaryManager, markSetsNaNOnlyOnUncomputedEnergyGhosts)
{
    EdgeSphereFixture f{InnerBoundaryConditionType::FixedDensity};
    auto const n = f.nonInterpolable();
    ASSERT_GT(n, 0u);

    std::vector<double> saved;
    f.fix.manager.markUncomputedMoments(f.fix.layout, f.st, saved);

    EXPECT_EQ(saved.size(), n);
    for (auto const& g : f.ghosts())
    {
        MeshIndex<2> const idx{g.index[0], g.index[1]};
        EXPECT_EQ(std::isnan(f.st.Etot1(idx)), !g.mirrorIsInterpolable);
        EXPECT_EQ(f.st.rho(idx), EdgeSphereFixture::rho);
        EXPECT_EQ(f.st.rhoV(Component::X)(idx), EdgeSphereFixture::rho * EdgeSphereFixture::vx);
    }
}

TEST(InnerBoundaryManager, restoreRefillsNaNAndFlagsStale)
{
    EdgeSphereFixture f{InnerBoundaryConditionType::FixedDensity};
    ASSERT_GT(f.nonInterpolable(), 1u);

    std::vector<double> saved;
    f.fix.manager.markUncomputedMoments(f.fix.layout, f.st, saved);

    MeshIndex<2> filled{};
    for (auto const& g : f.ghosts())
        if (!g.mirrorIsInterpolable)
        {
            filled = {g.index[0], g.index[1]};
            break;
        }
    f.st.Etot1(filled) = 42.0;

    f.fix.manager.restoreUncomputedMoments(f.fix.layout, f.st, saved);

    for (auto const& g : f.ghosts())
    {
        MeshIndex<2> const idx{g.index[0], g.index[1]};
        if (idx == filled)
        {
            EXPECT_EQ(f.st.Etot1(idx), 42.0);
            EXPECT_FALSE(g.stale);
        }
        else if (!g.mirrorIsInterpolable)
        {
            EXPECT_EQ(f.st.Etot1(idx), EdgeSphereFixture::etot());
            EXPECT_TRUE(g.stale);
        }
        else
            EXPECT_FALSE(g.stale);
    }
}

TEST(InnerBoundaryManager, fixedStateMarksNothing)
{
    EdgeSphereFixture f{InnerBoundaryConditionType::FixedState};
    ASSERT_GT(f.nonInterpolable(), 0u);

    std::vector<double> saved;
    f.fix.manager.markUncomputedMoments(f.fix.layout, f.st, saved);

    EXPECT_TRUE(saved.empty());
    for (auto const& g : f.ghosts())
        EXPECT_EQ(f.st.Etot1(MeshIndex<2>{g.index[0], g.index[1]}), EdgeSphereFixture::etot());
}

TEST(InnerBoundaryManager, electricFieldGhostEdgesAreZeroed)
{
    ManagerFixture fix;
    auto const& layout = fix.layout;
    TestStateStorage st{layout};
    fillVec(st.E, {3.0, 3.0, 3.0});

    fix.manager.applyBC(st.E, layout, InnerBCContext<TestStateStorage>{st, st, 0.0, 0.0});

    for (auto c : {Component::X, Component::Y, Component::Z})
    {
        auto const& ghosts
            = fix.manager.getMeshData().getGhostDataFromCentering(layout.centering(st.E(c)));
        ASSERT_FALSE(ghosts.empty());
        for (auto const& g : ghosts)
            EXPECT_NEAR(st.E(c)(g.index), 0.0, eps);
    }
}

TEST(InnerBoundaryManager, safeStateUsesB1)
{
    Manager::SafeStateValues safe;
    safe.density  = 1.5;
    safe.pressure = 2.0;
    safe.velocity = {2.0, 0.0, 0.0};
    safe.B1       = {0.5, 0.0, 0.0};
    ManagerFixture fix{safe};
    auto const& layout = fix.layout;

    StoredScalar etot1{"Etot1", MHDQuantity::Scalar::Etot1, layout};
    etot1.fill(9.0);
    fix.manager.setSafeState(etot1.field, layout);

    double const expected  = 2.0 / (adiabaticIndex - 1.0) + 0.5 * 1.5 * 4.0 + 0.5 * 0.25;
    auto const& cellStatus = fix.manager.getMeshData().cellStatusField();
    ASSERT_GT(fix.count(etot1.field, ElemStatus::Inactive), 0u);
    for (auto i = 0u; i < etot1.field.shape()[0]; ++i)
        for (auto j = 0u; j < etot1.field.shape()[1]; ++j)
            EXPECT_NEAR(etot1.field(i, j),
                        cellStatus(i, j) == toDouble(ElemStatus::Inactive) ? expected : 9.0, eps);
}

TEST(InnerBoundaryManager, setSafeStatePinsInactiveOnly)
{
    Manager::SafeStateValues safe;
    safe.density = 1.5;
    ManagerFixture fix{safe};
    auto const& layout = fix.layout;

    StoredScalar rho{"rho", MHDQuantity::Scalar::rho, layout};
    rho.fill(7.0);
    fix.manager.setSafeState(rho.field, layout);

    auto const& cellStatus = fix.manager.getMeshData().cellStatusField();
    ASSERT_GT(fix.count(rho.field, ElemStatus::Inactive), 0u);
    for (auto i = 0u; i < rho.field.shape()[0]; ++i)
        for (auto j = 0u; j < rho.field.shape()[1]; ++j)
            EXPECT_NEAR(rho.field(i, j),
                        cellStatus(i, j) == toDouble(ElemStatus::Inactive) ? 1.5 : 7.0, eps);
}

TEST(InnerBoundaryManager, setSafeStateThrowsForUnknownQuantity)
{
    ManagerFixture fix;
    StoredScalar p{"P2", MHDQuantity::Scalar::Vx, fix.layout};
    EXPECT_THROW(fix.manager.setSafeState(p.field, fix.layout), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
