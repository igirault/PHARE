


#include "simulator/simulator.hpp"

#include "core/inner_boundary/sphere_inner_boundary.hpp"
#include "tests/core/data/gridlayout/gridlayout_test.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <algorithm>

using namespace PHARE::amr;

TEST(test_tagger, fromFactoryValid)
{
    auto static constexpr opts = PHARE::SimOpts{1ul, 1ul, 2ul};
    using phare_types          = PHARE::solver::PHARE_Types<opts>;
    using hybrid_model         = phare_types::HybridModel_t;
    PHARE::initializer::PHAREDict dict;
    dict["hybrid_method"] = std::string{"default"};
    dict["threshold"]     = 0.2;
    auto hybridTagger     = TaggerFactory<hybrid_model>::make(dict);
    EXPECT_TRUE(hybridTagger != nullptr);
}

TEST(test_tagger, fromFactoryInvalid)
{
    auto static constexpr opts = PHARE::SimOpts{1ul, 1ul, 2ul};
    using phare_types          = PHARE::solver::PHARE_Types<opts>;
    using hybrid_model         = phare_types::HybridModel_t;
    PHARE::initializer::PHAREDict dict;
    dict["hybrid_method"] = std::string{"invalidStrat"};
    auto hybridTagger     = TaggerFactory<hybrid_model>::make(dict);
    auto badTagger        = TaggerFactory<hybrid_model>::make(dict);
    EXPECT_TRUE(badTagger == nullptr);
}


using Param   = std::vector<double>;
using RetType = std::shared_ptr<PHARE::core::Span<double>>;

RetType step1(Param const& x)
{
    std::vector<double> values(x.size());
    std::transform(std::begin(x), std::end(x), std::begin(values),
                   [](auto xx) { return std::tanh((xx - 0.52) / 0.05); });
    return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(values));
}

RetType step2(Param const& x, Param const& y)
{
    throw std::runtime_error("fix me");
}

template<std::size_t dim>
auto constexpr step_fn()
{
    if constexpr (dim == 1)
        return &step1;
    if constexpr (dim == 2)
        return &step2;
}


template<std::size_t dim>
PHARE::initializer::PHAREDict createDict()
{
    using InitFunctionT        = PHARE::initializer::InitFunction<dim>;
    auto static constexpr step = step_fn<dim>();

    PHARE::initializer::PHAREDict dict;
    dict["ions"]["nbrPopulations"] = std::size_t{2};
    dict["ions"]["pop0"]["name"]   = std::string{"protons"};
    dict["ions"]["pop0"]["mass"]   = 1.;
    dict["ions"]["pop0"]["particle_initializer"]["name"]
        = std::string{"MaxwellianParticleInitializer"};
    dict["ions"]["pop0"]["particle_initializer"]["density"] = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_x"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_y"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_z"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_x"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_y"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_z"]
        = static_cast<InitFunctionT>(step);


    dict["ions"]["pop0"]["particle_initializer"]["nbrPartPerCell"] = int{100};
    dict["ions"]["pop0"]["particle_initializer"]["charge"]         = -1.;
    dict["ions"]["pop0"]["particle_initializer"]["basis"]          = std::string{"Cartesian"};

    dict["ions"]["pop1"]["name"] = std::string{"alpha"};
    dict["ions"]["pop1"]["mass"] = 1.;
    dict["ions"]["pop1"]["particle_initializer"]["name"]
        = std::string{"MaxwellianParticleInitializer"};
    dict["ions"]["pop1"]["particle_initializer"]["density"] = static_cast<InitFunctionT>(step);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_x"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_y"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_z"]
        = static_cast<InitFunctionT>(step);


    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_x"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_y"]
        = static_cast<InitFunctionT>(step);

    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_z"]
        = static_cast<InitFunctionT>(step);


    dict["ions"]["pop1"]["particle_initializer"]["nbrPartPerCell"] = int{100};
    dict["ions"]["pop1"]["particle_initializer"]["charge"]         = -1.;
    dict["ions"]["pop1"]["particle_initializer"]["basis"]          = std::string{"Cartesian"};

    dict["electromag"]["name"]             = std::string{"EM"};
    dict["electromag"]["electric"]["name"] = std::string{"E"};
    dict["electromag"]["magnetic"]["name"] = std::string{"B"};

    dict["electromag"]["magnetic"]["initializer"]["x_component"] = static_cast<InitFunctionT>(step);
    dict["electromag"]["magnetic"]["initializer"]["y_component"] = static_cast<InitFunctionT>(step);
    dict["electromag"]["magnetic"]["initializer"]["z_component"] = static_cast<InitFunctionT>(step);

    dict["electrons"]["pressure_closure"]["name"] = std::string{"isothermal"};
    dict["electrons"]["pressure_closure"]["Te"]   = 0.12;

    return dict;
}



template<std::size_t dim_, std::size_t interp_, std::size_t refinedPartNbr_>
struct TaggingTestInfo
{
    auto static constexpr dim            = dim_;
    auto static constexpr interp         = interp_;
    auto static constexpr refinedPartNbr = refinedPartNbr_;
};


template<typename TaggingTestInfo_t>
struct TestTagger : public ::testing::Test
{
    auto static constexpr dim            = TaggingTestInfo_t::dim;
    auto static constexpr interp_order   = TaggingTestInfo_t::interp;
    auto static constexpr refinedPartNbr = TaggingTestInfo_t::refinedPartNbr;
    auto static constexpr opts           = PHARE::SimOpts{dim, interp_order, refinedPartNbr};

    using phare_types = PHARE::solver::PHARE_Types<opts>;
    using Electromag  = phare_types::Electromag_t;
    using Ions        = phare_types::Ions_t;
    using Electrons   = phare_types::Electrons_t;
    using GridLayoutT = GridLayout<GridLayoutImplYee<dim, interp_order>>;

    struct SinglePatchHybridModel
    {
        using gridlayout_type           = GridLayout<GridLayoutImplYee<dim, interp_order>>;
        static auto constexpr dimension = dim;
        HybridState<Electromag, Ions, Electrons> state;
    };

    GridLayoutT layout;

    UsableVecField<dim> B, E;

    SinglePatchHybridModel model;
    std::vector<int> tags;

    TestTagger()
        : layout{TestGridLayout<GridLayoutT>::make(20)}
        , B{"EM_B", layout, HybridQuantity::Vector::B}
        , E{"EM_E", layout, HybridQuantity::Vector::E}
        , model{createDict<dim>()}
        , tags(20 + layout.nbrGhosts(PHARE::core::QtyCentering::dual))
    {
        B.set_on(model.state.electromag.B);
        E.set_on(model.state.electromag.E);
        model.state.electromag.initialize(layout);
    }
};

using TaggingTestInfos = testing::Types<TaggingTestInfo<1, 1, 2> /*, TaggingTestInfo<2, 1, 4>*/>;
TYPED_TEST_SUITE(TestTagger, TaggingTestInfos);

// TODOmaybe find a way to test the tagging?
TYPED_TEST(TestTagger, scaledAvg)
{
    /*
      auto strat = DefaultHybridTaggerStrategy<SinglePatchHybridModel>();
      strat.tag(model, layout, tags.data());
      {
          auto start
              = layout.physicalStartIndex(PHARE::core::QtyCentering::dual,
      PHARE::core::Direction::X); auto end =
      layout.physicalEndIndex(PHARE::core::QtyCentering::dual, PHARE::core::Direction::X);

          auto endCell     = layout.nbrCells()[0] - 1;
          double threshold = 0.1;


          for (auto iCell = 0u, ix = start; iCell <= endCell; ++ix, ++iCell)
          {
              auto Bxavg = (Bx(ix - 1) + Bx(ix) + Bx(ix + 1)) / 3.;
              auto Byavg = (By(ix - 1) + By(ix) + By(ix + 1)) / 3.;
              auto Bzavg = (Bz(ix - 1) + Bz(ix) + Bz(ix + 1)) / 3.;

              auto diffx = std::abs(Bxavg - Bx(ix));
              auto diffy = std::abs(Byavg - By(ix));
              auto diffz = std::abs(Bzavg - Bz(ix));

              auto max = std::max({diffx, diffy, diffz});
              if (max > threshold)
              {
                  tags[iCell] = 1;
              }
              else
                  tags[iCell] = 0;
          }
      }
  */
}



namespace
{
constexpr std::size_t ib_tagger_dim    = 2;
constexpr std::size_t ib_tagger_interp = 1;
using IBTaggerGridLayout = GridLayout<GridLayoutImplYee<ib_tagger_dim, ib_tagger_interp>>;

struct IBTaggerMockModel
{
    using gridlayout_type           = IBTaggerGridLayout;
    static constexpr std::size_t dimension = ib_tagger_dim;

    struct FakeIBManager
    {
        SphereInnerBoundary<ib_tagger_dim>          sphere;
        InnerBoundaryGeometry<ib_tagger_dim> const& getGeometry() const { return sphere; }
    };

    std::unique_ptr<FakeIBManager> innerBoundaryManager = nullptr;
    UsableVecField<ib_tagger_dim>* B_ptr                = nullptr;

    bool hasInnerBoundary() const { return innerBoundaryManager != nullptr; }
    auto& get_B() { return *B_ptr; }
};
} // namespace

TEST(InnerBoundaryTagger, doesNotTagCellsNearBoundary)
{
    constexpr std::size_t dim = ib_tagger_dim;

    // 20x20 cells, dl=0.05, AMRBox [(0,0),(19,19)]
    // cell (ix,iy) has center at ((ix+0.5)*0.05, (iy+0.5)*0.05)
    auto const layout = TestGridLayout<IBTaggerGridLayout>::make(20);
    UsableVecField<dim> B{"B", layout, HybridQuantity::Vector::B};

    // Fill By with a linear ramp so gradient pass tags every cell to 1
    // criter_x = |By(ix+2)-By(ix)| / (1 + |By(ix+1)-By(ix)|) = 2/(1+1) = 1.0 > threshold
    auto& By             = B.getComponent(PHARE::core::Component::Y);
    auto const byAlloc   = layout.allocSize(PHARE::core::HybridQuantity::Scalar::By);
    for (std::size_t ix = 0; ix < byAlloc[0]; ++ix)
        for (std::size_t iy = 0; iy < byAlloc[1]; ++iy)
            By(ix, iy) = static_cast<double>(ix);

    constexpr double radius = 0.2;
    constexpr double halo   = 0.05;
    Point<double, dim> const center{0.5, 0.5};

    IBTaggerMockModel model;
    model.B_ptr = &B;
    model.innerBoundaryManager.reset(
        new IBTaggerMockModel::FakeIBManager{SphereInnerBoundary<dim>{"sphere", center, radius}});

    PHARE::initializer::PHAREDict dict;
    dict["threshold"]           = 0.1; // gradient pass tags all cells = 1
    dict["inner_boundary_halo"] = halo;

    DefaultTaggerStrategy<IBTaggerMockModel> strat{dict};

    auto const ncells = layout.nbrCells();
    std::vector<int> tags(ncells[0] * ncells[1], 0);
    strat.tag(model, layout, tags.data());

    bool constexpr fortran = false;
    auto tagsv             = NdArrayView<dim, int, fortran>(tags.data(), ncells);

    for (auto const [amr, tag] : boxes_iterator{layout.AMRBox(), boxFromNbrCells(ncells)})
    {
        auto const coords = layout.cellCenteredCoordinates(amr);
        double const dx   = coords[0] - center[0];
        double const dy   = coords[1] - center[1];
        double const sd   = std::sqrt(dx * dx + dy * dy) - radius;

        // IB pass clears (sets to 0) cells within halo of boundary to prevent refinement there
        int const expected = sd <= halo ? 0 : 1;
        EXPECT_EQ(tagsv(tag.toArray()), expected)
            << "cell (" << amr[0] << "," << amr[1] << ") sd=" << sd;
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int testResult = RUN_ALL_TESTS();
    return testResult;
}
