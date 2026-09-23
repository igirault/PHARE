#ifndef PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_DIPOLE_HPP
#define PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_DIPOLE_HPP

#include "core/models/external_field_updater_builtin.hpp"

#include <numbers>
#include <numeric>
#include <stdexcept>

namespace PHARE::core
{
/**
 * @brief Implement a static dipole external field.
 *
 * Provides the vector potential @f$\mathbf{A}_0@f$ of a magnetic dipole of moment
 * @f$\mathbf{m}@f$ placed at @f$\mathbf{x}_0@f$. In what follows @f$\mathbf{r} = \mathbf{x} -
 * \mathbf{x}_0@f$ and @f$r = \lVert\mathbf{r}\rVert@f$.
 *
 * The moment has one component per dimension: in 2D the configuration is invariant along
 * @f$z@f$, the moment lies in the @f$(x,y)@f$ plane, and only the @f$z@f$ component of the
 * potential is non-zero:
 * @f[
 *   A_{0,z}(\mathbf{x}) = \frac{1}{2\pi}\,
 *                         \frac{\left(\mathbf{m}\times\mathbf{r}\right)_z}{r^{2}}
 *                       = \frac{1}{2\pi}\,\frac{m_x r_y - m_y r_x}{r^{2}}
 * @f]
 * and in 3D:
 * @f[
 *   \mathbf{A}_0(\mathbf{x}) = \frac{1}{4\pi}\,
 *                              \frac{\mathbf{m}\times\mathbf{r}}{r^{3}}
 * @f]
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class ExternalFieldUpdaterDipole
    : public ExternalFieldUpdaterBuiltin<ExternalFieldUpdaterDipole<VecFieldT, GridLayoutT>,
                                         VecFieldT, GridLayoutT>
{
    using Super = ExternalFieldUpdaterBuiltin<ExternalFieldUpdaterDipole<VecFieldT, GridLayoutT>,
                                              VecFieldT, GridLayoutT>;

public:
    using vecfield_type  = Super::vecfield_type;
    using value_type     = Super::value_type;
    using point_type     = Super::point_type;
    using vector_type    = Point<value_type, GridLayoutT::dimension>;
    using component_type = Super::component_type;

    static constexpr std::size_t dimension = Super::dimension;

    ExternalFieldUpdaterDipole(point_type position, vector_type moment)
        : Super()
        , position_{position}
        , moment_{moment} {};

    template<component_type i>
    double potential(point_type const& coords, double /*time*/) const
    {
        if constexpr (dimension == 1)
        {
            throw std::runtime_error("a dipole in 1D makes no sense");
        }
        else if constexpr (dimension == 2)
        {
            if constexpr (i == component_type::Z)
            {
                constexpr double factor = 1. / (2. * std::numbers::pi);
                point_type const r      = coords - position_;
                double const rSquared   = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
                // z component of the cross product `moment_` times `r`
                return factor * (moment_[0] * r[1] - moment_[1] * r[0]) / rSquared;
            }
            else
                return 0.0;
        }
        else // 3D case
        {
            point_type const r      = coords - position_;
            double rSquared         = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
            double constexpr factor = 1. / (4. * std::numbers::pi);
            // elegant trick to express component i of cross product `moment_` times `r`
            constexpr auto j = (static_cast<std::size_t>(i) + 1) % 3;
            constexpr auto k = (static_cast<std::size_t>(i) + 2) % 3;
            return factor * (moment_[j] * r[k] - moment_[k] * r[j])
                   / (rSquared * std::sqrt(rSquared));
        }
    }

private:
    point_type position_; //!< where is placed the dipole in space
    vector_type moment_;  //!< the moment vector, one component per dimension
};

} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_EXTERNAL_FIELD_UPDATER_DIPOLE_HPP
