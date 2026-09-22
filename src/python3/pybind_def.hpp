#ifndef PHARE_PYTHON_PYBIND_DEF_HPP
#define PHARE_PYTHON_PYBIND_DEF_HPP

#include <tuple>
#include <cassert>
#include <cstdint>

#include "core/utilities/span.hpp"
#include "core/utilities/space_time_function.hpp"

#include "pybind11/numpy.h"


namespace PHARE::pydata
{
template<typename T>
using py_array_t = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;


using pyarray_particles_t = std::tuple<py_array_t<int32_t>, py_array_t<double>, py_array_t<double>,
                                       py_array_t<double>, py_array_t<double>>;

using pyarray_particles_crt
    = std::tuple<py_array_t<int32_t> const&, py_array_t<double> const&, py_array_t<double> const&,
                 py_array_t<double> const&, py_array_t<double> const&>;

template<typename PyArrayInfo>
std::size_t ndSize(PyArrayInfo const& ar_info)
{
    assert(ar_info.ndim >= 1 && ar_info.ndim <= 3);

    return std::accumulate(ar_info.shape.begin(), ar_info.shape.end(), 1,
                           std::multiplies<std::size_t>());
}


template<typename T>
class __attribute__((visibility("hidden"))) PyArrayWrapper : public core::Span<T>
{
public:
    PyArrayWrapper(PHARE::pydata::py_array_t<T> const& array)
        : core::Span<T>{static_cast<T*>(array.request().ptr), pydata::ndSize(array.request())}
        , _array{array}
    {
        assert(_array.request().ptr);
        assert(_array.request().ptr == array.request().ptr); // assert no copy
    }

protected:
    PHARE::pydata::py_array_t<T> _array;
};

template<typename T>
std::shared_ptr<core::Span<T>> makePyArrayWrapper(py_array_t<T> const& array)
{
    return std::make_shared<PyArrayWrapper<T>>(array);
}

template<typename T>
core::Span<T> makeSpan(py_array_t<T> const& py_array)
{
    auto ar_info = py_array.request();
    assert(ar_info.ptr);
    return {static_cast<T*>(ar_info.ptr), ndSize(ar_info)};
}


} // namespace PHARE::pydata


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/**
 * @brief Hands a core::CoordinateSpan to python as a zero-copy numpy view.
 *
 * This specialization must be visible before pybind11/functional.h instantiates the
 * std::function caster that uses it.
 */
template<>
struct type_caster<PHARE::core::CoordinateSpan>
{
    PYBIND11_TYPE_CASTER(PHARE::core::CoordinateSpan, const_name("numpy.ndarray"));

    /// spans only ever cross C++ -> python, nothing reads one back out of a python object
    bool load(handle, bool) { return false; }

    static handle cast(PHARE::core::CoordinateSpan const& span, return_value_policy, handle)
    {
        // the capsule base owns nothing and frees nothing, so numpy reads the buffer in place
        auto array = array_t<double>{static_cast<ssize_t>(span.size), span.ptr,
                                     capsule{span.ptr, [](void*) {}}};

        // pybind marks an array built over a non-array base as writeable whatever the
        // constness of the pointer it was given (see numpy.h). The coordinates are not the
        // user's to modify: a write would land in the buffer the next component is about
        // to be evaluated on.
        array_proxy(array.ptr())->flags &= ~npy_api::NPY_ARRAY_WRITEABLE_;

        return array.release();
    }
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif /*PHARE_PYTHON_PYBIND_DEF_H*/
