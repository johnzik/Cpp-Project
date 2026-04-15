#include "../inc/dense.hpp"
using namespace dense;

template<typename T>
size_t matrix<T>:: get_dim_size(size_t index) const{
    return dim_size[(int)index];
}
template size_t dense::matrix<float>::get_dim_size(size_t) const;
template size_t dense::matrix<double>::get_dim_size(size_t) const;
template size_t dense::matrix<std::complex<float>>::get_dim_size(size_t) const;
template size_t dense::matrix<std::complex<double>>::get_dim_size(size_t) const;
