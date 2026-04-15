#include <../inc/dense.hpp>
using namespace dense;

template<typename T>
matrix<T>::matrix(std::vector<T> v, std::vector<size_t> d, std::vector<int> p)
    : values(v), dim_size(d), precomputed_strides(p)
{

}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;