#include <../inc/dense.hpp>
using namespace dense;

template<typename T>
matrix<T>& matrix<T>::operator=(matrix<T> const& A)
{
    values = A.values;
    dim_size = A.dim_size;
    precomputed_strides = A.precomputed_strides;
    
    return *this;
}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;