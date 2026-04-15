#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T> matrix<T>::operator*(T x) {
    if (!valid_handle)
        this->refresh_handle();

    matrix<T> result(*this);

    // Scale the nonzero values (OpenMP-parallelized & vectorized)
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < result.values.size(); ++i)
        result.values[i] *= x;
    result.valid_handle = false;
    result.handle = nullptr;

    return result;
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;
