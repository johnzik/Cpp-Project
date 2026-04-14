#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
sparse::matrix<T> sparse::matrix<T>::operator*(T x){
    const MKL_INT n = this->row_size*this->col_size;
    std::vector<T> n_vals(this->values); //_scal operates in place
    if constexpr (std::is_same_v<T, float>)
        sscal(n, x, n_vals.data(), 1);
    else if constexpr (std::is_same_v<T, double>)
        dscal(n, x, n_vals.data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cscal(n, &x, n_vals.data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        zscal(n, &x, n_vals.data(), 1);
    //reused code from scal_mn.cpp
    return sparse::matrix<T>(n_vals, this->col_idx, this->row_idx, this->row_size, this->col_size);
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;