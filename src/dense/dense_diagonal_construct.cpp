#include "../inc/dense.hpp"
using namespace dense;

template<typename T>
matrix<T>::matrix(std::vector<T> diag, int n, int m){
    values = std::vector<T> (n*m, 0);
    precomputed_strides = {m,1};
    dim_size = {static_cast<size_t>(n),static_cast<size_t>(m)};

    for (int i=0; i<n;i++){
    values[i*m+i] = diag[i];
    }
}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;
