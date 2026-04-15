#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T>::matrix (matrix const& A)
    :values(A.values), col_idx(A.col_idx), row_idx(A.row_idx), 
     row_size(A.row_size), col_size(A.col_size), ownership(A.ownership)
{
    handle = nullptr;
    valid_handle = false;
}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;