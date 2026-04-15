#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T>& matrix<T>::operator=(matrix const& A)
{
    if (this == &A) return *this;  // self-assignment guard

    values = A.values;
    col_idx = A.col_idx;
    row_idx = A.row_idx;
    row_size = A.row_size;
    col_size = A.col_size;
    ownership = A.ownership;

    // Destroy existing handle if it was valid
    if ( valid_handle && handle != nullptr)
    {
        mkl_sparse_destroy(handle);
    }

    handle = nullptr;
    valid_handle = false;

    return *this;
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;