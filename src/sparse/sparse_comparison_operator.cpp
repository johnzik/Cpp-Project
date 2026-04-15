#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
bool matrix<T>::operator==(matrix const& A)
{
    // Check for self-assignment
    if (this == &A) return true; 

    // Dimensions check
    if (row_size != A.row_size || col_size != A.col_size) return false;

    // Check CSR vectors
    if (values != A.values) return false;
    if (row_idx != A.row_idx) return false;
    if (col_idx != A.col_idx) return false;

    return true;
}

template bool sparse::matrix<float>::operator==(matrix const& A);
template bool sparse::matrix<double>::operator==(matrix const& A);
template bool sparse::matrix<std::complex<float>>::operator==(matrix const& A);
template bool sparse::matrix<std::complex<double>>::operator==(matrix const& A);