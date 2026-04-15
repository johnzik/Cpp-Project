#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T> matrix<T>::I(int N)
{
    std::vector<T> values(N,1);
    std::vector<MKL_INT> col_idx(N);
    std::vector<MKL_INT> row_idx(N + 1);

    for (int count = 0; count < N; count++)
    {
        col_idx[count] = count;
        row_idx[count] = count;
    }
    
    row_idx[N] = values.size(); // The last element of the row_idx vector in CSR must always be nnz, i.e. the size of the value vector

    return matrix<T>(values, col_idx, row_idx, N, N);
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;