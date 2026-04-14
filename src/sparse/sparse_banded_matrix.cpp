#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T> matrix<T>::banded(int N, int kl, int ku, T val)
{
    // Estimated memory (over-estimation) for reservation
    MKL_INT estimated_nnz = N * (ku + kl + 1);

    std::vector<T> values;
    std::vector<MKL_INT> col_idx;
    std::vector<MKL_INT> row_idx(N+1);

    // Reserve memory
    values.reserve(estimated_nnz);
    col_idx.reserve(estimated_nnz);

    int current_nnz_count = 0;

    for (int i = 0; i < N; i++)
    {
        row_idx[i] = current_nnz_count;

        int start_col = std::max(0, i - kl);
        int end_col = std::min(N - 1, i + ku);

        for (int j = start_col; j <= end_col; j++)
        {
            values.push_back(val);
            col_idx.push_back(j);
            current_nnz_count++;
        }
    }

    row_idx[N] = current_nnz_count; // The last element of the row_idx vector in CSR must always be nnz, 
                                    // i.e. the size of the value vector

    // Shrink the over-estimated reserved memory
    values.shrink_to_fit();
    col_idx.shrink_to_fit();

    return matrix<T>(values, col_idx, row_idx, N, N);
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;