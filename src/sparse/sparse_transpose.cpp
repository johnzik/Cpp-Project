#include "sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T> matrix<T>::transpose() const{
    if (!valid_handle) refresh_handle();

    sparse_matrix_t At_handle = nullptr;
    sparse_convert_csr(handle, SPARSE_OPERATION_TRANSPOSE, &At_handle);

    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
    T *vals = nullptr;

    if constexpr (std::is_same_v<T, float>)
        sparse_s_export_csr(At_handle, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &vals);
    else if constexpr (std::is_same_v<T, double>)
        sparse_d_export_csr(At_handle, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &vals);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        sparse_c_export_csr(At_handle, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, reinterpret_cast<MKL_Complex8 **>(&vals));
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        sparse_z_export_csr(At_handle, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, reinterpret_cast<MKL_Complex16 **>(&vals));

    return matrix<T>(At_handle, rows, cols, true);
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;