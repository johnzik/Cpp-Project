#include "../inc/sparse.hpp"
#include <algorithm>
using namespace sparse;

template <typename T>
matrix<T>::matrix(const sparse_matrix_t h, size_t rows, size_t cols, bool own){
    this->handle = h;
    this->valid_handle = true;
    this->ownership = own;
    this->row_size = rows;
    this->col_size = cols;


    sparse_index_base_t indexing;
    MKL_INT mkl_rows, mkl_cols;
    MKL_INT* row_start;
    MKL_INT* row_end;
    MKL_INT* col_idx_ptr;

    if constexpr(std::is_same_v<T, float>){
        float* val_ptr;
        sparse_s_export_csr(h, &indexing, &mkl_rows, &mkl_cols,
                            &row_start, &row_end, &col_idx_ptr, &val_ptr);
        this->values = std::vector<float>(val_ptr, val_ptr + (row_end[mkl_rows-1] - row_start[0]));
    }
    else if constexpr(std::is_same_v<T, double>){
        double* val_ptr;
        sparse_d_export_csr(h, &indexing, &mkl_rows, &mkl_cols,
                            &row_start, &row_end, &col_idx_ptr, &val_ptr);
        this->values = std::vector<double>(val_ptr, val_ptr + (row_end[mkl_rows-1] - row_start[0]));
    }
    else if constexpr(std::is_same_v<T, std::complex<float>>){
        MKL_Complex8* val_ptr;
        sparse_c_export_csr(h, &indexing, &mkl_rows, &mkl_cols,
                            &row_start, &row_end, &col_idx_ptr, &val_ptr);
        this->values = std::vector<std::complex<float>>(
            reinterpret_cast<std::complex<float>*>(val_ptr),
            reinterpret_cast<std::complex<float>*>(val_ptr + (row_end[mkl_rows-1] - row_start[0]))
        );
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>){
        MKL_Complex16* val_ptr;
        sparse_z_export_csr(h, &indexing, &mkl_rows, &mkl_cols,
                            &row_start, &row_end, &col_idx_ptr, &val_ptr);
        this->values = std::vector<std::complex<double>>(
            reinterpret_cast<std::complex<double>*>(val_ptr),
            reinterpret_cast<std::complex<double>*>(val_ptr + (row_end[mkl_rows-1] - row_start[0]))
        );
    }


    this->col_idx = std::vector<MKL_INT>(col_idx_ptr,
        col_idx_ptr + (row_end[mkl_rows-1] - row_start[0]));

    std::vector<MKL_INT> row_idx(mkl_rows+1);
    row_idx[0] = row_start[0];
    for(size_t i=0;i<mkl_rows;i++){
        row_idx[i+1] = row_end[i];
    }
    this->row_idx = std::move(row_idx);
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;
