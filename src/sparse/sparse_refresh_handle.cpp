#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
void matrix<T>::refresh_handle() const{
    if (this->valid_handle && this->handle != nullptr){ return; }
    if (this->handle != nullptr){ sparse_destroy(this->handle); }

    sparse_status_t status = SPARSE_STATUS_SUCCESS;

    MKL_INT* row_start = const_cast<MKL_INT*>(this->row_idx.data());
    MKL_INT* row_end   = const_cast<MKL_INT*>(this->row_idx.data() + 1);
    MKL_INT* col_indx  = const_cast<MKL_INT*>(this->col_idx.data());

    if constexpr(std::is_same_v<T,float>){
        status = sparse_s_create_csr(&this->handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)this->row_size,(MKL_INT)this->col_size,
            row_start,row_end,col_indx,
            const_cast<float*>(this->values.data()));
    }
    else if constexpr(std::is_same_v<T,double>){
        status = sparse_d_create_csr(&this->handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)this->row_size,(MKL_INT)this->col_size,
            row_start,row_end,col_indx,
            const_cast<double*>(this->values.data()));
    }
    else if constexpr(std::is_same_v<T,std::complex<float>>){
        status = sparse_c_create_csr(&this->handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)this->row_size,(MKL_INT)this->col_size,
            row_start,row_end,col_indx,
            reinterpret_cast<MKL_Complex8*>(const_cast<std::complex<float>*>(this->values.data())));
    }
    else if constexpr(std::is_same_v<T,std::complex<double>>){
        status = sparse_z_create_csr(&this->handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)this->row_size,(MKL_INT)this->col_size,
            row_start,row_end,col_indx,
            reinterpret_cast<MKL_Complex16*>(const_cast<std::complex<double>*>(this->values.data())));
    }

    this->valid_handle = (status == SPARSE_STATUS_SUCCESS);
    return;
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;