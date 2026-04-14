#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
sparse::matrix<T> sparse::matrix<T>::operator+(sparse::matrix<T> const& A){
    // refresh handles if required
    if (!valid_handle)
        this->refresh_handle();
    if (!A.valid_handle)
        A.refresh_handle();
    this->valid_handle = true;
    A.valid_handle = true;  
    sparse_matrix_t ret;
    if constexpr (std::is_same_v<T, float>) {
        const float a = 1;
        sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, A.handle, a, this->handle, &ret);
    }
    else if constexpr (std::is_same_v<T, double>) {
        const double a = 1;
        sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, A.handle, a, this->handle, &ret);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        const MKL_Complex8 a = {1.0f, 0.0f};
        sparse_c_add(SPARSE_OPERATION_NON_TRANSPOSE, A.handle, a, this->handle, &ret);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        const MKL_Complex16 a = {1.0, 0.0};
        sparse_z_add(SPARSE_OPERATION_NON_TRANSPOSE, A.handle, a, this->handle, &ret);
    }
    return sparse::matrix<T>(ret, row_size, col_size, false);

}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;