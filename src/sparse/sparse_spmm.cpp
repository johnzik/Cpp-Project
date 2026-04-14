#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
matrix<T> matrix<T>:: operator*(matrix<T> const& A){
    // refresh handles if required
    if (!valid_handle)
        this->refresh_handle();
    if (!A.valid_handle)
        A.refresh_handle();
    this->valid_handle = true;
    A.valid_handle = true;
    sparse_matrix_t ret;
    sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, this->handle, A.handle, &ret);
    return matrix<T>(ret,this->row_size, A.col_size, false);
}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;