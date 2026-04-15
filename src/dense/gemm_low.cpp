#include "../inc/dense.hpp"
using namespace dense;

template<typename T>
void matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,const T beta, matrix<T> const& A, matrix<T> const &B, matrix<T> &C, const T alpha){
    if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)A.dim_size[1],alpha,A.get_values_data(),(MKL_INT)A.dim_size[1],B.get_values_data(),(MKL_INT)B.dim_size[1],beta,C.get_values_data(),(MKL_INT)C.dim_size[1]);
    else if constexpr (std::is_same_v<T,double>)
        cblas_dgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)A.dim_size[1],alpha,A.get_values_data(),(MKL_INT)A.dim_size[1],B.get_values_data(),(MKL_INT)B.dim_size[1],beta,C.get_values_data(),(MKL_INT)C.dim_size[1]);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)A.dim_size[1],reinterpret_cast<const void*>(&alpha),A.get_values_data(),(MKL_INT)A.dim_size[1],B.get_values_data(),(MKL_INT)B.dim_size[1],reinterpret_cast<const void*>(&beta),C.get_values_data(),(MKL_INT)C.dim_size[1]);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)A.dim_size[1],reinterpret_cast<const void*>(&alpha),A.get_values_data(),(MKL_INT)A.dim_size[1],B.get_values_data(),(MKL_INT)B.dim_size[1],reinterpret_cast<const void*>(&beta),C.get_values_data(),(MKL_INT)C.dim_size[1]);
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;