#include <../inc/dense.hpp>
using namespace dense;

template<typename T>
void matrix<T>::low_level_matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const T beta, matrix<T> const &A, matrix<T> const &B, matrix<T> &C, const T alpha){
    if constexpr (std::is_same_v<T, float>){  
        cblas_sgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)B.dim_size[0],alpha,A.values.data(),(MKL_INT)A.dim_size[1],B.values.data(),(MKL_INT)B.dim_size[1],beta,C.values.data(),(MKL_INT)C.dim_size[1]);
    }
    else if constexpr (std::is_same_v<T,double>){
        cblas_dgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)B.dim_size[0],alpha,A.values.data(),(MKL_INT)A.dim_size[1],B.values.data(),(MKL_INT)B.dim_size[1],beta,C.values.data(),(MKL_INT)C.dim_size[1]);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        cblas_cgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)B.dim_size[0],reinterpret_cast<const void*>(&alpha),A.values.data(),(MKL_INT)A.dim_size[1],B.values.data(),(MKL_INT)B.dim_size[1],reinterpret_cast<const void*>(&beta),C.values.data(),(MKL_INT)C.dim_size[1]);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        cblas_zgemm(CblasRowMajor, transa,transb,(MKL_INT)A.dim_size[0],(MKL_INT)B.dim_size[1],(MKL_INT)B.dim_size[0],reinterpret_cast<const void*>(&alpha),A.values.data(),(MKL_INT)A.dim_size[1],B.values.data(),(MKL_INT)B.dim_size[1],reinterpret_cast<const void*>(&beta),C.values.data(),(MKL_INT)C.dim_size[1]);
    }
}

template void dense::matrix<float>::low_level_matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,const float beta, matrix<float> const& A, matrix<float> const &B, matrix<float> &C, const float alpha);
template void dense::matrix<double>::low_level_matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,const double beta, matrix<double> const& A, matrix<double> const &B, matrix<double> &C, const double alpha);
template void dense::matrix<std::complex<double>>::low_level_matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,const std::complex<double> beta, matrix<std::complex<double>> const& A, matrix<std::complex<double>> const &B, matrix<std::complex<double>> &C, const std::complex<double> alpha);
template void dense::matrix<std::complex<float>>::low_level_matrix_multiplication(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,const std::complex<float> beta, matrix<std::complex<float>> const& A, matrix<std::complex<float>> const &B, matrix<std::complex<float>> &C, const std::complex<float> alpha);