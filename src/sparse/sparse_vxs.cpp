#include "sparse.hpp"

namespace sparse {

template <typename T>
std::vector<T> operator*(std::vector<T> x, const matrix<T>& A){
    A.refresh_handle();
    std::vector<T> result(A.shape().second);

    // Matrix description for mkl_sparse_?_mv
    matrix_descr descr; 
    descr.type = SPARSE_MATRIX_TYPE_GENERAL; 
    descr.mode = SPARSE_FILL_MODE_FULL; 
    descr.diag = SPARSE_DIAG_NON_UNIT;

    if constexpr(std::is_same_v<T, float>){
        sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0f, A.get_handle(), descr, x.data(), 0.0f, result.data());
    }
    else if constexpr(std::is_same_v<T, double>){
        sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A.get_handle(), descr, x.data(), 0.0, result.data());
    }
    else if constexpr(std::is_same_v<T, std::complex<float>>){
        MKL_Complex8 a = {1.0f, 0.0f};
        MKL_Complex8 b = {0.0f, 0.0f};
        sparse_c_mv(SPARSE_OPERATION_TRANSPOSE, a, A.get_handle(), descr, reinterpret_cast<const MKL_Complex8*>(x.data()), b, reinterpret_cast<MKL_Complex8*>(result.data()));
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>){
        MKL_Complex16 a = {1.0, 0.0};
        MKL_Complex16 b = {0.0, 0.0};
        sparse_z_mv(SPARSE_OPERATION_TRANSPOSE, a, A.get_handle(), descr, reinterpret_cast<const MKL_Complex16*>(x.data()), b, reinterpret_cast<MKL_Complex16*>(result.data()));
    }
    return result;
}
template std::vector<float> operator*<float>(std::vector<float>, const matrix<float>&);
template std::vector<double> operator*<double>(std::vector<double>, const matrix<double>&);
template std::vector<std::complex<float>> operator*<std::complex<float>>(std::vector<std::complex<float>>, const matrix<std::complex<float>>&);
template std::vector<std::complex<double>> operator*<std::complex<double>>(std::vector<std::complex<double>>, const matrix<std::complex<double>>&);}
