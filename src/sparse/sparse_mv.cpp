#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
std::vector<T> matrix<T>::operator*(const std::vector<T> x)
{
    // Result vector
    std::vector<T> result((*this).shape().first);

    // Refresh or regenerate handle
    (*this).refresh_handle();

    // Matrix description for mkl_sparse_?_mv
    matrix_descr descr; 
    descr.type = SPARSE_MATRIX_TYPE_GENERAL; 
    descr.mode = SPARSE_FILL_MODE_FULL; 
    descr.diag = SPARSE_DIAG_NON_UNIT;

    if constexpr(std::is_same_v<T, float>)
    {
        float a = 1.0;
        float b = 0;
        sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, (*this).handle, descr, x.data(), b, result.data());

        return result;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        double a = 1.0;
        double b = 0;
        sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, a, (*this).handle, descr, x.data(), b, result.data());
        
        return result;
    }
    else if constexpr(std::is_same_v<T, std::complex<float>>)
    {
        MKL_Complex8 a = {1.0f, 0.0f};
        MKL_Complex8 b = {0.0f, 0.0f};
        sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, 
                    a, 
                    (*this).handle, 
                    descr,
                    reinterpret_cast<const MKL_Complex8*>(x.data()), 
                    b, 
                    reinterpret_cast<MKL_Complex8*>(result.data()));

        return result;
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>)
    {
        MKL_Complex16 a = {1.0, 0.0};
        MKL_Complex16 b = {0.0, 0.0};
        sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 
                    a, 
                    (*this).handle, 
                    descr, 
                    reinterpret_cast<const MKL_Complex16*>(x.data()), 
                    b, 
                    reinterpret_cast<MKL_Complex16*>(result.data()));

        return result;
    }
}

template std::vector<float> matrix<float>::operator*(const std::vector<float> x);
template std::vector<double> matrix<double>::operator*(const std::vector<double> x);
template std::vector<std::complex<float>> matrix<std::complex<float>>::operator*(const std::vector<std::complex<float>> x);
template std::vector<std::complex<double>> matrix<std::complex<double>>::operator*(const std::vector<std::complex<double>> x);