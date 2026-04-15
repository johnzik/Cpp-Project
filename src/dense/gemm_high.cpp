#include "dense.hpp"
using namespace dense;

template<typename T>
template<typename U>
std::enable_if_t<std::is_same_v<U, T>, matrix<T>>
matrix<T>:: operator*(matrix<T> const& A) const{
    matrix<T> B((*this).dim_size[0], A.dim_size[1]);

    if constexpr (std::is_same_v<T, float>){
        float a = 1;
        sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (*this).dim_size[0], A.dim_size[1],
                (*this).dim_size[1], a, (*this).values.data(), (*this).precomputed_strides[0],
                A.values.data(), A.precomputed_strides[0], 0, B.values.data(), B.precomputed_strides[0]);}
    else if constexpr (std::is_same_v<T, double>){
        double a = 1;
        dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (*this).dim_size[0], A.dim_size[1],
                (*this).dim_size[1], a, (*this).values.data(), (*this).precomputed_strides[0],
                A.values.data(), A.precomputed_strides[0], 0, B.values.data(), B.precomputed_strides[0]);}
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        std::complex<float> a(1, 0);
        std::complex<float> b(0, 0);
        cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (*this).dim_size[0], A.dim_size[1],
                (*this).dim_size[1], &a, (*this).values.data(), (*this).precomputed_strides[0],
                A.values.data(), A.precomputed_strides[0], &b, B.values.data(), B.precomputed_strides[0]);}
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        std::complex<double> a(1, 0);
        std::complex<double> b(0, 0);
        zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (*this).dim_size[0], A.dim_size[1],
                (*this).dim_size[1], &a, (*this).values.data(), (*this).precomputed_strides[0],
                A.values.data(), A.precomputed_strides[0], &b, B.values.data(), B.precomputed_strides[0]);}
    return B;
}

template std::enable_if_t<std::is_same_v<float, float>, dense::matrix<float>>
dense::matrix<float>::operator*<float>(const dense::matrix<float>&) const;

template std::enable_if_t<std::is_same_v<double, double>, dense::matrix<double>>
dense::matrix<double>::operator*<double>(const dense::matrix<double>&) const;

template std::enable_if_t<std::is_same_v<std::complex<float>, std::complex<float>>, dense::matrix<std::complex<float>>>
dense::matrix<std::complex<float>>::operator*<std::complex<float>>(const dense::matrix<std::complex<float>>&) const;

template std::enable_if_t<std::is_same_v<std::complex<double>, std::complex<double>>, dense::matrix<std::complex<double>>>
dense::matrix<std::complex<double>>::operator*<std::complex<double>>(const dense::matrix<std::complex<double>>&) const;