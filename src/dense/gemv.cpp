#include "dense.hpp"
using namespace dense;

template <typename T>
std::vector<T> matrix<T>:: operator*(const std::vector<T> x){
    std::vector<T> result((*this).dim_size[0]);

    if constexpr (std::is_same_v<T, float>){
        sgemv(CblasRowMajor, CblasNoTrans, (*this).dim_size[0], (*this).dim_size[1], 1.0f, (*this).values.data(), (*this).dim_size[1], x.data(), 1, 0.0f, result.data(), 1);}
    else if constexpr (std::is_same_v<T, double>){
        dgemv(CblasRowMajor, CblasNoTrans, (*this).dim_size[0], (*this).dim_size[1], 1.0, (*this).values.data(), (*this).dim_size[1], x.data(), 1, 0.0, result.data(), 1);}
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        std::complex<float> a(1, 0);
        std::complex<float> b(0, 0);
        cgemv(CblasRowMajor, CblasNoTrans, (*this).dim_size[0], (*this).dim_size[1], &a, (*this).values.data(), (*this).dim_size[1], x.data(), 1, &b, result.data(), 1);}
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        std::complex<double> a(1, 0);
        std::complex<double> b(0, 0);
        zgemv(CblasRowMajor, CblasNoTrans, (*this).dim_size[0], (*this).dim_size[1], &a, (*this).values.data(), (*this).dim_size[1], x.data(), 1, &b, result.data(), 1);}
    return result;
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;