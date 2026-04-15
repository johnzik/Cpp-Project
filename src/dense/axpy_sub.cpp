#include "dense.hpp"
using namespace dense;

template <typename T>
matrix<T> matrix<T>:: operator-(matrix<T> const& A){
    matrix<T> B = *this;

    if constexpr (std::is_same_v<T, float>){
        float a = -1;
        saxpy(static_cast<int>(A.values.size()), a, A.values.data(), 1, B.values.data(), 1);}
    else if constexpr (std::is_same_v<T, double>){
        double a = -1;
        daxpy(static_cast<int>(A.values.size()), a, A.values.data(), 1, B.values.data(), 1);}
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        std::complex<float> a(-1, 0);
        caxpy(static_cast<int>(A.values.size()), &a, A.values.data(), 1, B.values.data(), 1);}
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        std::complex<double> a(-1, 0);
        zaxpy(static_cast<int>(A.values.size()), &a, A.values.data(), 1, B.values.data(), 1);}
    return B;
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;