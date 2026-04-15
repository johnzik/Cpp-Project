#include "dense.hpp"
using namespace dense;

template <typename T>
matrix<T> matrix<T>:: operator*(T x){
    matrix<T> B = *this;

    if constexpr (std::is_same_v<T, float>)
        sscal(B.values.size(), x, B.values.data(), 1);
    else if constexpr (std::is_same_v<T, double>)
        dscal(B.values.size(), x, B.values.data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cscal(B.values.size(), &x, B.values.data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        zscal(B.values.size(), &x, B.values.data(), 1);
    return B;
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;