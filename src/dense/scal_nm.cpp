#include "dense.hpp"

namespace dense {

template <typename T, typename U>
matrix<T> operator*(U x, const matrix<T>& A){
    dense::matrix<T> B = A;

    if constexpr (std::is_same_v<T, float>)
        sscal(B.get_size(), x, B.get_values_data(), 1);
    else if constexpr (std::is_same_v<T, double>)
        dscal(B.get_size(), x, B.get_values_data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cscal(B.get_size(), &x, B.get_values_data(), 1);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        zscal(B.get_size(), &x, B.get_values_data(), 1);
    return B;
}
template dense::matrix<float> operator*(int x, const dense::matrix<float>&);
template dense::matrix<float> operator*(float x, const dense::matrix<float>&);
template dense::matrix<float> operator*(double x, const dense::matrix<float>&);

template dense::matrix<double> operator*(int x, const dense::matrix<double>&);
template dense::matrix<double> operator*(float x, const dense::matrix<double>&);
template dense::matrix<double> operator*(double x, const dense::matrix<double>&);

template dense::matrix<std::complex<float>> operator*(int x, const dense::matrix<std::complex<float>>&);
template dense::matrix<std::complex<float>> operator*(float x, const dense::matrix<std::complex<float>>&);
template dense::matrix<std::complex<float>> operator*(double x, const dense::matrix<std::complex<float>>&);
template dense::matrix<std::complex<float>> operator*(std::complex<float> x, const dense::matrix<std::complex<float>>&);
template dense::matrix<std::complex<float>> operator*(std::complex<double> x, const dense::matrix<std::complex<float>>&);

template dense::matrix<std::complex<double>> operator*(int x, const dense::matrix<std::complex<double>>&);
template dense::matrix<std::complex<double>> operator*(float x, const dense::matrix<std::complex<double>>&);
template dense::matrix<std::complex<double>> operator*(double x, const dense::matrix<std::complex<double>>&);
template dense::matrix<std::complex<double>> operator*(std::complex<float> x, const dense::matrix<std::complex<double>>&);
template dense::matrix<std::complex<double>> operator*(std::complex<double> x, const dense::matrix<std::complex<double>>&);}