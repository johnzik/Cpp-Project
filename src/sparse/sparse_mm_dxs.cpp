#include "matrix.hpp"
#include <mkl.h>

namespace dense {

template<typename T>
dense::matrix<T> operator*(const dense::matrix<T>& A, const sparse::matrix<T>& B) {
    B.refresh_handle();

    // Εxisting sparse × dense operator
    auto result = (B.transpose() * A.transpose()).transpose();
    return result;
}
template dense::matrix<float> operator*(const dense::matrix<float>&, const sparse::matrix<float>&);
template dense::matrix<double> operator*(const dense::matrix<double>&, const sparse::matrix<double>&);
template dense::matrix<std::complex<float>> operator*(const dense::matrix<std::complex<float>>&, const sparse::matrix<std::complex<float>>&);
template dense::matrix<std::complex<double>> operator*(const dense::matrix<std::complex<double>>&, const sparse::matrix<std::complex<double>>&);
}