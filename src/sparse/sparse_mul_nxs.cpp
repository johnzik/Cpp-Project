#include "sparse.hpp"

namespace sparse {

template <typename T, typename U>
matrix<T> operator*(U x, const matrix<T>& A){
    A.refresh_handle();
    matrix<T> result(A);

    auto& vals = result.get_values_data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] *= x;
    result.invalidate_handle();
    return result;
}
template sparse::matrix<float> operator*(int x, const sparse::matrix<float>&);
template sparse::matrix<float> operator*(float x, const sparse::matrix<float>&);
template sparse::matrix<float> operator*(double x, const sparse::matrix<float>&);

template sparse::matrix<double> operator*(int x, const sparse::matrix<double>&);
template sparse::matrix<double> operator*(float x, const sparse::matrix<double>&);
template sparse::matrix<double> operator*(double x, const sparse::matrix<double>&);

template sparse::matrix<std::complex<float>> operator*(int x, const sparse::matrix<std::complex<float>>&);
template sparse::matrix<std::complex<float>> operator*(float x, const sparse::matrix<std::complex<float>>&);
template sparse::matrix<std::complex<float>> operator*(double x, const sparse::matrix<std::complex<float>>&);
template sparse::matrix<std::complex<float>> operator*(std::complex<float> x, const sparse::matrix<std::complex<float>>&);
template sparse::matrix<std::complex<float>> operator*(std::complex<double> x, const sparse::matrix<std::complex<float>>&);

template sparse::matrix<std::complex<double>> operator*(int x, const sparse::matrix<std::complex<double>>&);
template sparse::matrix<std::complex<double>> operator*(float x, const sparse::matrix<std::complex<double>>&);
template sparse::matrix<std::complex<double>> operator*(double x, const sparse::matrix<std::complex<double>>&);
template sparse::matrix<std::complex<double>> operator*(std::complex<float> x, const sparse::matrix<std::complex<double>>&);
template sparse::matrix<std::complex<double>> operator*(std::complex<double> x, const sparse::matrix<std::complex<double>>&);}