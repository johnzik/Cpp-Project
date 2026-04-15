#include "dense.hpp"
using namespace dense;

template <typename T>
matrix<T> matrix<T>:: transpose() const{

    // Transposed dimensions
    size_t rows = this->dim_size[1];
    size_t cols = this->dim_size[0];

    matrix<T> At(rows, cols);

    if constexpr (std::is_same_v<T, float>){
        float a = 1;
        somatcopy('R', 'T', rows, cols, a, this->values.data(), cols, At.values.data(), rows);}
    else if constexpr (std::is_same_v<T, double>){
        double a = 1;
        domatcopy('R', 'T', rows, cols, a, this->values.data(), cols, At.values.data(), rows);}
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        MKL_Complex8 a = {1.0f, 0.0f};
        comatcopy('R', 'T', rows, cols, a, reinterpret_cast<MKL_Complex8*>(const_cast<std::complex<float>*>(this->values.data())), cols, reinterpret_cast<MKL_Complex8*>(At.values.data()), rows);}
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        MKL_Complex16 a = {1.0, 0.0};
        zomatcopy('R', 'T', rows, cols, a, reinterpret_cast<MKL_Complex16*>(const_cast<std::complex<double>*>(this->values.data())), cols, reinterpret_cast<MKL_Complex16*>(At.values.data()), rows);}
    return At;
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;