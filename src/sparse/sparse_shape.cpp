#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
std::pair<size_t, size_t> matrix<T>::shape() const{
    return {row_size, col_size};
}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;