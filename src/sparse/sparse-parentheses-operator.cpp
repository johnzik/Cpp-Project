#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
const T matrix<T>::operator () (size_t r, size_t c) const{
    size_t row_start = row_idx[r];
    size_t row_end = row_idx[r+1];

    for (size_t i = row_start; i<row_end; i++){
        if (col_idx[i] == c){
            return values[i];
        }
    }
    return T{};
}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;