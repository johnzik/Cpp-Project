#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
void matrix<T>::set_value_at(size_t index, T value){
    //index = c + col_size*r
    size_t c = index%col_size;
    size_t r = index/col_size;
    size_t row_start = row_idx[r];
    size_t row_end = row_idx[r+1];

    for (size_t i = row_start; i<row_end; i++){
        if (col_idx[i] == c){
            values[i] = value;
            valid_handle = false;
            return;
        }
    }
    
    throw std::invalid_argument("Can only set non-zero values");
}
template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;
