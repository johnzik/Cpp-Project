#include "../inc/sparse.hpp"
using namespace sparse;

template <typename T>
void matrix<T>::print_matrix() 
{
    for(int i=0; i<row_size; i++){
        for (int j=0; j<col_size;j++){
            std::cout<<(*this)(i,j)<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

template void sparse::matrix<float>::print_matrix();
template void sparse::matrix<double>::print_matrix();
template void sparse::matrix<std::complex<float>>::print_matrix();
template void sparse::matrix<std::complex<double>>::print_matrix();