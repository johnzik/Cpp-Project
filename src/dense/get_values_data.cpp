#include "dense.hpp"
using namespace dense;

template <typename T>
T* matrix<T>::get_values_data(){
    return values.data();
}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;