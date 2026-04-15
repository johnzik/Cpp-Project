#include "../inc/dense.hpp"
using namespace dense;

template<typename T>
const T* matrix<T>::get_const_values_data() const{
    return values.data();
}
template const float* dense::matrix<float>::get_const_values_data() const;
template const double* dense::matrix<double>::get_const_values_data() const;
template const std::complex<float>* dense::matrix<std::complex<float>>::get_const_values_data() const;
template const std::complex<double>* dense::matrix<std::complex<double>>::get_const_values_data() const;
