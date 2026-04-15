#include "../inc/dense.hpp"
using namespace dense;

template<typename T>
T matrix<T>::get_value_at(size_t index) const{
    return values[(int)index];
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;