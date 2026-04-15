#include "dense.hpp"
using namespace dense;

template<typename T>
size_t matrix<T>::get_size() const{
    return values.size();
};

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;