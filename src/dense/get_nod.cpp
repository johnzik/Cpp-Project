#include <../inc/dense.hpp>
using namespace dense;

/**
 * @brief Gets number of dimensions.
 */
template<typename T>
size_t matrix<T>::get_nod() const
{
    return dim_size.size();
}

// template class matrix<int>;
template size_t dense::matrix<int>::get_nod() const;
template size_t dense::matrix<float>::get_nod() const;
template size_t dense::matrix<double>::get_nod() const;
template size_t dense::matrix<std::complex<float>>::get_nod() const;
template size_t dense::matrix<std::complex<double>>::get_nod() const;