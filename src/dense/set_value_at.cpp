#include <../inc/dense.hpp>
using namespace dense;

/**
 * @brief Sets value at a given index.
 * @param index The linear 0-based index in the flat values vector.
 * @param value The long double value to set.
 */
template <typename T>
void dense::matrix<T>::set_value_at(size_t index, T value)
{
    values.at(index) = value;
}

template void dense::matrix<int>::set_value_at(size_t index, int value);
template void dense::matrix<float>::set_value_at(size_t index, float value);
template void dense::matrix<double>::set_value_at(size_t index, double value);
template void dense::matrix<std::complex<float>>::set_value_at(size_t index, std::complex<float> value);
template void dense::matrix<std::complex<double>>::set_value_at(size_t index, std::complex<double> value);