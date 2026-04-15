#include <iostream>
#include <../inc/dense.hpp>
using namespace dense;

template<typename T>
void matrix<T>::print2D()
{
    if (dim_size.size() == 2)
    {
        std::cout << "[" ;
        int idx = 0;

        for (int i = 0; i < dim_size[0]; ++i)
        {
            for (int j = 0; j < dim_size[1]; ++j)
            {
                std::cout << values[idx];
                if (j != dim_size[1] - 1)
                {
                    std::cout << ", ";
                }
                idx++;
            }

            if (i != dim_size[0] - 1)
            {
                std::cout << "; ";
            }
        }
        std::cout << "]\n" << std::endl;
    }
}

template void dense::matrix<float>::print2D();
template void dense::matrix<double>::print2D();
template void dense::matrix<std::complex<float>>::print2D();
template void dense::matrix<std::complex<double>>::print2D();