#include "dense.hpp"
#include <cstring> 
using namespace dense;

template <typename T>
matrix<T> matrix<T>::operator ^ (int n){
    if (dim_size[0]!=dim_size[1] || dim_size.size()>2){
        throw std::invalid_argument("Can only exponentiate square 2D matrices");
    }
    if (n == 0){
        return matrix<T>().I(dim_size[1]);
    }
    if (n == 1){
        return (*this);
    }
    if (n>1){
        matrix<T> A = *this; //copy to alter safely without destroying *this
        matrix<T> res = matrix<T>().I(dim_size[1]);
        while (n>0){
            if(n&1) res = res*A;
            n = n>>1;
            A = A*A;
        }
        return res;
    }
    if (n == -1){
        lapack_int info;
        lapack_int r = dim_size[0];
        lapack_int * ipiv = new lapack_int[r];

        size_t mem1 = r*r*sizeof(T);
        T* A_data = (T*) mkl_malloc(mem1, 64);
        memcpy(A_data, (*this).get_values_data(), mem1);

        if constexpr (std::is_same_v<T, float>){
        info = LAPACKE_sgetrf (LAPACK_ROW_MAJOR , r , r, A_data ,r , ipiv);
        info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, r, A_data, r, ipiv);
        }
        else if constexpr (std::is_same_v<T, double>){
        info = LAPACKE_dgetrf (LAPACK_ROW_MAJOR , r , r, A_data ,r , ipiv);
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, r, A_data, r, ipiv);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>){
        info = LAPACKE_cgetrf (LAPACK_ROW_MAJOR , r , r, reinterpret_cast<MKL_Complex8*> (A_data) ,r , ipiv);
        info = LAPACKE_cgetri(LAPACK_ROW_MAJOR, r, reinterpret_cast<MKL_Complex8*> (A_data), r, ipiv);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>){
        info = LAPACKE_zgetrf (LAPACK_ROW_MAJOR , r , r, reinterpret_cast<MKL_Complex16*> (A_data) ,r , ipiv);
        info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, r, reinterpret_cast<MKL_Complex16*> (A_data), r, ipiv);
        }
        std::vector<T> v(A_data,A_data+r*r);
        matrix<T> ret(v,dim_size, precomputed_strides);
        return ret;
    }
    if (n<-1){
        return ((*this)^(-1))^(-n);
    }
}
template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;
