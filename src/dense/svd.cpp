#include "../inc/dense.hpp"
#include <cstring>
#include <variant>
using namespace dense;

template <typename T>
void matrix<T>::svd(matrix<T>& u, matrix<T>& s, matrix<T>& v){
    MKL_INT info;
    MKL_INT n = dim_size[1];
    MKL_INT m = dim_size[0];
    MKL_INT lda = n;
    MKL_INT ldu = m;
    MKL_INT ldv = n;
    
    size_t mem1 = n*m*sizeof(T);
    T* A_data = (T*) mkl_malloc(mem1, 64);
    memcpy(A_data, values.data(), mem1);
    
    std::vector<T> u_vec(m*m, 0);
    std::vector<T> v_vec(n*n, 0);
    if constexpr (std::is_same_v<T, float>){
        std::vector<float> s_vec (n,0);
        info = LAPACKE_sgesdd (LAPACK_ROW_MAJOR , 'A' , m, n , A_data , lda, s_vec.data(), u_vec.data(), ldu, v_vec.data(), ldv);
        s = matrix<float>(s_vec, m, n);
    }
    else if constexpr (std::is_same_v<T, double>){
        std::vector<double> s_vec (n,0);
        info = LAPACKE_dgesdd (LAPACK_ROW_MAJOR , 'A' , m, n ,A_data, lda, s_vec.data(), u_vec.data(), ldu, v_vec.data(), ldv);
        s = matrix<double>(s_vec, m, n);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>){
        std::vector<float> s_vec (n,0);
        info = LAPACKE_cgesdd (LAPACK_ROW_MAJOR , 'A' , m, n , reinterpret_cast<MKL_Complex8*> (A_data) , lda, s_vec.data(), reinterpret_cast<MKL_Complex8*> (u_vec.data()), ldu, reinterpret_cast<MKL_Complex8*> (v_vec.data()), ldv);
        std::vector<std::complex<float>> s_vec_c;
        for(auto i: s_vec){
            s_vec_c.push_back(std::complex<float>(i,0));
        }
        s = matrix<std::complex<float>>(s_vec_c, m, n);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>){
        std::vector<double> s_vec (n,0);
        info = LAPACKE_zgesdd (LAPACK_ROW_MAJOR , 'A' , m, n , reinterpret_cast<MKL_Complex16*> (A_data) , lda, s_vec.data(), reinterpret_cast<MKL_Complex16*> (u_vec.data()), ldu, reinterpret_cast<MKL_Complex16*> (v_vec.data()), ldv);
        std::vector<std::complex<double>> s_vec_z;
        for(auto i: s_vec){
            s_vec_z.push_back(std::complex<double>(i,0));
        }
        s = matrix<std::complex<double>>(s_vec_z, m, n);
    }
    u = matrix<T>(u_vec, std::vector<size_t>{(size_t)(m),(size_t)(m)}, std::vector<int>{m, 1});
    v = matrix<T>(v_vec, std::vector<size_t>{(size_t)(n),(size_t)(n)}, std::vector<int>{n, 1});
}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;
