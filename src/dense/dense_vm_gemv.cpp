#include "../inc/dense.hpp"

namespace dense
{
    template<typename T>
    std::vector<T> operator*(const std::vector<T> x, const matrix<T>& A)
    {
        // Dimension missmatch check
        if (A.get_nod() < 2) {
             throw std::runtime_error("v*A requires A to have at least 2 dimensions.");
        }
        const MKL_INT M = static_cast<MKL_INT>(A.get_dim_size(0));
        const MKL_INT N = static_cast<MKL_INT>(A.get_dim_size(1));

        if (static_cast<MKL_INT>(x.size()) != M) {
             throw std::runtime_error("Vector size (" + std::to_string(x.size()) + 
                                      ") does not match matrix's first dimension (" + std::to_string(M) + ").");
        }    

        std::vector<T> result(A.get_dim_size(1));

        if constexpr(std::is_same_v<T, float>) 
        {
            float a = 1.0;
            float b = 0.0;
            sgemv(CblasRowMajor, CblasTrans, A.get_dim_size(0), A.get_dim_size(1), a, A.get_const_values_data(),
                  A.get_dim_size(1), x.data(), 1, b, result.data(), 1);
            
            return result;
        }
        else if constexpr(std::is_same_v<T, double>) 
        {
            double a = 1.0;
            double b = 0.0;
            dgemv(CblasRowMajor, CblasTrans, A.get_dim_size(0), A.get_dim_size(1), a, A.get_const_values_data(),
                  A.get_dim_size(1), x.data(), 1, b, result.data(), 1);
            
            return result;
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>) 
        {
            std::complex<float> a(1, 0);
            std::complex<float> b(0, 0);
            cgemv(CblasRowMajor, CblasTrans, A.get_dim_size(0), A.get_dim_size(1), &a, A.get_const_values_data(),
                  A.get_dim_size(1), x.data(), 1, &b, result.data(), 1);
            
            return result;
        }
        else if constexpr(std::is_same_v<T, std::complex<double>>) 
        {
            std::complex<double> a(1, 0);
            std::complex<double> b(0, 0);
            zgemv(CblasRowMajor, CblasTrans, A.get_dim_size(0), A.get_dim_size(1), &a, A.get_const_values_data(),
                  A.get_dim_size(1), x.data(), 1, &b, result.data(), 1);
            
            return result;
        }
    }

    template std::vector<float> operator*(const std::vector<float> x, const dense::matrix<float>&);
    template std::vector<double> operator*(const std::vector<double> x, const dense::matrix<double>&);
    template std::vector<std::complex<float>> operator*(const std::vector<std::complex<float>> x, const dense::matrix<std::complex<float>>&);
    template std::vector<std::complex<double>> operator*(const std::vector<std::complex<double>> x, const dense::matrix<std::complex<double>>&);
}