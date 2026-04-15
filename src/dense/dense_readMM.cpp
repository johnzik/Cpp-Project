#include "../inc/dense.hpp"

using namespace dense;

template <typename T>
matrix<T> matrix<T>::dense_readMM(const std::string& MMpath)
{
    FILE *f;
    MM_typecode matcode;
    int MMrows, MMcols;

    // Open the file
    if ((f=fopen(MMpath.c_str(), "r")) == NULL)
    {
        fclose(f);
        throw std::runtime_error("Error: Could not open file " + MMpath);
    }

    // Read Banner
    if (mm_read_banner(f, &matcode) != 0)
    {
        fclose(f);
        throw std::runtime_error("Error: Could not process Matrix Market banner.");
    }

    // Check if it's a valid sparse matrix
    if (!mm_is_matrix(matcode) || !mm_is_array(matcode))
    {
        fclose(f);
        throw std::runtime_error("Error: This code only supports dense array matrices.");
    }

    // Read Size (Rows, Cols)
    if (mm_read_mtx_array_size(f, &MMrows, &MMcols) != 0) 
    {
        fclose(f);
        throw std::runtime_error("Error: Could not read matrix dimensions.");
    }

    std::cout << "Successfully opened " << MMpath << std::endl;
    std::cout << "Rows: " << MMrows << ", Cols: " << MMcols << std::endl;

    // Create the dimension matrix (with 0 values)
    // Save the dims reversed since MM uses Col-Major order
    matrix<T> A(MMcols, MMrows);

    if constexpr (std::is_same_v<T, float>)
    {
        for (int i = 0; i < (MMcols*MMrows); i++)
        {
            float val;
            if(fscanf(f, "%f\n", &val) != 1)
            {
                fclose(f);
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }

            A.set_value_at(i, val);
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        for (int i = 0; i < (MMcols*MMrows); i++)
        {
            double val;
            if(fscanf(f, "%lg\n", &val) != 1)
            {
                fclose(f);
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }

            A.set_value_at(i, val);
        }
    }
    else if constexpr(std::is_same_v<T, std::complex<float>>)
    {
        for (int i = 0; i < (MMcols*MMrows); i++)
        {
            float rVal, iVal;
            if(fscanf(f, "%f %f\n", &rVal, &iVal) != 2)
            {
                fclose(f);
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }

            std::complex<float> val(rVal, iVal);
            A.set_value_at(i, val);
        }
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>)
    {
        for (int i = 0; i < (MMcols*MMrows); i++)
        {
            double rVal, iVal;
            if(fscanf(f, "%lg %lg\n", &rVal, &iVal) != 2)
            {
                fclose(f);
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }

            std::complex<double> val(rVal, iVal);
            A.set_value_at(i, val);
        }
    }

    fclose(f);

    // MM stores dense matrices in Col-Major order so for Row-Major we must return the transpose
    return A.transpose();
}

template class dense::matrix<float>;
template class dense::matrix<double>;
template class dense::matrix<std::complex<float>>;
template class dense::matrix<std::complex<double>>;