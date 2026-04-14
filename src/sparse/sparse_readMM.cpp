#include "../inc/sparse.hpp"

using namespace sparse;

template <typename T>
matrix<T> matrix<T>::sparse_readMM(const std::string& MMpath)
{
    FILE *f;
    MM_typecode matcode;
    int MMrows, MMcols, MMnz;

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
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode))
    {
        fclose(f);
        throw std::runtime_error("Error: This code only supports sparse coordinate matrices.");
    }

    // Read Size (Rows, Cols, Non-zeros)
    if (mm_read_mtx_crd_size(f, &MMrows, &MMcols, &MMnz) != 0) 
    {
        fclose(f);
        throw std::runtime_error("Error: Could not read matrix dimensions.");
    }

    std::cout << "Successfully opened " << MMpath << std::endl;
    std::cout << "Rows: " << MMrows << ", Cols: " << MMcols << ", Non-Zeros: " << MMnz << std::endl;

    // Allocate Memory Vectors
    std::vector<int> rows(static_cast<size_t>(MMnz));
    std::vector<int> cols(static_cast<size_t>(MMnz));
    std::vector<T> val(MMnz);

    // Temp COO handle
    sparse_matrix_t coo_handle = nullptr; 

    if constexpr (std::is_same_v<T,float>)
    {
        for (int i = 0; i < MMnz; i++)
        {
            if(fscanf(f, "%d %d %f\n", &rows[i], &cols[i], &val[i]) != 3)
            {
                fclose(f);
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }
            // Convert 1-based to 0-based for C
            rows[i]--;
            cols[i]--;
        }
        fclose(f);

        mkl_sparse_s_create_coo(&coo_handle, SPARSE_INDEX_BASE_ZERO, MMrows, MMcols, MMnz, rows.data(), cols.data(), val.data());
    }
    else if constexpr (std::is_same_v<T,double>)
    {
        // Read Data Loop
        for (int i = 0; i < MMnz; i++)
        {
            if(fscanf(f, "%d %d %lg\n", &rows[i], &cols[i], &val[i]) != 3)
            {
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }
            // Convert 1-based to 0-based for C
            rows[i]--;
            cols[i]--;
        }
        fclose(f);

        mkl_sparse_d_create_coo(&coo_handle, SPARSE_INDEX_BASE_ZERO, MMrows, MMcols, MMnz, rows.data(), cols.data(), val.data());
    }
    else if constexpr(std::is_same_v<T,std::complex<float>>)
    {
        float realVal, imVal;
        for (int i = 0; i < MMnz; i++)
        {
            if(fscanf(f, "%d %d %f %f\n", &rows[i], &cols[i], &realVal, &imVal) != 4)
            {
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }
            // Convert 1-based to 0-based for C
            rows[i]--;
            cols[i]--;
            val[i] = std::complex<float>(realVal, imVal);
        }
        fclose(f);

        mkl_sparse_c_create_coo(&coo_handle, SPARSE_INDEX_BASE_ZERO, MMrows, MMcols, MMnz, rows.data(), cols.data(), reinterpret_cast<MKL_Complex8*>(val.data()));
    }
    else if constexpr(std::is_same_v<T,std::complex<double>>)
    {
        double realVal, imVal;
        for (int i = 0; i < MMnz; i++)
        {
            if(fscanf(f, "%d %d %lg %lg\n", &rows[i], &cols[i], &realVal, &imVal) != 4)
            {
                throw std::runtime_error("Error: File ended unexpectedly at line " + std::to_string(i));
            }
            // Convert 1-based to 0-based for C
            rows[i]--;
            cols[i]--;
            val[i] = std::complex<double>(realVal, imVal);
        }
        fclose(f);

        mkl_sparse_z_create_coo(&coo_handle, SPARSE_INDEX_BASE_ZERO, MMrows, MMcols, MMnz, rows.data(), cols.data(), reinterpret_cast<MKL_Complex16*>(val.data()));
    }

    sparse_matrix_t csr_handle = nullptr;
    mkl_sparse_convert_csr(coo_handle, SPARSE_OPERATION_NON_TRANSPOSE, &csr_handle);

    matrix<T> A = {csr_handle, static_cast<size_t>(MMrows), static_cast<size_t>(MMcols), true};

    mkl_sparse_destroy(coo_handle); // Destroy temp COO handle

    return A;
}

template class sparse::matrix<float>;
template class sparse::matrix<double>;
template class sparse::matrix<std::complex<float>>;
template class sparse::matrix<std::complex<double>>;