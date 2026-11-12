#ifndef SPARSE_HPP
#define SPARSE_HPP
#include <cstring>

/**
 * @file sparse.hpp
 * @brief Header file for the matrix class for sparse matrices and its associated functions.
 */

#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include "def.hpp"

// extern "C" 
// {
//       #include "../inc/mmio.h"
// }

extern "C" 
{
      #include "mmio.h"
}
namespace dense {
    template<typename T>
    class matrix; // forward declaration
}

namespace sparse{
/**
 * @class matrix
 * @brief A class to create a sparse matrix object in csr form
 * @tparam T The scalar type used for matrix elements.
 */
template <typename T>
class matrix
{
    private:

        /**
         * @brief Flat vector storing the values of non zero elements.
         */
        std::vector<T> values; 

        /**
         * @brief Flat vector storing the columns of non-zero elements
         */
        std::vector<MKL_INT> col_idx;

        /**
         * @brief Flat vector marking the begining of each row in the values vector
         */
        std::vector<MKL_INT> row_idx;

        /**
         * @brief Rows size.
         */
        size_t row_size;

        /**
         * @brief Columns size.
         */
        size_t col_size;
        
        /**
         * @brief MKL sparse matrix handle, saved for reusability and better chaining performance.
         */
        mutable sparse_matrix_t handle;

        /**
         * @brief False if the raw csr data (values, row_idx, col_idx) has been modified since the handle was last updated (say by a setter call), True if csr and handle data are expected to be in agreement.
         */
        mutable bool valid_handle;

        /**
         * @brief Keeps track of csr data (values, row_idx, col_idx) ownership.
         */
        bool ownership;

    public:
          /**
                * @brief Default sparse matrix constructor
                */
          matrix() = default;
        /**
         * @brief rebuilds a valid handle according to csr vectors.
         */
          void refresh_handle() const; //Thanasis
          /**
                * @brief Sparse matrix constructor
                * @param values sets the values vector
                * @param col_idx sets the col_idx vector
                * @param row_idx sets the row_idx vector
                * @param rows sets row size
                * @param cols sets column size
                */
          template<typename U>
          matrix(const std::vector<T>& values, const std::vector<U>& col_idx, const std::vector<U>& row_idx, size_t rows, size_t cols)
          : values(values), col_idx(col_idx), row_idx(row_idx), row_size(rows), col_size(cols), ownership(true), valid_handle(false), handle(nullptr)
          {}; //Thanasis

          /**
           *@brief Sparse matrix object constructor from MKL handle.
          * @param h MKL sparse matrix handle.
          * @param rows number of rows.
          * @param cols number of columns.
          * @param own CSR arrays ownership.
          */
          matrix(const sparse_matrix_t h, size_t rows, size_t cols, bool own); //Thanasis

          /**
          * @brief Sparse matrix destructor
          */
          ~matrix(){ // Victor
            if (valid_handle) {
                  mkl_sparse_destroy(handle);
                  valid_handle = false;}
            };

          /**
                * @brief Move constructor, transfers the ownership of matrix A
                * @param  A A matrix reference to be moved
                */
          matrix(matrix&& A) noexcept // Thanasis and Victor
            : values(std::move(A.values)), col_idx(std::move(A.col_idx)), row_idx(std::move(A.row_idx)),
            row_size(A.row_size), col_size(A.col_size), handle(A.handle), valid_handle(A.valid_handle), ownership(A.ownership){
            A.row_size = 0;
            A.col_size = 0;
            A.handle   = nullptr;
            A.valid_handle = false;
            A.ownership = false;
            }

          /**
                * @brief Matrix copy constructor
                * @param A A constant sparse matrix reference to be copied 
                */
          matrix (matrix const& A); // John

          /**
                * @brief Constructs an N by N identity matrix
                * @param N the number of rows/columns in the square identity matrix
                */
          static matrix I(int N); // John

          /**
           * @brief Contsructs an N by N banded matrix
           * @param N The number of rows/columns
           * @param kl The number of elements below the main diagonal
           * @param ku The number of the elements above the main diagonal
           * @param val The type <T> value of all the bands
           */
          static matrix banded(int N, int kl, int ku, T val);

          //operators
          
          /**
                * @brief Operator =, creates a deep copy of the matrix and assigns it
                * @param A A constant matrix reference used after the operator
                */
          matrix& operator=(matrix const& A); // John

          /**
                * @brief Operator +, adds the values of two matrices of same size and dimensions
                * @param A A constant matrix reference used after the operator
                */
          matrix operator+(matrix const& A); //Thanasis

          /**
                * @brief Operator *, multiplies two 2D sparse matrices
                * @param A A constant matrix reference used after the operator
                */
          matrix operator*(matrix const& A);// Thanasis

          /**
                * @brief Operator *, multiplies a sparse matrix by a number
                * @param x A numerical primitive used after the operator
                * @tparam T encompasses all primitive numerical data types
                */
          matrix operator*(T x);// Victor

          /**
                * @brief Operator *, multiplies a sparse matrix by a vector
                * @param x A vector used after the operator
                * @tparam T encompasses all primitive numerical data types
                */
          std::vector<T> operator*(const std::vector<T> x);// John

          /**
                * @brief Operator -, subtracts the values of second matrix from the first, both must have same size and dimensions
                * @param A A constant matrix reference used after the operator
                */
          matrix operator-(matrix const& A);// Thanasis

          /**
                * @brief Operator ==, compares two matricies
                * @param A A constant matrix reference used after the operator
                */
          bool operator==(matrix const& A);// John

          //getters-setters
          /**
                * @brief Gets value at a given index.
                * @param index The linear 0-based index in the flat values vector.
                */
          T get_value_at(size_t index) const;// Thanasis

          /**
                * @brief Sets value at a given index.
                * @param index The linear 0-based index in the flat values vector.
                * @param value The value to set.
                */ 
          void set_value_at(size_t index, T value);// Thanasis

          /**
                * @brief Returns a constant value in the matrix.
                * @param c The column of the value to be returned.
                * @param r The row of the value to be returned.
                */  
          const T operator() (size_t r, size_t c) const;// Thanasis

          /**
                * @brief Returns the shape (dimensions) of a matrix
                */  
          std::pair<size_t, size_t> shape() const;// Victor

          //sparse specific (for solvers etc)
          /**
                * @brief Returns a constant vector reference containing the diagonal elements of a matrix. For reading.
                */
          std::vector<T> diag() const;// Thanasis

          /**
               * @brief Prints the values of 2D array
          */
          void print2D(); // John
          /**
               * @brief Prints the sparse matrix row by row
          */
          void print_matrix(); // Thanasis

          /**
          * @brief Returns MKL handle (const access for read-only use in operations).
          */
           const sparse_matrix_t& get_handle() const { return handle; } // Victor

          /**
          * @brief Invalidates handle.
          */
          void invalidate_handle() { handle = nullptr; valid_handle = false; } // Victor

          /**
          * @brief Gets values' data const
          */
           const std::vector<T> get_values_data_const() const { return values; } // For reading Victor

          /**
          * @brief Gets values' non const
          */
          std::vector<T>& get_values_data() { return values; } // For writing Victor           
          
          /**
          * @brief Function that transposes the matrix in 2D
          */
          matrix transpose() const; // Victor

          /**
           * @brief Reads a Matrix Market (MM) sparse matrix, converts it from COO to CSR format 
           *        and returns a CSR matrix obj (with a handle cosntructor)
           * @param MMpath A string with the path to the MM matrix
           * @returns A sparse matrix object created with the handle constructor
           */
          static matrix sparse_readMM(const std::string& MMpath);

};
/**
 * @brief Operator *, multiplies a number by a sparse matrix
 * @param x A numerical primitive used before the operator
 * @tparam U encompasses all primitive numerical data types and T, the matrix type
 */
template<typename T, typename U>
matrix<T> operator*(U x, const matrix<T>& A); // Victor

/**
 * @brief Operator *, multiplies a vector by a sparse matrix
 * @param x A vector used before the operator
 * @tparam T encompasses all primitive numerical data types and the matrix type
 */
template<typename T>
std::vector<T> operator*(const std::vector<T> x, const matrix<T>& A); // Victor
}

std::vector<double> FGMRES_dsolve(sparse::matrix<double> & A, std::vector<double> & b, std::vector<double> & x0, bool preconditioned, sparse::matrix<double> & M);

/**
 * @brief Double Conjugate Gradient solver for a single right-hand side (SRHS) system
 * @param A SPD Coefficient Matrix 
 * @param b RHS vector
 * @param x0 Initial guess vector (usually its set to 0)
 */
std::vector<double> CG_dsolve(sparse::matrix<double> & A, std::vector<double> & b, std::vector<double> & x0);

#endif
