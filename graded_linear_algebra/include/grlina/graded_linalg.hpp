/**
 * @file graded_linalg.hpp
 * @author Jan Jendrysiak
 * @brief This library was created to provide a set of tools for working with (graded) matrices over F_2 for the
 *        AIDA algorithm which decomposes minimal presentations of persistence modules. 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#ifndef GRADED_LINALG_HPP
#define GRADED_LINALG_HPP


#include <vector>
#include <random>
#include <grlina/sparse_matrix.hpp>
#include <grlina/graded_matrix.hpp>
#include <grlina/matrix_base.hpp>
#include <grlina/graded_matrix.hpp>
#include <grlina/r2graded_matrix.hpp>
#include <grlina/r3graded_matrix.hpp>
#include <grlina/dense_matrix.hpp>

namespace graded_linalg {

template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;


/**
 * @brief Constructs a vector of R2GradedSparseMatrix objects from an input file stream.
 * 
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @param compute_batches whether to compute the column batches and k_max
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index>
std::vector<R2GradedSparseMatrix<index>> get_matrices_from_stream(std::ifstream& file_stream, bool lex_sort = false, bool compute_batches = false) {
    std::vector<R2GradedSparseMatrix<index>> matrices;

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index> matrix(file_stream, lex_sort, compute_batches);
        matrices.push_back(std::move(matrix));

        // Check if the stream is exhausted
        if (file_stream.eof()) {
            break;
        }
    }

    return matrices;
}

/**
 * @brief Constructs a vector of R2GradedSparseMatrix objects from an input file stream.
 * 
 * @param matrices vector to store the constructed matrices
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @param compute_batches whether to compute the column batches and k_max
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index, typename InputStream>
void construct_matrices_from_stream(std::vector<R2GradedSparseMatrix<index>>& matrices, InputStream& file_stream, bool lex_sort = false, bool compute_batches = false) {

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index> matrix(file_stream, lex_sort, compute_batches);
        matrices.push_back(std::move(matrix));

        // Skip empty lines
        std::string line;
        while (std::getline(file_stream, line)) {
            if (!line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
                file_stream.seekg(-static_cast<int>(line.length()) - 1, std::ios_base::cur);
                break;
            }
        }

        if (file_stream.eof()) {
            break;
        }
    }

}

/**
 * @brief Writes a vector of numbers (arithmetic type) to a .txt file.
 * 
 * @tparam T 
 * @param vec 
 * @param folder 
 * @param filename 
 */
template <typename T>
void write_vector_to_file(const std::vector<T>& vec, const std::string& folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    // Construct the full path by combining the folder and filename
    std::string full_path = folder + "/" + filename;

    std::ofstream outfile(full_path);
    
    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        outfile << vec[i];
        if (i != vec.size() - 1) {
            outfile << " ";  // Use space as a separator
        }
    }

    outfile.close();
    
    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}

/**
 * @brief Writes a vector of sets of numbers to a .txt file.
 * 
 * @tparam T 
 * @param vec 
 * @param relative_folder 
 * @param filename 
 */
template <typename T>
void write_vector_of_sets_to_file(const std::vector<std::set<T>>& vec, const std::string& relative_folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    std::cout << "Writing vector of length " << vec.size() << " to " << filename << " in folder " << relative_folder << std::endl;

    // Construct the full path by combining the relative folder and filename
    std::string full_path = relative_folder + "/" + filename;

    std::ofstream outfile(full_path);
    
    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (const auto& s : vec) {
        outfile << "{";
        for (auto it = s.begin(); it != s.end(); ++it) {
            outfile << *it;
            if (std::next(it) != s.end()) {
                outfile << ", ";
            }
        }
        outfile << "}\n";
    }

    outfile.close();
    
    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}

/**
 * @brief If the two streams contain lists of graded matrices, then
 * this function returns false if they have non-matching degrees.
 * 
 * @param stream1 
 * @param stream2 
 * @return true 
 * @return false 
 */
template <typename index>
bool compare_streams_of_graded_matrices(std::ifstream& stream1, std::ifstream& stream2) {
    vec<R2GradedSparseMatrix<index>> matrices1;
    construct_matrices_from_stream(matrices1, stream1, false, false);
    vec<R2GradedSparseMatrix<index>> matrices2;
    construct_matrices_from_stream(matrices2, stream1, false, false);
    if(matrices1.size() != matrices2.size()){
        return false;
    }
    std::sort(matrices1.begin(), matrices1.end(), Compare_by_degrees<degree, index>());
    std::sort(matrices2.begin(), matrices2.end(), Compare_by_degrees<degree, index>());
    for(index i = 0; i < matrices1.size(); i++){
        if( Compare_by_degrees<degree, index>::compare_three_way(matrices1[i], matrices2[i]) != 0){
            return false;
        }
    }
    return true;
}

/**
 * @brief If the two files contain lists of graded matrices, then
 * this function returns false if they have non-matching degrees.
 * 
 * @param path1 
 * @param path2 
 * @return true 
 * @return false 
 */
template <typename index>
bool compare_files_of_graded_matrices(std::string path1, std::string path2) {
    std::ifstream stream1(path1);
    std::ifstream stream2(path2);
    return compare_streams_of_graded_matrices<index>(stream1, stream2);
}
    




/**
 * @brief Gets the Minor of M with respect to the row and column indices and saves the result as a DenseMatrix.
 *  Not tested yet!
 * @tparam index 
 * @param M 
 * @param col_indices 
 * @param row_indices 
 * @return DenseMatrix 
 */
/**
template <typename index>
DenseMatrix restricted_dense_copy(const SparseMatrix<index>& M, vec<index> col_indices, vec<index> row_indices) {
    DenseMatrix<int> result(col_indices.size(), row_indices.size());
    for(index i = 0; i < col_indices.size(); i++){
        auto c = col_indices[i];
        auto it = std::lower_bound(M.data[c].begin(), M.data[c].end(), row_indices[0]);
        index j = 0;
        while(it != M.data[c].end() && j < row_indices.size()){
            auto r = row_indices[j];
            if(*it == r){
                result.data[i].set(j);
                j++;
                it++;
            } else if(*it < r){
                it++;
            } else {
                j++;
            }
        }
    }
}
*/

/**
 * @brief Computes A^{-1}*B by operating on B.
 * 
 * @tparam index 
 * @tparam COLUMN 
 * @tparam DERIVED 
 * @param A Needs to be a square invertible matrix
 * @param B 
 * @return DERIVED 
 */
template <typename index, typename COLUMN, typename DERIVEDfirst, typename DERIVEDsecond>
DERIVEDsecond divide(const MatrixUtil<index, COLUMN, DERIVEDfirst>& A, const MatrixUtil<index, COLUMN, DERIVEDsecond>& B){
    DERIVEDfirst A_copy(A);
    DERIVEDsecond B_copy(B);
    index m = B.get_num_cols();
    index n = B.get_num_rows();
    assert(A.get_num_cols() == A.get_num_rows() && A.get_num_rows() == n);


} 

/**
DenseMatrix::DenseMatrix(SparseMatrix<index> M) : MatrixUtil<bitset,index>(M.get_num_rows(),M.get_num_cols()){
    this->num_cols = M.get_num_cols();
    this->num_rows = M.get_num_rows();
    this->data = vec<bitset>(num_cols,bitset(num_rows));
    for(int i = 0; i < M.get_num_cols(); i++){
        for(index j : M.data[i]){
            this->data[i].set(j);
        }
    }
}
*/

// Function to randomly return false or true
bool random_bool() {
    static std::random_device rd;  // Seed for the random number engine
    static std::mt19937 gen(rd()); // Mersenne Twister random number engine
    static std::bernoulli_distribution d(0.5); // Distribution that produces true with probability 0.5

    return d(gen);
}

} // namespace graded_linalg

#endif // GRADED_LINALG_HPP
