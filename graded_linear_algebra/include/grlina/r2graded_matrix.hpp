/**
 * @file r2graded_matrix.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once

#ifndef R2GRADED_MATRIX_HPP
#define R2GRADED_MATRIX_HPP

#include <iostream>
#include <vector>
#include <grlina/graded_matrix.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>



namespace graded_linalg {

template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;


using degree = std::pair<double, double>;

template<>
struct Degree_traits<degree> {
    static bool equals(const degree& lhs, const degree& rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    static bool smaller(const degree& lhs, const degree& rhs) {
        if(lhs.first < rhs.first) {
            return (lhs.second <= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        } else {
            return false;
        }
    }

    static bool greater(const degree& lhs, const degree& rhs) {
        if(lhs.first > rhs.first) {
            return (lhs.second >= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second > rhs.second;
        } else {
            return false;
        }
    }

    static bool greater_equal(const degree& lhs, const degree& rhs) {
        return (lhs.first >= rhs.first) && (lhs.second >= rhs.second);
    }

    static bool smaller_equal(const degree& lhs, const degree& rhs) {
        return (lhs.first <= rhs.first) && (lhs.second <= rhs.second);
    }

    static bool lex_order(const degree& a, const degree& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

    static std::function<bool(const degree&, const degree&)> lex_lambda;

    static vec<double> position(const degree& a)  {
        return {a.first, a.second};
    }

    static void print_degree(const degree& a) {
        std::cout << "(" << a.first << ", " << a.second << ")";
    }

    static degree join(const degree& a, const degree& b)  {
        return {std::max(a.first, b.first), std::max(a.second, b.second)};
    }

    static degree meet(const degree& a, const degree& b) {
        return {std::min(a.first, b.first), std::min(a.second, b.second)};
    }

    
    /**
     * @brief Writes the degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const degree& a) {
        os << a.first << " " << a.second;
    }

    template <typename InputStream>
    static degree from_stream(InputStream& iss){
        degree deg;
        iss >> deg.first >> deg.second;
        return deg;
    }

}; //Degree_traits<degree>

/**
 * @brief Lambda function to compare lexicographically for sorting.
 */
std::function<bool(const degree&, const degree&)> Degree_traits<degree>::lex_lambda = [](const degree& a, const degree& b) {
    return Degree_traits<degree>::lex_order(a, b);
};

/**
 * @brief A graded matrix with degrees in R^2.
 * 
 * @tparam index 
 */
template <typename index>
struct R2GradedSparseMatrix : GradedSparseMatrix<degree, index> {

    R2GradedSparseMatrix() : GradedSparseMatrix<degree, index>() {}
    R2GradedSparseMatrix(index m, index n) : GradedSparseMatrix<degree, index>(m, n) {}
    

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) : GradedSparseMatrix<degree, index>() {

        size_t dotPosition = filepath.find_last_of('.');
        bool no_file_extension = false;
        if (dotPosition == std::string::npos) {
           // No dot found, invalid file format
           no_file_extension = true;
            std::cout << " File does not have an extension (.scc .firep .txt)?" << std::endl;
        }

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << " Error: Unable to open file " << filepath << std::endl;
            std::abort();
        }

        std::string extension;
        if(!no_file_extension) {
            extension=filepath.substr(dotPosition);
        }
        std::string line;

        // Check the file extension and perform actions accordingly
        if (extension == ".scc" || extension == ".firep" || extension == ".txt" || no_file_extension) {
            // std::cout << "Reading presentation file: " << filepath << std::endl;
        } else {
            // Invalid file extension
            std::cout << "Warning, extension does not match .scc, .firep, .txt, or no extension." << std::endl;
        }

        this->parse_stream(file, lex_sort, compute_batches);

    } // Constructor from file


    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<degree, index>() {
        this->parse_stream(file_stream, lex_sort, compute_batches);
    }

    

    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r2(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "2" << std::endl;
        output_stream << this->num_cols << " " << this->num_rows << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->num_cols; ++i) {
            Degree_traits<degree>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->num_rows; ++i) {
            Degree_traits<degree>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
            output_stream << std::endl;
        }
    }

    /**
     * @brief Returns a basis for the kernel of a 2d graded matrix.
     * 
     * @return SparseMatrix<index> 
     */
    SparseMatrix<index> kernel()  override {
        // Implement
        return SparseMatrix<index>();
    }




}; // R2GradedSparseMatrix




} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP
