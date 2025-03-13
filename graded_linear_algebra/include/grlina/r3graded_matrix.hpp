/**
 * @file r3graded_matrix.hpp
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

#ifndef R3GRADED_MATRIX_HPP
#define R3GRADED_MATRIX_HPP

#include <grlina/graded_matrix.hpp>

namespace graded_linalg {


struct triple {
    double x;
    double y;
    double z;

    triple() : x(0), y(0), z(0) {}
    triple(double x, double y, double z) : x(x), y(y), z(z) {}

    // Overload comparison operators if needed
    bool operator==(const triple& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const triple& other) const {
        return !(*this == other);
    }

    // Overload stream operators for easy input/output
    friend std::ostream& operator<<(std::ostream& os, const triple& d) {
        os << "(" << d.x << ", " << d.y << ", " << d.z << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, triple& d) {
        is >> d.x >> d.y >> d.z;
        return is;
    }
};



template<>
struct Degree_traits<triple> {
    static bool equals(const triple& lhs, const triple& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }

    static bool smaller(const triple& lhs, const triple& rhs) {
        if (lhs.x < rhs.x) {
            return (lhs.y <= rhs.y) && (lhs.z <= rhs.z);
        } else if (lhs.x == rhs.x) {
            if (lhs.y < rhs.y) {
                return (lhs.z <= rhs.z);
            } else if (lhs.y == rhs.y) {
                return (lhs.z < rhs.z);
            }
        }
        return false;
    }

    static bool greater(const triple& lhs, const triple& rhs) {
        if (lhs.x > rhs.x) {
            return (lhs.y >= rhs.y) && (lhs.z >= rhs.z);
        } else if (lhs.x == rhs.x) {
            if (lhs.y > rhs.y) {
                return (lhs.z >= rhs.z);
            } else if (lhs.y == rhs.y) {
                return (lhs.z > rhs.z);
            }
        }
        return false;
    }

    static bool greater_equal(const triple& lhs, const triple& rhs) {
        return (lhs.x >= rhs.x) && (lhs.y >= rhs.y) && (lhs.z >= rhs.z);
    }

    static bool smaller_equal(const triple& lhs, const triple& rhs) {
        return (lhs.x <= rhs.x) && (lhs.y <= rhs.y) && (lhs.z <= rhs.z);
    }

    static bool lex_order(const triple& lhs, const triple& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        } else if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        } else {
            return lhs.z < rhs.z;
        }
    }

    static std::function<bool(const triple&, const triple&)> lex_lambda;

    static vec<double> position(const triple& a) {
        return {a.x, a.y, a.z};
    }

    static void print_degree(const triple& a) {
        std::cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
    }

    static triple join(const triple& a, const triple& b) {
        return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
    }

    static triple meet(const triple& a, const triple& b) {
        return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
    }

    /**
     * @brief Writes the degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const triple& a) {
        os << a.x << " " << a.y << " " << a.z;
    }

    /**
     * @brief Gets the degree from an input stream.
     */
    template <typename InputStream>
    static triple from_stream(InputStream& iss) {
        triple deg;
        iss >> deg.x >> deg.y >> deg.z;
        return deg;
    }
};

/**
 * @brief Lambda function to compare lexicographically for sorting.
 */
std::function<bool(const triple&, const triple&)> Degree_traits<triple>::lex_lambda = [](const triple& a, const triple& b) {
    return Degree_traits<triple>::lex_order(a, b);
};

template <typename index>
struct R3GradedSparseMatrix : GradedSparseMatrix<triple, index> {

    R3GradedSparseMatrix() : GradedSparseMatrix<triple, index>() {}
    R3GradedSparseMatrix(index m, index n) : GradedSparseMatrix<triple, index>(m, n) {}

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R3GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) : GradedSparseMatrix<triple, index>() {

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
    R3GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<triple, index>() {
        this->parse_stream(file_stream, lex_sort, compute_batches);
    }


    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r3(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "3" << std::endl;
        output_stream << this->num_cols << " " << this->num_rows << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->num_cols; ++i) {
            Degree_traits<triple>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->num_rows; ++i) {
            Degree_traits<triple>::write_degree(output_stream, this->row_degrees[i]);
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


}; // R3GradedSparseMatrix

} // namespace graded_linalg

#endif // R3GRADED_MATRIX_HPP