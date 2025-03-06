// r2graded_matrix.hpp 

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
using degree_list = vec<degree>;

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
   
    std::pair<degree, std::vector<index>> parse_line(const std::string& line, bool hasEntries = true) {
        std::istringstream iss(line);
        degree deg;
        std::vector<index> rel;


        // Parse degree
        iss >> deg.first >> deg.second;


        // Consume the semicolon
        std::string tmp;
        iss >> tmp;
        if(tmp != ";"){
            std::cerr << "Error: Expecting a semicolon. Invalid format in the following line: " << line << std::endl;
            std::abort();
        }

        // Parse relation
        if(hasEntries){
    
            index num;
            while (iss >> num) {

                rel.push_back(num);
            }
        }

        return std::move(std::make_pair(deg, rel));
    }

    

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

        parse_stream(file, lex_sort, compute_batches);

    } // Constructor from file

    public:
    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<degree, index>() {
        parse_stream(file_stream, lex_sort, compute_batches);
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

private:
    void parse_stream(std::istream& file_stream, bool lex_sort, bool compute_batches) {
        std::string line;

        // Read the first line to determine the file type
        std::getline(file_stream, line);
        if (line.find("firep") != std::string::npos) {
            // Skip 2 lines for FIREP
            std::getline(file_stream, line);
            std::getline(file_stream, line);
        } else if (line.find("scc2020") != std::string::npos) {
            // Skip 1 line for SCC2020
            std::getline(file_stream, line);
        } else {
            // Invalid file type
            std::cerr << "Error: Unsupported file format. The first line must contain firep or scc2020." << std::endl;
            std::abort();
        }

        // Parse the first line after skipping
        std::getline(file_stream, line);
        std::istringstream iss(line);
        index num_rel, num_gen, thirdNumber;

        // Check that there are exactly 3 numbers
        if (!(iss >> num_rel >> num_gen >> thirdNumber) || thirdNumber != 0) {
            std::cerr << "Error: Invalid format in the third or fourth line. Expecting exactly 3 numbers with the last one being 0." << std::endl;
            std::abort();
        }

        this->num_cols = num_rel;
        this->num_rows = num_gen;

        this->col_degrees.reserve(num_rel);
        this->row_degrees.reserve(num_gen);
        this->data.reserve(num_gen);

        index rel_counter = 0;

        bool first_pass = true;
        degree last_degree;
        index k_counter = 1;
        index j = 0;
        if (compute_batches) {
            this->col_batches.reserve(num_rel);
            if(num_rel > 0){
                this->col_batches.push_back(vec<index>());
            }
        }

        while (rel_counter < num_rel + num_gen) {
            if(!std::getline(file_stream, line)){
                std::cout << "Error: Unexpected end of file. \n Make sure that the dimensions of the file are correctly given at the beginning of the file." << std::endl;
            }
            std::pair<degree, std::vector<index>> line_data;
            if (rel_counter < num_rel) {
                line_data = parse_line(line, true);
                if (compute_batches && !lex_sort) {
                    if (first_pass) {
                        last_degree = line_data.first;
                        first_pass = false;
                    } else if (line_data.first == last_degree) {
                        k_counter++;
                        if (k_counter > this->k_max) {
                            this->k_max = k_counter;
                        }
                    } else {
                        last_degree = line_data.first;
                        j++;
                        this->col_batches.push_back(vec<index>());
                        k_counter = 1;
                    }
                    this->col_batches[j].push_back(rel_counter);
                }
                this->col_degrees.push_back(line_data.first);
                this->data.push_back(line_data.second);
                rel_counter++;
            } else {
                line_data = parse_line(line, false);
                this->row_degrees.push_back(line_data.first);
                rel_counter++;
            }
        }

        if (compute_batches && !lex_sort) {
            // std::cout << "Loaded graded Matrix with k_max: " << this->k_max << std::endl;
        }

        if (lex_sort) {
            std::cout << "Sorting the matrix lexicographically" << std::endl;
            this->sort_columns_lexicographically();
            this->sort_rows_lexicographically();
            if (compute_batches) {
                this->compute_col_batches();
                // std::cout << "Loaded graded Matrix with k_max: " << this->k_max << std::endl;
            }
        }

        if (!compute_batches) {
            // std::cout << "Loaded graded Matrix without computing k_max" << std::endl;
        }
    } // Constructor from ifstream
    
}; // R2GradedSparseMatrix



} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP
