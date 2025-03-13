/**
 * @file dense_matrix.hpp
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

#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include "grlina/matrix_base.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>

namespace graded_linalg {


template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;



struct BitsetHash {
    unsigned long operator()(const boost::dynamic_bitset<>& bs) const {
        // Safe to use to_ulong() if the bitset size is guaranteed to be <= 32
        return bs.to_ulong();
    }
};

/**
 * @brief asks if a < b in order of the entries (reverse of standard comparison)
 * 
 * @param a 
 * @param b 
 * @return true 
 * @return false 
 */
bool compareBitsets(const boost::dynamic_bitset<>& a, const boost::dynamic_bitset<>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Bitsets must be of the same size for comparison.");
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.test(i) != b.test(i)) {
            return a.test(i) < b.test(i);
        }
    }

    return false; // The bitsets are equal
}

std::ostream& operator<< (std::ostream& ostr, const boost::dynamic_bitset<>& bs) {
    for (int i = 0; i < bs.size(); i++){
      ostr << bs[i] << " ";
    }
    return ostr;
}


/**
 * @brief prints the bitset from first to last entry
 * 
 * @param bs 
 */
void print_bitset(const boost::dynamic_bitset<>& bs) {
    std::cout << bs << std::endl;
}

/**
 * @brief prints the bitset from last to first entry
 * 
 * @param bs 
 */
void print_bitset_reverse(const boost::dynamic_bitset<>& bs) {
    for (int i = bs.size() - 1; i >= 0; --i) {
        std::cout << bs[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Converts a bitset to a string representation in reverse order
 * 
 * @param bs 
 * @return std::string 
 */
std::string bitsetToString_alt(const boost::dynamic_bitset<>& bs) {
    std::string result;
    result.reserve(bs.size());
    for (int i = bs.size() - 1; i >= 0; --i) {
        result.push_back(bs.test(i) ? '1' : '0');
    }
    return result;
}

/**
 * @brief Converts a bitset to a string representation in forward order
 * 
 * @param bs 
 * @return std::string 
 */
std::string bitsetToString(const boost::dynamic_bitset<>& bs) {
    std::string result;
    result.reserve(bs.size());
    for (size_t i = 0; i < bs.size(); ++i) {
        result.push_back(bs.test(i) ? '1' : '0');
    }
    return result;
}

/**
 * @brief Writes a dynamic_bitset to a file in reverse order
 * 
 * @param bs 
 * @param file 
 */
void serializeDynamicBitset(const boost::dynamic_bitset<>& bs, std::ofstream& file) {
    int length = bs.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    const auto& bs_data = bs.to_ulong();
    file.write(reinterpret_cast<const char*>(&bs_data), sizeof(bs_data));
}


/**
 * @brief Reads a dynamic_bitset from a file in reverse order
 * 
 * @param file 
 * @return boost::dynamic_bitset<> 
 */
boost::dynamic_bitset<> deserializeDynamicBitset(std::ifstream& file) {
    int length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    unsigned long bs_data;
    file.read(reinterpret_cast<char*>(&bs_data), sizeof(bs_data));
    return bitset(length, bs_data);
}

/**
 * @brief Adds bitset 'a' to bitset 'b' using the XOR operation.
 * 
 * @param a 
 * @param b 
 */
void add_to(boost::dynamic_bitset<> &a, boost::dynamic_bitset<> &b) {
    assert(a.size() != b.size());
    b ^= a;
}

/**
 * @brief Returns the int of the last non-zero entry in the bitset or -1 if there are no non-zero entries.
 * 
 * @param v 
 * @return int 
 */
int last_entry_index(bitset& v) {
    for (int i = v.size() - 1; i >= 0; --i) {
        if (v[i]) {
            return i;
        }
    }
    return -1;
};

vec<bitset> compute_standard_vectors(int k){
    vec<bitset> result;
    for (int i = 0; i < k; i++){
        result.emplace_back(bitset(k, 0).set(i));
    }
    return result;
}

/**
 * @brief returns { 1, 11, 111, ... }
 * 
 * @param k 
 * @return 
 */
vec<boost::dynamic_bitset<>> compute_sum_of_standard_vectors(int k){
    auto result = vec<boost::dynamic_bitset<>>();
    for (int i = 0; i < k; i++){
        boost::dynamic_bitset<> bitset(i + 1);
        bitset.set(); // Set all bits to 1
        result.push_back(bitset);
    }
    return result;
}

/**
 * @brief Returns a copy of a with the 1-entries replaced by b.
 * 
 * @param a 
 * @param b 
 * @return bitset 
 */
bitset glue(const bitset& a, const bitset& b){
    bitset result = a;
    assert(a.count() == b.size());
    size_t counter = 0;
    for(auto it = result.find_first(); it != bitset::npos; it = result.find_next(it)){
        if(!b[counter]){
            result.reset(it);
        }
        counter++;
    }
    return result;
}

/**
 * @brief Copies b onto the 1-entries of a.
 * 
 * @param a 
 * @param b 
 */
void glue_to(bitset& a, const bitset& b){
    assert(a.count() == b.size());
    size_t counter = 0;
    for(auto it = a.find_first(); it != bitset::npos; it = a.find_next(it)){
        if(!b[counter]){
            a.reset(it);
        }
        counter++;
    }
}

struct DenseMatrix : public MatrixUtil<bitset, int, DenseMatrix>{
        
    bool rowReduced = false;
    bool completeRowReduced = false;
    boost::dynamic_bitset<> pivot_vector;

    bool vis_nonzero_at(bitset& v, int i) override {
        return v[i];
    };

    void vadd_to(bitset& v, bitset& w) override {
        add_to(v, w);
    };

    /**
     * @brief Returns the int of the last non-zero entry in the bitset or -1 if there are no non-zero entries.
     * 
     * @param v
     * @return int 
     */
    int vlast_entry_index(bitset& v) override {
        for (int i = v.size() - 1; i >= 0; --i) {
            if (v[i]) {
                return i;
            }
        }
        return -1;
    };

    bool vis_equal(boost::dynamic_bitset<>& a, boost::dynamic_bitset<>& b) override {
        return(a == b);
    };

    bool vproduct(bitset& a, bitset& b) override {
        return (a & b).count() % 2;
    };

    bitset get_standard_vector(int i, int n)  {
        return bitset(n, 0).set(i);
    };

    bitset get_random_vector(int length, int perc) {
        bitset result(length, 0);
        for (int i = 0; i < length; i++){
            if (std::rand() % 100 < perc){
                result.set(i);
            }
        }
        return result;
    };
    

    void vset_entry(bitset& v, int j){
        v.flip(j);
    };

    /**
     * @brief Deletes the last i rows of the matrix.
     * 
     * @param i 
     */
    void cull_columns(int i){
        this->num_rows = this->num_rows - i;
        for (auto& column : this->data){
            column.resize(this->num_rows);
        }
    };

    DenseMatrix() : MatrixUtil<bitset, int, DenseMatrix>() {}

    DenseMatrix(int m, int n) : MatrixUtil<bitset, int, DenseMatrix>(m, n) {
        data = vec<bitset>(m, bitset(n, 0));
    }

    DenseMatrix(const vec<bitset>& data) : MatrixUtil<bitset, int, DenseMatrix>(data.size(), data[0].size(), data) {}

    DenseMatrix(const DenseMatrix& other) : MatrixUtil<bitset, int, DenseMatrix>(other)  {}

    DenseMatrix(int m, const std::string& type) : MatrixUtil<bitset, int, DenseMatrix>(m, m) {
        if(type == "Identity") {
            for(int i = 0; i < m; i++) {
                this->data.emplace_back(bitset(m, 0).set(i));
            }
        } else {
            throw std::invalid_argument("Unknown matrix type: " + type);
        }
    }

    /**
     * @brief Construct a new Dense Matrix object from an ifstream if it has the bitsets as 1-0-strings.
     * 
     * @param file 
     */
    DenseMatrix(std::ifstream& file) : MatrixUtil<bitset, int, DenseMatrix>(){
        // deserialize matrix from a file
        num_cols = 0;
        num_rows = 0;
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(int));
        data.reserve(num_cols);
        for (int i = 0; i < num_cols; ++i) {
            unsigned long bs_data;
            file.read(reinterpret_cast<char*>(&bs_data), sizeof(bs_data));
            data.emplace_back(bitset(num_rows, bs_data));
        }
    }

    /**
     * @brief Construct a new Dense Matrix consisting of only elementary basis vectors. 
     * 
     * @param i_pivots Marks the elementary basis vectors that are to be included in the matrix.
     */
    DenseMatrix(const boost::dynamic_bitset<>& i_pivots) : 
        MatrixUtil<bitset, int, DenseMatrix>(i_pivots.count(), i_pivots.size(), std::vector<bitset>(i_pivots.count(), bitset(i_pivots.size(), 0)) ),
        rowReduced(true),       // By construction, the matrix is in reduced form
        pivot_vector(i_pivots){ 
        // Populate the matrix to form an identity matrix on the pivot positions
        int pivotint = 0; // To keep track of which pivot we're on
        for (int r = 0; r < num_cols; ++r) {
            // Find the next set bit in i_pivots, which will be our next pivot
            while (!i_pivots[pivotint]) {
                ++pivotint; // Skip over unset bits
            }

            // Set the pivot position to 1
            data[r][pivotint] = 1;
            ++pivotint; // Move to the next pivot for the next row
        }
    }

    /**
     * @brief Construct a Dense Matrix in reduced form from a set of pivots_ and a number. 
     * The binary representation of this number fills the spots in the reduced matrix which are not necessarilly zero.
     * 
     * @param pivots_ 
     * @param positions 
     */
    DenseMatrix(const bitset& pivots_, std::vector<std::pair<int,int>> &positions, int filler) : DenseMatrix(pivots_) {
        size_t mul = positions.size();
        for (int j=0; j < mul; j++){
            if (filler & (1 << j)) {
                data[positions[j].first][positions[j].second] = 1;
            }
        }
    }

    

    /**
     * @brief Add this bitset to the i-th column.
     * 
     * @param i 
     * @param bitset 
     */
    void addOutsideRow(int i, const boost::dynamic_bitset<>& bitset) {
        data[i] ^= bitset; // XOR the i-th row with the bitset
    }

    /**
     * @brief This is a second algorithm to reduce column-wise. It finds a lower triangular reduced form of the matrix involving swaps.
     * 
     * @param complete 
     */
    void colReduce(bool complete = false) {
        int lead = 0;
        pivot_vector = boost::dynamic_bitset<>(num_rows, 0);
        for (int r = 0; r < num_cols; ++r) {
            if (lead >= num_rows) {
                break; // No more columns to work on, exit the loop
            }

            int i = r;
            // Find the first non-zero entry in the row(potential pivot)
            while (i < num_cols && !data[i][lead]) {
                ++i;
            }

            if (i < num_cols) {
                // Found a non-zero entry, so this row does have a pivot
                // If the pivot is not in the current col, swap the cols
                if (i != r) {
                    swap_cols(i, r);
                }
                pivot_vector[lead] = true; // Mark this row as having a pivot after confirming pivot

                // Eliminate all non-zero entries below this pivot
                for (int j = r + 1; j < num_cols; ++j) {
                    if (data[j][lead]) {
                        data[j] ^= data[r];
                    }
                }

                if (complete) {
                    // Eliminate all non-zero entries above this pivot
                    for (int j = 0; j < r; ++j) {
                        if (data[j][lead]) {
                            data[j] ^= data[r];
                        }
                    }
                }

                ++lead; // Move to the next row
            } else {
                // No pivot in this row, so we move to the next row without incrementing the col int
                ++lead;
                --r; // Stay on the same col for the next iteration
            }
        }
        rowReduced = true; 
        if(complete){completeRowReduced = true;}
    }

    /**
     * @brief Writes the matrix to an ofstream by converting it to a string.
     * 
     * @param file 
     */
    void serialize(std::ofstream& file) const {
        // Write the size of the vector
        assert(data.size() == num_cols);
        file.write(reinterpret_cast<const char*>(&num_cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_rows), sizeof(int));
        // Serialize each dynamic_bitset in the vector

        for (const auto& bs : data) {
            const auto& bs_data = bs.to_ulong();
            file.write(reinterpret_cast<const char*>(&bs_data), sizeof(bs_data));
        }
    }

    /**
     * @brief Computes this*other. Maybe a bit slow.
     * 
     * @param other 
     */
    DenseMatrix multiply_right(DenseMatrix& other) override {
        assert(this->get_num_cols() == other.get_num_rows());
        DenseMatrix result(other.get_num_cols(), this->get_num_rows());

        for (int i = 0; i < other.get_num_cols(); ++i) {
            for (int j = 0; j < this->get_num_rows(); ++j) {
                for (int k = 0; k < this->get_num_cols(); ++k) {
                    if (this->data[k][j] && other.data[i][k]) {
                        result.data[i].flip(j);
                    }
                }
            }
        }

        return result;
    }

    /**
     * @brief Returns a transpose of the matrix.
     * 
     * @return DenseMatrix 
     */
    DenseMatrix transposed_copy() override {
        DenseMatrix result(this->get_num_rows(), this->get_num_cols());
        for (int i = 0; i < this->get_num_cols(); i++){
            for (int j = 0; j < this->get_num_rows(); j++){
                result.data[j][i] = this->data[i][j];
            }
        }
        return result;
    }

}; // end of DenseMatrix



// A pair of DenseMatrices - the two subspaces of a decomposition.
using VecDecomp = std::pair<DenseMatrix, DenseMatrix>;

// Associates to each integer the subspace whos Plücker coordinates have the integer as a binary representation. 
// Then to each subspace there can be a list of complements.
using DecompBranch = std::vector< std::vector<VecDecomp> >;

// associates to each bitset the subspaces whose reduced form has the entries of the bitset as pivots.
using DecompTree = std::unordered_map<boost::dynamic_bitset<>, DecompBranch , BitsetHash>;

// A transition is an invertible matrix with 1s on the diagonal and a subset of the columns given as a bitset.
// The matrix stores the necessary column-operations to transform a decomposition into another one. 
// The nonzero entries of the bitset give the column-indices associated to the first subspace, 
// the zero entries to the second subspace.
using transition = std::pair<DenseMatrix, bitset>;

/**
 * @brief Writes the DecompTree to a file.
 * 
 * @param tree 
 * @param filename 
 */
void saveDecompTree(const DecompTree& tree, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "File opened successfully for writing: " << filename << std::endl;
    }

    
    size_t treeSize = tree.size();
    file.write(reinterpret_cast<const char*>(&treeSize), sizeof(treeSize));

    for ( auto& [key, branch] : tree) {

        serializeDynamicBitset(key, file);
        // key gives pivots
        
        size_t branchSize = branch.size();
        file.write(reinterpret_cast<const char*>(&branchSize), sizeof(branchSize));

        for (size_t p_coord = 0; p_coord< branch.size(); p_coord++) {
            
            // p_coord is the integer representation of the Plücker coordinate

            size_t vectorSize = branch[p_coord].size();
            file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

            for (const VecDecomp& vecs : branch[p_coord]) {
                // Careful, we are serializing the second file first, because the reader will reverse this again!
                vecs.second.serialize(file);

                vecs.first.serialize(file);
            }
        }
    }

    file.close();
}

/**
 * @brief Loads a DecompTree from a file.
 * 
 * @param filename 
 * @return DecompTree 
 */
DecompTree loadDecompTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file '" + filename + "' for reading.");
    } else {
        std::cout << "File opened successfully for reading: " << filename << std::endl;
    }
  
    size_t treeSize;
    file.read(reinterpret_cast<char*>(&treeSize), sizeof(treeSize));

    DecompTree tree;
    for (size_t i = 0; i < treeSize; ++i) {
        boost::dynamic_bitset<> key = deserializeDynamicBitset(file);

        size_t branchSize;
        file.read(reinterpret_cast<char*>(&branchSize), sizeof(branchSize));

        DecompBranch branch;
        for (size_t j = 0; j < branchSize; ++j) {

            size_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

            std::vector<VecDecomp> VecDecomps;
            VecDecomps.reserve(vectorSize);
            for (size_t k = 0; k < vectorSize; ++k) {
                //This reverses the order of the pair. Works because the writer has already reversed it, too.
                VecDecomps.emplace_back(DenseMatrix(file), DenseMatrix(file));
            }

            branch.emplace_back(std::move(VecDecomps));
        }

        tree.emplace(std::move(key), std::move(branch));
    }

    file.close();
    return tree;
}

/**
 * @brief Prints the whole content of the DecompTree to the console.
 * 
 * @param tree 
 */
void print_tree(DecompTree& tree){
    int num_bits = 1;
    std::cout << "Printing tree with the following branches: " << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << pivots_ << ", ";
    }
    std::cout << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << "Pivots: ";
        print_bitset(pivots_);
        for(int i = 0; i < branch.size(); i++){
            num_bits = static_cast<int>(std::log2(branch.size()));
            bitset binary(num_bits, i);
            std::cout << "Pluecker Coordinate as integer is " << i << ". And as binary is ";
            print_bitset(binary);
            for(auto& [first, second] : branch[i]){
                std::cout << "First: " << std::endl;
                first.print(true);
                std::cout << "Second: " << std::endl;
                second.print(true);
            }
        }
    }
}

/**
 * @brief Saves a transition list to a file.
 * 
 * @param transitions 
 * @param filename 
 */
void save_transition_list(const std::vector<transition>& transitions, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "Writing transitions to: " << filename << std::endl;
    }

    // Write the size of the vector
    size_t vectorSize = transitions.size();
    file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

    for (const auto& [matrix, bitset] : transitions) {
        matrix.serialize(file);
        serializeDynamicBitset(bitset, file);
    }

    file.close();
}

/**
 * @brief Loads a transition list from a file.
 * 
 * @param filename 
 * @return std::vector<transition> 
 */
std::vector<transition> load_transition_list(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename + " for reading.");
    } else {

    }

    size_t vectorSize;
    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));
    // std::cout << "loading #matrices: " << vectorSize << std::endl;
    std::vector<transition> transitions;
    transitions.reserve(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        auto T = DenseMatrix(file);
        auto bs = deserializeDynamicBitset(file);
        transitions.emplace_back(std::move(T), std::move(bs));
    }

    file.close();
    return transitions;
}

/**
 * @brief Tests if A == B wrt. their data. Shouldnt be needed with .equals() already existing
 * 
 * @param A 
 * @param B 
 * @return true 
 * @return false 
 */
bool compare_matrices(DenseMatrix& A, DenseMatrix& B){
    if (A.get_num_cols() != B.get_num_cols() || A.get_num_rows() != B.get_num_rows()){
        return false;
    }
    for (int i = 0; i < A.get_num_cols(); i++){
        for (int j = 0; j < A.get_num_rows(); j++){
            if (A.data[i][j] != B.data[i][j]){
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Generates a random bit matrix with the given dimensions.
 * 
 * @param cols 
 * @param rows 
 * @return DenseMatrix 
 */
DenseMatrix RandomBitMatrix(int cols, int rows) {
    DenseMatrix randomMatrix(cols, rows);

    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Fill the matrix with random data
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            randomMatrix.data[i][j] = std::rand() % 2;
        }
    }

    return randomMatrix;
}

} // namespace graded_linalg

#endif // DENSE_MATRIX_HPP