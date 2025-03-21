/**
 * @file aida.hpp
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

#ifndef AIDA_HPP
#define AIDA_HPP

#include <iostream>
#include <grlina/graded_linalg.hpp>
#include <unordered_set>
#include <list>
#include <boost/timer/timer.hpp>
#include <memory_resource>
#include <cstdlib>
#include <filesystem>
#include <queue>
#include <functional>
#include <stack>
#include <cmath>
#include <utility>

namespace aida{

using index = int; // Change to large enough int type.

using namespace graded_linalg;

namespace fs = std::filesystem;
using namespace boost::timer;

#ifndef TIMERS
#define TIMERS 0
#endif

#define DETAILS 1 // For debugging 
#define OBSERVE 0 // For debugging
#define CHECK_INT 0 // obsolete?
#define SYSTEM_SIZE 0 // (should) tracks sizes of linear systems solved to compare the "actual" computation time of the different algorithms without overhead.


#if OBSERVE
    index observe_batch_index = 218;
    vec<index> observe_row_indices;
    index observe_block_index = 263;
#endif

#if TIMERS
    cpu_timer hom_space_timer;
    cpu_timer hom_space_test_timer;
    cpu_timer full_timer;
    cpu_timer constructing_linear_system_timer;
    cpu_timer solve_linear_system_timer;
    cpu_timer dispose_S_timer;
    cpu_timer update_matrix_timer;
    cpu_timer update_hom_timer;
    cpu_timer load_matrices_timer;
    cpu_timer compute_N_timer;
    cpu_timer delete_with_col_timer;
    cpu_timer misc_timer;
    cpu_timer update_block_timer;
    cpu_timer compute_rows_timer;
    cpu_timer pre_alpha_decomp_optimisation_timer;
    cpu_timer alpha_decomp_timer;
    cpu_timer full_aida_timer;
    cpu_timer full_exhaustive_timer;
    cpu_timer full_block_reduce_timer;
#endif



template <typename T>
using vec = vec<T>;
template <typename T>
using array = vec<vec<T>>;

using Sparse_Matrix = SparseMatrix<index>;
using GradedMatrix = R2GradedSparseMatrix<index>;
using indtree = std::set<index>;
using CT = Column_traits<vec<index>, index>;
// a list of blocks with a corresponding subset of the columns of the current batch
// Can be treated like an indecomposable/block itself.
using Merge_data = std::pair<vec<index>, bitset>; 

    auto comparator = [](const Merge_data& a, const Merge_data& b) {
        return a.first[0] < b.first[0];
    };

// A list of blocks with a corresponding subset of the columsn of the current batch.
// Can be treated like a virtual block, as in that it is an indecomposable even before it is formally merged.
using Full_merge_info = vec<vec<Merge_data>>;
using pair = std::pair<index, index>;
using op_info = std::pair<pair, pair>;
using hom_info = std::pair<index, pair>;
using edge_list = array<index>;

// Every column of the sparse matrix is a homomorphism, where the pair corresponding to each entry 
    // is the source and target index (internal to the blocks) of a row operation.
using Hom_space = std::pair< Sparse_Matrix, vec<pair>>; 

    // The key is the block index, the value is the sub-matrix given by restricting the batch to this block.
using Sub_batch = std::unordered_map<index, Sparse_Matrix>; 


    // The key is a pair of block indices, the value is a basis for the space of homomorphism from the first to the second.
using Hom_map = std::unordered_map<pair, Hom_space, pair_hash<index>> ;
    
    // The Matrix encodes an admissible linear transformation of the (columns of the) batch
    // The vector of tuples encodes the associated homomorphisms: (c, b, i) means 
    // the i-th homomorphism from c to b in the linear representation of Hom(c,b) in the hom_map.
using Batch_transform = std::pair< DenseMatrix, vec<std::tuple<index,index,index>> > ;


    //  For some pair of blocks, this should contain a list of associated column operations. 
    //  Each entry is a pair of virtual blocks, 
    //  together with a linear combination of basis elements of the space of allowable column transformations
using Row_transform = vec< std::pair<pair, vec<index>> >; 

    // The key is a pair of block indices, the value is a Row_transform.
using Row_transform_map = std::unordered_map<pair, Row_transform, pair_hash<index>>;



struct AIDA_config {

    bool sort; // Lex-sorts the matrices before processing.
    bool exhaustive; // Uses the exhaustive algorithm for the alpha-decomposition.
    bool brute_force; // Uses the exhaustive algorithm and does not use the hom-spaces.
    bool sort_output; // Sorts the indecomposables of the decomposition by r2degree.
    bool compare_both; // Compares the hom space and direct version of block_reduce
    bool exhaustive_test; // Compares exhaustive with aida at runtime.
    bool progress; // Shows progress bar while deecomposing.
    bool save_base_change; // Saves the base changes for each decomposition.
    bool turn_off_hom_optimisation; // Turns off the hom-space optimisation.
    bool show_info; // prints information about the decomposition to console.
    bool compare_hom; // Compares the optimised and non-optimised hom space calculation.
    bool supress_col_sweep; // Does not try to delete subbatches with only the column operations.
    bool alpha_hom; // Turns the computation of alpha-homs on.
    vec<vec<index>> decomp_failure;
    
    AIDA_config(bool supress_col_sweep = false, bool sort_output = false, bool sort = false, bool save_base_change = false, bool exhaustive = false, bool brute_force = false, bool progress = false, bool compare_both = false, bool turn_off_hom_optimisation = false, bool show_info = true, bool exhaustive_test = false, bool compare_hom = false, bool alpha_hom = true)
        : supress_col_sweep(supress_col_sweep), save_base_change(save_base_change), sort_output(sort_output), sort(sort), exhaustive(exhaustive), brute_force(brute_force), compare_both(compare_both), progress(progress), turn_off_hom_optimisation(turn_off_hom_optimisation), show_info(show_info), exhaustive_test(exhaustive_test), compare_hom(compare_hom) { 
            decomp_failure = vec<vec<index>>();
        }

};

double calculateAverage(const std::vector<index>& indices) {
    if (indices.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(indices.begin(), indices.end(), 0.0);
    return sum / indices.size();
}

double calculateMedian(std::vector<index> indices) {
    if (indices.empty()) {
        return 0.0;
    }
    std::sort(indices.begin(), indices.end());
    size_t size = indices.size();
    if (size % 2 == 0) {
        return (indices[size / 2 - 1] + indices[size / 2]) / 2.0;
    } else {
        return indices[size / 2];
    }
}


/**
 * @brief Base class for base_change
 * 
 */
struct Base_change_virtual {
    vec<pair> performed_row_ops;
    virtual void add_row_op(index source, index target) = 0;
    virtual ~Base_change_virtual() = default; // Ensure a virtual destructor
};

/**
 * @brief In case we do not want to store the row_operations / basechange we need for decompostion.
 * 
 */
struct Null_base_change : public Base_change_virtual {
    void add_row_op(index source, index target) override {}
};

/**
 * @brief In case we do want to store the row_operations / basechange.
 * 
 */
struct Base_change : public Base_change_virtual {
    void add_row_op(index source, index target) override {
        performed_row_ops.push_back({source, target});
    }
};

/**
 * @brief handles all statistical information gathered at runtime of the AIDA algorithm.
 * 
 */
struct AIDA_runtime_statistics {

    vec<index> num_subspace_iterations = {1, 2, 7, 43, 186, 1965, 14605, 297181};
    index counter_no_comp;
    index counter_only_col;
    index counter_only_row;
    vec<index> num_of_pierced_blocks;
    index counter_naive_deletion;
    index counter_naive_full_iteration;
    index counter_extra_iterations;
    
    index counter_col_deletion;
    index counter_row_deletion;
    index resolvable_cyclic_counter;
    index cyclic_counter;
    index acyclic_counter;
    index alpha_cycle_avoidance;
    index local_k_max;
    index dim_hom_max;
    vec<index> dim_hom_vec;

    #if TIMERS
        double hom_space ;
        double hom_space_test ;
        double constructing_linear_system;
        double solve_linear_system ;
        double dispose_S ;
        double update_matrix;
        double update_hom ;
        double load_matrices ;
        double compute_N ;
        double delete_with_col ;
        double misc ;
        double update_block ;
        double compute_rows;
        double pre_alpha_decomp_optimisation;
        double alpha_decomp ;
        double full ;
        double accumulated ;
        double full_aida;
        double full_exhaustive ;
        double full_block_reduce ;
    #endif

    AIDA_runtime_statistics(){
        counter_no_comp = 0;
        counter_only_col = 0;
        counter_only_row = 0;
        counter_naive_deletion = 0;
        counter_naive_full_iteration = 0;
        counter_extra_iterations = 0;
        counter_col_deletion = 0;
        counter_row_deletion = 0;
        local_k_max = 0;
        resolvable_cyclic_counter = 0;
        cyclic_counter = 0;
        acyclic_counter = 0;
        alpha_cycle_avoidance = 0;
        dim_hom_max = 0;

        #if TIMERS
            hom_space = 0;
            hom_space_test = 0;
            constructing_linear_system = 0;
            solve_linear_system = 0;
            dispose_S = 0;
            update_matrix = 0;
            update_hom = 0;
            load_matrices = 0;
            compute_N = 0;
            delete_with_col = 0;
            misc = 0;
            update_block = 0;
            compute_rows = 0;
            pre_alpha_decomp_optimisation = 0;
            alpha_decomp = 0;
            full = 0;
            accumulated = 0;
            full_aida = 0;
            full_exhaustive = 0;
            full_block_reduce = 0;
        #endif
    }


    void operator+= (AIDA_runtime_statistics& other){
        counter_no_comp += other.counter_no_comp;
        counter_only_col += other.counter_only_col;
        counter_only_row += other.counter_only_row;
        counter_naive_deletion += other.counter_naive_deletion;
        counter_naive_full_iteration += other.counter_naive_full_iteration;
        counter_extra_iterations += other.counter_extra_iterations;
        counter_col_deletion += other.counter_col_deletion;
        counter_row_deletion += other.counter_row_deletion;
        resolvable_cyclic_counter += other.resolvable_cyclic_counter;
        cyclic_counter += other.cyclic_counter;
        acyclic_counter += other.acyclic_counter;
        alpha_cycle_avoidance += other.alpha_cycle_avoidance;
        local_k_max = std::max(local_k_max, other.local_k_max);
        dim_hom_max = std::max(dim_hom_max, other.dim_hom_max);
        num_of_pierced_blocks.insert(num_of_pierced_blocks.end(), other.num_of_pierced_blocks.begin(), other.num_of_pierced_blocks.end());
        // Performed row ops stores the row-operations performed on each matrix and is therefore dependend on the matrix it is used on.
        // Therefore, we do not add it here.
        dim_hom_vec.insert(dim_hom_vec.end(), other.dim_hom_vec.begin(), other.dim_hom_vec.end());

        #if TIMERS
            hom_space += other.hom_space;
            hom_space_test += other.hom_space_test;
            constructing_linear_system += other.constructing_linear_system;
            solve_linear_system += other.solve_linear_system;
            dispose_S += other.dispose_S;
            update_matrix += other.update_matrix;
            update_hom += other.update_hom;
            load_matrices += other.load_matrices;
            compute_N += other.compute_N;
            delete_with_col += other.delete_with_col;
            misc += other.misc;
            update_block += other.update_block;
            compute_rows += other.compute_rows;
            pre_alpha_decomp_optimisation += other.pre_alpha_decomp_optimisation;
            alpha_decomp += other.alpha_decomp;
            full += other.full;
            accumulated += other.accumulated;
            full_aida += other.full_aida;
            full_exhaustive += other.full_exhaustive;
            full_block_reduce += other.full_block_reduce;       
        #endif
    }

    void print(){
        std::cout << "  No computation: " << counter_no_comp << std::endl;
        std::cout << "  Only column operations: " << counter_only_col << std::endl;
        std::cout << "  Only row operations: " << counter_only_row << std::endl;
        std::cout << "  Naive deletion: " << counter_naive_deletion << std::endl;
        std::cout << "  Naive full iteration: " << counter_naive_full_iteration << std::endl;
        std::cout << "  Column Block deletions: " << counter_col_deletion << std::endl;
        std::cout << "  Row Block deletions: " << counter_row_deletion << std::endl;
        std::cout << "  Hom-spaces calculated: " << dim_hom_vec.size() << std::endl;
        std::cout << "  Total dimension of calculated hom-spaces: " << std::accumulate(dim_hom_vec.begin(), dim_hom_vec.end(), 0) << std::endl;
        std::cout << "  Maximum dimension of calculated hom-spaces " << dim_hom_max << std::endl;
        std::cout << "  Average dimension of calculated hom-spaces: " << calculateAverage(dim_hom_vec) << std::endl;
        std::cout << "  Median dimension of calculated hom-spaces: " << calculateMedian(dim_hom_vec) << std::endl;

        if(!num_of_pierced_blocks.empty()){
            std::cout << "  Maximum Number of pierced blocks: " << *std::max_element(num_of_pierced_blocks.begin(), num_of_pierced_blocks.end()) << std::endl;
        }
        std::cout << "  Local k_max: " << local_k_max << std::endl;
        std::cout << "  Acyclic batches " << acyclic_counter << std::endl;
        std::cout << "  Resolvable cyclic batches " << resolvable_cyclic_counter << std::endl;
        std::cout << "  Cyclic batches " << cyclic_counter << std::endl;
        std::cout << "  Alpha cycle avoidance " << alpha_cycle_avoidance << std::endl;
        std::cout << "  Extra iterations: " << counter_extra_iterations << std::endl;
    }

    #if TIMERS

        void initialise_timers(){
            aida::full_timer.start();
            aida::full_timer.stop();
            aida::hom_space_timer.start();
            aida::hom_space_timer.stop();
            aida::hom_space_test_timer.start();
            aida::hom_space_test_timer.stop();
            aida::constructing_linear_system_timer.start();
            aida::constructing_linear_system_timer.stop();
            aida::solve_linear_system_timer.start();
            aida::solve_linear_system_timer.stop();
            aida::dispose_S_timer.start();
            aida::dispose_S_timer.stop();
            aida::update_matrix_timer.start();
            aida::update_matrix_timer.stop();
            aida::update_hom_timer.start();
            aida::update_hom_timer.stop();
            aida::compute_N_timer.start();
            aida::compute_N_timer.stop();
            aida::delete_with_col_timer.start();
            aida::delete_with_col_timer.stop();
            aida::misc_timer.start();
            aida::misc_timer.stop();
            aida::update_block_timer.start();
            aida::update_block_timer.stop();
            aida::compute_rows_timer.start();
            aida::compute_rows_timer.stop();
            aida::pre_alpha_decomp_optimisation_timer.start();
            aida::pre_alpha_decomp_optimisation_timer.stop();
            aida::alpha_decomp_timer.start();
            aida::alpha_decomp_timer.stop();
            aida::full_aida_timer.start();
            aida::full_aida_timer.stop();
            aida::full_exhaustive_timer.start();
            aida::full_exhaustive_timer.stop();
            aida::full_block_reduce_timer.start();
            aida::full_block_reduce_timer.stop();
        }

        void evaluate_timers(){
            hom_space = aida::hom_space_timer.elapsed().wall/1e9;
            hom_space_test = aida::hom_space_test_timer.elapsed().wall/1e9;
            constructing_linear_system = aida::constructing_linear_system_timer.elapsed().wall/1e9;
            solve_linear_system = aida::solve_linear_system_timer.elapsed().wall/1e9;
            dispose_S = aida::dispose_S_timer.elapsed().wall/1e9;
            update_matrix = aida::update_matrix_timer.elapsed().wall/1e9;
            update_hom = aida::update_hom_timer.elapsed().wall/1e9;
            
            compute_N = aida::compute_N_timer.elapsed().wall/1e9;
            delete_with_col = aida::delete_with_col_timer.elapsed().wall/1e9 ;
            misc = aida::misc_timer.elapsed().wall/1e9 ;
            update_block = aida::update_block_timer.elapsed().wall/1e9 ;
            compute_rows = aida::compute_rows_timer.elapsed().wall/1e9;
            pre_alpha_decomp_optimisation = aida::pre_alpha_decomp_optimisation_timer.elapsed().wall/1e9 ;
            alpha_decomp = aida::alpha_decomp_timer.elapsed().wall/1e9 ;
            full = aida::full_timer.elapsed().wall/1e9;
            accumulated = hom_space + constructing_linear_system + solve_linear_system 
                + dispose_S + update_matrix + update_hom + compute_N + delete_with_col + misc + update_block + compute_rows + pre_alpha_decomp_optimisation + alpha_decomp;
            full_aida = aida::full_aida_timer.elapsed().wall/1e9;
            full_exhaustive = aida::full_exhaustive_timer.elapsed().wall/1e9;
            full_block_reduce = aida::full_block_reduce_timer.elapsed().wall/1e9;
        }

        void print_timers(){
            std::cout << "Timers: " << std::endl;
            std::cout << "  Hom-space " << hom_space << "s" << std::endl;
            std::cout << "  Hom-space test " << hom_space_test << "s" << std::endl;
            std::cout << "  Constructing linear system " << constructing_linear_system << "s" << std::endl;
            std::cout << "  Solve linear system " << solve_linear_system << "s" << std::endl;
            std::cout << "  Dispose S " << dispose_S << "s" << std::endl;
            std::cout << "  Update matrix " << update_matrix << "s" << std::endl;
            std::cout << "  Update Hom-Space Map " <<  update_hom << "s" << std::endl;
            std::cout << "  Compute N " << compute_N << "s" << std::endl;
            std::cout << "  Column Reduction at start " << delete_with_col <<  "s" << std::endl;
            std::cout << "  Misc " << misc << "s" << std::endl;
            std::cout << "  Update Block " << update_block << "s" << std::endl;
            std::cout << "  Compute Rows " << compute_rows << "s" << std::endl;
            std::cout << "  Pre_alpha_decomp_optimisation " << pre_alpha_decomp_optimisation << "s" << std::endl;
            std::cout << "  Alpha_decomp " << alpha_decomp << "s" << std::endl;
            std::cout << "  Total time: " << full << "s vs accumulated " << accumulated << "s (without loading time)" << std::endl;
            std::cout << "  block reduce: " << full_block_reduce << "s" << std::endl;
            std::cout << "  alpha decomp with Aida time: " << full_aida << "s" << std::endl;
            std::cout << "  alpha decomp with exhaustive time: " << full_exhaustive << "s" << std::endl;
        }
    #endif 

};

struct vec_index_hash {
    std::size_t operator()(const std::vector<index>& v) const {
        std::size_t seed = 0;
        for (const auto& elem : v) {
            seed ^= std::hash<index>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * @brief Careful, this only considers the indices of the blocks not the columns!
 * 
 */
struct virtual_block_pair_hash {
    std::size_t operator()(const std::pair<Merge_data, Merge_data>& p) const {
        vec_index_hash vector_hasher;
        auto hash1 = vector_hasher(p.first.first);
        auto hash2 = vector_hasher(p.second.first);
        return hash1 ^ (hash2 << 1); 
    }
};

using Transform_Map = std::unordered_map< std::pair<Merge_data, Merge_data>, vec<Batch_transform>, virtual_block_pair_hash>;


enum class BlockType {
    FREE, // 1 Generator, 0 Relations
    CYC,  // 1 Generator, >0 Relation (cyclic, non free)
    INT,  // Interval, non cyclic.
    NON_INT // Non-Interval
};


/**
 * @brief Holds the indecomposable summands in the matrix
 * 
 */
struct Block : GradedMatrix {

    vec<index> rows;
    vec<index> columns;

    BlockType type; // 0 for free module, 1 for cyclic, 2 for interval, 3 for non-interval

    vec<index> local_admissible_cols; // Stores the indices of the columns which can be used for column operations.
    vec<index> local_admissible_rows; // Stores the actual row_indices which can be used for row operations.
    vec<index> local_basislift_indices; // Stores a set of indices defining a subset of local_admissible_rows which forms a basis at the local r2degree.
    // Careful, these are indices only locally for this block!

    // This will store the columns which can be added to the current batch, i.e. local_data = data | admissible_cols.
    std::shared_ptr<Sparse_Matrix> local_data; 
    std::shared_ptr<Sparse_Matrix> local_data_normalised; // Stores the normalised local_data.
    std::shared_ptr<Sparse_Matrix> local_cokernel; // Stores the cokernel of the local_data.

    /**
     * @brief Get the type of indecomposable (free, interval, ...)
     * 
     * @return std::string 
     */
    std::string get_type(){
        if(type == BlockType::FREE){
            return "free";
        } else if(type == BlockType::CYC){
            return "cyclic";
        } else if(type == BlockType::INT){
            return "interval";
        } else {
            return "non-interval";
        }
    }

    /**
     * @brief Returns the indices of the relations with only one entry. 
     * If the presented module is an interval, these limit the end of the interval.
     * 
     * @return vec<index> 
     */
    vec<r2degree> endpoints(){
        vec<r2degree> result;
        for(index i = 0; i < get_num_cols(); i++){
            if(data[i].size() == 1){
                result.push_back(col_degrees[i]);
            }
        }
        return result;
    }



    /**
     * @brief If admissible_cols has been set, this fetches the columns for further processing.
     * 
     */
    void compute_local_data(r2degree d){
        // TO-DO: Can we make it so that this is already reduced?
        local_data = std::make_shared<Sparse_Matrix>(this->map_at_degree(d, local_admissible_cols));
    }

    /**
     * @brief stores the rows of r2degree <= d for repeated usage.
     * 
     * @param d 
     */
    void compute_local_generators(r2degree d){
        local_admissible_rows = vec<index>();
        for( index i = 0; i < get_num_rows(); i++){
            if( is_admissible_row_operation(d, i)){
                local_admissible_rows.push_back(rows[i]);
            }
        }
        local_data_normalised = std::make_shared<Sparse_Matrix>(*local_data);
        local_data_normalised->set_num_rows(local_admissible_rows.size());
        local_data_normalised->compute_normalisation_with_pivots(local_admissible_rows);
    }


    /**
     * @brief Computes the local basislift for the block.
     *  Assumes, that local data has been computed and reduced.
     * @param d 
     */
    void compute_local_basislift(r2degree d){
        compute_local_generators(d);
        // The following returns a subset of the indeices in local_admissible_row_indices.
        local_basislift_indices = local_data_normalised->coKernel_basis(local_admissible_rows, true);
    }

    void compute_local_cokernel(){
        local_cokernel = std::make_shared<Sparse_Matrix>(local_data_normalised->coKernel_transposed_without_prelim(local_basislift_indices));
    }
 

    /**
     * @brief Tries to delete N with the columns in local_data.
     * 
     * @param N 
     * @return true 
     * @return false 
     */
    bool reduce_N( Sparse_Matrix& N){
        // Tries to delete N with the columns in local_data.
        if(local_data->get_num_cols() == 0){
            return false;
        }
        return local_data->solve_col_reduction(N);
    }

    /**
     * @brief Tries to delete N with the columns in local_data.
     * 
     * @param N 
     * @return true 
     * @return false 
     */
    bool reduce_N_fully( Sparse_Matrix& N, bool is_diagonal){
        // Tries to delete N with the columns in local_data.
        if(local_data->get_num_cols() == 0){
            return false;
        }
        return local_data->reduce_fully(N, is_diagonal);
    }


    /**
     * @brief Use column-operations to try and delete the block (A_t)_B
     * 
     * @param N 
     */
    bool delete_with_col_ops(r2degree d, Sparse_Matrix& N, bool no_deletion = false) {

        this->compute_local_data(d);

        // Here I would like a version that also first tries column-reduction to be performed in-place
        // TO-DO: implement
        if(no_deletion){
            return false;
        }
        return local_data->solve_col_reduction(N);
    }

    /**
     * @brief Resets the local data after processing a batch.
     * 
     */
    void delete_local_data(){
        local_admissible_cols.clear();
        local_data.reset();
        local_data_normalised.reset();
        local_admissible_rows.clear();
        local_cokernel.reset();
        local_basislift_indices.clear();
    }

    

    /**
     * @brief Compute the list of rows format for the indecomposable matrix belonging to the block.
     * Careful, the entries of the rows are given with internal numbering of the columns!
     * If you want to change this, give the column indices as an argument to compute_rows_forward.
     * 
     */
    void compute_rows(vec<index>& row_map){
        compute_rows_forward_map(row_map);
    }


    //To-Do: Update the constructors, so that memory for the data  is reserved??
    Block(const vec<index>& c, const vec<index>& r) : GradedMatrix(c.size(), r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
    }

    Block(const vec<index>& c, const vec<index>& r, BlockType t) : GradedMatrix(c.size(), r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
        this->type = t;
    }

    
    Block() : GradedMatrix() {}
    
    void clear() {
      this->rows.clear();
      this->columns.clear();
    }
    
    /**
     * @brief Outputs either only the position or also the indecomposable of the block.
     * 
     * @param with_content 
     */
    void print_block(bool with_content=true) {
        std::cout << "  Columns: ";
        std::cout << columns << " ";

        std::cout << "\n  Rows: ";
        std::cout << rows << " ";

        if(with_content){
            std::cout << "  Data: " << std::endl;
            this->print(false, true);
        }
    }
}; //Block	



/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @param sub_batch_indices
 * @return vec<index> 
 */
vec<index> hom_action(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N, bitset& sub_batch_indices){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].rbegin(); it != N._rows[i].rend(); it++){
            if( sub_batch_indices.test(*it)){
                result.push_back( linearise_position_reverse_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
            } 
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}

/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @return vec<index> 
 */
vec<index> hom_action_full_support(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].rbegin(); it != N._rows[i].rend(); it++){
            result.push_back( linearise_position_reverse_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}

/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @return vec<index> 
 */
vec<index> hom_action_extension(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].begin(); it != N._rows[i].end(); it++){
            result.push_back( linearise_position_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}

/**
 * @brief Apply a homomorphism from c to b to all of A
 * 
 * @param A 
 * @param B 
 * @param C 
 * @param hom 
 * @param row_ops 
 */
void hom_action_A(GradedMatrix& A, vec<index>& source_rows, vec<index>& target_rows, vec<index>& hom, vec<pair>& row_ops, std::shared_ptr<Base_change_virtual>& base_change){
    for( index q : hom){
        auto [i,j] = row_ops[q];
        i = source_rows[i];
        j = target_rows[j];
        #if OBSERVE
            if( std::find(observe_row_indices.begin(), observe_row_indices.end(), i) != observe_row_indices.end() ){
                std::cout << "Row operation: " << i << " -> " << j << std::endl;
            }
        #endif
        base_change->add_row_op(i, j);
        assert(A.is_admissible_row_operation(i, j));
        A.fast_rev_row_op(i, j);
    }
}

/**
 * @brief Apply a homomorphism from c to b to N
 *  TO-DO: At the moment we change N everywhere, is that a problem?
 * 
 */
void hom_action_N(Block& B_target, Sparse_Matrix& N_source, Sparse_Matrix& N_target, vec<index>& hom, vec<pair>& row_ops){
    for( index q : hom){
        auto [i, j] = row_ops[q];
        CT::add_to(N_source._rows[i], N_target._rows[j]);
    }
    N_target.compute_columns_from_rows(B_target.rows);
    bool reduction = B_target.reduce_N_fully(N_target, true);
}

/**
 * @brief Deletes all hom-spaces whose (co)domain is merged or extended.
 *          TO-DO: We could, at this point, also compute which of the homomorphisms factor through the new block, 
 *          but since it is not always clear that we will need this information again, this might increase the total running time,
 *          instead of decreasing it.
 * 
 * @param block_partition 
 * @param hom_spaces 
 * @param domain_keys 
 * @param codomain_keys
 */
void update_hom_spaces( vec<Merge_data>& block_partition, Hom_map& hom_spaces, 
    std::unordered_map<index, vec<index>>& domain_keys, std::unordered_map<index, vec<index>>& codomain_keys){
    for(auto& partition : block_partition){
        vec<index>& block_indices = partition.first;
        for(index c : block_indices){
            for(index b : domain_keys[c]){
                hom_spaces.erase({c, b});
            }
            domain_keys.erase(c);
            for(index b : codomain_keys[c]){
                hom_spaces.erase({b, c});
            }
            codomain_keys.erase(c);
        }
    }
}

/**
 * @brief Constructs the digraph of non-zero homomorphisms between the blocks in vertex_labels.
 *        Not that the way hom_spaces is computed, meanse that there might be extra edges in the digraph, 
 *        where the corresponding homomorphisms are actually zero or zero at the r2degree of the current batch.
 * 
 * @param hom_spaces 
 * @param vertex_labels 
 * @return edge_list 
 */
Graph construct_hom_digraph( Hom_map& hom_spaces, vec<index>& vertex_labels){
    auto edge_checker = [&hom_spaces](const index& c, const index& b) -> bool {
        return hom_spaces[{c,b}].first.data.size(); 
    };
    return construct_boost_graph(vertex_labels, edge_checker);
}

Graph construct_batch_transform_graph(Transform_Map& batch_transforms, vec<Merge_data>& virtual_blocks){
    auto edge_checker = [&batch_transforms, virtual_blocks](const index& c, const index& b) -> bool {
        return ! batch_transforms[std::make_pair(virtual_blocks[c], virtual_blocks[b])].empty();
    };
    return construct_boost_graph(virtual_blocks.size(), edge_checker);
}

typedef std::list<Block> Block_list;
typedef Block_list::iterator Block_iterator;

/**
 * @brief Constructs the Blocks of an empty Matrix whose rows are given by A.
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 */
void initialise_block_list(const GradedMatrix& A, Block_list& B_list, vec<Block_list::iterator>& block_map) {
    B_list.clear();
    for(int i=0; i < A.get_num_rows(); i++) {
        Block B({},{i}, BlockType::FREE);
        B.set_num_rows(1);
        auto it = B_list.insert(B_list.end(), B);
        block_map.push_back(it);
        (*it).row_degrees[0] = A.row_degrees[i];
        (*it)._rows = vec<vec<index>>(1);
        (*it).rows_computed = true;
    }
}

/**
 * @brief Displays the degrees of each block in the block list.
 * 
 * @param B_list 
 */
void print_block_list_status(Block_list& B_list) {
    std::cout << "Status: " << B_list.size() << " blocks:\n";
    index count=0;
    for(Block& b : B_list) {
      std::cout << "Block " << count++ << ":" << std::endl;
      b.print_degrees();
      std::cout << std::endl;
    }
}

/**
 * @brief Extends the block B by the columns of N given by the batch_indices and the batch_positions.
 * 
 * @param B 
 * @param N 
 * @param batch_positions 
 * @param batch_indices 
 */
void extend_block(Block& B, Sparse_Matrix& N, vec<index> batch_indices, bitset& batch_positions, r2degree& alpha) {
    if(batch_positions.empty()){
        batch_positions = bitset(N.get_num_cols(), true);
    }
    
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        B.columns.push_back(batch_indices[i]);
        B.data.push_back(N.data[i]);
        B.col_degrees.push_back(alpha);
        // Directly compute the rows for efficiency:
        auto it = N.data[i].begin();
        for(index j = 0; j < B.rows.size() && it != N.data[i].end() ; j++){
            if(*it == B.rows[j]){
                B._rows[j].push_back(i);
                it++;
            }
        }   
    }
    B.increase_num_cols(batch_positions.count());
    assert(B.get_num_cols() == B.columns.size());
    assert(B.get_num_cols() == B.data.size());

    if(B.type == BlockType::FREE){
        B.type = BlockType::CYC;
    } 
    

}


using block_position = std::pair<index, Block_list::iterator>; 
/**
 * @brief Returns *lhs < *rhs
 * 
 */
struct compare_block_position_row {
    bool operator()(const block_position& lhs, const block_position& rhs) const {
        return lhs.second->rows[lhs.first] > rhs.second->rows[rhs.first];
    }
};

struct compare_block_position_column {
    bool operator()(const block_position& lhs, const block_position& rhs) const {
        return lhs.second->columns[lhs.first] > rhs.second->columns[rhs.first];
    }
};

/**
 * @brief Merges the content of all blocks and a restriction of N into a new block.
 * While the rows stay sorted, the columns are not.
 *          
 * @param block_indices 
 * @param block_map 
 * @param new_block 
 * @param N_map 
 * @param batch_positions 
 * @param batch_indices 
 */
void merge_blocks_into_block(vec<index>& block_indices, vec<Block_list::iterator>& block_map, Block& new_block, 
                            Sub_batch& N_map, bitset& batch_positions, vec<index>& batch_indices, 
                            vec<index>& row_map, r2degree& alpha){
    
    std::priority_queue<block_position, vec<block_position>, compare_block_position_row> row_heap;
    std::priority_queue<block_position, vec<block_position>, compare_block_position_column> column_heap;

    // maps the initial index of the block to a vector containing the pairs: 
    // (batch index, iterator to the associated column of N)
    /*
    std::map<index, vec<std::pair<index, vec<index>::iterator>> > N_iterators;
    */


    bool input_is_interval = true;

    for(index i : block_indices){
        auto B = block_map[i];
        if(B->type == BlockType::NON_INT){
            input_is_interval = false;
        }
        row_heap.push({0, B});
        if(!(B->columns.empty())){
            column_heap.push({0, B});
        }
        /*
        new_block.columns.insert(new_block.columns.end(), B->columns.begin(), B->columns.end());
        new_block.data.insert(new_block.data.end(), B->data.begin(), B->data.end());
        new_block.col_degrees.insert(new_block.col_degrees.end(), B->col_degrees.begin(), B->col_degrees.end());
        */
        new_block.increase_num_rows( B->get_num_rows());
        new_block.increase_num_cols( B->get_num_cols());
    }
    
    new_block.type = BlockType::NON_INT;


    // Check if we stay being an interval
    Degree_traits<r2degree> traits;
    if(batch_positions.count() == 1 && block_indices.size() == 2 && input_is_interval){
        auto i = batch_positions.find_first();
        bool N_is_length_two = true;
        vec<r2degree> generators;
        for(index b : block_indices){
            if( N_map[b].data[i].size() != 1 ){
                N_is_length_two = false;
                break;
            } else {
                generators.push_back(block_map[b]->row_degrees[row_map[N_map[b].data[i].front()]]);
            }
        }
        if(N_is_length_two){
            if( traits.equals(alpha,  traits.join(generators[0], generators[1])) ){
                new_block.type = BlockType::INT;
            }
        }
    }



    index batch_threshold = new_block.get_num_cols();

    new_block.rows.reserve(new_block.get_num_rows());
    new_block._rows.reserve(new_block.get_num_rows());
    new_block.row_degrees.reserve(new_block.get_num_rows());
    new_block.columns.reserve(new_block.get_num_cols());
    new_block.data.reserve(new_block.get_num_cols());
    new_block.col_degrees.reserve(new_block.get_num_cols());

    // Add columns of the blocks according to the column_heap.

    if( !column_heap.empty() ){
    for(auto current_col = column_heap.top(); !column_heap.empty(); current_col = column_heap.top()){
        new_block.columns.push_back(current_col.second->columns[current_col.first]);
        new_block.col_degrees.push_back(current_col.second->col_degrees[current_col.first]);
        new_block.data.push_back(current_col.second->data[current_col.first]);
        column_heap.pop();
        if (current_col.first + 1 < current_col.second->columns.size()) {
            column_heap.push({current_col.first + 1, current_col.second});
        }
    }
    }

    // Add columns of N and initialise all iterators to the columns of N.
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        new_block.columns.push_back(batch_indices[i]);
        new_block.col_degrees.push_back(alpha);
        new_block.data.push_back(vec<index>());
        for(index j : block_indices){
            new_block.data.back().insert(new_block.data.back().end(), N_map[j].data[i].begin(), N_map[j].data[i].end());
            /*
            if( N_map[j].data[i].empty() ){
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            } else {
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            }
            */
        }
        std::sort(new_block.data.back().begin(), new_block.data.back().end());
    }
    new_block.increase_num_cols( batch_positions.count());
    // The minheap sorts the row-indices of the blocks in ascending order. 
    // Iteratively, we add this row index, the associated row from the block, and append entries, if the columns of N permit us.
    // TO-DO: Maybe this is much slower than simply sorting everything, I do not know.

    new_block._rows = vec<vec<index>>(new_block.get_num_rows());
    index row_counter = 0;
    while (!row_heap.empty()) {
        block_position current = row_heap.top();
        Block& B = *current.second;
        row_heap.pop();
        new_block.rows.push_back(B.rows[current.first]);
        new_block.row_degrees.push_back(B.row_degrees[current.first]);
        // Reevaluating the map from A.rows to new_block.rows
        row_map[B.rows[current.first]] = row_counter;
        /* Since we are now using internally indexed rows it doe not make sense to merge them like this without a map from col_indices to internal indexes.
        Instead we should recompute them, although this might be costly. Maybe optimise in a later version.
        new_block._rows.push_back(B._rows[current.first]);
        
        auto& itvec = N_iterators[B.rows.front()];
        index internal_col_index = batch_positions.find_first();
        for(index i = 0; i < itvec.size(); i++){
            if( itvec[i].second == N_map[B.rows.front()].data[internal_col_index].end() ){
                internal_col_index = batch_positions.find_next(internal_col_index);
                continue;    
            }
            internal_col_index = batch_positions.find_next(internal_col_index);
            if( *itvec[i].second == B.rows[current.first]){
                new_block.data[batch_threshold + i].push_back(*itvec[i].second);
                new_block._rows[row_counter].push_back(itvec[i].first);
                itvec[i].second++;
            }
        }
        */
        if (current.first + 1 < B.rows.size()) {
            row_heap.push({current.first + 1, current.second});
        }
        row_counter++;
    }   

    assert(new_block._rows.size() == new_block.row_degrees.size());
    assert(new_block.rows.size() == new_block.row_degrees.size());
    assert(new_block.columns.size() == new_block.data.size());
    assert(new_block.data.size() == new_block.get_num_cols());

}

/**
 * @brief Merges the blocks and updates the B_list.
 * 
 * @param A 
 * @param B_list 
 * @param N_map 
 * @param block_map 
 * @param block_partition 
 */
void merge_blocks(Block_list& B_list, Sub_batch& N_map, 
                    vec<Block_list::iterator>& block_map, vec<Merge_data>& block_partition, vec<index>& batch_indices, 
                    vec<index>& row_map, r2degree& alpha){ 
    for(auto& partition : block_partition){
        vec<index>& block_indices = partition.first;
        bitset& batch_positions = partition.second; 
        index first = *block_indices.begin();
        if(block_indices.size() == 1){
            extend_block(*block_map[first], N_map[first], batch_indices, batch_positions, alpha);
        } else {
            if(block_indices.size() < 1){
                std::cout << "  Warning: No Merge info at batch indices " <<  batch_indices << std::endl;
                assert(false);
            }
            Block new_block({}, {});
            // TO-DO: In many cases it might be better to find the largest block and merge the others into it.
            auto new_it = B_list.insert(B_list.end(), new_block);
            merge_blocks_into_block(block_indices, block_map, *new_it, N_map, batch_positions, batch_indices, row_map, alpha); 

            for(index i : block_indices){
                auto del_it = block_map[i];
                for(index j : block_map[i]->rows){
                    block_map[j] = new_it;
                }
                B_list.erase(del_it);

            }
        }          
    }
}

/**
 * @brief Adds the content of source to target, thereby merging virtual blocks.
 * 
 * @param target 
 * @param source 
 */
void merge_virtual_blocks(Merge_data& target, Merge_data& source ){
    assert( (target.second & source.second).none() );
    target.first.insert(target.first.end(), source.first.begin(), source.first.end());
    target.second |= source.second;
}

/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns, vec<index>& batch_indices, vec<long>& result, bitset& sub_batch_indices) {
    
    assert(batch_indices.size() == sub_batch_indices.size());
    for(auto& ref : Ns){
        Sparse_Matrix& N = ref.get();
        for(index i = sub_batch_indices.size()-1; i >= 0; i--){
            if(sub_batch_indices.test(i)){
                for(index j : N.data[i]){
                    result.push_back(A.linearise_position_reverse(batch_indices[i], j));
                }
            }
        }
    }
    std::sort(result.begin(), result.end());
}

/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior_full_support( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns, vec<index>& batch_indices, vec<long>& result) {
    
    for(auto& ref : Ns){
        Sparse_Matrix& N = ref.get();
        for(index i = batch_indices.size()-1; i >= 0; i--){
                for(index j : N.data[i]){
                    result.push_back(A.linearise_position_reverse(batch_indices[i], j));
                }
        }
    }
    // Why was this here? -> std::sort(result.begin(), result.end());
}



void test_rows_of_A(GradedMatrix& A, index batch){
    for(index i = 0; i < A.get_num_rows(); i++){
        if(!A._rows[i].empty()){
            for(index j : A._rows[i]){
                if(j < batch){
                    std::cout << "Warning: The row " << i << " has an entry smaller than the current batch: " << j << std::endl;
                }
                if(j > A.get_num_cols()){
                    std::cout << "Warning: The row " << i << " has an entry larger than the number of columns: " << j << std::endl;
                }
            }
        }
    }
}

/**
 * @brief Constructs the linear system to delete the block N=(A_t)_B with row and column operations. Concretly:
 *          Solve for matrices: P, P_i, Q_i, (i != j in relevant blocks)
 *          B * P_i = Q_i * B_i 
 *          B * P + Q_i * N_i = N
 * 
 * @param A 
 * @param Ns
 * @param sub_batch_indices
 * @param restricted_batch
 * @param relevant_blocks
 * @param block_map
 * @param S
 * @param ops
 * @param b_vec
 * @param N_map
 */
void construct_linear_system(GradedMatrix& A, vec<index>& batch_indices, bitset& sub_batch_indices, bool restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            SparseMatrix<long>& S, vec<op_info>& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            const bitset& extra_columns = bitset(0)){
    //TO-DO: Parallelise this whole subroutine.

    Block& B_first = *block_map[*b_vec.begin()];  
    Block& B_probe = *block_map[*relevant_blocks.begin()];
    size_t buffer = 0;
    size_t max_buffer_size = 200000;
    if(B_first.local_data != nullptr){
        buffer = B_first.local_data->data.size();
    }
    buffer += b_vec.size()*relevant_blocks.size()*(B_first.rows.size()*B_probe.rows.size() 
    + B_probe.columns.size()*B_first.columns.size());
    buffer = std::min(buffer, max_buffer_size);
    S.data.reserve(buffer);
    
    // First find all blocks which can actually contribute by having a non-zero admissible row operation to any row of B:
    // While doing that, construct the associated columns of S belonging to these row operations.
    index S_index = 0; indtree admissible_relevant_blocks;

    bool no_new_inserts = false;
    for(index b: b_vec){
        Block& B = *block_map[b];
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            for(index i = 0; i < C.rows.size(); i++){
                auto source_index = C.rows[i];
                for(index j = 0; j < B.rows.size(); j++){
                    auto target_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_index, target_index)){
                        S.data.push_back(vec<long>());
                        ops.emplace_back( std::make_pair(std::make_pair(i , j), std::make_pair(c, b)) );
                        // Fill the column of S belonging to the operation first with the row in N_C, 
                        // then with the row in C, so that no sorting is needed.
                        for(auto row_it = N_C._rows[i].rbegin(); row_it != N_C._rows[i].rend(); row_it++){
                            if(!restricted_batch || sub_batch_indices.test(*row_it)){
                                S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[*row_it], target_index));
                            }
                        }    

                        if(!C._rows[i].empty()){
                        for(auto row_it2 = C._rows[i].rbegin(); row_it2 != C._rows[i].rend(); row_it2++){
                            // only insert if the row operation has an effect on B.rows*C.columns.
                            if(!no_new_inserts){
                                auto result = admissible_relevant_blocks.insert(c);
                                no_new_inserts = result.second;
                            }
                            auto effect_position = A.linearise_position_reverse(C.columns[*row_it2], target_index);
                            S.data[S_index].emplace_back(effect_position);
                        }
                        }
                        S_index++;
                    } 
                }
            }
            no_new_inserts = false;
        }
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        // Next add all col ops from all blocks in b_vec to the columns of the blocks which could contribute.

        for(index c : admissible_relevant_blocks){
            auto it = block_map[c];
            Block& C = *it;
            for(index i = 0; i < C.columns.size(); i++){
                for(index j = 0; j < B.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[j], C.columns[i])){
                        S.data.push_back(vec<long>());
                        for(index row_index : B.data[j]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[i], row_index));
                        }
                        S_index++;
                    }
                }
            }
        }
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This doesnt work yet, somehow local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = sub_batch_indices.find_first(); i != bitset::npos ; i = sub_batch_indices.find_next(i)){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<long>());
                    for(index j : column){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[i], j)); 
                    }
                    S_index++;
                }
            }
        }
    }

    // If we're in the last step of naive decomposition, 
    // need to the additional column operation from extra columns to sub_batch_indices
    if(extra_columns.any()){
        for(index i = extra_columns.find_first(); i != bitset::npos; i = extra_columns.find_next(i)){
            for(index j = sub_batch_indices.find_first(); j != bitset::npos; j = sub_batch_indices.find_next(j)){
                S.data.push_back(vec<long>());
                for(auto b : b_vec){
                    for(index row_index : N_map[b].data[i]){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[j], row_index));
                    }
                }
                std::sort(S.data[S_index].begin(), S.data[S_index].end());
                S_index++;
            }          
        }
    }

} //construct_linear_system

/**
 * @brief Constructs the linear system to delete the block N=(A_t)_B with row and column operations. Concretly:
 *          Solve for matrices: P, P_i, Q_i, (i != j in relevant blocks)
 *          B * P_i = Q_i * B_i 
 *          B * P + Q_i * N_i = N
 * 
 * @param A 
 * @param Ns
 * @param sub_batch_indices
 * @param restricted_batch
 * @param relevant_blocks
 * @param block_map
 * @param S
 * @param ops
 * @param b_vec
 * @param N_map
 */
void construct_linear_system_full_support(GradedMatrix& A, vec<index>& batch_indices, bool restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            SparseMatrix<long>& S, vec<op_info>& ops, 
                            vec<index>& b_vec, Sub_batch& N_map){
    //TO-DO: Parallelise this whole subroutine.

    Block& B_first = *block_map[*b_vec.begin()];  
    Block& B_probe = *block_map[*relevant_blocks.begin()];
    size_t buffer = 0;
    size_t max_buffer_size = 200000;
    if(B_first.local_data != nullptr){
        buffer = B_first.local_data->data.size();
    }
    buffer += b_vec.size()*relevant_blocks.size()*(B_first.rows.size()*B_probe.rows.size() 
    + B_probe.columns.size()*B_first.columns.size());
    buffer = std::min(buffer, max_buffer_size);
    S.data.reserve(buffer);
    
    // First find all blocks which can actually contribute by having a non-zero admissible row operation to any row of B:
    // While doing that, construct the associated columns of S belonging to these row operations.
    index S_index = 0; indtree admissible_relevant_blocks;

    bool no_new_inserts = false;
    for(index b: b_vec){
        Block& B = *block_map[b];
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            for(index i = 0; i < C.rows.size(); i++){
                auto source_index = C.rows[i];
                for(index j = 0; j < B.rows.size(); j++){
                    auto target_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_index, target_index)){
                        S.data.push_back(vec<long>());
                        ops.emplace_back( std::make_pair(std::make_pair(i , j), std::make_pair(c, b)) );
                        // Fill the column of S belonging to the operation first with the row in N_C, 
                        // then with the row in C, so that no sorting is needed.
                        for(auto row_it = N_C._rows[i].rbegin(); row_it != N_C._rows[i].rend(); row_it++){
                            
                            S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[*row_it], target_index));

                        }    

                        if(!C._rows[i].empty()){
                        for(auto row_it2 = C._rows[i].rbegin(); row_it2 != C._rows[i].rend(); row_it2++){
                            // only insert if the row operation has an effect on B.rows*C.columns.
                            if(!no_new_inserts){
                                auto result = admissible_relevant_blocks.insert(c);
                                no_new_inserts = result.second;
                            }
                            auto effect_position = A.linearise_position_reverse(C.columns[*row_it2], target_index);
                            S.data[S_index].emplace_back(effect_position);
                        }
                        }
                        S_index++;
                    } 
                }
            }
            no_new_inserts = false;
        }
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        // Next add all col ops from all blocks in b_vec to the columns of the blocks which could contribute.

        for(index c : admissible_relevant_blocks){
            auto it = block_map[c];
            Block& C = *it;
            for(index i = 0; i < C.columns.size(); i++){
                for(index j = 0; j < B.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[j], C.columns[i])){
                        S.data.push_back(vec<long>());
                        for(index row_index : B.data[j]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[i], row_index));
                        }
                        S_index++;
                    }
                }
            }
        }
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This doesnt work yet, somehow local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i_b : batch_indices){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<long>());
                    for(index j : column){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(i_b, j)); 
                    }
                    S_index++;
                }
            }
        }
    }


} //construct_linear_system_full_support

/**
 * @brief  
 * 
 * @param batch_indices 
 * @param sub_batch_indices 
 * @param restricted_batch 
 * @param relevant_blocks 
 * @param block_map 
 * @param S 
 * @param ops 
 * @param b_vec 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @param y 
 * @param extra_columns 
 */
void construct_linear_system_hom(vec<index>& batch_indices, bitset& sub_batch_indices, bool& restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            Sparse_Matrix& S, vec< hom_info >& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            Hom_map& hom_spaces,
                            vec<index>& row_map,
                            vec<index>& y,
                            const bitset& extra_columns = bitset(0)){
    
    //TO-DO: Parallelise this
    index row_glueing = 0;
    index total_num_rows = 0;
    index S_index = 0;
    for(index b : b_vec){
        total_num_rows += block_map[b]->get_num_rows();
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        Sparse_Matrix& N_B = N_map[b];
        // Populate y
        for(index row_index = 0; row_index < B.rows.size(); row_index++){
            for(auto int_col_it = N_B._rows[row_index].rbegin(); int_col_it != N_B._rows[row_index].rend(); int_col_it++){
                if(sub_batch_indices.test(*int_col_it)){
                    y.emplace_back(linearise_position_reverse_ext<index>(*int_col_it, row_index + row_glueing, N_B.get_num_cols(), total_num_rows));
                }
            }
        }
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            Hom_space& hom_cb = hom_spaces[{c,b}];
            for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
                ops.emplace_back( i_B , std::make_pair(c, b) );
                S.data.emplace_back(hom_action(row_glueing, total_num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C, sub_batch_indices) );
            }
        }
        row_glueing += B.get_num_rows();
    }
    std::sort(y.begin(), y.end());
    S_index = S.data.size();
    row_glueing = 0;
    for(index b: b_vec){
        Block& B = *block_map[b];
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This isnt fully optimised yet, local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<index>());
                    for(index j : column){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( i, row_map[j]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    S_index++;
                }
            }
        }
        row_glueing += B._rows.size();
    }
    // If we're in the last step of naive decomposition, use also column-operations internal to the batch:
    
    if(extra_columns.any()){
        for(index i = extra_columns.find_first(); i != bitset::npos; i = extra_columns.find_next(i)){
            for(index j = sub_batch_indices.find_first(); j != bitset::npos; j = sub_batch_indices.find_next(j)){
                S.data.push_back(vec<index>());
                row_glueing = 0;
                for(auto b : b_vec){
                    for(index row_index : N_map[b].data[i]){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( j, row_map[row_index]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    row_glueing += N_map[b]._rows.size();
                }
                //TO-DO: with smarter book-keeping this could be avoided:
                std::sort(S.data[S_index].begin(), S.data[S_index].end());
                S_index++;
            }
        }
    }

} //construct_linear_system_hom

/**
 * @brief  
 * 
 * @param batch_indices 
 * @param sub_batch_indices 
 * @param restricted_batch 
 * @param relevant_blocks 
 * @param block_map 
 * @param S 
 * @param ops 
 * @param b_vec 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @param y 
 * @param extra_columns 
 */
void construct_linear_system_hom_full_support(vec<index>& batch_indices, bool& restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            Sparse_Matrix& S, vec< hom_info >& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            Hom_map& hom_spaces,
                            vec<index>& row_map,
                            vec<index>& y){
    
    //TO-DO: Parallelise this
    index row_glueing = 0;
    index total_num_rows = 0;
    index S_index = 0;
    for(index b : b_vec){
        total_num_rows += block_map[b]->get_num_rows();
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        Sparse_Matrix& N_B = N_map[b];
        // Populate y
        for(index row_index = 0; row_index < B.rows.size(); row_index++){
            for(auto int_col_it = N_B._rows[row_index].rbegin(); int_col_it != N_B._rows[row_index].rend(); int_col_it++){

                y.emplace_back(linearise_position_reverse_ext<index>(*int_col_it, row_index + row_glueing, N_B.get_num_cols(), total_num_rows));

            }
        }
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            Hom_space& hom_cb = hom_spaces[{c,b}];
            for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
                ops.emplace_back( i_B , std::make_pair(c, b) );
                S.data.emplace_back(hom_action_full_support(row_glueing, total_num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C) );
            }
        }
        row_glueing += B.get_num_rows();
    }
    std::sort(y.begin(), y.end());
    S_index = S.data.size();
    row_glueing = 0;
    for(index b: b_vec){
        Block& B = *block_map[b];
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This isnt fully optimised yet, local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = 0; i < batch_indices.size(); i++){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<index>());
                    for(index j : column){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( i, row_map[j]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    S_index++;
                }
            }
        }
        row_glueing += B._rows.size();
    }

} //construct_linear_system_hom_full_support

/**
 * @brief Stores all entries of N[b] at the column_indices given in a single vector of size N[b].rows*N[b].columns 
 * 
 * @param b 
 * @param batch_column_indices 
 * @param N_map 
 * @param row_map 
 * @return vec<index> 
 */
void linearise_sub_batch_entries(vec<index>& result, Sparse_Matrix& N, bitset& batch_column_indices, vec<index>& row_map){
    for(index i = batch_column_indices.find_first(); i != bitset::npos; i = batch_column_indices.find_next(i) ){
        for(index r : N.data[i]){
            result.emplace_back( linearise_position_ext<index>(i, row_map[r], batch_column_indices.size(), N.get_num_rows()) );
        }
    }
}

void construct_linear_system_extension(Sparse_Matrix& S, vec<hom_info>& hom_storage, index& E_threshold,
    index& N_threshold, index& M_threshold, index& b, bitset& b_non_zero_columns, 
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){

    vec<index>& pro_block_blocks = pro_block.first;
    bitset& target = pro_block.second;
    index num_rows = block_map[b]->get_num_rows();
    assert( block_map[b]->get_num_rows() == N_map[b].get_num_rows());
    assert( N_map[b].get_num_rows() == N_map[b]._rows.size());


    // First add row-operations from the virtual processed block.

    for( index c : pro_block_blocks){
        Sparse_Matrix& N_C = N_map[c];
        Hom_space& hom_cb = hom_spaces[{c,b}];
        for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
            hom_storage.emplace_back( i_B , std::make_pair(c, b) );
            index row_glue = 0;
            S.data.emplace_back(hom_action_extension(row_glue, num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C) );
            assert(is_sorted(S.data.back()));
        }
    }

    E_threshold = hom_storage.size();
    // Then the internal column-operations

    for(index i : incoming_vertices){
        // Do not need to consider this, if the respective cocycle has been deleted
        if(deleted_cocycles_b.test(i)){
            Merge_data& E = pro_blocks[i];
            vec<Batch_transform>& internal_col_ops = batch_transforms[{E, pro_block}];
            assert( !internal_col_ops.empty());
            for( index j = 0; j < internal_col_ops.size(); j++){
                Batch_transform col_ops = internal_col_ops[j];
                DenseMatrix& T = col_ops.first;
                // The following is bloaty, could be fixed by not having T as a Dense_Matrix or directly reading of the result.
                Sparse_Matrix N_b = N_map[b];
                N_b.multiply_dense(T);
                S.data.push_back(vec<index>());
                linearise_sub_batch_entries(S.data.back(), N_b, target, row_map);
                assert(is_sorted(S.data.back()));
                hom_storage.push_back({j, {i, b}});
            }
        } else {
            // Nothing to do. might count how often this happens.
        }
    }

    N_threshold = hom_storage.size();

    // Column-operations from the support of B in the batch:
    
    for(index i = b_non_zero_columns.find_first(); i != bitset::npos; i = b_non_zero_columns.find_next(i)){
        for(index j = target.find_first(); j != bitset::npos; j = target.find_next(j)){
            S.data.push_back(vec<index>());
            for(index row_index : N_map[b].data[i]){
                S.data.back().emplace_back( linearise_position_ext<index>( j, row_map[row_index], target.size(), num_rows));
            }
            assert(is_sorted(S.data.back()));
            hom_storage.push_back({i, {j, b}});
        }
    }

    M_threshold = hom_storage.size();
    // Add the basic column-operations from B to N which have already been computed
    
    if(block_map[b]->local_data != nullptr){
    for(index i = target.find_first(); i != bitset::npos; i = target.find_next(i)){
        for(vec<index>& column : (block_map[b]->local_data)->data){
            S.data.push_back(vec<index>());
            for(index j : column){
                S.data.back().emplace_back( linearise_position_ext<index>( i, row_map[j], target.size(), num_rows));
                assert(is_sorted(S.data.back()));
            }
        }
    }
    }

    

} //construct_linear_system_extension

/**
 * @brief Computes a basis for the hom-space Hom(C, B).
 * 
 * @param A 
 * @param C
 * @param B 
 * @return vec<Sparse_Matrix> 
 */
Hom_space compute_hom_space(GradedMatrix& A, Block& C, Block& B, r2degree& alpha, const bool& alpha_hom = false){

    vec< pair > row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.

    Sparse_Matrix K(0,0);

    switch (C.type){
        case BlockType::FREE : {
            index counter = 0;
            for( index i = 0; i < C.get_num_rows(); i++){
                // indices in rows_alpha are internal to C. For external change to .., true) 
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(C.row_degrees[i]);
                vec<index> basislift = B_alpha.coKernel_basis(rows_alpha, B.rows);
                for( index j : basislift){
                    row_ops.push_back( {i, j} );
                    K.data.push_back( {counter} );
                    counter++;
                }
            }
            K.compute_num_cols();
            return {K, row_ops};
            break;
        }

        case BlockType::CYC : {
            //TO-DO: Implement
            return hom_space(C, B, C.rows, B.rows);
            break;
        }

        case BlockType::INT : {
            
            return hom_space(C, B, C.rows, B.rows);
            break;
            #if CHECK_INT
            if( B.type == BlockType::INT ){
                degree_list endpoints_C = C.endpoints();
                degree_list endpoints_B = B.endpoints();
                // Assuming rows are already lexicographically sorted, while columns are not because of the merging.
                std::sort(endpoints_C.begin(), endpoints_C.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                std::sort(endpoints_B.begin(), endpoints_B.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                
                vec<vec<r2degree>> intersection;
                index segment_counter;

                //TO-DO: finish

                auto B_it = B.row_degrees.begin();
                for(auto C_it = C.row_degrees.begin(); C_it != C.row_degrees.end();  ){
                    if( lex_order(*C_it, *B_it) ){
                        C_it++;
                    } else if ( (*C_it).second < (*B_it).second ) {
                        B_it++;
                    } else {
                        intersection[segment_counter].push_back(*C_it);
                    }
                }

            } else {
                // Find a fast algorithm to compute Hom_alpha(M, -) for M an interval? Do I need the codomain to be an intervall too if i want it fast?
            }
            #endif

        }

        case BlockType::NON_INT : {
            return hom_space(C, B, C.rows, B.rows);
            /*
            Non-optimised version
            Sparse_Matrix S(0,0);
            S.data.reserve( C.rows.size() + B.rows.size() + 1);
            index S_index = 0;
            // First add all row-operations from C to B
            for(index i = 0; i < C.rows.size(); i++){
                for(index j = 0; j < B.rows.size(); j++){
                    auto source_row_index = C.rows[i];
                    auto target_row_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                        row_ops.push_back({source_row_index, target_row_index});
                        S.data.push_back(vec<index>());
                        for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                            auto& internal_column_index = *rit;
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[internal_column_index], target_row_index));
                        }
                        S_index++;
                    }
                }
            }

            index row_op_threshold = S_index;
            assert( row_ops.size() == S_index );

            if(row_op_threshold == 0){
                // If there are no row-operations, then the hom-space is zero.
                return {Sparse_Matrix(0, 0), row_ops};
            }

            

            // Then all column-operations from B to C
            for(index i = 0; i < B.columns.size(); i++){
                for(index j = 0; j < C.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[i], C.columns[j])){
                        S.data.push_back(vec<index>());
                        for(index row_index : B.data[i]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[j], row_index));
                        }
                        S_index++;
                    }
                }
            }
 
            S.compute_get_num_cols()();
            
            // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
            K = S.get_kernel();
            // To see how much the following reduces K: index K_size = K.data.size();
            // Now we need to delete the entries of K which correspond to the row-operations.
            K.cull_columns(row_op_threshold, false);
            
            // Last we need to quotient out those Q where for every i the column Q_i - with its r2degree alpha_i -
            // lies in the image of N, that is, it lies in the image of N|alpha_i.
            // That is equivalent to locally reducing every column of Q.
            
            // Given a row-op, this gives the position in a vector corresponding to it.
            std::unordered_map< std::pair<index, index>, index, pair_hash<index> > pair_to_index = pair_to_index_map(row_ops);

            for(index i = 0; i < C.rows.size(); i++){
                r2degree alpha = C.row_degrees[i];
                vec<index> local_admissible_columns;
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(alpha);
                if(rows_alpha.empty() || B_alpha.get_num_cols() == 0){
                    continue;
                }
                std::unordered_map<index, index> reIndexMap;

                for(index j : rows_alpha){
                    reIndexMap[B.rows[j]] = pair_to_index[{C.rows[i], B.rows[j]}];
                }
                B_alpha.transform_data(reIndexMap);
                B_alpha.reduce_fully(K);
            }
            
            //  delete possible linear dependencies.
            K.column_reduction_triangular(true);
            */
            break;
        }
    }
    return {Sparse_Matrix(0,0), {}};
}


/**
 * @brief Computes a basis for the hom-space Hom(C, B).
 * 
 * @param A 
 * @param C
 * @param B 
 * @return vec<Sparse_Matrix> 
 */
Hom_space compute_hom_space_no_optimisation(GradedMatrix& A, Block& C, Block& B, r2degree& alpha, Sparse_Matrix& S_compare, const bool& alpha_hom = false){
    //TO-DO: This function is not working correctly yet, but it is only for testing anyways.

    vec< pair > row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.

    Sparse_Matrix K(0,0);

    switch (C.type){
        case BlockType::FREE : {
            index counter = 0;
            for( index i = 0; i < C.get_num_rows(); i++){
                // indices in rows_alpha are internal to C. For external change to .., true) 
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(C.row_degrees[i]);
                vec<index> basislift = B_alpha.coKernel_basis(rows_alpha, B.rows);
                for( index j : basislift){
                    row_ops.push_back( {i, j} );
                    K.data.push_back( {counter} );
                    counter++;
                }
            }
            K.compute_num_cols();
            return {K, row_ops};
            break;
        }

        case BlockType::INT : {
            #if CHECK_INT
            if( B.type == BlockType::INT ){
                degree_list endpoints_C = C.endpoints();
                degree_list endpoints_B = B.endpoints();
                // Assuming rows are already lexicographically sorted, while columns are not because of the merging.
                std::sort(endpoints_C.begin(), endpoints_C.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                std::sort(endpoints_B.begin(), endpoints_B.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                
                vec<vec<r2degree>> intersection;
                index segment_counter;

                //TO-DO: finish

                auto B_it = B.row_degrees.begin();
                for(auto C_it = C.row_degrees.begin(); C_it != C.row_degrees.end();  ){
                    if( lex_order(*C_it, *B_it) ){
                        C_it++;
                    } else if ( (*C_it).second < (*B_it).second ) {
                        B_it++;
                    } else {
                        intersection[segment_counter].push_back(*C_it);
                    }
                }

            } else {
                // Find a fast algorithm to compute Hom_alpha(M, -) for M an interval? Do I need the codomain to be an intervall too if i want it fast?
            }
            #endif

        }

        case BlockType::CYC : {
            //TO-DO: Implement maybe
        }

        case BlockType::NON_INT : {
            
            SparseMatrix<long> S(0,0);
            S.data.reserve( C.rows.size() + B.rows.size() + 1);
            index S_index = 0;
            // First add all row-operations from C to B
            for(index i = 0; i < C.rows.size(); i++){
                for(index j = 0; j < B.rows.size(); j++){
                    auto source_row_index = C.rows[i];
                    auto target_row_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                        row_ops.push_back({source_row_index, target_row_index});
                        S.data.push_back(vec<long>());
                        for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                            auto& internal_column_index = *rit;
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[internal_column_index], target_row_index));
                        }
                        S_index++;
                    }
                }
            }

            // If there are no row-operations, then the hom-space is zero.
            if(S_index == 0){
                return {Sparse_Matrix(0, 0), row_ops};
            }

            index row_op_threshold = S_index;
            assert( row_ops.size() == S_index );

            if(row_op_threshold == 0){
                // If there are no row-operations, then the hom-space is zero.
                return {Sparse_Matrix(0, 0), row_ops};
            }

            

            // Then all column-operations from B to C
            for(index i = 0; i < B.columns.size(); i++){
                for(index j = 0; j < C.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[i], C.columns[j])){
                        S.data.push_back(vec<long>());
                        for(index row_index : B.data[i]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[j], row_index));
                        }
                        S_index++;
                    }
                }
            }
 
            S.compute_num_cols();
            #if SYSTEM_SIZE
                std::cout << "System size: " << S.get_num_cols() << std::endl;
            #endif

            if(S_compare.get_num_cols() != 0){
                S.print();
                S_compare.print();
            }
            // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
            
            K = S.get_kernel_int<index>();
            // To see how much the following reduces K: index K_size = K.data.size();
            // Now we need to delete the entries of K which correspond to the row-operations.
            K.cull_columns(row_op_threshold, false);
            
            // Last we need to quotient out those Q where for every i the column Q_i - with its r2degree alpha_i -
            // lies in the image of N, that is, it lies in the image of N|alpha_i.
            // That is equivalent to locally reducing every column of Q.
            
            // Given a row-op, this gives the position in a vector corresponding to it.
            std::unordered_map< std::pair<index, index>, index, pair_hash<index> > pair_to_index = pair_to_index_map(row_ops);

            for(index i = 0; i < C.rows.size(); i++){
                r2degree alpha = C.row_degrees[i];
                vec<index> local_admissible_columns;
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(alpha);
                if(rows_alpha.empty() || B_alpha.get_num_cols() == 0){
                    continue;
                }
                std::unordered_map<index, index> reIndexMap;

                for(index j : rows_alpha){
                    reIndexMap[B.rows[j]] = pair_to_index[{C.rows[i], B.rows[j]}];
                }
                B_alpha.transform_data(reIndexMap);
                B_alpha.reduce_fully(K);
            }
            
            //  delete possible linear dependencies.
            K.column_reduction_triangular(true);
            return {K, row_ops};
            break;
        }
    }

    return {Sparse_Matrix(0,0), {}};
}



/**
 * @brief This computes all hom-spaces (possibly only the alpha-homs) between active blocks and stores them in hom_spaces.
 *      It also keeps track of which hom-spaces have been computed in domain_keys and codomain_keys.
 * 
 * @param A 
 * @param block_map 
 * @param active_blocks 
 * @param hom_spaces 
 * @param domain_keys 
 * @param codomain_keys 
 * @param alpha 
 * @param compare_hom_space_computation 
 */
void compute_hom_to_b (GradedMatrix& A, index& b, vec<Block_list::iterator>& block_map, indtree& active_blocks, 
                                Hom_map& hom_spaces, std::unordered_map<index, vec<index>>& domain_keys, 
                                std::unordered_map<index, vec<index>>& codomain_keys, r2degree& alpha, AIDA_runtime_statistics& statistics, 
                                AIDA_config& config){ //turn_off_hom_optimisation, bool compare_hom_space_computation = false, bool compute_alpha_hom = true
    Block& B = *block_map[b];
    for(index c : active_blocks){
        Block& C = *block_map[c];
        if(b != c){
            // Do not compute again unless needed; This should save a lot of time hopefully.
            // TO-DO: Run some tests, cannot use if hom_alpha is used
            if( (hom_spaces.find({c,b}) == hom_spaces.end() || false) || config.alpha_hom){
                #if TIMERS
                    hom_space_timer.resume();
                    misc_timer.stop();
                #endif
                
                if(config.turn_off_hom_optimisation){
                    Sparse_Matrix S = Sparse_Matrix(0,0);
                    hom_spaces.emplace(std::make_pair(c,b), compute_hom_space_no_optimisation(A, C, B, alpha, S, config.alpha_hom));}
                else {
                    hom_spaces.emplace(std::make_pair(c,b), compute_hom_space(A, C, B, alpha, config.alpha_hom));    
                }
                
                #if TIMERS
                    hom_space_timer.stop();
                    misc_timer.resume();
                #endif
                
                
                int dim = hom_spaces[{c,b}].first.get_num_cols();
                statistics.dim_hom_vec.push_back(dim);
                if(dim > statistics.dim_hom_max){
                    statistics.dim_hom_max = dim;
                }

                #if DETAILS
                    std::cout << "      Hom-space " << c << " -> " << b << " computed, dim " << dim << std::endl;
                #endif
                 
                
                // We need to know for which blocks b we have computed a hom-space from c to b.
                // When c merges/extends, then these hom-spaces become void.
                // TO-DO: Instead of recomputing, one can check for each computed morphism if it factors/extend.
                if(domain_keys.find(c) == domain_keys.end()){
                    domain_keys[c] = vec<index>();
                }
                if(codomain_keys.find(b) == codomain_keys.end()){
                    codomain_keys[b] = vec<index>();
                }
                domain_keys[c].push_back(b);	
                codomain_keys[b].push_back(c);

                
                if(config.compare_hom){
                    #if TIMERS
                        hom_space_test_timer.resume();
                        misc_timer.stop();
                    #endif
                    Sparse_Matrix S(0,0);
                    auto non_optimised_hom = compute_hom_space_no_optimisation(A, C, B, alpha, S);
                    #if TIMERS
                        hom_space_test_timer.stop();
                        misc_timer.resume();
                    #endif
                    if(hom_spaces[{c,b}].first.get_num_cols() != non_optimised_hom.first.get_num_cols()){
                        std::cout << "Error: Hom-spaces " << c << " -> " << b << " do not match: " << dim << " optimised, vs " << non_optimised_hom.first.get_num_cols() << "brute_force" << std::endl;
                    }
                }
            } else {
                #if DETAILS
                    std::cout << "      Hom-space " << c << " -> " << b << " already computed, dim " << hom_spaces[{c,b}].first.get_num_cols() << std::endl;
                #endif  
            }
        }
    }
}

/**
 * @brief Changes A and N according to the row_operations computed by block_reduce
 * 
 */
void update_matrix(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, vec<index>& batch_indices, 
            vec<index>& solution, index& row_op_limit, vec<op_info>& ops, bool& restricted_batch, bool& delete_N){

    for(index operation_index : solution){

        if(operation_index >= row_op_limit){

        } else {
            auto op = ops[operation_index];
            auto& B_source = *block_map[op.second.first];
            auto& B_target = *block_map[op.second.second];
            

            #if OBSERVE
                if( std::find(observe_row_indices.begin(), observe_row_indices.end(), B_source.rows[op.first.first]) != observe_row_indices.end() ){
                    std::cout << "Row operation: " << B_source.rows[op.first.first] << " -> " << B_target.rows[op.first.second] << std::endl;
                }
            #endif
            A.fast_rev_row_op(B_source.rows[op.first.first], B_target.rows[op.first.second]);
            if(restricted_batch && !delete_N){
                auto& N_source = N_map[op.second.first];
                auto& N_target = N_map[op.second.second];
                // TO-DO: Here we only change _rows of N. We should also change the columns/data. 
                // We also dont need to change the part of N which we are looking at, because it can be reduced to zero with the column operations.
                auto& source_row = N_source._rows[op.first.first];
                CT::add_to(source_row, N_target._rows[op.first.second]);     
            }
        }
    }
}

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param extra_columns 
 */                     
bool block_reduce(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, bitset& sub_batch_indices, std::shared_ptr<Base_change_virtual>& base_change,   
                vec<index>& row_map, 
                bool compare_both = false, const bitset& extra_columns = bitset(0), bool delete_N = false) {
    vec<long> solution_long; vec<index> solution; index row_op_limit; vec<op_info> ops; bool reduced_to_zero = false;
    vec<long> c;  std::vector<std::reference_wrapper<Sparse_Matrix>> Ns; SparseMatrix<long> S(0,0);

    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }

    #if DETAILS
        std::cout << "  Block_reduce called on (b_vec) ";
        for(index b : b_vec){ std::cout << b << " ";}
            std::cout << "  Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }

    #endif  
    

    #if TIMERS
        misc_timer.stop();      
        constructing_linear_system_timer.resume();
    #endif
    linearise_prior(A, Ns, batch_indices, c, sub_batch_indices);
    construct_linear_system(A, batch_indices, sub_batch_indices, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map, extra_columns);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    
    #if SYSTEM_SIZE
        std::cout << "Solving linear system of size: " << S.get_num_cols() << std::endl;
        // S.print();
        // std::cout << "c: " <<  c << std::endl;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution_long);
    for(long i : solution_long){
        solution.push_back(i);
    }
    #if TIMERS
        solve_linear_system_timer.stop();
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero && !compare_both){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << "x (" << sub_batch_indices << ")" << std::endl;
        #endif
        update_matrix(A, N_map, block_map, batch_indices, solution, row_op_limit, ops, restricted_batch, delete_N);
        if(restricted_batch){
            // TO-DO: This could be done faster.
            for(index b : b_vec){
                N_map[b].compute_columns_from_rows((*block_map[b]).rows);
                for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                    N_map[b].data[i].clear();
                }
                N_map[b].compute_rows_forward_map(row_map);
            }
        }
    }
    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param extra_columns 
 */                     
bool block_reduce_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, std::shared_ptr<Base_change_virtual>& base_change,   
                vec<index>& row_map, 
                bool compare_both = false, bool delete_N = false) {
    vec<long> solution_long; vec<index> solution; index row_op_limit; vec<op_info> ops; bool reduced_to_zero = false;
    vec<long> c;  std::vector<std::reference_wrapper<Sparse_Matrix>> Ns; SparseMatrix<long> S(0,0);

    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }

    #if DETAILS
        std::cout << "  Block_reduce called on (b_vec) ";
        for(index b : b_vec){ std::cout << b << " ";}
            std::cout << "  Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
    #endif  
    

    #if TIMERS
        misc_timer.stop();      
        constructing_linear_system_timer.resume();
    #endif
    linearise_prior_full_support(A, Ns, batch_indices, c);
    construct_linear_system_full_support(A, batch_indices, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    
    #if SYSTEM_SIZE
        std::cout << "Solving linear system of size: " << S.get_num_cols() << std::endl;
        // S.print();
        // std::cout << "c: " <<  c << std::endl;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution_long);
    for(long i : solution_long){
        solution.push_back(i);
    }

    #if TIMERS
        solve_linear_system_timer.stop();
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero && !compare_both){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << "x (" << batch_indices << ")" << std::endl;
        #endif
        update_matrix(A, N_map, block_map, batch_indices, solution, row_op_limit, ops, restricted_batch, delete_N);
    }
    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_full_support


/**
 * @brief Updates A and N according to the hom-operations computed by block_reduce_hom
 * 
 * @param A 
 * @param N_map 
 * @param block_map 
 * @param solution 
 * @param row_op_limit 
 * @param ops 
 * @param restricted_batch 
 * @param naive_first 
 */
void update_matrix_hom(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, 
            vec<index>& batch_indices, Hom_map& hom_spaces, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map,
            vec<index>& solution, index& row_op_limit, vec<hom_info>& ops,
            bool restricted_batch = false, bool delete_N = false){
    

    for(index operation_index : solution){

        if(operation_index >= row_op_limit){
           
        } else {
            auto op = ops[operation_index];
            auto& B_source = *block_map[op.second.first];
            auto& B_target = *block_map[op.second.second];
            
                
            auto& C = B_source;
            auto& B = B_target;
            Hom_space& hom_cb = hom_spaces[{op.second.first, op.second.second}];

            hom_action_A(A, B_source.rows, B_target.rows, hom_cb.first.data[op.first], hom_cb.second, base_change);

            if(restricted_batch && !delete_N){
                auto& N_source = N_map[op.second.first];
                auto& N_target = N_map[op.second.second];
                hom_action_N(B, N_source, N_target, hom_cb.first.data[op.first], hom_cb.second);
            } 
        }
    }
}

void update_matrix_extension(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, 
    Hom_map& hom_spaces, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& non_processed_blocks, 
    vec<index>& row_map, vec<index>& solution, index& E_threshold, index& N_threshold, index& M_threshold, vec<hom_info>& hom_storage,
    Transform_Map& batch_transforms, vec<Merge_data>& pro_blocks, Merge_data& pro_block){

    for(index operation_index : solution){

        if(operation_index < E_threshold){
            auto op = hom_storage[operation_index];
            auto& C = *block_map[op.second.first];
            auto& B = *block_map[op.second.second];
           
            Hom_space& hom_cb = hom_spaces[{op.second.first, op.second.second}];
            hom_action_A(A, C.rows, B.rows, hom_cb.first.data[op.first], hom_cb.second, base_change);
        } else if (operation_index < N_threshold){
            auto op = hom_storage[operation_index];
            Batch_transform& transform_space = batch_transforms[std::make_pair(pro_blocks[op.second.first], pro_block)][op.first];
            auto& hom_infos = transform_space.second;
            auto& T = transform_space.first;
            for(index c : non_processed_blocks){
                N_map[c].multiply_id_triangular(T);
            }
            for(auto hom_info : hom_infos){
                auto& C = *block_map[std::get<0>(hom_info)];
                auto& B = *block_map[std::get<1>(hom_info)];
                Hom_space& hom_cb = hom_spaces[{std::get<0>(hom_info), std::get<1>(hom_info)}];
                hom_action_A(A, C.rows, B.rows, hom_cb.first.data[std::get<2>(hom_info)], hom_cb.second, base_change);
            }
        } else if (operation_index < M_threshold){
            auto& op = hom_storage[operation_index];
            index& source = op.first;
            index& target = op.second.first;
            for(index c : non_processed_blocks){
                N_map[c].col_op(source, target);
            }
        }
    }
}

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param morphisms The morphisms between blocks.
 * @param extra_columns if naive_first is true, this is the set of columns-indices of the batch which belong to the second subspace tested in naive decomposition.
 *                      If naive_first is false, this is the set of columns-indices of the batch which belong to the first subspace tested in naive decomposition.
 *                      It should be empty if block reduce is not called from naive decomposition.
 */                     
bool block_reduce_hom(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, bitset& sub_batch_indices,
                std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map, Hom_map& hom_spaces,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){ 
    vec<index> solution; index row_op_limit; vec<hom_info> ops; bool reduced_to_zero = false;
    vec<index> c;  Sparse_Matrix S(0,0); 
    std::vector<std::reference_wrapper<Sparse_Matrix>> Ns;
    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }
    #if DETAILS
        std::cout << "  block_reduce_hom called on blocks ";
        for(index b : b_vec){ std::cout << b << " ";}
        std::cout << " - Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
        // std::cout << "      batch_indices: " << batch_indices << std::endl;
        if(restricted_batch){
            // std::cout << "      sub_batch_indices: " << sub_batch_indices << std::endl;
        }
    #endif

    #if TIMERS 
        misc_timer.stop();     
        constructing_linear_system_timer.resume();
    #endif
    construct_linear_system_hom(batch_indices, sub_batch_indices, restricted_batch, 
            relevant_blocks, block_map, S, ops, b_vec, N_map, hom_spaces, row_map, c, extra_columns);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
        // S.print(true, true);
        // std::cout << "  c: " <<  c;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution);

    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << " and " << sub_batch_indices << std::endl;
        #endif
        update_matrix_hom(A, N_map, block_map, batch_indices, hom_spaces, base_change, row_map, solution, row_op_limit, ops, restricted_batch, delete_N);
        for(index b : b_vec){
            if(restricted_batch && !delete_N){
                N_map[b].compute_columns_from_rows((*block_map[b]).rows);
            }
            for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
            }
            N_map[b].compute_rows_forward_map(row_map);
        }
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_hom


/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param morphisms The morphisms between blocks.
 * @param extra_columns if naive_first is true, this is the set of columns-indices of the batch which belong to the second subspace tested in naive decomposition.
 *                      If naive_first is false, this is the set of columns-indices of the batch which belong to the first subspace tested in naive decomposition.
 *                      It should be empty if block reduce is not called from naive decomposition.
 */                     
bool block_reduce_hom_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map,
                std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map, Hom_map& hom_spaces,  
                bool delete_N = false){ 
    vec<index> solution; index row_op_limit; vec<hom_info> ops; bool reduced_to_zero = false;
    vec<index> c;  Sparse_Matrix S(0,0); 
    std::vector<std::reference_wrapper<Sparse_Matrix>> Ns;
    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }
    #if DETAILS
        std::cout << "  block_reduce_hom called on blocks ";
        for(index b : b_vec){ std::cout << b << " ";}
        std::cout << " - Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
        // std::cout << "      batch_indices: " << batch_indices << std::endl;
        if(restricted_batch){
            // std::cout << "      sub_batch_indices: " << sub_batch_indices << std::endl;
        }
    #endif

    #if TIMERS 
        misc_timer.stop();     
        constructing_linear_system_timer.resume();
    #endif
    construct_linear_system_hom_full_support(batch_indices, restricted_batch, 
            relevant_blocks, block_map, S, ops, b_vec, N_map, hom_spaces, row_map, c);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
        // S.print(true, true);
        // std::cout << "  c: " <<  c;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution);

    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << " and " << batch_indices << std::endl;
        #endif
        update_matrix_hom(A, N_map, block_map, batch_indices, hom_spaces, base_change, row_map, solution, row_op_limit, ops, restricted_batch, delete_N);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_hom_full_support

/**
 * @brief This function is used to test both block_reduce and block_reduce_hom and compare the results.
 * 
 * @param A 
 * @param b_vec 
 * @param N_map 
 * @param batch_indices 
 * @param restricted_batch 
 * @param blocks 
 * @param block_map 
 * @param support 
 * @param base_change 
 * @param row_map 
 * @param hom_spaces 
 * @param brute_force 
 * @param compare_both 
 * @param extra_columns 
 * @param delete_N If true, then we will not perform row operations on N, 
 *                 but only delete the part of N belonging to b_vec and sub_batch_indices.
 * @return true 
 * @return false 
 */
bool use_either_block_reduce(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& blocks, vec<Block_iterator>& block_map, bitset& support, std::shared_ptr<Base_change_virtual>& base_change, 
                vec<index>& row_map, Hom_map& hom_spaces, bool brute_force = false, bool compare_both = false,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){

    bool block_reduce_result = false;
    bool block_reduce_result_hom = false;

    if (brute_force || compare_both) {
        block_reduce_result = block_reduce(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, support, base_change, row_map, compare_both, extra_columns, delete_N);
    }

    if (!brute_force || compare_both){
        block_reduce_result_hom = block_reduce_hom(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, support, base_change, row_map, hom_spaces, extra_columns, delete_N);
    }
    
    if(compare_both){
        if(block_reduce_result != block_reduce_result_hom){
            std::cout << "Error: Block reduce and block reduce hom do not agree." << std::endl;
            std::cout << "Blocks to delete: " << b_vec << std::endl;
            std::cout << "All blocks: " << blocks << std::endl;
            std::cout << "Block reduce result: " << block_reduce_result << " Block reduce hom result: " << block_reduce_result_hom << std::endl;
        }
        assert(block_reduce_result == block_reduce_result_hom);
    }
    return block_reduce_result || block_reduce_result_hom;
}

/**
 * @brief This function is used to test both block_reduce and block_reduce_hom and compare the results.
 * 
 * @param A 
 * @param b_vec 
 * @param N_map 
 * @param batch_indices 
 * @param restricted_batch 
 * @param blocks 
 * @param block_map 
 * @param support 
 * @param base_change 
 * @param row_map 
 * @param hom_spaces 
 * @param brute_force 
 * @param compare_both 
 * @param extra_columns 
 * @param delete_N If true, then we will not perform row operations on N, 
 *                 but only delete the part of N belonging to b_vec and sub_batch_indices.
 * @return true 
 * @return false 
 */
bool use_either_block_reduce_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& blocks, vec<Block_iterator>& block_map, std::shared_ptr<Base_change_virtual>& base_change, 
                vec<index>& row_map, Hom_map& hom_spaces, bool brute_force = false, bool compare_both = false,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){

    bool block_reduce_result = false;
    bool block_reduce_result_hom = false;

    if (brute_force || compare_both) {
        block_reduce_result = block_reduce_full_support(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, base_change, row_map, compare_both, delete_N);
    }

    if (!brute_force || compare_both){
        block_reduce_result_hom = block_reduce_hom_full_support(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, base_change, row_map, hom_spaces, delete_N);
    }
    
    if(compare_both){
        if(block_reduce_result != block_reduce_result_hom){
            std::cout << "Error: Block reduce and block reduce hom do not agree." << std::endl;
            std::cout << "Blocks to delete: " << b_vec << std::endl;
            std::cout << "All blocks: " << blocks << std::endl;
            std::cout << "Block reduce result: " << block_reduce_result << " Block reduce hom result: " << block_reduce_result_hom << std::endl;
        }
        assert(block_reduce_result == block_reduce_result_hom);
    }
    return block_reduce_result || block_reduce_result_hom;
}

/**
 * @brief Considers two virtual blocks (blocks + columns in batch) and computes all batch_internal 
 *          column operations from source to target which are part of a homomorphism (which has to go in the opposite direction).
 * 
 * @param virtual_source_block 
 * @param virtual_target_block 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @return array<index> 
 */
vec< Batch_transform > compute_internal_col_ops(Merge_data& virtual_source_block, Merge_data& virtual_target_block, 
    Sub_batch& N_map, Hom_map& hom_spaces, vec<index>& row_map, index& k, vec<Block_iterator>& block_map){

    bitset& source_batch_indices = virtual_source_block.second;
    vec<index>& source_block_indices = virtual_source_block.first;
    bitset& target_batch_indices = virtual_target_block.second;
    vec<index>& target_block_indices = virtual_target_block.first;
    // Let N_source and N_target be the sub-batches of N corresponding to the virtual blocks.
    // Then a set of batch-internal column operations from source to target is a matrix P such that 
    // N_source * P = Q * N_target for some Q in Hom(virtual_target_block, virtual_source_block).

    // First add the column operations to the linear system.
    
    index current_row = 0;
    vec<index> source_block_row_map; // Maps the rows of the blocks in the virtual source block to the rows of the linear system.
    for(index i : source_block_indices){
        source_block_row_map.push_back( current_row );
        current_row += N_map[i].get_num_rows();
    }
    
    // Need a linearisation scheme for the entries which can be touched by Q N_target.
    auto position = [source_block_row_map, target_batch_indices, current_row, k](index& column_index, index& row_index, index& target_block_number) -> index {
        return linearise_position_reverse_ext<index>(column_index, source_block_row_map[target_block_number] + row_index, k, current_row);
    };

    Sparse_Matrix S(0, target_batch_indices.count()*current_row);
    index col_op_threshold = 0;
    index hom_threshold = 0;
    vec<pair> col_op_keys;
    vec< std::tuple<index, index, index> > hom_keys;

    // First the homomorphisms aka row-ops from blocks in the target to blocks in the source.

    for(index c : target_block_indices){
        for(index i = 0; i< source_block_indices.size(); i++){
            index& b = source_block_indices[i];
            Hom_space& hom = hom_spaces[{c, b}];
            Sparse_Matrix& hom_matrices = hom.first;
            vec<pair>& row_op_keys = hom.second;
            for(index hom_counter = 0; hom_counter < hom_matrices.data.size(); hom_counter++){
                vec<index>& Q = hom_matrices.data[hom_counter];
                S.data.emplace_back(hom_action_full_support(source_block_row_map[i], current_row, Q, row_op_keys, N_map[c]));
                hom_keys.push_back({c, b, hom_counter});
            }
        }
    }

    hom_threshold = S.data.size();

    // Then add the sets of internal column operations from source to target.
    for(index i = source_batch_indices.find_first(); i != bitset::npos; i = source_batch_indices.find_next(i)){
        for(index j = target_batch_indices.find_first(); j != bitset::npos; j = target_batch_indices.find_next(j)){
            col_op_keys.push_back({i,j});
            S.data.push_back(vec<index>());
            for(index block_counter = 0; block_counter < source_block_indices.size(); block_counter++){
                for(index row_index : N_map[source_block_indices[block_counter]].data[i]){
                    S.data.back().push_back(position(j, row_map[row_index], block_counter));
                }
            }
        }
    }
    col_op_threshold = S.data.size();

    // At last, local column operations from the source.

    for(index i = 0; i< source_block_indices.size(); i++){
        Block& B = *block_map[source_block_indices[i]];
        for(vec<index> column : B.local_data -> data){
            for(index j = target_batch_indices.find_first(); j != bitset::npos; j = target_batch_indices.find_next(j)){
                S.data.push_back(vec<index>());
                for(index row_index : column){
                    S.data.back().push_back(position(j, row_map[row_index], i));
                }
            }
        }
    }

    // Reduction to independent column operations:
    S.compute_num_cols();
    auto K = S.kernel();
    K.cull_columns(col_op_threshold, false);
    K.column_reduction_triangular(hom_threshold, true);

    // Build corresponding matrices.

    vec< std::pair<DenseMatrix, vec<std::tuple<index, index, index>> > > result;
    for(vec<index> column : K.data){
        result.push_back({DenseMatrix(k, k), {}});
        for(index i : column){
            if(i < hom_threshold){
                result.back().second.push_back(hom_keys[i]);
            } else {
                index j = i - hom_threshold;
                auto& [source_col, target_col] = col_op_keys[j];
                result.back().first.data[target_col].set(source_col);
            }
        }
    }

    return result;  
}

/**
 * @brief This computes a column-form of the entries in the batch, splits it up into the blocks which are touched by the batch, and stores the information.
 * 
 * @param active_blocks Stores the touched blocks.
 * @param block_map 
 * @param A the matrix
 * @param batch 
 */
void compute_touched_blocks(indtree& active_blocks, vec<Block_iterator>& block_map, 
                            GradedMatrix& A, vec<index>& batch, Sub_batch& N_map) {

    for(index j = 0; j < batch.size(); j++){
        index bj = batch[j];
        // Q: Is this the fastest way to do it? It is possible we want a specific sorting function.
        A.sort_column(bj);
        convert_mod_2(A.data[bj]); 
        #if DETAILS
            std::cout << "  Batch-col " << j << " after sorting: " << A.data[bj] << std::endl;
        #endif
        for(index i : A.data[bj]){
            Block& B = *block_map[i];
            index initial = B.rows.front();
	        auto new_touched = active_blocks.insert(initial); 
            if(new_touched.second){
                N_map.emplace(initial, Sparse_Matrix(batch.size(), B.rows.size()));
                N_map[initial].data = vec<vec<index>>(batch.size(), vec<index>());
            }
            N_map[initial].data[j].push_back(i);
            // Maybe we should also store the rows, but not sure right now.
            assert( A._rows[i].back() == bj);
            A._rows[i].pop_back();
        }
    }
} // compute_touched_blocks

/**
 * @brief Changes N to achieve a partial decomposition of (A_B | N) by recursively finding a decomposition into two components.
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 * @param pierced blocks
 * @param batch_indices
 * @param N_column_indices 
 * @param e_vec 
 * @param N_map 
 * @param vector_space_decompositions 
 */
vec< Merge_data > exhaustive_alpha_decomp(
                        GradedMatrix& A, Block_list& B_list, vec<Block_iterator>& block_map, 
                        vec<index>& pierced_blocks, vec<index>& batch_indices, const bitset& N_column_indices, 
                        vec<bitset>& e_vec, Sub_batch& N_map, vec<vec<transition>>& vector_space_decompositions,
                        AIDA_config& config, std::shared_ptr<Base_change_virtual>& base_change,
                        Hom_map& hom_spaces, vec<index> row_map, bool brute_force = false, bool compare_both = false) {
                           
    //TO-DO: Implement a branch and bound strategy such that we do not need to iterate over those decompositions
    //TO-DO: Need to first decompose/reduce the left-hand columns, create an updated temporary block-merge and then decompose the right-hand columns.
    int k = N_column_indices.count();    
    int num_b = pierced_blocks.size();
    assert(batch_indices.size() == N_column_indices.size());
    #if DETAILS
        std::cout << "  Calling exhaustive_alpha_decomp with " << k << " columns and " << 
        num_b << " blocks at N_column_indices: " << N_column_indices << std::endl;
    #endif
    if( k == 1 ){
        vec< Merge_data > result;
        result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
        return result;
    }
    if( k > vector_space_decompositions.size() + 1 ){
        config.decomp_failure.push_back(batch_indices);
        if(true){
            std::cout << "  No vector space decompositions for the local (!) dimension " << k << 
            " provided. \n Warning: Decomposition is almost surely only partial from here on." << std::endl;
        }
        // TO-DO: We could call methods from generate decompositions here to try some column-operations 
        // to find a decomposition, then break if we dont find one after a certain time.
        vec< Merge_data > result;
        result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
        return result;
    }

    // Iterate over all decompositions of GF(2)^k into two subspaces.
    for(auto transition : vector_space_decompositions[k-2]){
        auto& basechange = transition.first;
        auto& partition_indices = transition.second;
        bitset indices_1 = glue(N_column_indices, partition_indices);
        bitset indices_2 = glue(N_column_indices, partition_indices.flip());
        #if DETAILS
            std::cout << "  Indices 1: " << indices_1 << " Indices 2: " << indices_2 << std::endl;
        #endif
        vec<index> blocks_1;
        vec<index> blocks_2;
        indtree blocks_conflict;
        for(index b : pierced_blocks){
            N_map[b].multiply_dense_with_e_check(basechange, e_vec, N_column_indices);
            Block& B = *block_map[b];
            N_map[b].compute_rows_forward_map(row_map);
        }

        for(index b : pierced_blocks){
            if(N_map[b].is_zero(indices_1)){
                blocks_2.push_back(b);
            } else if (N_map[b].is_zero(indices_2)){
                blocks_1.push_back(b);
            } else {
                blocks_conflict.insert(b);
                blocks_1.push_back(b);
                blocks_2.push_back(b);
            }
        }

        #if DETAILS
            for(index b : pierced_blocks){
                std::cout << "  Block " << b << ": ";
                N_map[b].print_rows();
            }
            std::cout << "B_1: " << blocks_1 << " B_2: " << blocks_2 << " B_conflict: " << blocks_conflict << std::endl;
        #endif
        bool conflict_resolution = true;

        // Optimisation step: If we could not delete the lhs of a block in conflict, then try to delete the rhs without all operations. 
        // Only stop doing this if this does not work for one block because in this case we will have to try to delete all of these blcoks at once anyways.
        #if DETAILS
            vec<pair> deletion_tracker;
        #endif
        if (blocks_conflict.size() > 0){

            for(auto itr = blocks_conflict.rbegin(); itr != blocks_conflict.rend();){
                index b = *itr;
                vec<index> b_vec = {b};

                #if TIMERS
                    alpha_decomp_timer.stop();
                    misc_timer.resume();  
                #endif

                bool deleted_N1 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                    blocks_1, block_map, indices_1, base_change, row_map, hom_spaces, brute_force, compare_both);

                #if TIMERS
                    dispose_S_timer.stop();
                    alpha_decomp_timer.resume();
                #endif
                if(deleted_N1){
                    assert(N_map[b].is_zero(indices_1));
                    erase_from_sorted_vector(blocks_1, b);
                    #if DETAILS
                        deletion_tracker.push_back({b, 1});
                    #endif
                    auto it = blocks_conflict.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it); 
                } else if ( conflict_resolution ) {
                    
                    #if TIMERS
                        alpha_decomp_timer.stop();
                        misc_timer.resume();
                    #endif
                    bool deleted_N2 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                        blocks_2, block_map, indices_2, base_change, row_map, hom_spaces, brute_force, compare_both);

                    #if TIMERS
                        dispose_S_timer.stop();
                        alpha_decomp_timer.resume();
                    #endif
                    if(deleted_N2){
                        assert(N_map[b].is_zero(indices_2));
                        erase_from_sorted_vector(blocks_2, b);
                        #if DETAILS
                            deletion_tracker.push_back({b, 2});
                        #endif
                        auto it = blocks_conflict.erase(--itr.base());
                        itr = std::reverse_iterator<decltype(it)>(it);
                    } else {
                        conflict_resolution = false;
                        itr++;
                    }
                } else {
                    itr++;
                
                }
            }
        }
        #if DETAILS
            for(auto [b, side] : deletion_tracker){
                    std::cout << "    Deleted " << b << " from side " << side << std::endl;
                }
        #endif
        if(conflict_resolution){

            #if DETAILS
                std::cout << "  Conflict resolved directly." << std::endl;
            #endif
        } else {
            #if DETAILS
                std::cout << "  Conflict could not be resolved. First reducing N_1 as much as possible." << std::endl;
            #endif
            
            indtree blocks_excl_1;
            std::set_difference(blocks_1.begin(), blocks_1.end(), blocks_conflict.begin(), blocks_conflict.end(), std::inserter(blocks_excl_1, blocks_excl_1.begin()));
            // Only need to further reduce those blocks which are not in conflict, because they would have already been deleted in the first step.
            for(auto itr = blocks_excl_1.rbegin(); itr != blocks_excl_1.rend();){
                index b = *itr;
                vec<index> b_vec = {b};

                #if TIMERS
                    alpha_decomp_timer.stop();
                    misc_timer.resume();
                #endif
                bool deleted_more_1 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                    blocks_1, block_map, indices_1, base_change, row_map, hom_spaces, brute_force, compare_both);
                #if TIMERS
                    dispose_S_timer.stop();
                    alpha_decomp_timer.resume();
                #endif
                if(deleted_more_1 ){
                    assert(N_map[b].is_zero(indices_1));
                    if(!N_map[b].is_zero(indices_2)){
                        insert_into_sorted_vector(blocks_2, b);
                    }
                    erase_from_sorted_vector(blocks_1, b);
                    auto it = blocks_excl_1.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it);  
                    #if DETAILS
                        std::cout << "    In Step 2, deleted " << b << " from side 1." << std::endl;
                        if(std::find(blocks_2.begin(), blocks_2.end(), b) != blocks_2.end()){
                            std::cout << "    ..but added " << b << " to side 2." << std::endl;
                        }
                    #endif
                } else {
                    itr++;
                }
            }
            assert( blocks_conflict.size() > 0);
            // Now need to treat all of blocks_1 as one block and delete its rhs of N.
            #if DETAILS
                std::cout << "B_1: " << blocks_1 << " B_2: " << blocks_2 << " B_conflict: " << blocks_conflict << std::endl;
            #endif

            #if TIMERS
                alpha_decomp_timer.stop();
                misc_timer.resume();
            #endif
            conflict_resolution = use_either_block_reduce(A, blocks_1, N_map, batch_indices, true, 
                pierced_blocks, block_map, indices_2, base_change, row_map, hom_spaces, brute_force, compare_both, indices_1, true);
            #if TIMERS
                dispose_S_timer.stop();
                alpha_decomp_timer.resume();
            #endif
            if(conflict_resolution){
                for(index b : blocks_1){
                    erase_from_sorted_vector(blocks_2, b);
                    blocks_conflict.erase(b);
                    assert(N_map[b].is_zero(indices_2));
                }
            }
        }
        
        if (conflict_resolution){ 
            assert(blocks_conflict.size() == 0);
            // A valid decomposition has been found. Continue here.
            auto partition_1 = exhaustive_alpha_decomp(A, B_list, block_map, blocks_1, batch_indices, indices_1, 
                e_vec, N_map, vector_space_decompositions, config, base_change,
                hom_spaces, row_map, brute_force, compare_both);
            auto partition_2 = exhaustive_alpha_decomp(A, B_list, block_map, blocks_2, batch_indices, indices_2, 
                e_vec, N_map, vector_space_decompositions, config, base_change,
                hom_spaces, row_map, brute_force, compare_both);
            partition_1.insert(partition_1.end(), partition_2.begin(), partition_2.end());
            return partition_1;
        }
    }
    vec< Merge_data > result;
    result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
    return result;
} // exhaustive_alpha_decomp


/**
 * @brief Tries to delete a cocycle defining an extension from pro_block to b
 * 
 * @param A 
 * @param b 
 * @param b_non_zero_columns 
 * @param non_processed_blocks 
 * @param pro_block 
 * @param incoming_vertices 
 * @param pro_blocks 
 * @param deleted_cocycles_b 
 * @param hom_graph 
 * @param hom_spaces 
 * @param batch_transforms 
 * @param base_change 
 * @param block_map 
 * @param row_map 
 * @param N_map 
 * @return true 
 * @return false 
 */
bool alpha_extension_decomposition(GradedMatrix& A, index& b, bitset& b_non_zero_columns, vec<index>& non_processed_blocks,
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& external_incoming_vertices, Row_transform_map& component_transforms, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){
    
    Block& B = *block_map[b];
    bitset& target_batch_indices = pro_block.second;

    // If N_b is already zero, then we do not need to do anything.
    if(N_map[b].is_zero(target_batch_indices)){
        return true;        
    }

    #if DETAILS
        std::cout << "  alpha_extension_decomposition called on blocks ";
        std::cout << b << " x " << pro_block.first << " / " << target_batch_indices << std::endl;
    #endif

    
    #if TIMERS
        misc_timer.stop();    
        constructing_linear_system_timer.resume();
    #endif
    vec<index> y;
    linearise_sub_batch_entries(y, N_map[b], target_batch_indices, row_map);
    vec<index> solution;    
    Sparse_Matrix S(0,0); vec<hom_info> hom_storage; index E_threshold;  index N_threshold; index M_threshold;
    construct_linear_system_extension(S, hom_storage, E_threshold, N_threshold, M_threshold,
    b, b_non_zero_columns, 
    pro_block, incoming_vertices, pro_blocks, deleted_cocycles_b,
    hom_graph, hom_spaces, batch_transforms, 
    block_map, row_map, N_map);


    
    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
    #endif
    S.compute_num_cols();
    bool reduced_to_zero = S.solve_col_reduction(y, solution);
    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b << " and " << target_batch_indices << std::endl;
        #endif
        update_matrix_extension(A, N_map, block_map, hom_spaces, base_change, non_processed_blocks, row_map, 
        solution, E_threshold, N_threshold, M_threshold, hom_storage, batch_transforms, pro_blocks, pro_block);
        for(index i = target_batch_indices.find_first(); i != bitset::npos; i = target_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
        }
        N_map[b].compute_rows_forward_map(row_map);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    #if DETAILS
        if(reduced_to_zero){
            std::cout << "Deleted Cocycle" << std::endl;
        } else {
            std::cout << "Did not delete Cocycle" << std::endl;
        }
    #endif
    return reduced_to_zero;

}

/**
 * @brief tries to solve the linear system needed to delete a cocycle and returns a solution if successful.
 * 

 */
bool virtual_alpha_extension_decomposition(GradedMatrix& A, index& b, bitset& b_non_zero_columns, vec<index>& non_processed_blocks,
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, std::shared_ptr<Base_change_virtual>& base_change, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){
    
    Block& B = *block_map[b];
    bitset& target_batch_indices = pro_block.second;
    //In general version the following should be the effect of a hom on a block.
    Sparse_Matrix& Cocycle_container = N_map[b]; 

    // If N_b is already zero, then we do not need to do anything. 
    // Should be checked before calling, so not necessary to check again.
    // if(N_map[b].is_zero(target_batch_indices)){
    //     return true;        
    // }

    //TO-DO: Continue here.
    #if DETAILS
        std::cout << "  alpha_extension_decomposition called on blocks ";
        std::cout << b << " x " << pro_block.first << " / " << target_batch_indices << std::endl;
    #endif

    #if TIMERS
        misc_timer.stop();    
        constructing_linear_system_timer.resume();
    #endif
    vec<index> y;
    linearise_sub_batch_entries(y, Cocycle_container, target_batch_indices, row_map);
    vec<index> solution;    
    Sparse_Matrix S(0,0); vec<hom_info> hom_storage; index E_threshold;  index N_threshold; index M_threshold;
    construct_linear_system_extension(S, hom_storage, E_threshold, N_threshold, M_threshold,
    b, b_non_zero_columns, 
    pro_block, incoming_vertices, pro_blocks, deleted_cocycles_b,
    hom_graph, hom_spaces, batch_transforms, 
    block_map, row_map, N_map);
    
    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
    #endif
    S.compute_num_cols();
    bool reduced_to_zero = S.solve_col_reduction(y, solution);
    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b << " and " << target_batch_indices << std::endl;
        #endif
        update_matrix_extension(A, N_map, block_map, hom_spaces, base_change, non_processed_blocks, row_map, 
        solution, E_threshold, N_threshold, M_threshold, hom_storage, batch_transforms, pro_blocks, pro_block);
        for(index i = target_batch_indices.find_first(); i != bitset::npos; i = target_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
        }
        N_map[b].compute_rows_forward_map(row_map);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif

    return reduced_to_zero;
}

Row_transform compute_internal_row_ops(bitset& source_deleted_cocycles, bitset& target_deleted_cocycles){
    // TO-DO: Implement this.
    // Question is: can the collection of cocycles in source_deleted_cocycles be deleted when in target_deleted_cocycles?
    // Right now I will only program this for the case where we deal with components which are cyclic modules.
    
    // We should be able to iterate over all non-zero entries in source_deleted_cocycles, 
    // pretend that their entries lie in the target row and 
    // check if we can delete each seperately via alpha-extension decomposition.

    for(index i = source_deleted_cocycles.find_first(); i != bitset::npos; i = source_deleted_cocycles.find_next(i)){
        // bool deleteion = virtual_alpha_extension_decomposition(
                            
    }

    return Row_transform();
}

/**
 * @brief 
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 * @param pierced_blocks 
 * @param batch_indices 
 * @param N_column_indices 
 * @param e_vec 
 * @param N_map 
 * @param config
 * @param base_change 
 * @param hom_spaces 
 * @param row_map 
 * @param computation_order 
 * @param scc 
 * @param condensation 
 * @param is_resolvable_cycle
 * @return vec< Merge_data > 
 */
vec< Merge_data > automorphism_sensitive_alpha_decomp( GradedMatrix& A, Block_list& B_list, vec<Block_iterator>& block_map, 
    vec<index>& local_pierced_blocks, vec<index>& batch_indices, const bitset& N_column_indices, 
    vec<bitset>& e_vec, Sub_batch& N_map, vec<vec<transition>>& vector_space_decompositions,
    AIDA_config& config, std::shared_ptr<Base_change_virtual>& base_change,
    Hom_map& hom_spaces, vec<index>& row_map, Graph& hom_graph,
    vec<index>& computation_order, vec<vec<index>>& scc, Graph& condensation, vec<bool>& is_resolvable_cycle){

    index k = batch_indices.size();

    Transform_Map batch_transforms; // storage for transforms; maps pairs of virtual (processed) blocks to the admissible batch-transforms.
    vec<Merge_data> pro_blocks; // Processed blocks may have already merged, though not formally so.
    Graph pro_graph; // graph on processed blocks given by existence of associated column-operations.
    // There should be an optimisation which mostly avoids computing this ->
    array<index> pro_scc;
    Graph pro_condensation; 
    vec<index> pro_computation_order;
    bitset non_pro = N_column_indices;
    vec<index> non_processed_blocks = local_pierced_blocks;
    

    #if DETAILS
        std::cout << "      Automorphism Sensitive Alpha Decomposition" << std::endl;
        std::cout << "Blocks: " << local_pierced_blocks << ", col-support: " << N_column_indices << " with hom-graph: " << std::endl;
        print_graph_with_labels(hom_graph, local_pierced_blocks);
        vec<indtree> scc_labels;
        // Create the labels for the SCCs for better readability.
        for( vec<index>& s : scc){
            scc_labels.push_back(indtree());
            for(index number : s){
                scc_labels.back().insert(local_pierced_blocks[number]);
            }
        }
        std::cout << " and condensation: " << std::endl;
        print_graph_with_labels<indtree>(condensation, scc_labels);
        vec<vec<bool>> N_map_indicator = vec<vec<bool>>(local_pierced_blocks.size(), vec<bool>(k, 0));
        for(index i = 0; i < local_pierced_blocks.size(); i++){
            for(index j = 0; j < k; j++){
                N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
            }
        }
    #endif

    for(auto it = computation_order.rbegin(); it != computation_order.rend(); it++){
        
        vec<bitset> component_non_zero_columns; // Stores the column supports for N^{zero} for each block in the component.
        index component_index = *it;
        vec<index>& component_blocks_numbers = scc[component_index];
        vec<index> component_blocks = vec_restriction(local_pierced_blocks, component_blocks_numbers);
        // This actually removes component blocks from non_processed_blocks of course.
        CT::add_to(component_blocks, non_processed_blocks);
        #if DETAILS
            std::cout << "Processing component " << component_index << " with vertices { " << 
            component_blocks_numbers << "} and blocks { " << component_blocks << "}"  << std::endl;
            std::cout << "Processed virtual blocks: ";
            for(auto& p : pro_blocks){
                std::cout << " ( " << p.first << ") ";
            }
            std::cout << std::endl;
            std::cout << "Internal col-op graph:" << std::endl;
                vec<vec<index>> pro_graph_labels = vec<vec<index>>();
                for(Merge_data virtual_block : pro_blocks){
                    pro_graph_labels.push_back(virtual_block.first);
                }
                print_graph_with_labels(pro_graph, pro_graph_labels);
        #endif
        //Should not need to do this, if there arent many entries-> TO-DO: Check if this is necessary.
        bitset non_zero_columns = non_pro;
        if(non_pro.any()){
            non_zero_columns = simultaneous_column_reduction(N_map, component_blocks, local_pierced_blocks, non_pro);
            non_pro ^= non_zero_columns;
            // simultaneous_align(N_map, local_pierced_blocks, processed_support, non_zero_columns); -> Do we need this?  Update non-zero cols if align is used. 
        }
        for( index b : component_blocks){
            N_map[b].compute_rows_forward_map(row_map);
        }
        #if DETAILS
            std::cout << " N^{0}-support: " << non_zero_columns << std::endl;
            std::cout << " Non-processed columns: " << non_pro << std::endl;
            for(index i = 0; i < local_pierced_blocks.size(); i++){
                for(index j = 0; j < k; j++){
                    N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                }
            }
        #endif

        Graph component_graph;
        // TO-DO: Need to store which colums-ops correspond to an operation that is legal via this graph.
        Row_transform_map component_transforms;

        if(component_blocks.size() == 1){

            component_non_zero_columns = vec<bitset>(1, non_zero_columns);
            // No further reduction needed.
        } else {
            // If the component has more than one block, we need to consider the automorphism group of the blocks when extended with their new columns.
            // This means forming a new sub graph of \B
            component_graph = induced_subgraph(hom_graph, component_blocks_numbers);
            #if DETAILS
                std::cout << "  Induced subgraph of hom-graph: " << std::endl;
                print_graph_with_labels(component_graph, component_blocks);
            #endif

            if(is_resolvable_cycle[component_index]){
                component_non_zero_columns = vec<bitset>(component_blocks.size(), bitset(k, 0));
                // With this we can track which col ops need to be done if we perform a row op internal to the component.
                // -> We can also use all row-operations and thus reduce the sub-matrix N^zero even more if needed without calling Naive Decomposition
                if(non_zero_columns.count() < component_blocks.size() && non_zero_columns.any()){
                    // If the reduced Matrix has less columns than rows (each block is one row), only then are additional row-operations necessary:
                    simultaneous_row_reduction_on_submatrix(N_map, component_blocks, non_zero_columns, A);
                    for(index b : component_blocks){
                        N_map[b].compute_columns_from_rows((*block_map[b]).rows);
                    }
                }
                if(non_zero_columns.any()){
                    for(index j = 0; j < component_blocks.size(); j++){
                        bool all_zero = true;
                        for(index i = non_zero_columns.find_first(); i != bitset::npos; i = non_zero_columns.find_next(i)){
                            if( N_map[component_blocks[j]].is_nonzero(i)){
                                all_zero = false;
                                component_non_zero_columns[j].set(i);
                                assert(component_non_zero_columns[j].count() == 1);
                            }
                        }
                        if(all_zero){
                            delete_incoming_edges(component_graph, j);
                        }
                    }
                }

            } else {
                // At the moment, this should never be called, because I do not have the time to 
                // write a program which generates the sub-group of GL_k(F_2) acting on the columns via 
                // homomorphisms and from there generate all decompositions accesible from a group action.
                assert(false);
            }
        }

        // After N^{0} has been reduced we use these new columns to reduce N_b for the processed blocks.
        // This bitset has a true/1 entry whenever a cocycle has not(!) been deleted.
        vec<bitset> deleted_cocycles = vec<bitset>(component_blocks.size(), bitset(pro_blocks.size(), 0));
        for(index i = 0; i < component_blocks.size(); i++){
            for(index j = 0; j < pro_blocks.size(); j++){
                if(N_map[component_blocks[i]].is_nonzero(pro_blocks[j].second)){
                    deleted_cocycles[i].set(j);
                }
            }
        }

        for(auto it = pro_computation_order.rbegin(); it != pro_computation_order.rend(); it++){
            index& pro_component_index = *it;
            vec<index>& pro_component_blocks_numbers = pro_scc[pro_component_index];
            Vertex current_vertex = boost::vertex(pro_component_index, pro_condensation);
            vec<Merge_data> current_pro_blocks = vec_restriction(pro_blocks, pro_component_blocks_numbers); 

            #if DETAILS
                std::cout << "  Deleting above " << pro_component_index << " with ";
                for(auto& p : current_pro_blocks){
                    std::cout << " ( " << p.first << ") -> ";
                    std::cout << p.second;
                }
                std::cout << std::endl;
            #endif
            if (component_blocks.size() == 1) {
                index& b = component_blocks[0];
                bitset& b_non_zero_columns = component_non_zero_columns[0];
                for(index pro_b : pro_component_blocks_numbers){
                    Merge_data& pro_block = pro_blocks[pro_b];
                    // vec<index>& pro_block_blocks = pro_block.first;
                    // bitset& pro_support = pro_block.second;
                    Vertex internal_current_vertex = boost::vertex(pro_b, pro_graph);
                    vec<index> incoming_vertices = incoming_edges<index>(pro_graph, internal_current_vertex); // Those virtual blocks from which there are internal column operations to the current block.
                    vec<index> external_incoming_vertices = vec<index>(); // No other blocks in the component.
                    if(deleted_cocycles[0][pro_b]){
                        deleted_cocycles[0][pro_b] = ! alpha_extension_decomposition(
                            A, b, b_non_zero_columns, non_processed_blocks, pro_block, incoming_vertices, pro_blocks, deleted_cocycles[0],
                            hom_graph, hom_spaces, batch_transforms, base_change, external_incoming_vertices, component_transforms,
                            block_map, row_map, N_map);
                    } else {
                        #if DETAILS
                        std::cout << "      0 cocyle at " << b << " and " << pro_block << std::endl;
                        #endif
                    }
                    #if DETAILS
                        for(index i = 0; i < local_pierced_blocks.size(); i++){
                            for(index j = 0; j < k; j++){
                                N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                            }
                        }
                    #endif
                }
            } else if (pro_component_blocks_numbers.size() == 1){
                index& pro_b = pro_component_blocks_numbers[0];
                Merge_data& pro_block = pro_blocks[pro_b];
                // vec<index>& pro_block_blocks = pro_block.first;
                // bitset& pro_support = pro_block.second;
                Vertex internal_current_vertex = boost::vertex(pro_b, pro_graph);
                vec<index> internal_incoming_vertices = incoming_edges<index>(pro_graph, internal_current_vertex); // Those virtual blocks from which there are internal column operations to the current pro block.

                for(index i = 0; i < component_blocks.size(); i++){
                    index& b = component_blocks[i];
                    Vertex external_current_vertex = boost::vertex(i, component_graph);
                    assert(external_current_vertex == i);
                    vec<index> external_incoming_vertices = incoming_edges<index>(component_graph, external_current_vertex); // Those blocks in the component from which there are row operations to the current block.
                    bitset& b_non_zero_columns = component_non_zero_columns[i];
                    if(deleted_cocycles[i][pro_b]){
                        deleted_cocycles[i][pro_b] = ! alpha_extension_decomposition(
                            A, b, b_non_zero_columns, non_processed_blocks, pro_block, internal_incoming_vertices, pro_blocks, deleted_cocycles[i],
                            hom_graph, hom_spaces, batch_transforms, base_change, external_incoming_vertices, component_transforms,
                            block_map, row_map, N_map); 
                        #if DETAILS
                            for(index i = 0; i < local_pierced_blocks.size(); i++){
                                for(index j = 0; j < k; j++){
                                    N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                                }
                            }
                        #endif
                    } else {
                        #if DETAILS
                        std::cout << "      0 cocyle at " << b << " and " << pro_block << std::endl;
                        #endif
                    }
                }
                // Now recompute allowed row-operations for the component.  
                
                for(index i = 0; i < component_blocks.size(); i++){
                    index& b = component_blocks[i];
                    bitset& b_non_zero_columns = component_non_zero_columns[i];
                    bitset& b_cocycles = deleted_cocycles[i];
                    vec<index> sources = incoming_edges<index>(component_graph, i);
                    for(index j : sources){
                        index& c = component_blocks[j];
                        bitset& c_non_zero_columns = component_non_zero_columns[j];
                        bitset& c_cocycles = deleted_cocycles[j];
                        // This should compute if the effect of a row operation on the cocycles can be reverted by column operations.
                        // component_transforms[{i,j}] = compute_internal_row_ops(b_cocycles, c_cocycles);
                        //TO-DO: Implement this.
                    }
                }
            } else {
                assert(false);
                // Right now, I dont think I want this to happen.
                // If the graph on the processed blocks has no cycles this should be done block-by-block, otherwise:
                // TO-DO: Determine the the automorphism group of the blocks when extended with their new columns in
                // component_non_zero_columns as a subgroup of GL_comp_blocks.size ( F_2 )
                // and iterate over these ? 
                // -> Maybe call exhaustive decomposition instead.
            }
        }
        // Virtually merge the current blocks with the processed virtual blocks based on deleted cocycles.
        if (component_blocks.size() == 1){
            assert(deleted_cocycles.size() == 1);
            assert(component_non_zero_columns.size() == 1);
            Merge_data new_pro_block = {component_blocks, component_non_zero_columns[0]};
            for(index j = deleted_cocycles[0].size()-1; j > -1 ; j--){
                if(deleted_cocycles[0][j]){
                    merge_virtual_blocks(new_pro_block, pro_blocks[j]);
                    pro_blocks.erase(pro_blocks.begin() + j);
                }
            }
            pro_blocks.push_back(new_pro_block);
        } else {
            vec<Merge_data> new_pro_blocks = vec<Merge_data>();
            for(index i = 0; i < component_blocks.size(); i++){
                new_pro_blocks.push_back({{component_blocks[i]}, component_non_zero_columns[i]});
            }
            assert(new_pro_blocks.size() == component_blocks.size());
            for(index j = pro_blocks.size() -1 ; j > -1; j--){
                assert(deleted_cocycles.size() == new_pro_blocks.size());
                index first_false = -1;
                vec<index> component_block_merges = vec<index>();
                for(index b = 0; b < new_pro_blocks.size(); b++){
                    if(deleted_cocycles[b][j]){
                        if(first_false == -1){
                            first_false = b;
                            merge_virtual_blocks(new_pro_blocks[b], pro_blocks[j]);
                            pro_blocks.erase(pro_blocks.begin() + j);
                        } else {
                            component_block_merges.push_back(b);
                        }
                    }
                }
                if (!component_block_merges.empty()){
                    // We can merge the blocks in the component and the bitsets indicating which of the cocycles have been deleted.
                    for(auto it = component_block_merges.rbegin(); it != component_block_merges.rend(); it++){
                        merge_virtual_blocks(new_pro_blocks[first_false], new_pro_blocks[*it]);
                        new_pro_blocks.erase(new_pro_blocks.begin() + *it); // Is this a problem? will the iterator become invalid?
                        deleted_cocycles[first_false] |= deleted_cocycles[*it];
                        deleted_cocycles.erase(deleted_cocycles.begin() + *it);
                    }
                }
            }
            pro_blocks.insert(pro_blocks.end(), new_pro_blocks.begin(), new_pro_blocks.end());  
        }
        // Update the processed graph. Right now this is very non-optimised.   
        for(auto it = pro_blocks.rbegin(); it != pro_blocks.rend(); it++){
            for( auto it2 = pro_blocks.rbegin(); it2 != pro_blocks.rend(); it2++){
                if( it != it2){
                if( batch_transforms.find({*it, *it2}) == batch_transforms.end() ){
                    batch_transforms[{*it, *it2}] = compute_internal_col_ops(*it, *it2, N_map, hom_spaces, row_map, k, block_map);
                }
                }
            }
        }
        pro_graph = construct_batch_transform_graph(batch_transforms, pro_blocks);
        vec<index> component = vec<index>(boost::num_vertices(pro_graph));
        pro_condensation = compute_scc_and_condensation(pro_graph, component, pro_scc);
        pro_computation_order = compute_topological_order<index>(pro_condensation); 

    }
    return pro_blocks;
}


/**
 * @brief Check for directed cycles
 * 
 * @param pierced_blocks 
 * @param scc 
 * @param has_cycle 
 * @param has_unresolvable_cycle 
 * @param has_multiple_cycles 
 */
void get_cycle_information(vec<index>& pierced_blocks, vec<vec<index>>& scc, vec<Block_iterator>& block_map,
    bool& has_cycle,
    bool& has_unresolvable_cycle,
    bool& has_multiple_cycles,
    vec<bool>& is_resolvable_cycle){

    for(index i = 0; i < scc.size(); i++){
        if(scc[i].size() > 1){
            if(has_cycle){
                has_multiple_cycles = true;
                #if DETAILS
                    std::cout << "Multiple cycles detected." << std::endl;
                #endif
                break;
            }
            has_cycle = true;
            #if DETAILS
            std::cout << "Blocks in cycle: ";
            #endif
            for(index vertex_number : scc[i]){
                index block_index = pierced_blocks[vertex_number];
                #if DETAILS
                std::cout << block_index << " ";
                #endif
                if(block_map[block_index]->type == BlockType::NON_INT){
                    has_unresolvable_cycle = true;
                    is_resolvable_cycle[i] = false;
                    break;
                }
            }
            #if DETAILS
            std::cout << std::endl;
            #endif
        }
    }
}

void reduce_hom_alpha_graph(Hom_map& hom_spaces, vec<index>& local_pierced_blocks, 
    Graph& hom_digraph, r2degree& alpha, vec<Block_iterator>& block_map) {

    auto edges = boost::edges(hom_digraph);
    std::vector<std::pair<Graph::vertex_descriptor, Graph::vertex_descriptor>> edges_to_remove;
    for (auto edge_it = edges.first; edge_it != edges.second; ++edge_it) {
        // Get the source and target vertices of the edge
        auto source_vertex = boost::source(*edge_it, hom_digraph);
        auto target_vertex = boost::target(*edge_it, hom_digraph);

        index c = local_pierced_blocks[source_vertex];
        index b = local_pierced_blocks[target_vertex];

        // Access the blocks C and B
        Block& C = *block_map[c];
        Block& B = *block_map[b];

        // Perform the steps already present
        if (B.local_basislift_indices.empty()) {
            B.compute_local_basislift(alpha);
        }
        if (C.local_basislift_indices.empty()) {
            C.compute_local_basislift(alpha);
        }
        if (B.local_cokernel == nullptr) {
            B.compute_local_cokernel();
        }

        bool is_zero = hom_quotient_zero(hom_spaces[{c,b}] , *B.local_cokernel, C.local_basislift_indices, C.local_admissible_rows, B.local_admissible_rows, C.rows);
        
        if (is_zero) {
            #if DETAILS
                std::cout << "  alpha-reduction: Deleted " << c << " to " << b << std::endl;
            #endif
            hom_spaces[{c,b}].first.data.clear();
            hom_spaces[{c,b}].first.set_num_cols(0);
            edges_to_remove.emplace_back(source_vertex, target_vertex);
        }
    }

    for (const auto& edge : edges_to_remove) {
        boost::remove_edge(edge.first, edge.second, hom_digraph);
    }
}


/**
 * @brief Given a list of blocks and all hom_spaces, constructs the digraph on the blocks 
 * where there is a directed edge from block i to block j if Hom(B_i, B_j) != 0, as well as
 * a condensation of this graph and a topological order on the condensation.
 * 
 * @param hom_digraph 
 * @param component 
 * @param scc 
 * @param condensation 
 * @param computation_order 
 * @param pierced_blocks 
 * @param hom_spaces 
 * @param cyclic_counter 
 * @param resolvable_cyclic_counter 
 * @param acyclic_counter 
 * @return false if there are unresolvable cycles 
 */
bool construct_graphs_from_hom(Graph& hom_digraph, std::vector<index>& component, vec<vec<index>>& scc, vec<bool>& is_resolvable_cycle,
        Graph& condensation, vec<index>& computation_order, vec<index>& pierced_blocks, Hom_map& hom_spaces, 
        AIDA_runtime_statistics& statistics, AIDA_config& config, index& t, vec<Block_iterator>& block_map, r2degree& alpha){
    
    // Graph on pierced blocks representing Hom(C,B) != 0 
    hom_digraph = construct_hom_digraph(hom_spaces, pierced_blocks);
    
    bool test_alpha_cycles = true;

    bool test_has_cycle = false;
    bool test_has_unresolvable_cycle = false;
    bool test_has_multiple_cycles = false;
    vec<bool> test_is_resolvable_cycle;

    if(test_alpha_cycles){
        std::vector<index> test_component = vec<index>(boost::num_vertices(hom_digraph));
        vec<vec<index>> test_scc;
        Graph test_condensation = compute_scc_and_condensation(hom_digraph, test_component, test_scc);
        test_is_resolvable_cycle = vec<bool>(test_scc.size(), true);
        get_cycle_information(pierced_blocks, test_scc, block_map, test_has_cycle, test_has_unresolvable_cycle, test_has_multiple_cycles, test_is_resolvable_cycle);
    }

    if(config.alpha_hom){
      //  reduce_hom_alpha_graph(hom_spaces, pierced_blocks, hom_digraph, alpha, block_map);
    }

    // Components assigns to each block the index of the strongly connected component it is in.
    component = vec<index>(boost::num_vertices(hom_digraph));
    // SCC is a vec< set<index> >, where each set contains the indices of the blocks in the SCC., condensation is a graph on the SCCs.
    condensation = compute_scc_and_condensation(hom_digraph, component, scc);
    // Contains the order in which to process the SCCs in reverse
    computation_order = compute_topological_order<index>(condensation); 
    
    is_resolvable_cycle = vec<bool>(scc.size(), true);

    #if DETAILS
        print_graph(hom_digraph);
        std::cout << "Component " <<  component << std::endl;
        std::cout << "SCCs " << scc << std::endl;
        print_graph(condensation);
        std::cout << "Computation order " << computation_order << std::endl;
    #endif


    // Check if there are any cycles in the hom-digraph by checking if there is a strongly connected component of size > 1
    bool has_cycle = false;
    bool has_unresolvable_cycle = false;
    bool has_multiple_cycles = false;

    get_cycle_information(pierced_blocks, scc, block_map, has_cycle, has_unresolvable_cycle, has_multiple_cycles, is_resolvable_cycle);

    if(test_alpha_cycles){
        if(test_has_unresolvable_cycle && !has_unresolvable_cycle){
            statistics.alpha_cycle_avoidance ++;
        } else if (test_has_multiple_cycles && !has_multiple_cycles){
            // record this?
        } else if (test_has_cycle && !has_cycle){
            // record this?
        }
    }

    #if DETAILS
        std::cout << "Batch " << t << " is ";
    #endif
    if(has_unresolvable_cycle){
        statistics.cyclic_counter++;
        #if DETAILS
            std::cout << "un-resolvable cyclic." << std::endl;
        #endif
    } else if(has_multiple_cycles){
        #if DETAILS
            std::cout << "polycyclic." << std::endl;
        #endif
    } else if(has_cycle) {
        statistics.resolvable_cyclic_counter++;
        #if DETAILS
            std::cout << "resolvable cyclic." << std::endl;
        #endif
    } else {
        statistics.acyclic_counter++;
        #if DETAILS
            std::cout << "acyclic." << std::endl;
        #endif
    }
    return !has_unresolvable_cycle && !has_multiple_cycles;
}

/**
 * @brief Groups the blocks so that every group occupies a seperate subset of the columns.
 * 
 * @param N_map 
 * @param blocks 
 * @param support 
 * @return vec<Merge_data>
 */
vec<Merge_data> find_prelim_decomposition(Sub_batch& N_map, const vec<index>& blocks, const bitset& support){
    index k = support.size();
    assert(k == N_map[blocks[0]].get_num_cols());
    vec<Merge_data> block_to_columns; 
    for(index i = 0; i< blocks.size(); i++){
        index b = blocks[i];
        block_to_columns.push_back({{b}, bitset(k, 0)});
        for(index col = support.find_first(); col != bitset::npos ; col = support.find_next(col)){
            if(N_map[b].col_last(col) != -1){
                block_to_columns[i].second.set(col);
            }
        }
    }
    for(index col = support.find_first(); col!= bitset::npos; col = support.find_next(col)){
        auto first_occurence = block_to_columns.end();
        for(auto it = block_to_columns.begin(); it != block_to_columns.end();){
            if((*it).second.test(col)){
                if(first_occurence == block_to_columns.end()){
                    first_occurence = it;
                    it++;
                } else {
                    // Merge the two blocks
                    (*first_occurence).first.insert((*first_occurence).first.end(), (*it).first.begin(), (*it).first.end() );
                    (*first_occurence).second |= (*it).second;
                    it = block_to_columns.erase(it);
                }
            } else {
                it++;
            }
        }       
    }
    return block_to_columns;
}



/**
 * @brief Decomposes the matrix A into a direct sum of indecomposable submatrices.
 * 
 * @param A 
 * @param B_list 
 * @param base_change 
 * @param statistics 
 * @param config
 * @param merge_info 
 */
void AIDA(GradedMatrix& A, Block_list& B_list, vec<vec<transition>>& vector_space_decompositions, std::shared_ptr<Base_change_virtual>& base_change, AIDA_runtime_statistics& statistics,
    AIDA_config& config, Full_merge_info& merge_info) {
    
    #if TIMERS
        aida::full_timer.start();
        aida::misc_timer.start();
    #endif

    index batches = A.col_batches.size();
    

    // Only for analysis and optimisation ->
    

    #if OBSERVE
        // Continuous monitoring of the content of a batch
        observe_row_indices = vec<index>();
        Degree_traits<r2degree> degree_traits;
        std::cout << "Observing batch " << observe_batch_index << " with columns " << A.col_batches[observe_batch_index] << " at " << std::endl;
        degree_traits.print_degree(A.col_degrees[A.col_batches[observe_batch_index][0]]);
        std::cout << " and with content:" << std::endl;
        Sparse_Matrix observed_batch_comparison = A.restricted_domain_copy(A.col_batches[observe_batch_index]);
        observed_batch_comparison.print();
        // Save indices to see where the first change to this batch occurs.
        for(vec<index> column : observed_batch_comparison.data){
            observe_row_indices.insert(observe_row_indices.end(), column.begin(), column.end());
        }
    #endif

    vec<index> row_map(A.get_num_rows(), 0);
    vec<Block_iterator> block_map;
    #if TIMERS
        misc_timer.stop();
        update_block_timer.resume();
    #endif
    initialise_block_list(A, B_list, block_map);
    if(batches == 0){
        if(config.show_info){
            std::cout << "The entered matrix has no columns and is thus trivially decomposable w.r.t. any basis." << std::endl;   
        }
        return;
    }
    #if TIMERS
        misc_timer.resume();
        update_block_timer.stop();
    #endif

    Sub_batch N_map;
    Hom_map hom_spaces; // Stores the hom-spaces for each pair of initials (c, b) where necessary
    std::unordered_map< index, vec<index>> domain_keys; // For some c, stores the indices b for which hom_spaces has a key (c, b).
    std::unordered_map< index, vec<index>> codomain_keys; // For some b, stores the indices c for which hom_spaces has a key (c, b).    
    vec<bitset> e_vec = compute_standard_vectors(A.k_max);
    vec<bitset> count_vector = compute_sum_of_standard_vectors(A.k_max);

    #if TIMERS
        misc_timer.stop();
        compute_rows_timer.resume();
    #endif
    A.compute_revrows();
    #if TIMERS
        misc_timer.resume();
        compute_rows_timer.stop();
    #endif

    for(index t = 0; t < batches; t++){
        #if !DETAILS
            if (config.progress) {
                static index last_percent = -1;
                // (-)^{1.5} progress bar for now, but not clear that computational time increases with this exponent.
                index percent = static_cast<index>(pow(static_cast<double>(t + 1) / batches, 1.5) * 100);
                if (percent != last_percent) {
                    // Calculate the number of symbols to display in the progress bar
                    int num_symbols = percent / 2;
                    std::cout << "\r" << t + 1 << " of " << batches << " batches : [";
                    // Print the progress bar
                    for (int i = 0; i < 50; ++i) {
                        if (i < num_symbols) {
                            std::cout << "#";
                        } else {
                            std::cout << " ";
                        }
                    }
                    std::cout << "] " << percent << "%";
                    std::flush(std::cout);
                    last_percent = percent;
                }
                if (t == batches - 1) {
                    std::cout << std::endl;
                }
            }
        #endif

        bool one_block_left = false;
        vec<index> batch_indices = A.col_batches[t]; // Indices of the columns in the batch
        int k_ = batch_indices.size(); // Number of columns in the batch
        // TO-DO: Have seen rounding errors here, investigate.
        r2degree alpha = A.col_degrees[batch_indices[0]]; // r2degree of the batch
        
        #if DETAILS
            std::cout << "Processing batch " << t << " with " << k_ << " columns at the indices " << batch_indices <<  std::endl;
        #endif
        N_map.clear(); bool no_further_comp = false; vec<Merge_data> block_partition = {}; indtree active_blocks;
        
        // Get the batch as a set of columns from the rows and identify the blocks which need to be processed
        #if TIMERS
            misc_timer.stop();
            compute_N_timer.resume();
        #endif
        compute_touched_blocks(active_blocks, block_map, A, batch_indices, N_map); 

        #if OBSERVE
        if( t == observe_batch_index ){
            std::cout << "Analysing batch " << t << " at " << batch_indices << " - Printing B and N:" << std::endl;
            for(index b : active_blocks){
                std::cout << "Block " << b << ": " << std::endl;
                block_map[b]->print();
                std::cout << "N[" << b << "]: " << std::endl;
                N_map[b].print();
            }
        }
        #endif
        #if TIMERS
            misc_timer.resume();
            compute_N_timer.stop();
        #endif
        #if DETAILS
             std::cout << "  !! There are "  << active_blocks.size() << " touched blocks with the following indices: ";
            for(index i : active_blocks){
                std::cout << i << " ";
            }
            std::cout << std::endl;
        #endif
        // First try to delete every whole sub-batch only with column operations. That is, compute the *affected* blocks.
        if(active_blocks.size() != 1) {
            for(auto it = active_blocks.begin(); it != active_blocks.end();){
                index j = *it;
                Block& B = *block_map[j];
                // No need to do anything here if the block is empty. 
                    if(B.columns.size() == 0){
                        it++;
                        B.local_data = std::make_shared<Sparse_Matrix>(0,0);
                        N_map[j].compute_rows_forward_map(row_map, B.rows.size());
                        continue;}
                auto& N = N_map[j];
                #if TIMERS
                    delete_with_col_timer.resume();
                    misc_timer.stop();
                #endif 
                bool only_col_ops = B.delete_with_col_ops(alpha, N, config.supress_col_sweep);
                #if TIMERS
                    delete_with_col_timer.stop();
                    misc_timer.resume();
                #endif 
                if(only_col_ops){
                    #if DETAILS
                        std::cout << "      Deleted N at index " << j << " with column ops." << std::endl;
                    #endif
                    statistics.counter_col_deletion++;
                    B.delete_local_data();
                    N_map.erase(j);
                    it = active_blocks.erase(it);
                } else {
                    it++;
                    // If the block could not be deleted with column operations, then it is still active.
                    // We will need its row-information later.
                    #if TIMERS
                    misc_timer.stop();
                    compute_rows_timer.resume();
                    #endif 
                    B.compute_rows(row_map); 
                    N_map[j].compute_rows_forward_map(row_map, B.rows.size());
                    #if TIMERS
                    compute_rows_timer.stop();
                    misc_timer.resume();
                    #endif
                }
            }
        } else {
            one_block_left = true;
            statistics.counter_no_comp++;
        }
        
        // Next try to delete every whole sub-batch also with the help of row operations.
        // To find all row-operations needed, we need to compute the hom-spaces between the blocks.

        if(active_blocks.size() != 1){ 
            assert(active_blocks.size() > 0);
            #if OBSERVE
                for(index r : observe_row_indices){
                    if( active_blocks.find(block_map[r]->rows[0]) != active_blocks.end() ){
                        std::cout << "Row index " << r << " belongs to block " << block_map[r]->rows[0] << std::endl;
                    }
                }
            #endif
            #if DETAILS
                std::cout << "   !! There are "  << active_blocks.size() << " affected blocks with the following row indices: ";
                for(index i : active_blocks){
                    std::cout << " " << block_map[i]->get_type() << " ";
                    std::cout << (*block_map[i]).rows << " - ";
                }
                std::cout << std::endl;
                for(index i : active_blocks){
                    std::cout << " N[" << i << "]: ";
                    N_map[i].print_rows();
                }
            #endif

            
            // Then delete with previously computed hom-spaces.
            // We want to start with the blocks of lowest r2degree, because the spaces of homomorphisms with these codomains are smaller.

            for(auto itr = active_blocks.rbegin(); itr != active_blocks.rend();) {
                index b = *itr; vec<index> b_vec = {b}; 
                // This is ugly. Need to find a solution for the set / vector problem.
                vec<index> active_blocks_vec(active_blocks.begin(), active_blocks.end());

                if( !config.brute_force ){
                    compute_hom_to_b(A, b, block_map, active_blocks, hom_spaces, domain_keys, codomain_keys, alpha, statistics, config);
                }

                #if TIMERS
                    full_block_reduce_timer.start();
                #endif
                bool deleted_N_b = use_either_block_reduce_full_support(A, b_vec, N_map, batch_indices, false, 
                    active_blocks_vec, block_map, base_change, row_map, hom_spaces, config.brute_force, config.compare_both);

                #if TIMERS
                    full_block_reduce_timer.stop();
                    dispose_S_timer.stop();
                    misc_timer.resume();
                #endif
                
                if( deleted_N_b){
                    #if DETAILS
                        std::cout << "      Deleted N at index " << b << " with row ops." << std::endl;
                    #endif
                    statistics.counter_row_deletion++;
                    block_map[b]->delete_local_data();
                    N_map.erase(b);
                    auto it = active_blocks.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it);
                } else {
                    // It is imporant to have the local data be completely reduced 
                    // ( that is, have a canonical representation in the quotient space )
                    bool further_reduce = block_map[b] -> reduce_N_fully(N_map[b], false);
                    
                    #if TIMERS
                        misc_timer.stop();
                        compute_rows_timer.resume();
                    #endif               
                    N_map[b].compute_rows_forward_map(row_map, block_map[b]->rows.size());
                    #if TIMERS
                        misc_timer.resume();
                        compute_rows_timer.stop();
                    #endif
                    assert(further_reduce == false);
                    itr++;
                }
            }
        } else {
            if(!one_block_left){
                statistics.counter_only_col++;
                one_block_left = true;
            }
        }

        if( k_ != 1 &&  active_blocks.size() != 1) {
            assert(active_blocks.size() > 0);
            statistics.num_of_pierced_blocks.push_back(active_blocks.size());
            #if DETAILS
                std::cout << "There are "  << active_blocks.size() << " pierced blocks with the following row indices: ";
                for(index i : active_blocks){
                    std::cout << " " << block_map[i]->get_type() << " ";
                    std::cout << (*block_map[i]).rows << " - ";
                }
                std::cout << std::endl;
                for(index i : active_blocks){
                    std::cout << " N[" << i << "]: ";
                    N_map[i].print_rows();
                }
                std::cout << std::endl;
            #endif
            vec<index> pierced_blocks(active_blocks.begin(), active_blocks.end());

            #if TIMERS
                pre_alpha_decomp_optimisation_timer.resume();
                misc_timer.stop();
            #endif
            bitset non_zero_indices = simultaneous_column_reduction_full_support(N_map, pierced_blocks, pierced_blocks);
            if(non_zero_indices.count() < k_){
                std::cout << "Either the input file was not minimised correctly or there is a bug in the algorithm." << std::endl;
                #if DETAILS
                for(index b: pierced_blocks){
                    std::cout << "Block and N for " << b << std::endl;
                    block_map[b]->print();
                    N_map[b].print_rows();
                }
                #endif

                assert(false);
            }
            // TO-DO: Check how decomposed the matrix already is and pass the pieces to the next algorithm:
            
            vec<Merge_data> prelim_decomposition = find_prelim_decomposition(N_map, pierced_blocks, count_vector[k_-1]);
            // Ordering is not strictly necessary, but some parts might (will!) not work because they have been coded in a way which assumes it.
            for(Merge_data& merge : prelim_decomposition){
                std::sort(merge.first.begin(), merge.first.end());
            }
            #if DETAILS
                std::cout << "Prelim decomposition: " << std::endl;
                for(auto& pair : prelim_decomposition){
                    std::cout << "  " << pair.first << " -> " << pair.second << std::endl;
                }
            #endif

            #if TIMERS
                pre_alpha_decomp_optimisation_timer.stop();
                alpha_decomp_timer.resume();
            #endif
            for( Merge_data& pair : prelim_decomposition){
                vec<index>& local_pierced_blocks = pair.first;
                bitset& N_column_indices = pair.second;
                if(N_column_indices.count() > statistics.local_k_max){
                    statistics.local_k_max = N_column_indices.count();
                }
                #if OBSERVE
                    if(std::find(local_pierced_blocks.begin(), local_pierced_blocks.end(), observe_block_index) != local_pierced_blocks.end()){
                        
                        std::cout << " Found " << observe_block_index << " in local_pierced_blocks." << std::endl;
                        std::cout << "  Pierced blocks: " << active_blocks << std::endl;
                        std::cout << "  Local pierced blocks: " << local_pierced_blocks << std::endl;
                        std::cout << "  Batch indices: " << batch_indices << std::endl;
                    }
                #endif

                if(local_pierced_blocks.size() == 1 || N_column_indices.count() == 1){
                    block_partition.push_back(pair);
                } else {
                    Graph hom_digraph; std::vector<index> component; vec<vec<index>> scc; Graph condensation;
                    vec<bool> is_resolvable_cycle;
                    vec<index> computation_order;
                    
                    bool is_resolvable = construct_graphs_from_hom(hom_digraph, component, scc, is_resolvable_cycle, condensation, computation_order, 
                            local_pierced_blocks, hom_spaces, statistics, config, t, block_map, alpha);

                    vec<Merge_data> result;
                    if(is_resolvable && !config.exhaustive && !config.brute_force && !config.compare_both){
                        #if TIMERS
                            full_aida_timer.resume();
                        #endif
                        result = automorphism_sensitive_alpha_decomp(A, B_list, block_map, local_pierced_blocks, batch_indices, N_column_indices, 
                            e_vec, N_map, vector_space_decompositions, config, base_change, hom_spaces, row_map, hom_digraph,
                            computation_order, scc, condensation, is_resolvable_cycle);
                        #if TIMERS
                            full_aida_timer.stop();
                        #endif
                    } else {
                        #if TIMERS
                            full_exhaustive_timer.resume();
                        #endif
                        result = exhaustive_alpha_decomp(A, B_list, block_map, local_pierced_blocks, batch_indices, N_column_indices, 
                            e_vec, N_map, vector_space_decompositions, config, base_change,
                            hom_spaces, row_map, config.brute_force, config.compare_both);
                        #if TIMERS
                            full_exhaustive_timer.stop();
                        #endif
                        if(result.size() == 1){
                            statistics.counter_naive_full_iteration++;
                        }
                        for(auto& merge : result){
                            if(merge.second.count() > 1){
                                statistics.counter_extra_iterations += statistics.num_subspace_iterations[merge.second.count()-1];
                            }
                        }
                        statistics.counter_naive_deletion += result.size()-1;
                    }
                    block_partition.insert(block_partition.end(), std::make_move_iterator(result.begin()), std::make_move_iterator(result.end()));
                }
            }
            #if TIMERS
                alpha_decomp_timer.stop();
                misc_timer.resume();
            #endif
        } else {
            if(!one_block_left){
                statistics.counter_only_row++;
                one_block_left = true;
            }
            // If at some point active_blocks contains only one block OR k == 1, we should eventually land here, thus:
            block_partition = {{vec<index>(active_blocks.begin(), active_blocks.end()) , count_vector[k_-1]}};
        }
        // Deleting remaining local data.
        for(index i : active_blocks){
            block_map[i]->delete_local_data();
        }
        #if TIMERS
            update_block_timer.resume();
            misc_timer.stop();
        #endif
        

        merge_blocks(B_list, N_map, block_map, block_partition, batch_indices, row_map, alpha);
        #if TIMERS
            update_block_timer.stop();
            misc_timer.resume();
        #endif
        // If a block has changed, we delete its hom space. TO-DO: if the change is small, we could update them instead.
        if(!config.brute_force){
            #if TIMERS
                misc_timer.stop();  
                update_hom_timer.resume();
            #endif
            update_hom_spaces(block_partition, hom_spaces, domain_keys, codomain_keys);
            #if TIMERS
                update_hom_timer.stop();
                misc_timer.resume();  
            #endif
        }
        merge_info.push_back(block_partition);

        #if OBSERVE
        Sparse_Matrix observed_batch = A.restricted_domain_copy(A.col_batches[observe_batch_index]);
        index changed_relation = observed_batch.equals_with_entry_check(observed_batch_comparison, true);
        if( changed_relation > -1 ){
            std::cout << " Observed batch was altered in batch " << t << " at " << A.col_batches[t][changed_relation] << std::endl;
            vec<index> sorted_current = observed_batch.data[changed_relation];
            vec<index> sorted_old = observed_batch_comparison.data[changed_relation];
            std::sort(sorted_current.begin(), sorted_current.end());   
            std::sort(sorted_old.begin(), sorted_old.end());
            convert_mod_2(sorted_current);
            convert_mod_2(sorted_old);
            vec<index> diff = sorted_current + sorted_old; 
            std::cout << "  First difference: " << diff << std::endl;
            std::cout << "  Before: " << std::endl;
            observed_batch_comparison.print();
            std::cout << "  After: " << std::endl;
            observed_batch.print();
            std::cout << " Merge data: " << std::endl;
            for(auto& merge : block_partition){
                std::cout << "  " << merge.first << " -> " << merge.second << std::endl;
            }
            observed_batch_comparison = SparseMatrix(observed_batch);
        }
        #endif

    } 

    // Normalise the blocks

    for(auto it = B_list.begin(); it != B_list.end(); it++){
        Block& B = *it;
        B.transform_data(row_map);
    }

    #if TIMERS
        full_timer.stop();
        misc_timer.stop();
    #endif
    
    /** 
    #if DETAILS
        std::cout << "Full merge details: " << std::endl;
        for(index i = 0; i < merge_info.size(); i++){
            std::cout << "  #Merges at batch " << i << ": ";
            for(auto& merge : merge_info[i]){
                std::cout << merge.first << " ";
            }
            std::cout << std::endl;
        }
    #endif
    */
} //AIDA


/**
 * @brief Compares the sets of merges at each batch. The first input is assumed to be the stable one.
 * 
 * @param merge_info_1 
 * @param merge_info_2 
 */
void compare_merge_info(Full_merge_info& merge_info_1, Full_merge_info& merge_info_2){
    assert(merge_info_1.size() == merge_info_2.size());
    bool success = true;
    for(index i = 0; i < merge_info_1.size(); i++){
        auto& merge_vec_1 = merge_info_1[i];
        auto& merge_vec_2 = merge_info_2[i];

        for(Merge_data& merge : merge_vec_1){
            std::sort(merge.first.begin(), merge.first.end());
            assert(is_sorted(merge.first));
        }
        for(Merge_data& merge : merge_vec_2){
            std::sort(merge.first.begin(), merge.first.end());
            assert(is_sorted(merge.first));
        }
        std::sort(merge_vec_1.begin(), merge_vec_1.end(), comparator);
        std::sort(merge_vec_2.begin(), merge_vec_2.end(), comparator);

        if(merge_vec_1.size() != merge_vec_2.size()){
            success = false;
            std::cout << "Different number of merges at batch " << i << std::endl;
            for(auto& merge : merge_vec_1){
                std::cout << "(" << merge.first << ") ";
            }
            std::cout << std::endl;
            std::cout << " vs " << std::endl;
            for(auto& merge : merge_vec_2){
                std::cout << "(" << merge.first << ") ";
            }
            std::cout << std::endl;
        } else {
            
            for(index j = 0; j < merge_vec_1.size(); j++){
                Merge_data& merge_1 = merge_vec_1[j];
                Merge_data& merge_2 = merge_vec_2[j];
                if(merge_1.first.size() != merge_2.first.size()){
                    success = false;
                    std::cout << "Different number of blocks in merge at batch " << i << std::endl;
                    for(auto& block : merge_1.first){
                        std::cout << block << " ";
                    }
                    std::cout << std::endl;
                    std::cout << " vs " << std::endl;
                    for(auto& block : merge_2.first){
                        std::cout << block << " ";
                    }
                    std::cout << std::endl;
                } else {
                    /** 
                    auto it1 = merge_1.first.begin();
                    auto it2 = merge_2.first.begin();
                    for(; it1 != merge_1.first.end(); it1++, it2++){
                        if(*it1 != *it2){
                            std::cout << "Different blocks in merge at batch " << i << std::endl;
                            for(auto& block : merge_1.first){
                                std::cout << block << " ";
                            }
                            std::cout << std::endl;
                            std::cout << " vs " << std::endl;
                            for(auto& block : merge_2.first){
                                std::cout << block << " ";
                            }
                            std::cout << std::endl;
                        }
                    }
                    */
                }
            }
        }
    }
    if(success){
        std::cout << "The number of blocks at each merge point is the same." << std::endl;
    }
} // Compare_merge_info


} // namespace aida



#endif // AIDA_HPP
