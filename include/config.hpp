/**
 * @file config.hpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Configuration options for AIDA library
  * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#ifndef AIDA_CONFIG_HPP
#define AIDA_CONFIG_HPP

#include "types.hpp"
#include "block.hpp"

namespace aida{

#define DETAILS 0 // For debugging 
#define OBSERVE 0 // For debugging
#define CHECK_INT 0 // obsolete?
#define SYSTEM_SIZE 0 // (should) tracks sizes of linear systems solved to compare the "actual" computation time of the different algorithms without overhead.

// For debugging purposes
#if OBSERVE
    index observe_batch_index = 193;
    vec<index> observe_row_indices;
    index observe_block_index = 51;
#endif

// Do not touch this - used to enable/disable timers with cmake option AIDA_WITH_STATS
#ifndef AIDA_WITH_STATS
#define TIMERS 0
#else
#if AIDA_WITH_STATS
#define TIMERS 1
#else
#define TIMERS 0
#endif
#endif


// Helper functions for statistics

#if TIMERS
    extern cpu_timer hom_space_timer;
    extern cpu_timer hom_space_test_timer;
    extern cpu_timer full_timer;
    extern cpu_timer constructing_linear_system_timer;
    extern cpu_timer solve_linear_system_timer;
    extern cpu_timer dispose_S_timer;
    extern cpu_timer update_matrix_timer;
    extern cpu_timer update_hom_timer;
    extern cpu_timer load_matrices_timer;
    extern cpu_timer compute_N_timer;
    extern cpu_timer delete_with_col_timer;
    extern cpu_timer misc_timer;
    extern cpu_timer update_block_timer;
    extern cpu_timer compute_rows_timer;
    extern cpu_timer pre_alpha_decomp_optimisation_timer;
    extern cpu_timer alpha_decomp_timer;
    extern cpu_timer full_aida_timer;
    extern cpu_timer full_exhaustive_timer;
    extern cpu_timer full_block_reduce_timer;
#endif

double calculateAverage(const vec<index>& values);
double calculateMedian(vec<index> values);


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
    
    AIDA_config(bool supress_col_sweep = false, bool sort_output = false, 
        bool sort = false, bool save_base_change = false, 
        bool exhaustive = false, bool brute_force = false, 
        bool progress = false, bool compare_both = false, 
        bool turn_off_hom_optimisation = false, 
        bool show_info = false, bool exhaustive_test = false, 
        bool compare_hom = false, bool alpha_hom = false)
        : supress_col_sweep(supress_col_sweep), save_base_change(save_base_change), sort_output(sort_output), sort(sort), exhaustive(exhaustive), brute_force(brute_force), compare_both(compare_both), progress(progress), turn_off_hom_optimisation(turn_off_hom_optimisation), show_info(show_info), exhaustive_test(exhaustive_test), compare_hom(compare_hom) { 
            decomp_failure = vec<vec<int>>();
        }

};


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
 * @brief Computes and processes statistics about indecomposables
 */
struct AIDA_statistics {
    index total_num_rows;
    index num_of_summands;
    index num_of_free;
    index num_of_cyclic;
    index num_of_intervals;
    index num_of_non_intervals;
    index gen_max;
    index size_of_intervals;
    index size_of_non_intervals;
    double interval_ratio;
    double interval_size_ratio;

    AIDA_statistics();
    
    void compute_statistics(Block_list& B_list);
    void operator+=(const AIDA_statistics& other);
    void print_statistics();
};

/**
 * @brief Handles all statistical information gathered at runtime of the AIDA algorithm
 */
struct AIDA_runtime_statistics {
    vec<index> num_subspace_iterations;
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
        double hom_space;
        double hom_space_test;
        double constructing_linear_system;
        double solve_linear_system;
        double dispose_S;
        double update_matrix;
        double update_hom;
        double load_matrices;
        double compute_N;
        double delete_with_col;
        double misc;
        double update_block;
        double compute_rows;
        double pre_alpha_decomp_optimisation;
        double alpha_decomp;
        double full;
        double accumulated;
        double full_aida;
        double full_exhaustive;
        double full_block_reduce;
    #endif

    AIDA_runtime_statistics();
    
    void operator+=(AIDA_runtime_statistics& other);
    void print();

    #if TIMERS
        void initialise_timers();
        void evaluate_timers();
        void print_timers();
    #endif
};

}

#endif // AIDA_CONFIG_HPP