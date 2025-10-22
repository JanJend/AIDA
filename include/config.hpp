/**
 * @file config.hpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Configuration options for AIDA library
 */

#pragma once

#ifndef AIDA_CONFIG_HPP
#define AIDA_CONFIG_HPP

// Do not touch this - used to enable/disable timers with cmake option AIDA_WITH_STATS
#ifndef AIDA_WITH_STATS
#define TIMERS 1
#else
#if AIDA_WITH_STATS
#define TIMERS 1
#else
#define TIMERS 0
#endif
#endif


// Helper functions for statistics
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
    
    AIDA_config(bool supress_col_sweep = false, bool sort_output = false, bool sort = false, bool save_base_change = false, bool exhaustive = false, bool brute_force = false, bool progress = false, bool compare_both = false, bool turn_off_hom_optimisation = false, bool show_info = true, bool exhaustive_test = false, bool compare_hom = false, bool alpha_hom = true)
        : supress_col_sweep(supress_col_sweep), save_base_change(save_base_change), sort_output(sort_output), sort(sort), exhaustive(exhaustive), brute_force(brute_force), compare_both(compare_both), progress(progress), turn_off_hom_optimisation(turn_off_hom_optimisation), show_info(show_info), exhaustive_test(exhaustive_test), compare_hom(compare_hom) { 
            decomp_failure = vec<vec<index>>();
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


#endif // AIDA_CONFIG_HPP