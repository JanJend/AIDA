/**
 * @file config.cpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Configuration options for AIDA library
 */

#include "config.hpp"
#include <unistd.h>
#include <climits>
#include <regex>
#include <filesystem>
#include <stdexcept>

namespace aida {

// ============================================================================
// Helper functions for statistics
// ============================================================================


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
    
double calculateAverage(const vec<index>& values) {
    if (values.empty()) return 0.0;
    return static_cast<double>(std::accumulate(values.begin(), values.end(), 0)) / values.size();
}

double calculateMedian(vec<index> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

// ============================================================================
// AIDA_statistics implementation
// ============================================================================

AIDA_statistics::AIDA_statistics() 
    : num_of_summands(0)
    , num_of_free(0)
    , num_of_cyclic(0)
    , num_of_intervals(0)
    , num_of_non_intervals(0)
    , size_of_intervals(0)
    , size_of_non_intervals(0)
    , interval_ratio(0)
    , interval_size_ratio(0)
    , total_num_rows(0)
    , gen_max(0) 
{}

void AIDA_statistics::compute_statistics(Block_list& B_list) {
    num_of_summands = B_list.size();
    for(Block& B : B_list) {
        if(B.type == BlockType::FREE) {
            num_of_free++;
            total_num_rows++;
        } else if (B.type == BlockType::CYC) {
            num_of_cyclic++;
            total_num_rows++;
        } else if (B.type == BlockType::INT) {
            num_of_intervals++;
            size_of_intervals += B.rows.size();
            total_num_rows += B.rows.size();
            if(B.rows.size() > gen_max) {
                gen_max = B.rows.size();
            }
        } else {
            size_of_non_intervals += B.rows.size();
            total_num_rows += B.rows.size();
            if(B.rows.size() > gen_max) {
                gen_max = B.rows.size();
            }
        }
    }
    interval_ratio = static_cast<double>(num_of_intervals + num_of_free + num_of_cyclic) / num_of_summands;
    interval_size_ratio = static_cast<double>(size_of_intervals + num_of_free + num_of_cyclic) / total_num_rows;
}

void AIDA_statistics::operator+=(const AIDA_statistics& other) {
    num_of_summands += other.num_of_summands;
    num_of_free += other.num_of_free;
    num_of_cyclic += other.num_of_cyclic;
    num_of_intervals += other.num_of_intervals;
    num_of_non_intervals += other.num_of_non_intervals;
    size_of_intervals += other.size_of_intervals;
    size_of_non_intervals += other.size_of_non_intervals;
    total_num_rows += other.total_num_rows;
    gen_max = std::max(gen_max, other.gen_max);
}

void AIDA_statistics::print_statistics() {
    std::cout << "Statistics for indecomposable summands: " << std::endl;
    std::cout << "  # indecomposables: " << num_of_summands << std::endl;
    std::cout << "  # free summands: " << num_of_free << std::endl;
    std::cout << "  # non-free cyclic summands: " << num_of_cyclic << std::endl;
    std::cout << "  # non-cyclic intervals: " << num_of_intervals << std::endl;
    std::cout << "  # non-intervals: " << num_of_summands - (num_of_intervals + num_of_cyclic + num_of_free) << std::endl;
    std::cout << "  # generators of non-cyclic intervals: " << size_of_intervals << std::endl;
    std::cout << "  # generators of non-intervals: " << size_of_non_intervals << std::endl;
    std::cout << "  # generators of largest (non-cyclic) indecomposable: " << gen_max << std::endl;
    std::cout << "  Ratio of intervals: " << interval_ratio << std::endl;
    std::cout << "  Ratio of generators belonging to intervals: " << interval_size_ratio << std::endl;
}

// ============================================================================
// AIDA_runtime_statistics implementation
// ============================================================================

AIDA_runtime_statistics::AIDA_runtime_statistics() 
    : num_subspace_iterations{1, 2, 7, 43, 186, 1965, 14605, 297181}
    , counter_no_comp(0)
    , counter_only_col(0)
    , counter_only_row(0)
    , counter_naive_deletion(0)
    , counter_naive_full_iteration(0)
    , counter_extra_iterations(0)
    , counter_col_deletion(0)
    , counter_row_deletion(0)
    , local_k_max(0)
    , resolvable_cyclic_counter(0)
    , cyclic_counter(0)
    , acyclic_counter(0)
    , alpha_cycle_avoidance(0)
    , dim_hom_max(0)
{
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

void AIDA_runtime_statistics::operator+=(AIDA_runtime_statistics& other) {
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

void AIDA_runtime_statistics::print() {
    std::cout << "  No computation: " << counter_no_comp << std::endl;
    std::cout << "  Only column operations: " << counter_only_col << std::endl;
    std::cout << "  Only row operations: " << counter_only_row << std::endl;
    std::cout << "  Naive deletion: " << counter_naive_deletion << std::endl;
    std::cout << "  Naive full iteration: " << counter_naive_full_iteration << std::endl;
    std::cout << "  Column Block deletions: " << counter_col_deletion << std::endl;
    std::cout << "  Row Block deletions: " << counter_row_deletion << std::endl;
    std::cout << "  Hom-spaces calculated: " << dim_hom_vec.size() << std::endl;
    std::cout << "  Total dimension of calculated hom-spaces: " << std::accumulate(dim_hom_vec.begin(), dim_hom_vec.end(), 0) << std::endl;
    std::cout << "  Maximum dimension of calculated hom-spaces: " << dim_hom_max << std::endl;
    std::cout << "  Average dimension of calculated hom-spaces: " << calculateAverage(dim_hom_vec) << std::endl;
    std::cout << "  Median dimension of calculated hom-spaces: " << calculateMedian(dim_hom_vec) << std::endl;

    if(!num_of_pierced_blocks.empty()) {
        std::cout << "  Maximum Number of pierced blocks: " << *std::max_element(num_of_pierced_blocks.begin(), num_of_pierced_blocks.end()) << std::endl;
    }
    std::cout << "  Local k_max: " << local_k_max << std::endl;
    std::cout << "  Acyclic batches: " << acyclic_counter << std::endl;
    std::cout << "  Resolvable cyclic batches: " << resolvable_cyclic_counter << std::endl;
    std::cout << "  Cyclic batches: " << cyclic_counter << std::endl;
    std::cout << "  Alpha cycle avoidance: " << alpha_cycle_avoidance << std::endl;
    std::cout << "  Extra iterations: " << counter_extra_iterations << std::endl;
}

#if TIMERS

void AIDA_runtime_statistics::initialise_timers() {

    full_timer.start();
    full_timer.stop();
    hom_space_timer.start();
    hom_space_timer.stop();
    hom_space_test_timer.start();
    hom_space_test_timer.stop();
    constructing_linear_system_timer.start();
    constructing_linear_system_timer.stop();
    solve_linear_system_timer.start();
    solve_linear_system_timer.stop();
    dispose_S_timer.start();
    dispose_S_timer.stop();
    update_matrix_timer.start();
    update_matrix_timer.stop();
    update_hom_timer.start();
    update_hom_timer.stop();
    compute_N_timer.start();
    compute_N_timer.stop();
    delete_with_col_timer.start();
    delete_with_col_timer.stop();
    misc_timer.start();
    misc_timer.stop();
    update_block_timer.start();
    update_block_timer.stop();
    compute_rows_timer.start();
    compute_rows_timer.stop();
    pre_alpha_decomp_optimisation_timer.start();
    pre_alpha_decomp_optimisation_timer.stop();
    alpha_decomp_timer.start();
    alpha_decomp_timer.stop();
    full_aida_timer.start();
    full_aida_timer.stop();
    full_exhaustive_timer.start();
    full_exhaustive_timer.stop();
    full_block_reduce_timer.start();
    full_block_reduce_timer.stop();
}

void AIDA_runtime_statistics::evaluate_timers() {
    hom_space = hom_space_timer.elapsed().wall/1e9;
    hom_space_test = hom_space_test_timer.elapsed().wall/1e9;
    constructing_linear_system = constructing_linear_system_timer.elapsed().wall/1e9;
    solve_linear_system = solve_linear_system_timer.elapsed().wall/1e9;
    dispose_S = dispose_S_timer.elapsed().wall/1e9;
    update_matrix = update_matrix_timer.elapsed().wall/1e9;
    update_hom = update_hom_timer.elapsed().wall/1e9;
    compute_N = compute_N_timer.elapsed().wall/1e9;
    delete_with_col = delete_with_col_timer.elapsed().wall/1e9;
    misc = misc_timer.elapsed().wall/1e9;
    update_block = update_block_timer.elapsed().wall/1e9;
    compute_rows = compute_rows_timer.elapsed().wall/1e9;
    pre_alpha_decomp_optimisation = pre_alpha_decomp_optimisation_timer.elapsed().wall/1e9;
    alpha_decomp = alpha_decomp_timer.elapsed().wall/1e9;
    full = full_timer.elapsed().wall/1e9;
    accumulated = hom_space + constructing_linear_system + solve_linear_system
        + dispose_S + update_matrix + update_hom + compute_N + delete_with_col 
        + misc + update_block + compute_rows + pre_alpha_decomp_optimisation + alpha_decomp;
    full_aida = full_aida_timer.elapsed().wall/1e9;
    full_exhaustive = full_exhaustive_timer.elapsed().wall/1e9;
    full_block_reduce = full_block_reduce_timer.elapsed().wall/1e9;
}

void AIDA_runtime_statistics::print_timers() {
    std::cout << "Timers: " << std::endl;
    std::cout << "  Hom-space: " << hom_space << "s" << std::endl;
    std::cout << "  Hom-space test: " << hom_space_test << "s" << std::endl;
    std::cout << "  Constructing linear system: " << constructing_linear_system << "s" << std::endl;
    std::cout << "  Solve linear system: " << solve_linear_system << "s" << std::endl;
    std::cout << "  Dispose S: " << dispose_S << "s" << std::endl;
    std::cout << "  Update matrix: " << update_matrix << "s" << std::endl;
    std::cout << "  Update Hom-Space Map: " <<  update_hom << "s" << std::endl;
    std::cout << "  Compute N: " << compute_N << "s" << std::endl;
    std::cout << "  Column Reduction at start: " << delete_with_col <<  "s" << std::endl;
    std::cout << "  Misc: " << misc << "s" << std::endl;
    std::cout << "  Update Block: " << update_block << "s" << std::endl;
    std::cout << "  Compute Rows: " << compute_rows << "s" << std::endl;
    std::cout << "  Pre_alpha_decomp_optimisation: " << pre_alpha_decomp_optimisation << "s" << std::endl;
    std::cout << "  Alpha_decomp: " << alpha_decomp << "s" << std::endl;
    std::cout << "  Total time: " << full << "s vs accumulated " << accumulated << "s (without loading time)" << std::endl;
    std::cout << "  Block reduce: " << full_block_reduce << "s" << std::endl;
    std::cout << "  Alpha decomp with Aida time: " << full_aida << "s" << std::endl;
    std::cout << "  Alpha decomp with exhaustive time: " << full_exhaustive << "s" << std::endl;
}

#endif // TIMERS

} // namespace aida
