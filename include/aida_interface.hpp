/**
 * @file aida_interface.hpp
 * @author Jan Jendrysiak
 * @brief How to use the AIDA Functor:
 *
 * 1. Create an instance of the AIDA_functor.
 * 2. If you want to use only parts of the algorithm or need to sort input matrices, use the set_config method.
 * 3. Call the functor with an input stream and an output stream.
 * Results:
 * 1. The output stream will contain the decompositions.
 * 2. The statistics_vec will contain statistics about the indecomposables.
 * 3. The runtime_statistics will contain information about the presentation matrix while decomposing.
 *
 * Configuration:
 * 1. sort_output: Sorts the indecomposable summands by row degrees, then column degrees.
 * 2. sort: Sorts the columns of the input matrices lexicographically.
 * 3. exhaustive: Uses the exhaustive algorithm for the alpha-decomposition.
 * 4. brute_force: Uses the exhaustive algorithm and also does not compute hom-spaces explicitly.
 * 5. compare_both: Compares the hom space and direct version of block_reduce. Only for debugging.
 *
 * @version 0.9.0
 * @date 2025-10-21
  * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#ifndef AIDA_INTERFACE_HPP
#define AIDA_INTERFACE_HPP

#include "config.hpp"
#include "aida_decompose.hpp"
#include "grid_aida_adapter.hpp"
#include <grlina/matrices/coord.hpp>
#include <grlina/subspaces.hpp>
#include <grlina/matrices/io.hpp>

namespace aida {

template<typename index>
struct multipers_interface_input{
    vec<r2degree> col_degrees;
    vec<r2degree> row_degrees;
    vec<vec<index>> matrix;
};

template<typename index>
struct multipers_interface_output {
    vec<multipers_interface_input<index>> summands;
};

/**
 * @brief Applies AIDA to streams.
 */
struct AIDA_functor {
    AIDA_config config;
    vec<AIDA_statistics> statistics_vec;
    AIDA_statistics cumulative_statistics;
    vec<AIDA_runtime_statistics> runtime_statistics;
    vec<vec<transition>> vector_space_decompositions;
    AIDA_runtime_statistics cumulative_runtime_statistics;
    vec<Full_merge_info> merge_data_vec;
    vec<std::shared_ptr<Base_change_virtual>> base_changes;

    AIDA_functor();

    void load_vector_space_decompositions(int max_dim, std::string decomp_path);
    void load_existing_decompositions(int& k_max);
    void compute_vector_space_decompositions_faulty(const std::string& path);
    void clear_decompositions();
    
    void operator()(GradedMatrix& A, Block_list& B_list);
    // Edit vector columns once, then rebuild the caller's CSC storage in a batch.
    void operator()(CSCGradedMatrix& matrix,
                    Block_list& blocks) {
        GradedMatrix editable(matrix.get_num_cols(), matrix.get_num_rows(),
                              matrix.col_degrees, matrix.row_degrees);
        array<index> columns;
        for (index i = 0; i < matrix.get_num_cols(); ++i) columns.push_back(matrix.get_col(i));
        editable.assign_data(std::move(columns));
        editable.compatibly_sorted = matrix.compatibly_sorted;
        operator()(editable, blocks);
        matrix = CSCGradedMatrix(editable.get_num_cols(), editable.get_num_rows(), editable.data,
                                 editable.real_col_degrees(), editable.real_row_degrees());
        matrix.compatibly_sorted = editable.compatibly_sorted;
    }
    void operator()(graded_linalg::Module<CSCGradedMatrix>& module,
                    Block_list& blocks) {
        module.proj_res_.differentials().resize(1);
        module.res_complete_ = false;
        module.inj_res_.clear();
        operator()(module.presentation(), blocks);
    }
    // Typed summand presentations preserve scalar and index types.
    template<class Scalar, class Index>
    void operator()(GridGradedSparseMatrix<Scalar, 2, Index>& matrix,
                    vec<GridGradedSparseMatrix<Scalar, 2, Index>>& summands) {
        grid_aida_decompose(*this, matrix, summands);
    }
    template<class Scalar, class Index>
    void operator()(graded_linalg::Module<GridGradedSparseMatrix<Scalar, 2, Index>>& module,
                    vec<GridGradedSparseMatrix<Scalar, 2, Index>>& summands) {
        module.proj_res_.differentials().resize(1);
        module.res_complete_ = false;
        module.inj_res_.clear();
        operator()(module.presentation(), summands);
    }
    // Block_list stores double/int; use vec<Matrix> for other scalar/index types.
    template<class Scalar, class Index,
             std::enable_if_t<!std::is_same_v<GridGradedSparseMatrix<Scalar, 2, Index>, GradedMatrix>, int> = 0>
    void operator()(GridGradedSparseMatrix<Scalar, 2, Index>&, Block_list&) = delete;

    /** Module-owning entry point; the matrix overload remains for compatibility. */
    void operator()(Module& module, Block_list& B_list) {
        module.proj_res_.differentials().resize(1);
        module.res_complete_ = false;
        module.inj_res_.clear();
        operator()(module.presentation(), B_list);
    }
    Sparse_Matrix get_row_basis(index i, index m);
    Sparse_Matrix get_row_basis(index m);

    template<typename InputStream,
             std::enable_if_t<std::is_base_of_v<std::istream, InputStream>, int> = 0>
    void operator()(InputStream& ifstr, Block_list& B_list_cumulative) {
        vec<GradedMatrix> matrices;
        #if TIMERS
            aida::load_matrices_timer.start();
        #endif
        construct_matrices_from_stream(matrices, ifstr, config.sort);
        vec<Column_batches> batch_data;
        batch_data.reserve(matrices.size());
        int k_max = 0;
        for (const auto& matrix : matrices) {
            batch_data.emplace_back(matrix);
            k_max = std::max(k_max, batch_data.back().k_max);
        }
        vec<Module> modules;
        modules.reserve(matrices.size());
        for (auto& matrix : matrices) modules.emplace_back(std::move(matrix));
        #if TIMERS
            aida::load_matrices_timer.stop();
            double load_matrices = aida::load_matrices_timer.elapsed().wall/1e9;
            if(config.show_info){
                std::cout << "Loaded " << matrices.size() << " matrix/ces in time: "
                << load_matrices << " s." << std::endl;
            }
        #endif
        load_existing_decompositions(k_max);

        for (std::size_t i = 0; i < modules.size(); ++i) {
            GradedMatrix& A = modules[i].presentation();
            const Column_batches& batch_info = batch_data[i];
            if(config.show_info && modules.size() == 1){
                std::cout << " Matrix has " << A.get_num_rows() << " rows and " << A.get_num_cols() <<
                " columns, k_max is " << batch_info.k_max << ", and there are " << batch_info.col_batches.size() << " batches." << std::endl;
            }
            std::shared_ptr<Base_change_virtual> base_change;
            if(config.save_base_change){
                base_change = std::make_shared<Base_change>();
            } else {
                base_change = std::make_shared<Null_base_change>();
            }
            base_changes.push_back(base_change);
            statistics_vec.push_back(AIDA_statistics());
            runtime_statistics.push_back(AIDA_runtime_statistics());
            Block_list B_list;
            Full_merge_info merge_info;

            #if TIMERS
                runtime_statistics.back().initialise_timers();
            #endif

            AIDA(A, B_list, vector_space_decompositions, base_changes.back(), runtime_statistics.back(), config, merge_info, batch_info);
            
            #if TIMERS
                runtime_statistics.back().evaluate_timers();
            #endif

            merge_data_vec.push_back(merge_info);
            statistics_vec.back().compute_statistics(B_list);
            B_list_cumulative.splice(B_list_cumulative.end(), B_list);
        }
        if(config.show_info){
            std::cout << B_list_cumulative.size() << " indecomposable summands." << std::endl;
        }

        cumulative_statistics = AIDA_statistics();
        cumulative_statistics.compute_statistics(B_list_cumulative);
        cumulative_runtime_statistics = AIDA_runtime_statistics();
        for(auto& runtime_stat : runtime_statistics){
            cumulative_runtime_statistics += runtime_stat;
        }

        if(config.sort_output){
            B_list_cumulative.sort(CompareBlocksByDegrees{});
        }
    }

    template<typename InputStream, typename OutputStream>
    void to_stream(InputStream& ifstr, OutputStream& ofstr) {
        Block_list B_list;
        operator()(ifstr, B_list);
        ofstr << "scc2020sum" << "\n";
        ofstr << B_list.size() << "\n";
        for (Block& B : B_list) {
            ofstr << "\n";
            ofstr << B.get_type() << "\n";
            B.to_stream(ofstr);
        }
    }

    template<typename index>
    multipers_interface_output<index> multipers_interface(multipers_interface_input<index>& input){
        static_assert(std::is_same_v<index, aida::index>, "AIDA uses int matrix indices");
        CoordGradedSparseMatrix<double, 2, index> coordinates(
            input.col_degrees.size(), input.row_degrees.size());
        coordinates.data = input.matrix;
        for (index i = 0; i < coordinates.get_num_cols(); ++i)
            coordinates.col_degrees[i] = input.col_degrees[i];
        for (index i = 0; i < coordinates.get_num_rows(); ++i)
            coordinates.row_degrees[i] = input.row_degrees[i];
        GradedMatrix presentation(coordinates);
        presentation.sort_compatibly();
        Module module(std::move(presentation));
        Block_list B_list;
        this->operator()(module, B_list);
        multipers_interface_output<index> result;
        for(auto& B : B_list){
            multipers_interface_input<index> summand;
            for (index i = 0; i < B.get_num_cols(); ++i) {
                auto degree = B.real_col_degree(i);
                summand.col_degrees.push_back({degree[0], degree[1]});
            }
            for (index i = 0; i < B.get_num_rows(); ++i) {
                auto degree = B.real_row_degree(i);
                summand.row_degrees.push_back({degree[0], degree[1]});
            }
            summand.matrix = B.data;
            result.summands.push_back(summand);
        }
        return result;
    }


};

} // namespace aida

#endif // AIDA_INTERFACE_HPP
