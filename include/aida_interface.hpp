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
template<typename Index = int>
struct AIDA_functor {
    using GradedMatrix = GridGradedSparseMatrix<double, 2, Index>;
    using CSCGradedMatrix = R2GradedSparseMatrix<Index, CSCStorage<Index>>;
    using Block_list = Block_list_t<GradedMatrix>;
    using Full_merge_info = Full_merge_info_t<Index>;
    using Base_change_virtual = Base_change_virtual_t<Index>;
    using Base_change = Base_change_t<Index>;
    using Null_base_change = Null_base_change_t<Index>;

    AIDA_config config;
    vec<AIDA_statistics> statistics_vec;
    AIDA_statistics cumulative_statistics;
    vec<AIDA_runtime_statistics> runtime_statistics;
    vec<vec<transition>> vector_space_decompositions;
    AIDA_runtime_statistics cumulative_runtime_statistics;
    vec<Full_merge_info> merge_data_vec;
    vec<std::shared_ptr<Base_change_virtual>> base_changes;

    void load_vector_space_decompositions(std::size_t max_dim, const std::string& decomp_path);
    void load_existing_decompositions(std::size_t k_max);
    void compute_vector_space_decompositions_faulty(const std::string& path);
    void clear_decompositions() { vector_space_decompositions.clear(); }

    template<class Matrix,
             std::enable_if_t<std::is_same_v<matrix_index_t<Matrix>, Index>, int> = 0>
    void operator()(Matrix& matrix, Block_list_t<Matrix>& blocks) {
        decompose(matrix, blocks);
    }

    template<class Matrix>
    void operator()(Matrix& matrix, vec<Matrix>& summands) {
        Block_list_t<Matrix> blocks;
        operator()(matrix, blocks);
        vec<Matrix> result;
        result.reserve(blocks.size());
        for (auto& block : blocks) result.emplace_back(std::move(static_cast<Matrix&>(block)));
        summands = std::move(result);
    }

    // Edit vector columns once, then rebuild the caller's CSC storage in a batch.
    void operator()(CSCGradedMatrix& matrix, Block_list& blocks) {
        GradedMatrix editable(matrix.get_num_cols(), matrix.get_num_rows(),
                              matrix.col_degrees, matrix.row_degrees);
        array<Index> columns;
        for (Index i = 0; i < matrix.get_num_cols(); ++i) columns.push_back(matrix.get_col(i));
        editable.assign_data(std::move(columns));
        editable.compatibly_sorted = matrix.compatibly_sorted;
        operator()(editable, blocks);
        matrix = CSCGradedMatrix(editable.get_num_cols(), editable.get_num_rows(), editable.data,
                                 editable.real_col_degrees(), editable.real_row_degrees());
        matrix.compatibly_sorted = editable.compatibly_sorted;
    }

    template<class Matrix, class Output>
    void operator()(graded_linalg::Module<Matrix>& module, Output& output) {
        module.proj_res_.differentials().resize(1);
        module.res_complete_ = false;
        module.inj_res_.clear();
        operator()(module.presentation(), output);
    }

    template<class InputStream,
             std::enable_if_t<std::is_base_of_v<std::istream, InputStream>, int> = 0>
    void operator()(InputStream& input, Block_list& blocks) {
        vec<GradedMatrix> matrices;
        #if TIMERS
            load_matrices_timer.start();
        #endif
        construct_matrices_from_stream(matrices, input, config.sort);
        #if TIMERS
            load_matrices_timer.stop();
            if (config.show_info)
                std::cout << "Loaded " << matrices.size() << " matrices in "
                          << load_matrices_timer.elapsed().wall / 1e9 << " s.\n";
        #endif
        for (auto& matrix : matrices) {
            Block_list summands;
            operator()(matrix, summands);
            blocks.splice(blocks.end(), summands);
        }
        cumulative_statistics = AIDA_statistics();
        cumulative_statistics.compute_statistics(blocks);
        if (config.show_info) std::cout << blocks.size() << " indecomposable summands.\n";
        if (config.sort_output) blocks.sort(CompareBlocksByDegrees{});
    }

    template<class InputStream, class OutputStream>
    void to_stream(InputStream& input, OutputStream& output) {
        Block_list blocks;
        operator()(input, blocks);
        output << "scc2020sum\n" << blocks.size() << "\n";
        for (auto& block : blocks) {
            output << "\n" << block.get_type() << "\n";
            block.to_stream(output);
        }
    }

    multipers_interface_output<Index> multipers_interface(multipers_interface_input<Index>& input) {
        using Coordinates = CoordGradedSparseMatrix<double, 2, Index>;
        Coordinates matrix(input.col_degrees.size(), input.row_degrees.size());
        matrix.data = input.matrix;
        for (Index i = 0; i < matrix.get_num_cols(); ++i) matrix.col_degrees[i] = input.col_degrees[i];
        for (Index i = 0; i < matrix.get_num_rows(); ++i) matrix.row_degrees[i] = input.row_degrees[i];
        matrix.sort_compatibly();
        Block_list_t<Coordinates> blocks;
        operator()(matrix, blocks);
        multipers_interface_output<Index> result;
        for (auto& block : blocks) {
            multipers_interface_input<Index> summand;
            for (const auto& degree : block.col_degrees) summand.col_degrees.push_back({degree[0], degree[1]});
            for (const auto& degree : block.row_degrees) summand.row_degrees.push_back({degree[0], degree[1]});
            summand.matrix = std::move(block.data);
            result.summands.push_back(std::move(summand));
        }
        return result;
    }

    SparseMatrix<Index> get_row_basis(Index i, Index m) {
        const auto& operations = base_changes[i]->performed_row_ops;
        SparseMatrix<Index> result(m, m, "Identity");
        for (auto op = operations.rbegin(); op != operations.rend(); ++op)
            result.row_op_on_cols(op->first, op->second);
        return result;
    }
    SparseMatrix<Index> get_row_basis(Index m) {
        return get_row_basis(static_cast<Index>(base_changes.size() - 1), m);
    }

private:
    template<class Matrix>
    void decompose(Matrix& matrix, Block_list_t<Matrix>& blocks);
};

} // namespace aida

#include "aida_interface_impl.hpp"

#endif // AIDA_INTERFACE_HPP
