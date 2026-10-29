/**
 * @file aida_decompose.hpp
    * @author Jan Jendrysiak
    * @brief Decomposes a presentation matrix using AIDA
    * @version 0.9.0
    * @date 2025-10-21
    *   This file is part of the AIDA library.
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/
#pragma once

#ifndef AIDA_DECOMPOSE_HPP
#define AIDA_DECOMPOSE_HPP

#include "aida_functions.hpp"
#include "column_batches.hpp"

namespace aida {

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
template<class Matrix, typename index = matrix_index_t<Matrix>>
void AIDA(Matrix& A, Block_list_t<Matrix>& blocks, vec<vec<transition>>& decompositions,
          std::shared_ptr<Base_change_virtual_t<index>>& base_change, AIDA_runtime_statistics& statistics,
          AIDA_config& config, Full_merge_info_t<index>& merge_info, const Column_batches_t<index>& batches);

// Decompose stored integer degrees and attach the unchanged physical grid to each result.
template<class Scalar, std::size_t Dimension, typename index>
void AIDA(GridGradedSparseMatrix<Scalar, Dimension, index>& matrix,
          Block_list_t<GridGradedSparseMatrix<Scalar, Dimension, index>>& blocks,
          vec<vec<transition>>& decompositions,
          std::shared_ptr<Base_change_virtual_t<index>>& base_change, AIDA_runtime_statistics& statistics,
          AIDA_config& config, Full_merge_info_t<index>& merge_info, const Column_batches_t<index>& batches) {
    using GridMatrix = GridGradedSparseMatrix<Scalar, Dimension, index>;
    using Coordinates = CoordGradedSparseMatrix<index, Dimension, index>;
    Coordinates coordinates(matrix.get_num_cols(), matrix.get_num_rows(), matrix.data,
                            matrix.col_degrees, matrix.row_degrees);
    coordinates.compatibly_sorted = matrix.compatibly_sorted;
    Block_list_t<Coordinates> coordinate_blocks;
    AIDA(coordinates, coordinate_blocks, decompositions, base_change, statistics, config, merge_info, batches);

    Block_list_t<GridMatrix> result;
    for (auto& block : coordinate_blocks) {
        result.emplace_back(GridMatrix(static_cast<const Coordinates&>(block), matrix.grid_));
        result.back().rows = std::move(block.rows);
        result.back().columns = std::move(block.columns);
        result.back().type = block.type;
    }
    matrix = GridMatrix(coordinates, matrix.grid_);
    blocks = std::move(result);
}

template<class Matrix, typename index = matrix_index_t<Matrix>>
void AIDA(Matrix& A, Block_list_t<Matrix>& blocks, vec<vec<transition>>& decompositions,
          std::shared_ptr<Base_change_virtual_t<index>>& base_change, AIDA_runtime_statistics& statistics,
          AIDA_config& config, Full_merge_info_t<index>& merge_info) {
    AIDA(A, blocks, decompositions, base_change, statistics, config, merge_info, Column_batches_t<index>(A));
}

// The editable algorithm keeps vector columns; CSC is rebuilt only at the boundary.
template<typename index>
void AIDA(R2GradedSparseMatrix<index, CSCStorage<index>>& matrix,
          Block_list_t<GridGradedSparseMatrix<double, 2, index>>& blocks, vec<vec<transition>>& decompositions,
          std::shared_ptr<Base_change_virtual_t<index>>& base_change,
          AIDA_runtime_statistics& statistics, AIDA_config& config,
          Full_merge_info_t<index>& merge_info) {
    using GradedMatrix = GridGradedSparseMatrix<double, 2, index>;
    using CSCGradedMatrix = R2GradedSparseMatrix<index, CSCStorage<index>>;
    GradedMatrix editable(matrix.get_num_cols(), matrix.get_num_rows(),
                          matrix.col_degrees, matrix.row_degrees);
    array<index> columns;
    for (index i = 0; i < matrix.get_num_cols(); ++i) columns.push_back(matrix.get_col(i));
    editable.assign_data(std::move(columns));
    editable.compatibly_sorted = matrix.compatibly_sorted;
    editable.sort_compatibly();
    AIDA(editable, blocks, decompositions, base_change, statistics, config, merge_info);
    matrix = CSCGradedMatrix(editable.get_num_cols(), editable.get_num_rows(), editable.data,
                             editable.real_col_degrees(), editable.real_row_degrees());
    matrix.compatibly_sorted = editable.compatibly_sorted;
}

} // namespace aida

#include "aida_decompose_impl.hpp"

#endif // AIDA_DECOMPOSE_HPP
