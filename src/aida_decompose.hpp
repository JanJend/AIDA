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
void AIDA(GradedMatrix& A, Block_list& B_list, vec<vec<transition>>& vector_space_decompositions, std::shared_ptr<Base_change_virtual>& base_change, AIDA_runtime_statistics& statistics,
    AIDA_config& config, Full_merge_info& merge_info, const Column_batches& batch_info);

inline void AIDA(GradedMatrix& A, Block_list& B_list, vec<vec<transition>>& vector_space_decompositions,
                 std::shared_ptr<Base_change_virtual>& base_change, AIDA_runtime_statistics& statistics,
                 AIDA_config& config, Full_merge_info& merge_info) {
    AIDA(A, B_list, vector_space_decompositions, base_change, statistics, config, merge_info, Column_batches(A));
}

// NEW: the editable algorithm keeps vector columns; CSC is rebuilt only at the boundary.
inline void AIDA(CSCGradedMatrix& matrix,
                 Block_list& blocks, vec<vec<transition>>& decompositions,
                 std::shared_ptr<Base_change_virtual>& base_change,
                 AIDA_runtime_statistics& statistics, AIDA_config& config,
                 Full_merge_info& merge_info) {
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

#endif // AIDA_DECOMPOSE_HPP
