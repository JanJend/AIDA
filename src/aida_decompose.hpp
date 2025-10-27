/**
 * @file aida_decompose.hpp
    * @author Jan Jendrysiak
    * @brief Decomposes a presentation matrix using AIDA
    * @version 0.2
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
    AIDA_config& config, Full_merge_info& merge_info);

} // namespace aida

#endif // AIDA_DECOMPOSE_HPP