/** @file grid_aida_adapter.hpp
 * @brief Preserve scalar precision and wider indices around AIDA's int core.
 */
#pragma once
#include "aida_decompose.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace aida {
namespace detail {

template <class Number>
int aida_checked_int(Number value) {
    if constexpr (std::is_signed_v<Number>)
        if (value < 0)
            throw std::length_error("AIDA requires nonnegative matrix indices");
    if (static_cast<std::uintmax_t>(value) >
        static_cast<std::uintmax_t>(std::numeric_limits<int>::max()))
        throw std::length_error("AIDA's int core requires ranks and grid sizes at most INT_MAX");
    return static_cast<int>(value);
}

template <class Matrix>
Matrix restore_aida_grid(const GradedMatrix& ordinal,
                         const typename Matrix::grid_type& grids) {
    using Index = typename Matrix::index_type;
    using Editable = GridGradedSparseMatrix<typename Matrix::real_degree_type::value_type, 2, Index>;
    Editable result(ordinal.get_num_cols(), ordinal.get_num_rows());
    // Restore the original scalar coordinates, including unused grid points.
    // AIDA's output degrees still index these same axes.
    result.grids = grids;
    auto degree = [&](auto view) {
        typename Matrix::degree_type owned{};
        for (std::size_t axis = 0; axis < view.size(); ++axis)
            owned[axis] = static_cast<Index>(view[axis]);
        return owned;
    };
    for (int i = 0; i < ordinal.get_num_rows(); ++i)
        result.row_degrees[i] = degree(ordinal.row_degrees[i]);
    for (int i = 0; i < ordinal.get_num_cols(); ++i) {
        result.col_degrees[i] = degree(ordinal.col_degrees[i]);
        for (int row : ordinal.column(i))
            result.data[i].push_back(static_cast<Index>(row));
    }
    result.compatibly_sorted = ordinal.compatibly_sorted;
    return result;
}

} // namespace detail

/** AIDA only uses coordinate order. Integer ordinals retain that order exactly,
 * including scalar values closer together than double precision can represent.
 */
template <class Decomposer, class Scalar, class Index>
void grid_aida_decompose(Decomposer& decomposer,
                            GridGradedSparseMatrix<Scalar, 2, Index>& matrix,
                            vec<GridGradedSparseMatrix<Scalar, 2, Index>>& summands) {
    using Matrix = GridGradedSparseMatrix<Scalar, 2, Index>;
    // Work in editable int columns. Physical coordinates stay in matrix.grids;
    // the core sees only their order, encoded as consecutive integer values.
    GradedMatrix ordinal(detail::aida_checked_int(matrix.get_num_cols()),
                         detail::aida_checked_int(matrix.get_num_rows()));
    for (std::size_t axis = 0; axis < matrix.parameter_count(); ++axis) {
        const int count = detail::aida_checked_int(matrix.grid(axis).size());
        ordinal.grids[axis].resize(count);
        for (int i = 0; i < count; ++i)
            ordinal.grids[axis][i] = i;
    }
    auto degree = [&](auto view) {
        Degree owned{};
        for (std::size_t axis = 0; axis < view.size(); ++axis)
            owned[axis] = detail::aida_checked_int(view[axis]);
        return owned;
    };
    for (Index i = 0; i < matrix.get_num_rows(); ++i)
        ordinal.row_degrees[i] = degree(matrix.row_degrees[i]);
    for (Index i = 0; i < matrix.get_num_cols(); ++i) {
        ordinal.col_degrees[i] = degree(matrix.col_degrees[i]);
        for (Index row : matrix.column(i))
            ordinal.data[i].push_back(detail::aida_checked_int(row));
    }
    ordinal.compatibly_sorted = matrix.compatibly_sorted;
    Block_list blocks;
    decomposer(ordinal, blocks);

    // Finish all conversions before replacing the caller's matrix and summands.
    auto restored_matrix = detail::restore_aida_grid<Matrix>(ordinal, matrix.grids);
    vec<Matrix> result;
    result.reserve(blocks.size());
    for (const Block& block : blocks)
        result.push_back(detail::restore_aida_grid<Matrix>(block, matrix.grids));
    matrix = std::move(restored_matrix);
    summands = std::move(result);
}

} // namespace aida
