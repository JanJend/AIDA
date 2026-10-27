/** Equal-degree column batches used during an AIDA decomposition. */
#pragma once

#include <grlina/posets/traits.hpp>
#include <algorithm>
#include <vector>

namespace aida {

struct Column_batches {
    std::vector<std::vector<int>> col_batches;
    int k_max = 0;

    // Equal degrees must be consecutive; AIDA's entry points handle sorting.
    template <typename Matrix>
    explicit Column_batches(const Matrix& matrix) {
        using Degree = typename Matrix::degree_type;
        for (int i = 0; i < matrix.get_num_cols(); ++i) {
            if (i == 0 || !graded_linalg::Degree_traits<Degree>::equals(
                    matrix.col_degrees[i], matrix.col_degrees[i - 1]))
                col_batches.emplace_back();
            col_batches.back().push_back(i);
            k_max = std::max(k_max, static_cast<int>(col_batches.back().size()));
        }
    }
};

} // namespace aida
