/** Equal-degree column batches used during an AIDA decomposition. */
#pragma once

#include <grlina/matrices/graded.hpp>
#include <algorithm>
#include <vector>

namespace aida {

template<typename Index>
struct Column_batches_t {
    std::vector<std::vector<Index>> col_batches;
    Index k_max = 0;

    // Equal degrees must be consecutive; AIDA's entry points handle sorting.
    template <typename Matrix>
    explicit Column_batches_t(const Matrix& matrix) {
        using Degree = graded_linalg::base_degree<Matrix>;
        for (Index i = 0; i < matrix.get_num_cols(); ++i) {
            if (i == 0 || !graded_linalg::Degree_traits<Degree>::equals(
                    matrix.col_degrees[i], matrix.col_degrees[i - 1]))
                col_batches.emplace_back();
            col_batches.back().push_back(i);
            k_max = std::max(k_max, static_cast<Index>(col_batches.back().size()));
        }
    }
};

using Column_batches = Column_batches_t<int>;

} // namespace aida
