#ifdef NDEBUG
#undef NDEBUG
#endif
#include <aida_interface.hpp>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

template<typename Index>
void decompose_with_index() {
    using Coordinates = graded_linalg::CoordGradedSparseMatrix<Index, 2, Index>;
    using Grid = graded_linalg::GridGradedSparseMatrix<long double, 2, Index>;
    static_assert(std::is_same_v<graded_linalg::matrix_index_t<Coordinates>, Index>);
    static_assert(std::is_same_v<decltype(aida::Column_batches_t<Index>{Coordinates{}}.k_max), Index>);

    // Force hom-space calculations as well as the exhaustive and brute-force paths.
    for (int mode = 0; mode < 4; ++mode) {
        Grid grid(3, 3, {{0}, {1}, {0, 1, 2}},
                  {{0.25L, 1.5L}, {1.25L, 1.5L}, {2.25L, 3.5L}},
                  {{0.25L, -0.5L}, {1.25L, -0.5L}, {2.25L, 2.5L}});
        grid.include_real_degrees({{-7.0L, 99.75L}});
        const auto axes = grid.grid_;
        Coordinates coordinates(grid.get_num_cols(), grid.get_num_rows(), grid.data,
                                grid.col_degrees, grid.row_degrees);
        aida::AIDA_functor<Index> from_grid, from_coordinates;
        from_grid.config.sort = from_grid.config.sort_output = true;
        from_grid.config.supress_col_sweep = from_grid.config.save_base_change = true;
        from_grid.config.turn_off_hom_optimisation = mode == 1;
        from_grid.config.exhaustive = mode >= 2;
        from_grid.config.brute_force = mode == 3;
        from_coordinates.config = from_grid.config;
        aida::Block_list_t<Grid> grid_blocks;
        aida::Block_list_t<Coordinates> coordinate_blocks;
        from_grid(grid, grid_blocks);
        from_coordinates(coordinates, coordinate_blocks);
        assert(grid_blocks.size() == 3 && coordinate_blocks.size() == 3);
        assert(grid.grid_ == axes);
        assert(grid.data == coordinates.data);
        assert(grid.col_degrees == coordinates.col_degrees);
        assert(grid.row_degrees == coordinates.row_degrees);
        auto c = coordinate_blocks.begin();
        for (const auto& block : grid_blocks) {
            assert(block.grid_ == axes);
            assert(block.is_graded_matrix());
            assert(block.type == c->type && block.rows == c->rows && block.columns == c->columns);
            assert(block.data == c->data);
            assert(block.row_degrees == c->row_degrees && block.col_degrees == c->col_degrees);
            ++c;
        }
        const auto basis = from_grid.get_row_basis(3);
        static_assert(std::is_same_v<graded_linalg::matrix_index_t<decltype(basis)>, Index>);
        assert(basis.data == from_coordinates.get_row_basis(3).data);
    }

    // A repeated-degree batch exercises transition tables with this index type.
    Coordinates repeated(2, 2, {{0}, {1}}, {{1, 1}, {1, 1}}, {{0, 0}, {0, 0}});
    aida::AIDA_functor<Index> decompose;
    decompose.config.sort = true;
    std::vector<Coordinates> summands;
    decompose(repeated, summands);
    assert(summands.size() == 2);

    // The free case returns before the core's usual final block-normalisation loop.
    Grid free(0, 2, {}, {}, {{-3.5L, 1.0L}, {7.25L, 2.0L}});
    const auto free_axes = free.grid_;
    std::vector<Grid> free_summands;
    decompose(free, free_summands);
    assert(free_summands.size() == 2);
    for (const auto& summand : free_summands) assert(summand.grid_ == free_axes);
}

int main() {
    decompose_with_index<int>();
    decompose_with_index<long>();
    decompose_with_index<long long>();

    using Index = std::int64_t;
    using Coordinates = graded_linalg::CoordGradedSparseMatrix<Index, 2, Index>;
    const Index high = static_cast<Index>(std::numeric_limits<int>::max()) + 17;
    Coordinates matrix(1, 2, {{0, 1}}, {{high, high}}, {{high - 1, high}, {high, high - 1}});
    aida::AIDA_functor<Index> decompose;
    decompose.config.sort = true;
    std::vector<Coordinates> summands;
    decompose(matrix, summands);
    assert(summands.size() == 1);
    assert(summands[0].col_degrees[0][0] == high);
    assert(summands[0].col_degrees[0][1] == high);
    aida::Base_change_t<Index> operations;
    operations.add_row_op(high, high + 1);
    assert(operations.performed_row_ops[0].first == high);
    assert(operations.performed_row_ops[0].second == high + 1);

    using CSC = graded_linalg::R2GradedSparseMatrix<Index, graded_linalg::CSCStorage<Index>>;
    CSC compressed(1, 2, {{0, 1}}, {{1, 1}}, {{0, 1}, {1, 0}});
    auto raw = compressed;
    aida::AIDA_functor<Index>::Block_list compressed_summands;
    decompose(compressed, compressed_summands);
    assert(compressed_summands.size() == 1);
    assert(compressed_summands.front().get_num_rows() == 2);

    aida::AIDA_functor<Index>::Block_list raw_summands;
    std::shared_ptr<aida::Base_change_virtual_t<Index>> basis =
        std::make_shared<aida::Null_base_change_t<Index>>();
    aida::AIDA_runtime_statistics statistics;
    aida::Full_merge_info_t<Index> merges;
    aida::AIDA(raw, raw_summands, decompose.vector_space_decompositions,
               basis, statistics, decompose.config, merges);
    assert(raw_summands.size() == compressed_summands.size());
    assert(raw_summands.front().data == compressed_summands.front().data);
}
