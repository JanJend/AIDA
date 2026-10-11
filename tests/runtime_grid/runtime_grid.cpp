/** AIDA regression for grid degrees and CSC conversion boundaries. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <aida_interface.hpp>
#include <cassert>
#include <sstream>
#include <type_traits>

static_assert(GRLINA_CSC_RUNTIME_WARNINGS == 1, "CSC boundary checks require active diagnostics");

using CSCMatrix = aida::CSCGradedMatrix;
static_assert(std::is_invocable_v<aida::AIDA_functor&, CSCMatrix&, aida::Block_list&>,
              "AIDA accepts CSC through an editable copy");
static_assert(std::is_invocable_v<aida::AIDA_functor&, graded_linalg::Module<CSCMatrix>&, aida::Block_list&>,
              "AIDA accepts modules with CSC presentations");

static void interval_and_free() {
    std::ostringstream source;
    source << "scc2020\n2\n1 3 0\n";
    auto degree = [&](double x, double y) {
        source << x << ' ' << y;
    };
    degree(0.25, 0.25); source << "; 0 1\n";
    degree(-0.5, 0.25); source << ";\n";
    degree(0.25, -0.5); source << ";\n";
    degree(10.75, 10.75); source << ";\n";
    std::istringstream input(source.str());
    aida::Module module{aida::GradedMatrix(input)};
    aida::AIDA_functor decompose;
    decompose.config.sort = true;
    decompose.config.sort_output = true;
    aida::Block_list blocks;
    decompose(module, blocks);
    assert(blocks.size() == 2);
    std::size_t intervals = 0, free = 0;
    for (const auto& block : blocks) {
        assert(block.is_graded_matrix());
        assert(block.parameter_count() == 2);
        if (block.type == aida::BlockType::INT) {
            ++intervals;
            assert(block.get_num_rows() == 2 && block.get_num_cols() == 1);
            const auto actual = block.real_col_degree(0);
            assert(actual[0] == 0.25 && actual[1] == 0.25);
        } else {
            assert(block.type == aida::BlockType::FREE);
            ++free;
        }
    }
    assert(intervals == 1 && free == 1);
}

static void cyclic_hom_spaces(bool optimised) {
    // Two cyclic summands and one free generator meet the final relation.
    // Disabling the column sweep exercises hom spaces with global row indices.
    std::istringstream input(
        "scc2020\n2\n3 3 0\n"
        "0.25 1.5; 0\n"
        "1.25 1.5; 1\n"
        "2.25 3.5; 0 1 2\n"
        "0.25 -0.5;\n"
        "1.25 -0.5;\n"
        "2.25 2.5;\n");
    aida::GradedMatrix matrix(input);
    aida::AIDA_functor decompose;
    decompose.config.sort = true;
    decompose.config.supress_col_sweep = true;
    decompose.config.turn_off_hom_optimisation = !optimised;
    aida::Block_list blocks;
    decompose(matrix, blocks);
    assert(blocks.size() == 3);
    for (const auto& block : blocks) {
        assert(block.is_graded_matrix());
        assert(block.type == aida::BlockType::CYC);
        assert(block.get_num_rows() == 1);
        assert(block.parameter_count() == 2);
    }
}

static void hom_between_independent_grids() {
    aida::GradedMatrix lower(0, 1, {}, {{0.0, 0.0}});
    aida::GradedMatrix higher(0, 1, {}, {{1.0, 0.0}});
    lower.compute_rows_forward(); higher.compute_rows_forward();
    assert(graded_linalg::hom_space_no_opt(lower, higher).first.get_num_cols() == 0);
    assert(graded_linalg::hom_space_optimised(lower, higher).first.get_num_cols() == 0);
    auto lifts = graded_linalg::hom_space_basis_new(higher, lower);
    assert(lifts.size() == 1);
    assert(lifts.front().is_graded_matrix());
    assert(lifts.front().get_col(0) == aida::vec<aida::index>{0});
    assert(lifts.front().real_col_degree(0) == higher.real_row_degree(0));
    assert(lifts.front().real_row_degree(0) == lower.real_row_degree(0));
    aida::GradedMatrix cyclic(1, 1, {{0}}, {{1.0, 0.0}}, {{0.0, 0.0}});
    assert(graded_linalg::hom_space_basis_new(higher, cyclic).size() == 1);
    assert(graded_linalg::hom_space_basis_new(higher, cyclic, false, false, true).empty());
    assert(graded_linalg::hom_space_basis_new(higher, cyclic, true).empty());
}

static void wider_index_and_precise_scalar() {
    using Matrix = graded_linalg::GridGradedSparseMatrix<long double, 2, long>;
    const long double first = 1.0L;
    const long double next = std::nextafter(first, 2.0L);
    assert(next > first);
    Matrix matrix(1, 2, {{1}}, {{next, 1.0L}},
                              {{first, 0.0L}, {next, 0.0L}});
    const auto original_grids = matrix.grids;
    aida::AIDA_functor decompose;
    decompose.config.sort = true;
    aida::vec<Matrix> summands;
    decompose(matrix, summands);
    assert(summands.size() == 2);
    assert(matrix.grids == original_grids);
    bool found_first = false, found_next = false;
    for (const auto& summand : summands) {
        assert(summand.is_graded_matrix());
        assert(summand.grids == original_grids);
        const auto degree = summand.real_row_degree(0);
        if (degree[0] == first) found_first = true;
        if (degree[0] == next) found_next = true;
    }
    assert(found_first && found_next);
    bool refused = false;
    try { aida::detail::aida_checked_int(static_cast<long>(std::numeric_limits<int>::max()) + 1L); }
    catch (const std::length_error&) { refused = true; }
    assert(refused);
    static_assert(!std::is_invocable_v<aida::AIDA_functor&, Matrix&, aida::Block_list&>);
}

template<class Left, class Right>
static void same_matrix(const Left& left, const Right& right) {
    assert(left.get_num_cols() == right.get_num_cols());
    assert(left.get_num_rows() == right.get_num_rows());
    assert(graded_linalg::detail::geometric_col_degrees(left) ==
           graded_linalg::detail::geometric_col_degrees(right));
    assert(graded_linalg::detail::geometric_row_degrees(left) ==
           graded_linalg::detail::geometric_row_degrees(right));
    for (int column = 0; column < left.get_num_cols(); ++column)
        assert(left.get_col(column) == right.get_col(column));
}

static void same_blocks(const aida::Block_list& left, const aida::Block_list& right) {
    assert(left.size() == right.size());
    auto l = left.begin(), r = right.begin();
    for (; l != left.end(); ++l, ++r) {
        assert(l->type == r->type);
        assert(l->rows == r->rows && l->columns == r->columns);
        same_matrix(*l, *r);
    }
}

static void batches_and_stream_input() {
    aida::GradedMatrix empty(0, 2);
    const aida::Column_batches empty_batches(empty);
    assert(empty_batches.col_batches.empty() && empty_batches.k_max == 0);
    aida::GradedMatrix grouped(6, 0);
    grouped.col_degrees = {{0, 0}, {0, 0}, {1, 1}, {2, 2}, {2, 2}, {2, 2}};
    const aida::Column_batches groups(grouped);
    assert((groups.col_batches == aida::array<int>{{0, 1}, {2}, {3, 4, 5}}));
    assert(groups.k_max == 3);
    grouped.col_degrees = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
    assert(aida::Column_batches(grouped).k_max == 1);
    aida::GradedMatrix wide(101, 0);
    assert(aida::Column_batches(wide).k_max == 101);

    const std::string repeated =
        "scc2020\n2\n2 2 0\n1 1; 0\n1 1; 1\n0 0;\n0 0;\n";
    const std::string free = "scc2020\n2\n0 1 0\n3 3;\n";
    aida::AIDA_functor expected;
    expected.config.sort = true;
    aida::Block_list expected_blocks;
    for (const auto& source : {repeated, free}) {
        std::istringstream input(source);
        aida::GradedMatrix matrix(input);
        aida::Block_list blocks;
        expected(matrix, blocks);
        expected_blocks.splice(expected_blocks.end(), blocks);
    }
    assert(expected_blocks.size() == 3);
    // The stream entry point also accepts input already supplied in order.
    for (bool sort : {false, true}) {
        std::istringstream input(repeated + "\n" + free);
        aida::AIDA_functor actual;
        actual.config.sort = sort;
        aida::Block_list actual_blocks;
        actual(input, actual_blocks);
        same_blocks(expected_blocks, actual_blocks);
        assert(actual.statistics_vec.size() == 2);
        assert(!actual.vector_space_decompositions.empty());
    }
}

static void csc_matches_vector_columns() {
    aida::GradedMatrix original(3, 3, {{0}, {1}, {0, 1, 2}},
        {{0.25, 1.5}, {1.25, 1.5}, {2.25, 3.5}},
        {{0.25, -0.5}, {1.25, -0.5}, {2.25, 2.5}});
    original.include_real_degrees({{99.75, 99.75}});
    auto vector = original;
    std::ostringstream diagnostics;
    auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
    auto csc = CSCMatrix(original.get_num_cols(), original.get_num_rows(), original.data,
                         original.real_col_degrees(), original.real_row_degrees());
    auto raw = csc;
    graded_linalg::Module<CSCMatrix> module(csc);
    aida::AIDA_functor expected, actual;
    expected.config.sort = true;
    expected.config.sort_output = true;
    expected.config.supress_col_sweep = true;
    actual.config = expected.config;
    aida::Block_list expected_blocks, actual_blocks;
    expected(vector, expected_blocks);
    actual(csc, actual_blocks);
    same_matrix(vector, csc);
    same_blocks(expected_blocks, actual_blocks);
    actual(module, actual_blocks);
    same_matrix(vector, module.presentation());
    same_blocks(expected_blocks, actual_blocks);
    aida::vec<aida::vec<graded_linalg::transition>> transitions;
    std::shared_ptr<aida::Base_change_virtual> basis = std::make_shared<aida::Null_base_change>();
    aida::AIDA_runtime_statistics statistics;
    aida::Full_merge_info merges;
    aida::AIDA(raw, actual_blocks, transitions, basis, statistics, expected.config, merges);
    same_matrix(vector, raw);
    same_blocks(expected_blocks, actual_blocks);
    std::cerr.rdbuf(previous);
    assert(diagnostics.str().find("CSC performance warning") == std::string::npos);
}

int main() {
    interval_and_free();
    bool rejected = false;
    try { std::istringstream input("scc2020\n3\n0 0 0\n"); aida::GradedMatrix unsupported(input); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    batches_and_stream_input();
    cyclic_hom_spaces(false);
    cyclic_hom_spaces(true);
    hom_between_independent_grids();
    csc_matches_vector_columns();
    wider_index_and_precise_scalar();
}
