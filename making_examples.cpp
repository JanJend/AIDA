#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;

int main(int argc, char** argv) {
    R2GradedSparseMatrix<int> M(14,9);
    M.row_degrees = {
        {26,79}, {23,103}, {40,173}, {22,217}, {35,230},
        {37,299}, {21,307}, {36,309}, {20,324}
    };

    M.col_degrees = {
        {26,103}, {23,217}, {35,247}, {39,299}, {40,299},
        {37,302}, {22,307}, {37,309}, {21,324}, {36,338},
        {29,429}, {24,430}, {23,433}, {20,448}
    };
    M.data = {
        {0,1},     // col 0
        {1,3},     // col 1
        {4},       // col 2
        {5},       // col 3
        {2},       // col 4
        {5},       // col 5
        {3,6},     // col 6
        {5,7},     // col 7
        {6,8},     // col 8
        {7},       // col 9
        {0},       // col 10
        {1},       // col 11
        {1},       // col 12
        {8}        // col 13
    };

    std::cout << M.is_graded_matrix() << std::endl;
    M.print_graded();
    std::filesystem::path output_path("/home/wsljan/AIDA/Persistence-Algebra/test_presentations/two_small_circles_2.scc");
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return 1;
    }
    M.to_stream(output_file);
    output_file.close();
} // main