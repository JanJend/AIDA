#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;

void death(std::filesystem::path input_path, std::filesystem::path output_path, double epsilon) {
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    presentation.sort_columns_lexicographically();
    presentation.sort_rows_lexicographically();
    presentation.minimize();
    r2degree step = {epsilon, epsilon};
    auto original = presentation;
    presentation.shift(step);
    R2GradedSparseMatrix<int> zero = R2GradedSparseMatrix<int>(0, presentation.get_num_rows());
    zero.col_degrees = vec<r2degree>();
    zero.row_degrees = presentation.row_degrees;
    zero.data = vec<vec<int>>();
    R2GradedSparseMatrix<int> shifted = shifted_identity<r2degree, R2GradedSparseMatrix<int>>(presentation.row_degrees, step);
    auto ker_epsilon = shifted.inverse_image_copy(presentation, zero);
    auto death = ker_epsilon.presentation_of_submodule(original);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        death.to_stream(output_file);
        output_file.close();
        std::cout << "death curve computed and saved to: " << output_path << std::endl;
    }

}


void birth(std::filesystem::path input_path, std::filesystem::path output_path, double epsilon) {
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    presentation.sort_columns_lexicographically();
    presentation.sort_rows_lexicographically();
    presentation.minimize();
    r2degree step = {epsilon, epsilon};
    R2GradedSparseMatrix<int> shifted = shifted_identity<r2degree, R2GradedSparseMatrix<int>>(presentation.row_degrees, step);
    presentation.quotient_by(shifted);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        presentation.to_stream(output_file);
        output_file.close();
        std::cout << "birth curve computed and saved to: " << output_path << std::endl;
    }
}


int main(int argc, char** argv) {
    
    std::string filepath;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        filepath = "/home/wsljan/AIDA/tests/test_presentations/ex1.scc";
    } else {
        filepath = argv[1];
    }

    std::filesystem::path input_path(filepath);
    
    std::string birth_path = insert_suffix_before_extension(filepath, "_birth");
    std::filesystem::path birth_output_path(birth_path);

    std::string death_path = insert_suffix_before_extension(filepath, "_death");
    std::filesystem::path death_output_path(death_path);

    double epsilon = 0.01;
    birth(input_path, birth_output_path, epsilon);
    death(input_path, death_output_path, epsilon);
    return 0;
} // main