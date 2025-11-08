#include "grlina/r2graded_matrix.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_minimization(std::filesystem::path input_path, std::filesystem::path output_path) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    presentation.minimize();
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        presentation.to_stream(output_file);
        output_file.close();
        std::cout << "Resolution computed and saved to: " << output_path << std::endl;
    }
}

std::string insert_suffix_before_extension(const std::string& filepath, const std::string& suffix) {
    std::filesystem::path path(filepath);
    std::string stem = path.stem().string();             // filename without extension
    std::string extension = path.extension().string();   // e.g., ".txt"
    std::filesystem::path new_path = path.parent_path() / (stem + suffix + extension);
    return new_path.string();
}

int main(int argc, char** argv) {
    
    std::string filepath;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        filepath = "/home/wsljan/AIDA/tests/test_presentations/davids_annulus_min.sccsum";
    } else {
        filepath = argv[1];
    }

    std::filesystem::path input_path(filepath);
    
    std::string modified_path = insert_suffix_before_extension(filepath, "_min");
    std::filesystem::path output_path(modified_path);
    
    
    return 0;
} // main