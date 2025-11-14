#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;

void get_minimal_presentation(std::filesystem::path input_path, std::filesystem::path output_path) {
    
    R2Sequence<int> complex(input_path.string());
    R2GradedSparseMatrix<int> minimal_presentation = complex.get_homology();
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return;
    }
    minimal_presentation.to_stream(output_file);
    output_file.close();

}

std::string insert_suffix_before_extension(const std::string& filepath, const std::string& suffix) {
    std::filesystem::path path(filepath);
    std::string stem = path.stem().string();             // filename without extension
    std::string extension = path.extension().string();   // e.g., ".txt"
    std::filesystem::path new_path = path.parent_path() / (stem + suffix + extension);
    return new_path.string();
}

int main(int argc, char** argv) {

    std::string example = "/home/wsljan/MP-Workspace/data/CompPer25/chain_cpx/circles_20_60_dim1.scc";

    std::string filepath = example ;
    std::filesystem::path input_path(filepath);

    std::string modified_path = insert_suffix_before_extension(filepath, "_minpres");
    std::filesystem::path output_path(modified_path);

    get_minimal_presentation(input_path, output_path);
    std::cout << "Minimal presentation saved to: " << output_path << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
} // main