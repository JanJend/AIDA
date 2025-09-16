#include "grlina/r2graded_matrix.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;



bool is_decomp_file(const std::filesystem::path& filepath) {
    return filepath.extension() == ".sccsum";
}

void compute_decomp_resolutions_streaming(std::filesystem::path input_path, std::filesystem::path output_path) {
    std::ifstream input_file(input_path);
    std::ofstream output_file(output_path);
    
    if (!input_file.is_open() || !output_file.is_open()) {
        std::cerr << "Error opening files" << std::endl;
        return;
    }
    
    std::string line;
    
    // Read and verify header
    if (!std::getline(input_file, line) || line != "scc2020sum") {
        std::cerr << "Error: Expected 'scc2020sum' as first line" << std::endl;
        return;
    }
    
    // Read number of sections
    int declared_sections;
    if (!(input_file >> declared_sections)) {
        std::cerr << "Error: Could not read number of sections" << std::endl;
        return;
    }
    input_file.ignore(); // consume newline after number
    
    // Read empty line
    std::getline(input_file, line);
    if (!line.empty()) {
        std::cerr << "Warning: Expected empty line after section count" << std::endl;
    }
    
    int processed_sections = 0;
    int section_index = 0;
    
    // Process sections until EOF or declared count reached
    while (section_index < declared_sections && !input_file.eof()) {
        // Try to read type
        if (!std::getline(input_file, line)) {
            if (section_index < declared_sections) {
                std::cerr << "Warning: Reached EOF but expected " << declared_sections 
                         << " sections, only found " << section_index << std::endl;
            }
            break;
        }
        
        // Skip empty lines between sections
        if (line.empty()) {
            continue;
        }
        
        std::string type = line;
        section_index++;
        
        // Update progress
        std::cout << "\rProcessing section " << section_index << "/" << declared_sections 
                  << " (" << type << ")..." << std::flush;
                  
        // Check if next line is scc2020 or firep
        std::streampos pos_before_header = input_file.tellg();
        if (!std::getline(input_file, line) || (line != "scc2020" && line != "firep")) {
            std::cerr << "Warning: Expected 'scc2020' or 'firep' after type: " << type 
                     << " in section " << section_index << std::endl;
            continue;
        }
        
        try {
            // Reset to position before header and let constructor handle parsing
            input_file.seekg(pos_before_header);
            
            R2GradedSparseMatrix<int> minimal_presentation(input_file);
            R2Resolution<int> resolution(minimal_presentation, false);
            
            // Output format: type line + resolution output
            output_file << type << "\n";
            resolution.to_stream(output_file);
            output_file << "\n";
            
            processed_sections++;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << type << " section " << section_index 
                     << ": " << e.what() << std::endl;
        }
    }
    
    // Check for remaining content after declared sections
    if (section_index >= declared_sections && !input_file.eof()) {
        std::string remaining;
        while (std::getline(input_file, remaining)) {
            if (!remaining.empty()) {
                std::cerr << "Warning: Found additional content after " << declared_sections 
                         << " declared sections. File may contain more sections than declared." << std::endl;
                break;
            }
        }
    }
    
    // Final report
    if (processed_sections != declared_sections) {
        std::cerr << "Warning: Declared " << declared_sections << " sections, but successfully processed " 
                 << processed_sections << " sections." << std::endl;
    }
    
    std::cout << "Processed " << processed_sections << "/" << declared_sections 
              << " sections, saved to: " << output_path << std::endl;
}

void compute_resolution(std::filesystem::path input_path, std::filesystem::path output_path) {
    
    R2GradedSparseMatrix<int> minimal_presentation = R2GradedSparseMatrix<int>(input_path.string());
    R2Resolution<int> resolution(minimal_presentation, false);
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        resolution.to_stream(output_file);
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

    std::string example = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/presentation/non_cyclic_summands/comp3.scc";

    std::string filepath = example ;
    std::filesystem::path input_path(filepath);

    std::string modified_path = insert_suffix_before_extension(filepath, "_resolution");
    std::filesystem::path output_path(modified_path);

    if (is_decomp_file(input_path)) {
        compute_decomp_resolutions_streaming(input_path, output_path);
    } else {
        compute_resolution(input_path, output_path);
    }

    return 0;
} // main