/**
 * @file aida.cpp
 * @author Jan Jendrysiak
 * @version 0.2.1
 * @date 2025-10-21
 * @copyright 2025 TU Graz
 *  This file is part of the AIDA library.
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include <aida_interface.hpp>
#include <option_parser.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// Helper function declarations
void write_to_file(std::ostringstream& ostream, 
                   const std::string& input_directory,
                   const std::string& file_without_extension,
                   const std::string& extension,
                   const std::string& output_string);
void run_on_test_files(aida::AIDA_functor& decomposer, std::ostringstream& ostream);

int main(int argc, char** argv) {
    // Create decomposer with default config
    aida::AIDA_functor decomposer;
    decomposer.config.progress = true;
    decomposer.config.show_info = true;
    decomposer.config.sort_output = true;
    decomposer.config.sort = true;
    
    // Parse options
    aida::OptionParser parser;
    parser.enable_stats_and_output();

    if (!parser.parse(argc, argv, decomposer.config)) {
        return 0; // Help/version shown or error
    }
    
    // Extract file information
    std::string matrix_path;
    std::string input_directory;
    std::string filename;
    std::string file_without_extension;
    std::string extension;
    
    if (parser.has_input_file()) {
        fs::path fs_path(parser.get_input_file());
        
        if (fs_path.is_relative()) {
            matrix_path = fs::current_path().string() + "/" + parser.get_input_file();
        } else {
            matrix_path = parser.get_input_file();
        }
        
        input_directory = fs_path.parent_path().string();
        filename = fs_path.filename().string();
        
        size_t dot_position = filename.find_last_of('.');
        if (dot_position == std::string::npos) {
            file_without_extension = filename;
            extension = "";
        } else {
            file_without_extension = filename.substr(0, dot_position);
            extension = filename.substr(dot_position);
        }
    }
    
    // Run decomposition
    std::ostringstream ostream;
    
    if (!parser.test_files()) {
        std::ifstream istream(matrix_path);
        if (!istream.is_open()) {
            std::cerr << "Error: Could not open input file: " << matrix_path << std::endl;
            return 1;
        }
        
        std::cout << "Decomposing " << filename << std::endl;
        decomposer.to_stream(istream, ostream);
    } else {
        run_on_test_files(decomposer, ostream);
    }
    
    // Show statistics
    if (parser.show_indecomp_statistics()) {
        decomposer.cumulative_statistics.print_statistics();
    }
    
    if (parser.show_runtime_statistics()) {
        decomposer.cumulative_runtime_statistics.print();
        #if TIMERS
            decomposer.cumulative_runtime_statistics.print_timers();
        #endif
    }
    
    if (decomposer.config.save_base_change) {
        int total_row_ops = 0;
        for (auto& base_change : decomposer.base_changes) {
            total_row_ops += base_change->performed_row_ops.size();
        }
        if (decomposer.config.show_info) {
            std::cout << "Basechange: Performed " << total_row_ops 
                     << " row operations in total." << std::endl;
        }
    }
    
    // Write output
    if (parser.has_output()) {
        write_to_file(ostream, input_directory, file_without_extension, 
                     extension, parser.get_output_string());
    }
    
    // Comparison tests
    if (decomposer.config.compare_both || decomposer.config.exhaustive_test) {
        std::ifstream istream_test(matrix_path);
        std::ostringstream ostream_test;
        
        aida::AIDA_functor test_decomposer;
        test_decomposer.config = decomposer.config;
        
        if (decomposer.config.exhaustive_test) {
            test_decomposer.config.exhaustive_test = false;
            test_decomposer.config.exhaustive = true;
        }
        
        if (decomposer.config.compare_both) {
            test_decomposer.config.compare_both = false;
            test_decomposer.config.exhaustive = true;
            test_decomposer.config.brute_force = true;
        }
        
        test_decomposer.to_stream(istream_test, ostream_test);
        
        aida::Full_merge_info merge_data = decomposer.merge_data_vec[0];
        aida::Full_merge_info merge_data_test = test_decomposer.merge_data_vec[0];
        
        aida::index num_indecomp = decomposer.cumulative_statistics.num_of_summands;
        aida::index num_indecomp_test = test_decomposer.statistics_vec.back().num_of_summands;
        
        if (num_indecomp != num_indecomp_test) {
            std::cout << "Decomposition is different. AIDA: " << num_indecomp 
                     << ", test: " << num_indecomp_test << std::endl;
        }
        
        aida::compare_merge_info(merge_data, merge_data_test);
    }
    
    return 0;
}

// Helper function implementations
void write_to_file(std::ostringstream& ostream,
                   const std::string& input_directory,
                   const std::string& file_without_extension,
                   const std::string& extension,
                   const std::string& output_string) {
    std::string output_path;
    
    if (output_string.empty()) {
        output_path = input_directory + "/" + file_without_extension + ".sccsum";
    } else if (fs::is_directory(output_string)) {
        output_path = output_string + "/" + file_without_extension + ".sccsum";
    } else {
        output_path = output_string;
    }
    
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_path << std::endl;
        return;
    }
    
    output_file << ostream.str();
    output_file.close();
    
    std::cout << "Output written to: " << output_path << std::endl;
}

void run_on_test_files(aida::AIDA_functor& decomposer, std::ostringstream& ostream) {
    // Implementation for running test files
    std::cout << "Running on test files..." << std::endl;
    // Your test file logic here
}