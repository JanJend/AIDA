/**
 * @file aida_helpers.hpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Interface and statistics for AIDA library
 *
 */

#pragma once
#ifndef AIDA_HELPERS_HPP
#define AIDA_HELPERS_HPP

#include "aida_helpers.hpp"



std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

std::string getExecutableDir() {
    std::string execPath = getExecutablePath();
    return execPath.substr(0, execPath.find_last_of("/\\"));
}

std::string findDecompositionsDir() {
    std::string base_path = getExecutableDir();
    std::string relative_path_1 = "/../lists_of_decompositions";
    std::string relative_path_2 = "/lists_of_decompositions";

    std::string full_path_1 = base_path + relative_path_1;
    std::string full_path_2 = base_path + relative_path_2;

    if (fs::exists(full_path_1)) {
        return full_path_1;
    } else if (fs::exists(full_path_2)) {
        return full_path_2;
    } else {
        throw std::runtime_error("Could not find the lists_of_decompositions directory in either of the following locations:\n" +
                                 full_path_1 + "\n" + full_path_2 + "\n"
                                 "Ensure that the the executable is located in the AIDA folder or one level higher.");
    }
}

int findLargestNumberInFilenames(const std::string& directory) {
    std::regex pattern(R"(transitions_reduced_(\d+)\.bin)");
    int largest_number = -1;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                int number = std::stoi(match[1].str());
                if (number > largest_number) {
                    largest_number = number;
                }
            }
        }
    }

    return largest_number;
}


std::size_t vec_index_hash::operator()(const std::vector<index>& v) const {
    std::size_t seed = 0;
    for (const auto& elem : v) {
        seed ^= std::hash<index>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

/**
 * @brief Careful, this only considers the indices of the blocks not the columns!
 * 
 */

std::size_t virtual_block_pair_hash::operator()(const std::pair<Merge_data, Merge_data>& p) const {
    vec_index_hash vector_hasher;
    auto hash1 = vector_hasher(p.first.first);
    auto hash2 = vector_hasher(p.second.first);
    return hash1 ^ (hash2 << 1); 
}



#endif // AIDA_HELPERS_HPP
