/*
 * @file aida_helpers.hpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Helper functions for AIDA library
 */
#pragma once
#ifndef AIDA_HELPERS_HPP
#define AIDA_HELPERS_HPP    


std::string getExecutablePath();

std::string getExecutableDir();

std::string findDecompositionsDir();

int findLargestNumberInFilenames(const std::string& directory);


struct vec_index_hash {
    std::size_t operator()(const std::vector<index>& v);
};

/**
 * @brief Careful, this only considers the indices of the blocks not the columns!
 * 
 */
struct virtual_block_pair_hash {
    std::size_t operator()(const std::pair<Merge_data, Merge_data>& p);
};

#endif // AIDA_HELPERS_HPP