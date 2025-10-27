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

#include <types.hpp>
#include <string>
#include <regex>
#include <filesystem>

namespace aida {

std::string getExecutablePath();

std::string getExecutableDir();

std::string findDecompositionsDir();

int findLargestNumberInFilenames(const std::string& directory);


}

#endif // AIDA_HELPERS_HPP