/**
 * @file aida_helpers.hpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief Interface and statistics for AIDA library
 *
 */


#include "aida_helpers.hpp"

namespace aida {

namespace fs = std::filesystem;

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
    std::string this_file = __FILE__;
    std::string aida_source_dir = fs::path(this_file).parent_path().string();
    
    std::string relative_path_1 = "/../lists_of_decompositions";
    std::string relative_path_2 = "/lists_of_decompositions";

    std::string full_path_1 = aida_source_dir + relative_path_1;
    std::string full_path_2 = aida_source_dir + relative_path_2;

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


}

