#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./use_hn_on_folder <folder_path>" << std::endl;
        return 1;
    }

    std::string folder = argv[1];

    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        std::cerr << "Error: Folder '" << folder << "' does not exist or is not a directory." << std::endl;
        return 1;
    }

    std::string hn_filtration = "/home/wsljan/AIDA/build/hn_filtration";

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.ends_with("min_pres.scc") || filename.ends_with("min_pres.firep")) {
                files.push_back(entry.path().string());
            }
        }
    }

    if (files.empty()) {
        std::cout << "No matching files found." << std::endl;
        return 0;
    }

    for (const auto& file : files) {
        std::string command = hn_filtration + " -pe " + file;
        std::cout << "Running: " << command << std::endl;
        int ret_code = std::system(command.c_str());
        if (ret_code != 0) {
            std::cerr << "Error executing command: " << command << std::endl;
        }
    }

    return 0;
}
