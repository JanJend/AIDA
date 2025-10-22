#include "filt_landscape.hpp"
#include <iostream>


int main(int argc, char* argv[]) {
    if (argc > 6) {
        std::cerr << "Usage: " << argv[0] << " [<input.sky>] [<output.txt>] [<theta>] [<k>] \n";
        return 1;
    }
    
    std::string input_file = (argc >= 2) ? argv[1] : "/home/wsljan/AIDA/tests/test_presentations/two_circles.sky";
    
    std::string output_file;
    if (argc >= 3) {
        output_file = argv[2];
    } else {
        // Remove extension and add "_landscape.txt"
        size_t last_dot = input_file.find_last_of('.');
        output_file = (last_dot != std::string::npos) ? input_file.substr(0, last_dot) : input_file;
        output_file += "_landscape.txt";
    }
    
    double theta = (argc >= 4) ? std::stod(argv[3]) : 0.0;
    int k = (argc >= 5) ? std::stoi(argv[4]) : 1;
    
    try {
        GridData data = bars_from_sky(input_file);
        compute_landscape(data, output_file, theta, k);
        std::cout << "Landscape computed successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}