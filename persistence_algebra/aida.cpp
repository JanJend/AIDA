/**
 * @file aida.cpp
 * @author Jan Jendrysiak
 * @brief  How to use the AIDA program
 * 
 * 1. create a folder build in the root directory of the project
 * 2. run cmake -DCMAKE_BUILD_TYPE=Release .. in the build directory
 * 3. run make 
 * 4. run ./aida <path_to_file> to decompose a file
 * 
 * OPTIONS:
 * -o : writes the output to a file with the name <input_file_name>_decomposition.scc in the current directory.
 * -o <output_file> : writes the output to the specified file. Can be a relative or absolute path.
 * -o <output_directory> : writes the output to the specified directory with the name <input_file_name>_decomposition.scc
 * -o<argument>: As above but without the space between the option and the argument.
 * -sort : sorts the input matrices lexicographically
 * -c : saves the base change for each decomposition
 * -p : shows progress bar while decomposing 
 * -t : shows statistics about the indecomposables
 * -r : shows runtime statistics
 * -e : uses the exhaustive algorithm
 * -b : uses the DeyXin (bruteforce) algorithm. This implies -e automatically.
 * -m : compares the hom space and direct version (i.e -b) of block_reduce. Only for debugging.
 * -l : suppresses most console output
 * -a : compares the exhaustive and aida alpha-decomp. Only for debugging.
 * -i : compare optimised and non-optimised hom space calculation.
 * --h : display detailed description
 * --v : display version
 * 
 * @version 1.0
 * @date 2024-10-07
 * 
 * @copyright ?
 * 
 */

#include "aida_interface.hpp"
#include <unistd.h> 
#include <getopt.h>

namespace fs = std::filesystem;

void display_help() {
    std::cout << "Usage: ./aida <input_file> [options]\n"
              << "Options:\n"
              << "  -h, --help           Display this help message\n"
              << "  -v, --version        Display version information\n"
              << "  -b, --bruteforce     Stops hom-space calculation and thus all optimisation. \n"
              << "  -s, --sort           Lexicographically sorts the relations of the input\n"
              << "  -e, --exhaustive     Always iterates over all decompositions of a batch\n"
              << "  -t, --statistics     Show statistics about indecomposable summands\n"
              << "  -r, --runtime        Show runtime statistics and timers\n"
              << "  -p, --progress       Show progressbar\n"
              << "  -c, --basechange     Save base change\n"
              << "  -o, --output <file>  Specify output file\n"
              << "  -l, --less_console   Suppreses most console output\n"
              << "  -m, --compare_b      Compares with -b at runtime, then runs with only -b and compares.\n"
              << "  -a, --compare_e      Compares exhaustive and brute force at runtime.\n"
              << "  -i, --compare_hom    Compares optimised and non-opt hom space calculation at runtime.\n"
              << "  -j, --no_hom_opt     Does not use the optimised hom space calculation.\n"
              << "  -w, --no_col_sweep   Does not use the column sweep optimisation.\n"
              << "  -f, --no_alpha       Turns the computation of alpha-homs off.\n"
              << "      <file> is optional and will default to the <input_file> with _decomposed appended\n"
              << "      You can pass relative and absolute paths as well as only a directory."
              << "Further Instructions: \n Make sure that the inputfile is a (sequence of) scc or firep presentations that are minimised.\n"
              << std::endl;
}

void display_version() {
    std::cout << "AIDA version 1.0 -- 21st Oct 2024\n";
}

int main(int argc, char** argv){
    aida::AIDA_functor decomposer = aida::AIDA_functor();
    
    decomposer.config.exhaustive = false;
    decomposer.config.brute_force = false;
    decomposer.config.sort = false;
    decomposer.config.sort_output = true;
    decomposer.config.alpha_hom = true;
    bool write_output = false;

    bool show_indecomp_statistics = false;
    bool show_runtime_statistics = false;
    decomposer.config.show_info = true;

    decomposer.config.compare_both = false; // Compares normal functioning and brute force at runtime, then also compares output.
    bool compare_time = false;
    decomposer.config.exhaustive_test = false;

    bool compare_hom_internal = false; // Cannot be used with the functor right now.


    std::string input_directory;
    std::string filename;
    std::string matrix_path;
    std::string output_string;
    std::string output_file_path;

    if (argc < 2) {
        std::cerr << "No input file specified. Please provide an input file." << std::endl;
        display_help();
        std::cout << "Please provide options/arguments: ";
        std::string input;
        std::getline(std::cin, input);
        std::vector<std::string> args;
        args.push_back(argv[0]);
        std::istringstream iss(input);
        std::string token;
        while (iss >> token) {
            args.push_back(token);
        }
        argc = args.size();
        argv = new char*[argc + 1];
        for (size_t i = 0; i < args.size(); ++i) {
            argv[i] = new char[args[i].size() + 1];
            std::strcpy(argv[i], args[i].c_str());
        }
        argv[argc] = nullptr;
        optind = 1; // Reset getopt
    }

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"output", optional_argument, 0, 'o'},
        {"bruteforce", no_argument, 0, 'b'},
        {"sort", no_argument, 0, 's'},
        {"exhaustive", no_argument, 0, 'e'},
        {"statistics", no_argument, 0, 't'},
        {"runtime", no_argument, 0, 'r'},
        {"progress", no_argument, 0, 'p'},
        {"basechange", no_argument, 0, 'c'},
        {"less_console", no_argument, 0, 'l'},
        {"compare_b", no_argument, 0, 'm'},
        {"compare_e", no_argument, 0, 'a'},
        {"compare_hom", no_argument, 0, 'i'},
        {"no_hom_opt", no_argument, 0, 'j'},
        {"no_col_sweep", no_argument, 0, 'w'},
        {"no_alpha", no_argument, 0, 'f'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "ho::bsetrpclmvaijwf", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                display_help();
                return 0;
            case 'v':
                display_version();
                return 0;
            case 'o':
                write_output = true;
                if (optarg) {
                    output_string = std::string(optarg);
                } else if (optind < argc && argv[optind][0] != '-') {
                    output_string = std::string(argv[optind]);
                    optind++;
                } else {
                    output_string.clear(); // Set output_string to empty
                }
                break;
            case 'b':
                decomposer.config.brute_force = true;
                decomposer.config.exhaustive = true;
                break;
            case 's':
                decomposer.config.sort = true;
                break;
            case 'e':
                decomposer.config.exhaustive = true;
                break;
            case 't':
                show_indecomp_statistics = true;
                break;
            case 'r':
                show_runtime_statistics = true;
                break;
            case 'p':
                decomposer.config.progress = true;
                break;
            case 'c':
                decomposer.config.save_base_change = true;
                break;
            case 'l':
                decomposer.config.show_info = false;
                break;
            case 'm':
                decomposer.config.compare_both = true;
                break;
            case 'a':
                decomposer.config.exhaustive_test = true;
                break;
            case 'i':
                decomposer.config.compare_hom = true;
                break;
            case 'j':
                decomposer.config.turn_off_hom_optimisation = true;
                break;
            case 'w':
                decomposer.config.supress_col_sweep = true;
                break;
            case 'f':
                decomposer.config.alpha_hom = false;
                break;
            default:
                return 1;
        }
    }

    std::string file_without_extension;
    std::string extension;

    if (optind < argc) {
        std::filesystem::path fs_path(argv[optind]);
        if (fs_path.is_relative()) {
            matrix_path = std::filesystem::current_path().string() + "/" + argv[optind];
        } else {
            matrix_path = argv[optind];
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
    } else {
        std::cerr << "No input file specified. Please provide an input file." << std::endl;
        return 1;
    }

    std::ifstream istream(matrix_path);
    if (!istream.is_open()) {
            std::cerr << "Error: Could not open input file: " << matrix_path << std::endl;
            return 0;
    }

    std::cout << "Decomposing " + filename << std::endl;

    std::ostringstream ostream;
    decomposer(istream, ostream);
    if(show_indecomp_statistics){
        decomposer.cumulative_statistics.print_statistics();
    }
    if(show_runtime_statistics){
        decomposer.cumulative_runtime_statistics.print();
        #if TIMERS
            decomposer.cumulative_runtime_statistics.print_timers();
        #endif
    }
    if(decomposer.config.save_base_change){
        int total_row_ops = 0;
        for(auto& base_change : decomposer.base_changes){
           total_row_ops += base_change->performed_row_ops.size();
        }
        if(decomposer.config.show_info){
            std::cout << "Basechange: Performed " << total_row_ops << " row operations in total." << std::endl;
        }
    }
    
    aida::Full_merge_info merge_data = decomposer.merge_data_vec[0];
    aida::index num_indecomp = decomposer.cumulative_statistics.num_of_summands;
    
    if(write_output){
        if(output_string.empty()){
            output_file_path = input_directory + "/" + file_without_extension + "_decomposition" + extension;
        } else {
            std::filesystem::path output_path(output_string);
            if (output_path == ".") {
                output_file_path = std::filesystem::current_path().string() + "/" + file_without_extension + "_decomposition" + extension;
            } else if (output_path.is_relative()) {
                output_file_path = std::filesystem::current_path().string() + "/" + output_string;
            } else if (std::filesystem::is_directory(output_path)) {
                output_file_path = output_path.string() + "/" + file_without_extension + "_decomposition" + extension;
            } else if (output_path.is_absolute()) {
                output_file_path = output_string;
            } else {
                output_file_path = input_directory + "/" + output_string;
            }
        }

        std::filesystem::create_directories(std::filesystem::path(output_file_path).parent_path());

        std::ofstream file_out(output_file_path);
        if(file_out.is_open()){
            file_out << ostream.str();
            file_out.close();
            if(decomposer.config.show_info){
                std::cout << "Decomposition written to " << output_file_path << std::endl;
            }
        } else {
            std::cout << "Error: Could not write decomposition to file: " << output_file_path << std::endl;
        }
    }

    if(decomposer.config.compare_both|| compare_time || decomposer.config.exhaustive_test){
        
        std::ifstream istream_test(matrix_path);
        std::ostringstream ostream_test;
        aida::AIDA_functor test_decomposer = aida::AIDA_functor();
        test_decomposer.config = decomposer.config;
        if(decomposer.config.exhaustive_test){
            decomposer.config.exhaustive_test = false;
            test_decomposer.config.exhaustive = true;
        }
        if(decomposer.config.compare_both){
            test_decomposer.config.compare_both = false;
            test_decomposer.config.exhaustive = true;
            test_decomposer.config.brute_force = true;
        }
        std::ifstream test_istream(matrix_path);

        test_decomposer(test_istream, ostream);
        aida::Full_merge_info merge_data_test = test_decomposer.merge_data_vec[0];

        aida::index num_indecomp_test = test_decomposer.statistics_vec.back().num_of_summands;
        if(num_indecomp != num_indecomp_test){
            std::cout << "Decomposition is different. AIDA: " << num_indecomp << ", test: " << num_indecomp_test << std::endl;
        }
        for(int t = 0; t < merge_data.size(); t++){
            if(merge_data[t].empty()){
                std::cout << "Warning: Empty merge data at " << t << std::endl;
            }
        }
        if(true){
            aida::compare_merge_info(merge_data, merge_data_test);
        }
    } 

    return 0;
} //main


