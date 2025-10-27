#include "option_parser.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <getopt.h>

namespace aida {

OptionParser::OptionParser(const OptionSet& options)
    : options_(options) {
}

// Also add this overload if you want default construction:
OptionParser::OptionParser() 
    : OptionParser(OptionSet{}) {
}

void OptionParser::build_option_strings(std::string& short_opts, 
                                       std::vector<::option>& long_opts) {
    short_opts = "hv";
    
    // Help and version always included
    long_opts.push_back({"help", no_argument, 0, 'h'});
    long_opts.push_back({"version", no_argument, 0, 'v'});
    
    if (options_.include_output) {
        long_opts.push_back({"output", optional_argument, 0, 'o'});
        short_opts += "o::";
    }
    
    if (options_.include_bruteforce) {
        long_opts.push_back({"bruteforce", no_argument, 0, 'b'});
        short_opts += "b";
    }
    
    if (options_.include_sort) {
        long_opts.push_back({"sort", no_argument, 0, 's'});
        short_opts += "s";
    }
    
    if (options_.include_exhaustive) {
        long_opts.push_back({"exhaustive", no_argument, 0, 'e'});
        short_opts += "e";
    }
    
    if (options_.include_statistics) {
        long_opts.push_back({"statistics", no_argument, 0, 't'});
        short_opts += "t";
    }
    
    if (options_.include_runtime) {
        long_opts.push_back({"runtime", no_argument, 0, 'r'});
        short_opts += "r";
    }
    
    if (options_.include_progress) {
        long_opts.push_back({"progress", no_argument, 0, 'p'});
        short_opts += "p";
    }
    
    if (options_.include_basechange) {
        long_opts.push_back({"basechange", no_argument, 0, 'c'});
        short_opts += "c";
    }
    
    if (options_.include_console_control) {
        long_opts.push_back({"less_console", no_argument, 0, 'l'});
        short_opts += "l";
    }
    
    if (options_.include_alpha) {
        long_opts.push_back({"alpha", no_argument, 0, 'f'});
        short_opts += "f";
    }
    
    if (options_.include_hom_options) {
        long_opts.push_back({"no_hom_opt", no_argument, 0, 'j'});
        long_opts.push_back({"no_col_sweep", no_argument, 0, 'w'});
        short_opts += "jw";
    }
    
    if (options_.include_debug_options) {
        long_opts.push_back({"compare_b", no_argument, 0, 'm'});
        long_opts.push_back({"compare_e", no_argument, 0, 'a'});
        long_opts.push_back({"compare_hom", no_argument, 0, 'i'});
        short_opts += "mai";
    }
    
    if (options_.include_test_files) {
        long_opts.push_back({"test_files", no_argument, 0, 'x'});
        short_opts += "x";
    }
    
    // Terminator
    long_opts.push_back({0, 0, 0, 0});
}

bool OptionParser::handle_no_input(int argc, char**& argv) {
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
    
    // Rebuild argv
    char** new_argv = new char*[args.size() + 1];
    for (size_t i = 0; i < args.size(); ++i) {
        new_argv[i] = new char[args[i].size() + 1];
        std::strcpy(new_argv[i], args[i].c_str());
    }
    new_argv[args.size()] = nullptr;
    
    // Update argc and argv
    argc = args.size();
    argv = new_argv;
    optind = 1; // Reset getopt
    
    return true;
}

bool OptionParser::parse(int argc, char** argv, AIDA_config& config) {
    // Handle no arguments case
    if (argc < 2) {
        if (!handle_no_input(argc, argv)) {
            return false;
        }
    }
    
    // Build option lists
    std::string short_opts;
    std::vector<struct option> long_opts;
    build_option_strings(short_opts, long_opts);
    
    // Parse options
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, short_opts.c_str(), 
                              long_opts.data(), &option_index)) != -1) {
        switch (opt) {
            case 'h':
                display_help();
                return false;
                
            case 'v':
                display_version();
                return false;
                
            case 'o':
                if (!options_.include_output) break;
                write_output_ = true;
                if (optarg) {
                    output_string_ = std::string(optarg);
                } else if (optind < argc && argv[optind][0] != '-') {
                    output_string_ = std::string(argv[optind]);
                    optind++;
                } else {
                    output_string_.clear();
                }
                break;
                
            case 'b':
                if (!options_.include_bruteforce) break;
                config.brute_force = true;
                config.exhaustive = true;
                break;
                
            case 's':
                if (!options_.include_sort) break;
                config.sort = true;
                break;
                
            case 'e':
                if (!options_.include_exhaustive) break;
                config.exhaustive = true;
                break;
                
            case 't':
                if (!options_.include_statistics) break;
                show_indecomp_stats_ = true;
                break;
                
            case 'r':
                if (!options_.include_runtime) break;
                show_runtime_stats_ = true;
                break;
                
            case 'p':
                if (!options_.include_progress) break;
                config.progress = false;
                break;
                
            case 'c':
                if (!options_.include_basechange) break;
                config.save_base_change = true;
                break;
                
            case 'l':
                if (!options_.include_console_control) break;
                config.show_info = false;
                break;
                
            case 'f':
                if (!options_.include_alpha) break;
                config.alpha_hom = true;
                break;
                
            case 'j':
                if (!options_.include_hom_options) break;
                config.turn_off_hom_optimisation = true;
                break;
                
            case 'w':
                if (!options_.include_hom_options) break;
                config.supress_col_sweep = true;
                break;
                
            case 'm':
                if (!options_.include_debug_options) break;
                config.compare_both = true;
                break;
                
            case 'a':
                if (!options_.include_debug_options) break;
                config.exhaustive_test = true;
                break;
                
            case 'i':
                if (!options_.include_debug_options) break;
                config.compare_hom = true;
                break;
                
            case 'x':
                if (!options_.include_test_files) break;
                test_files_ = true;
                break;
                
            default:
                return false;
        }
    }
    
    // Get input file
    if (optind < argc) {
        input_file_ = argv[optind];
    } else if (!test_files_) {
        std::cerr << "No input file specified." << std::endl;
        return false;
    }
    
    return true;
}

void OptionParser::display_help() const {
    std::cout << "AIDA - Decomposition Tool\n\n";
    std::cout << "Usage: aida <input_file> [options]\n\n";
    std::cout << "Input: The input file must be a minimised presentation in scc2020 or firep format.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help           Display this help message\n";
    std::cout << "  -v, --version        Display version information\n";
    
    if (options_.include_output) {
        std::cout << "  -o, --output [path]  Write output to file or directory\n";
    }
    
    if (options_.include_progress || options_.include_console_control) {
        std::cout << "\nGeneral Options:\n";
        if (options_.include_progress) {
            std::cout << "  -p, --progress       Show progress bar\n";
        }
        if (options_.include_console_control) {
            std::cout << "  -l, --less_console   Suppress most console output\n";
        }
    }
    
    if (options_.include_sort || options_.include_bruteforce || 
        options_.include_exhaustive || options_.include_basechange) {
        std::cout << "\nAlgorithm Options:\n";
        if (options_.include_bruteforce) {
            std::cout << "  -b, --bruteforce     Stop hom-space calculation (most optimization)\n";
        }
        if (options_.include_sort) {
            std::cout << "  -s, --sort           Lexicographically sort input relations\n";
        }
        if (options_.include_exhaustive) {
            std::cout << "  -e, --exhaustive     Always iterate over all decompositions\n";
        }
        if (options_.include_basechange) {
            std::cout << "  -c, --basechange     Save base change\n";
        }
        if (options_.include_alpha) {
            std::cout << "  -f, --alpha          Turn on alpha-homs computation\n";
        }
    }
    
    if (options_.include_statistics || options_.include_runtime) {
        std::cout << "\nOutput Options:\n";
        if (options_.include_statistics) {
            std::cout << "  -t, --statistics     Show statistics about indecomposable summands\n";
        }
        if (options_.include_runtime) {
            std::cout << "  -r, --runtime        Show runtime statistics and timers\n";
        }
    }
    
    if (options_.include_hom_options) {
        std::cout << "\nOptimization Options:\n";
        std::cout << "  -j, --no_hom_opt     Disable optimized hom space calculation\n";
        std::cout << "  -w, --no_col_sweep   Disable column sweep optimization\n";
    }
    
    if (options_.include_debug_options) {
        std::cout << "\nDebug/Testing Options:\n";
        std::cout << "  -m, --compare_b      Compare with bruteforce at runtime\n";
        std::cout << "  -a, --compare_e      Compare exhaustive and brute force\n";
        std::cout << "  -i, --compare_hom    Compare optimized/non-opt hom space calculation\n";
    }
    
    if (options_.include_test_files) {
        std::cout << "  -x, --test_files     Run algorithm on test files\n";
    }
    
    std::cout << std::endl;
}

void OptionParser::display_version() const {
    std::cout << "AIDA version 0.2.1\n";
    std::cout << "Copyright 2025 TU Graz\n";
}

} // namespace aida