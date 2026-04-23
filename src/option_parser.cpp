#include "option_parser.hpp"
#include "CLI11.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace aida {

bool OptionParser::parse(int argc, char** argv, AIDA_config& config) {
    CLI::App app{"AIDA - Decomposition Tool"};
    app.set_version_flag("-v,--version", "AIDA version 0.2.1\nCopyright 2025 TU Graz");

    // Positional input file
    app.add_option("input_file", input_file_,
                   "Minimised presentation in scc2020 or firep format");

    // Output
    CLI::Option* out_opt = nullptr;
    if (options_.include_output) {
        out_opt = app.add_option("-o,--output", output_string_,
                                 "Write output to file or directory")
                      ->expected(0, 1);
    }

    // Algorithm options
    if (options_.include_bruteforce) {
        app.add_flag_callback("-b,--bruteforce",
            [&config]() { config.brute_force = true; config.exhaustive = true; },
            "Stop hom-space calculation (most optimization)");
    }
    if (options_.include_sort) {
        app.add_flag("-s,--sort", config.sort,
                     "Lexicographically sort input relations");
    }
    if (options_.include_exhaustive) {
        app.add_flag("-e,--exhaustive", config.exhaustive,
                     "Always iterate over all decompositions");
    }
    if (options_.include_basechange) {
        app.add_flag("-c,--basechange", config.save_base_change,
                     "Save base change");
    }
    if (options_.include_alpha) {
        app.add_flag("-f,--alpha", config.alpha_hom,
                     "Turn on alpha-homs computation");
    }

    // Progress / console
    if (options_.include_progress) {
        // Preserving original behaviour: -p sets progress = false
        app.add_flag_callback("-p,--progress",
            [&config]() { config.progress = false; },
            "Toggle progress bar");
    }
    if (options_.include_console_control) {
        app.add_flag_callback("-l,--less_console",
            [&config]() { config.show_info = false; },
            "Suppress most console output");
    }

    // Hom optimization
    if (options_.include_hom_options) {
        app.add_flag("-j,--no_hom_opt", config.turn_off_hom_optimisation,
                     "Disable optimized hom space calculation");
        app.add_flag("-w,--no_col_sweep", config.supress_col_sweep,
                     "Disable column sweep optimization");
    }

    // Statistics
    if (options_.include_statistics) {
        app.add_flag("-t,--statistics", show_indecomp_stats_,
                     "Show statistics about indecomposable summands");
    }
    if (options_.include_runtime) {
        app.add_flag("-r,--runtime", show_runtime_stats_,
                     "Show runtime statistics and timers");
    }

    // Debug
    if (options_.include_debug_options) {
        app.add_flag("-m,--compare_b", config.compare_both,
                     "Compare with bruteforce at runtime");
        app.add_flag("-a,--compare_e", config.exhaustive_test,
                     "Compare exhaustive and brute force");
        app.add_flag("-i,--compare_hom", config.compare_hom,
                     "Compare optimized/non-opt hom space calculation");
    }

    if (options_.include_test_files) {
        app.add_flag("-x,--test_files", test_files_,
                     "Run algorithm on test files");
    }

    // If no arguments, prompt interactively (matches original behaviour)
    if (argc < 2) {
        std::cout << app.help() << std::endl;
        std::cout << "Please provide options/arguments: ";
        std::string line;
        std::getline(std::cin, line);

        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }

        try {
            // CLI11 parses vector<string> in reverse order
            std::vector<std::string> reversed(tokens.rbegin(), tokens.rend());
            app.parse(reversed);
        } catch (const CLI::ParseError& e) {
            app.exit(e);
            return false;
        }
    } else {
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            app.exit(e);
            return false;
        }
    }

    // Detect whether -o was passed (with or without a value)
    if (out_opt && out_opt->count() > 0) {
        write_output_ = true;
    }

    // Validate: need an input file unless running on test files
    if (input_file_.empty() && !test_files_) {
        std::cerr << "No input file specified." << std::endl;
        return false;
    }

    return true;
}

} // namespace aida