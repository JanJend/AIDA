/**
 * @file option_parser.hpp
 * @author Jan Jendrysiak
 * @version 0.3
 * @date 2025-10-21
 * @brief parses input options for the AIDA decomposer
 * @copyright 2025 TU Graz
 */
#pragma once
#ifndef AIDA_OPTION_PARSER_HPP
#define AIDA_OPTION_PARSER_HPP

#include "config.hpp"
#include <string>

namespace aida {

class OptionParser {
public:
    struct OptionSet {
        // Algorithmic options — enable when used as a library
        bool include_bruteforce = true;
        bool include_sort = true;
        bool include_exhaustive = true;
        bool include_progress = true;
        bool include_basechange = true;
        bool include_alpha = true;
        bool include_hom_options = true;

        // Main-program options
        bool include_output = false;
        bool include_statistics = false;
        bool include_console_control = false;
        bool include_debug_options = false;
        bool include_test_files = false;
        bool include_runtime = false;
    };

    OptionParser() = default;
    explicit OptionParser(const OptionSet& options) : options_(options) {}

    void enable_stats_and_output() {
        options_.include_output = true;
        options_.include_statistics = true;
        options_.include_runtime = true;
        options_.include_console_control = true;
        options_.include_debug_options = true;
        options_.include_test_files = false;
    }

    // Returns true if parsing succeeded and execution should continue.
    // Returns false if --help/--version was shown or on parse error.
    bool parse(int argc, char** argv, AIDA_config& config);

    bool has_input_file() const { return !input_file_.empty(); }
    std::string get_input_file() const { return input_file_; }

    bool has_output() const { return write_output_; }
    std::string get_output_string() const { return output_string_; }

    bool show_indecomp_statistics() const { return show_indecomp_stats_; }
    bool show_runtime_statistics() const { return show_runtime_stats_; }
    bool test_files() const { return test_files_; }

private:
    OptionSet options_;

    std::string input_file_;
    std::string output_string_;
    bool write_output_ = false;
    bool show_indecomp_stats_ = false;
    bool show_runtime_stats_ = false;
    bool test_files_ = false;
};

} // namespace aida

#endif // AIDA_OPTION_PARSER_HPP