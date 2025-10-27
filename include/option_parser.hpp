#pragma once

#ifndef AIDA_OPTION_PARSER_HPP
#define AIDA_OPTION_PARSER_HPP

#include <string>
#include <vector>
#include "config.hpp"

namespace aida {
    
    class OptionParser {
    public:
        struct OptionSet {
            bool include_output = true;
            bool include_bruteforce = true;
            bool include_sort = true;
            bool include_exhaustive = true;
            bool include_statistics = true;
            bool include_runtime = true;
            bool include_progress = true;
            bool include_basechange = true;
            bool include_console_control = true;
            bool include_alpha = true;
            bool include_hom_options = true;
            bool include_debug_options = true;
            bool include_test_files = false;
            
            OptionSet() = default;    

        };

    private:
        OptionSet options_;

    public:
        OptionParser();  // Default constructor
        OptionParser(const OptionSet& options); 
        
        // Parse command line arguments and populate config
        bool parse(int argc, char** argv, AIDA_config& config);
        
        // Getters for parsed values
        bool has_input_file() const { return !input_file_.empty(); }
        std::string get_input_file() const { return input_file_; }
        
        bool has_output() const { return write_output_; }
        std::string get_output_string() const { return output_string_; }
        
        bool show_indecomp_statistics() const { return show_indecomp_stats_; }
        bool show_runtime_statistics() const { return show_runtime_stats_; }
        bool test_files() const { return test_files_; }
        
        void display_help() const;
        void display_version() const;
        
    private:
        
        // Parsed state
        std::string input_file_;
        std::string output_string_;
        bool write_output_ = false;
        bool show_indecomp_stats_ = false;
        bool show_runtime_stats_ = false;
        bool test_files_ = false;
        
        void build_option_strings(std::string& short_opts, 
                                  std::vector<struct option>& long_opts);
        bool handle_no_input(int argc, char**& argv);
    };
    
} // namespace aida

#endif // AIDA_OPTION_PARSER_HPP