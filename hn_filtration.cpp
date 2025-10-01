#include "aida_interface.hpp"
#include <unistd.h> 
#include <getopt.h>
#include "hn_filtration.hpp"

namespace fs = std::filesystem;

void display_help() {
    std::cout << "Usage: ./aida <input_file> [options]\n"
              << "Options:\n"
              << "  -h, --help           Display this help message\n"
              << "  -g, --diagonal       Save a copy where each subquotient is restricted to the diagonal to compute landscapes\n"
              << "  -d, --is_decomposed  Specify if the input is already decomposed\n"
              << "  -v, --version        Display version information\n"
              << "  -b, --bruteforce     Stops hom-space calculation and thus most optimisation. \n"
              << "  -s, --sort           Lexicographically sorts the relations of the input\n"
              << "  -e, --exhaustive     Always iterates over all decompositions of a batch\n"
              << "  -t, --statistics     Show statistics about indecomposable summands\n"
              << "  -r, --runtime        Show runtime statistics and timers\n"
              << "  -p, --progress       Turn off progressbar\n"
              << "  -c, --basechange     Save base change\n"
              << "  -o, --output <file>  Specify output file\n"
              << "  -l, --less_console   Suppreses most console output\n"
              << "  -m, --compare_b      Compares with -b at runtime, then runs with only -b and compares.\n"
              << "  -a, --compare_e      Compares exhaustive and brute force at runtime.\n"
              << "  -i, --compare_hom    Compares optimised and non-opt hom space calculation at runtime.\n"
              << "  -j, --no_hom_opt     Does not use the optimised hom space calculation.\n"
              << "  -w, --no_col_sweep   Does not use the column sweep optimisation.\n"
              << "  -f, --alpha       Turns the computation of alpha-homs on.\n"
              << "  -x, -test_files          Runs the algorithm on some test files.\n"
              << "      <file> is optional and will default to the <input_file> with _decomposed appended\n"
              << "      You can pass relative and absolute paths as well as only a directory."
              << "Further Instructions: \n Make sure that the inputfile is a (sequence of) scc or firep presentations that are minimised.\n"
              << std::endl;
}

void display_version() {
    std::cout << "AIDA version 0.2 -- 21st Mar 2025\n";
}



void write_to_file(std::ostringstream& ostream, std::string& output_file_path, std::string& input_directory, std::string& file_without_extension, std::string& extension, std::string& output_string){

    if(output_string.empty()){
        output_file_path = input_directory + "/" + file_without_extension + extension;
    } else {
        std::filesystem::path output_path(output_string);
        if (output_path == ".") {
            output_file_path = std::filesystem::current_path().string() + "/" + file_without_extension +  extension;
        } else if (output_path.is_relative()) {
            output_file_path = std::filesystem::current_path().string() + "/" + output_string;
        } else if (std::filesystem::is_directory(output_path)) {
            output_file_path = output_path.string() + "/" + file_without_extension + extension;
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
        std::cout << "HN filtration written to " << output_file_path << std::endl;
    } else {
        std::cout << "Error: Could not write HN filtration to file: " << output_file_path << std::endl;
    }
}

int main(int argc, char** argv){
    aida::AIDA_functor decomposer = aida::AIDA_functor();
    
    decomposer.config.exhaustive = false;
    decomposer.config.brute_force = false;
    decomposer.config.sort = false;
    decomposer.config.sort_output = true;
    decomposer.config.alpha_hom = false;
    decomposer.config.progress = true;
    bool write_output = false;
    bool diagonal_output = false;

    bool show_indecomp_statistics = false;
    bool show_runtime_statistics = false;
    decomposer.config.show_info = true;

    decomposer.config.compare_both = false; // Compares normal functioning and brute force at runtime, then also compares output.
    // bool compare_time = false;
    decomposer.config.exhaustive_test = false;

    // bool compare_hom_internal = false; // Cannot be used with the functor right now.
    bool test_files = false;
    bool is_decomposed = false;

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
        {"alpha", no_argument, 0, 'f'},
        {"test_files", no_argument, 0, 'x'},
        {"is_decomposed", no_argument, 0, 'd'},
        {"diagonal", no_argument, 0, 'g'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "ho::gbsetrpclmvaijwfxd", long_options, &option_index)) != -1) {
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
                decomposer.config.progress = false;
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
                decomposer.config.alpha_hom = true;
                break;
            case 'x':
                test_files = true;
                break;
            case 'd':
                is_decomposed = true;
                break;
            case 'g':
                diagonal_output = true;
                break;
            default:
                return 1;
        }
    }

    std::string file_without_extension;
    std::string extension = ".sky";


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
        } else {
            file_without_extension = filename.substr(0, dot_position);
        }
    } else if (test_files) {
        // Do nothing
    } else {
        fs::path cpp_path = fs::path(__FILE__).parent_path();
        fs::path test_file_folder = cpp_path / "Persistence-Algebra/test_presentations";
        fs::path ex1 = "two_circles_2_dim1_minpres.scc";
        matrix_path = test_file_folder / ex1;
        input_directory = test_file_folder.string();
        filename = ex1.string();
        size_t dot_position = filename.find_last_of('.');
        file_without_extension = filename.substr(0, dot_position);
        std::cout << "No input file specified. Running on test file: " << matrix_path << std::endl;
    }
    
    std::ostringstream ostream;
    if(!test_files){
        std::ifstream istream(matrix_path);
        if (!istream.is_open()) {
                std::cerr << "Error: Could not open input file: " << matrix_path << std::endl;
                return 0;
        }
         std::cout << "Decomposing, then generating submodules at grid points and decomposing them again of " + filename << std::endl;
         ostream << std::fixed << std::setprecision(17);
        hnf::full_grid_induced_decomposition(decomposer, istream, ostream, show_indecomp_statistics, show_runtime_statistics, true, is_decomposed);
         
     } else {
        // Run on some test files.
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
     
     // aida::index num_indecomp = decomposer.cumulative_statistics.num_of_summands;
     
     if(write_output){
         write_to_file(ostream, output_file_path, input_directory, file_without_extension, extension, output_string);
     }
 
     return 0;
 } //main
 
 
 