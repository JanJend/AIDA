/**
 * @file aida_interface.hpp
 * @author Jan Jendrysiak 
 * @brief How to use the AIDA Functor:
 * 
 * 1. Create an instance of the AIDA_functor.
 * 2. If you want to use only parts of the algorithm or need to sort input matrices, use the set_config method.
 * 3. Call the functor with an input stream and an output stream.
 * Results:
 * 1. The output stream will contain the decompositions.
 * 2. The statistics_vec will contain statistics about the indecomposables.
 * 3. The runtime_statistics will contain information about the presentation matrix while decomposing.
 * 
 * Configuration: 
 * 1. sort_output: Sorts the indecomposable summands by row degrees, then column degrees.
 * 2. sort: Sorts the columns of the input matrices lexicographically.
 * 3. exhaustive: Uses the exhaustive algorithm for the alpha-decomposition.
 * 4. brute_force: Uses the exhaustive algorithm and also does not compute hom-spaces explicitly.
 * 5. compare_both: Compares the hom space and direct version of block_reduce. Only for debugging.
 * 
 * GLOBAL VARIABLES
 * 
 * TIMERS : If set to 1, runtime_statistics will also contain information about the time spent in the different parts of the algorithm.
 * 
 * Example usage:
 * 
 * std::ifstream istream(matrix_path);
 * std::ostringstream ostream;
 * aida::AIDA_functor decomposer = aida::AIDA_functor();
 * decomposer.config.progress = true;
 * decomposer(istream, ostream);
 * decomposer.cumulative_statistics.print_statistics();
 * decomposer.cumulative_runtime_statistics.print();
 * 
 * Can use Compare_by_degrees<degree, index> to compare the degrees of graded matrices.
 * If you want to compare two streams of graded matrices with this method, 
 * use compare_streams_of_graded_matrices
 * and compare_files_of_graded_matrices for scc/firep files.
 * 
 * 
 * @version 0.1
 * @date 2024-10-07
 * 
 * @copyright ?
 * 
 * 
 */


#pragma once

#ifndef AIDA_INTERFACE_HPP
#define AIDA_INTERFACE_HPP

#ifndef AIDA_WITH_STATS
// If the flag is not set, they can be changed here (for developing and debugging)
#define TIMERS 1

#else
// Do not touch this, otherwise the executables
// "aida_with_stats" and "aida_without_stats" won't be compiled correctly
#if AIDA_WITH_STATS
#define TIMERS 1
#else
#define TIMERS 0
#endif
// end of "do not touch this"
#endif

#include "aida.hpp"
#include <regex>

namespace aida{

namespace fs = std::filesystem;

/**
 * @brief Computes and processes statistics about indecomposables
 * 
 */
struct AIDA_statistics {

    index total_num_rows;
    index num_of_summands;
    index num_of_free;
    index num_of_cyclic; // non-free
    index num_of_intervals; // non-cyclic
    index num_of_non_intervals;
    index gen_max;
    
    index size_of_intervals;
    index size_of_non_intervals;
    double interval_ratio;
    double interval_size_ratio;

    AIDA_statistics() : num_of_summands(0), num_of_free(0), num_of_cyclic(0), num_of_intervals(0), num_of_non_intervals(0), size_of_intervals(0), size_of_non_intervals(0), interval_ratio(0), interval_size_ratio(0), total_num_rows(0), gen_max(0) {}

    void compute_statistics(Block_list& B_list ){
        num_of_summands = B_list.size();
        for(Block& B : B_list){
            if(B.type == BlockType::FREE){
                num_of_free++;
                total_num_rows++;
            } else if (B.type == BlockType::CYC){
                num_of_cyclic++;
                total_num_rows++;
            } else if (B.type == BlockType::INT ){
                num_of_intervals++;
                size_of_intervals += B.rows.size();
                total_num_rows += B.rows.size();
                if(B.rows.size() > gen_max){
                    gen_max = B.rows.size();
                }
            } else {
                size_of_non_intervals += B.rows.size();
                total_num_rows += B.rows.size();
                if(B.rows.size() > gen_max){
                    gen_max = B.rows.size();
                }
            }
        }
        interval_ratio = static_cast<double>(num_of_intervals + num_of_free + num_of_cyclic) / num_of_summands;
        interval_size_ratio = static_cast<double>(size_of_intervals + num_of_free + num_of_cyclic) / (total_num_rows);
    }

    void operator+=(const AIDA_statistics& other){
        num_of_summands += other.num_of_summands;
        num_of_free += other.num_of_free;
        num_of_cyclic += other.num_of_cyclic;
        num_of_intervals += other.num_of_intervals;
        num_of_non_intervals += other.num_of_non_intervals;
        size_of_intervals += other.size_of_intervals;
        size_of_non_intervals += other.size_of_non_intervals;
        total_num_rows += other.total_num_rows;
        gen_max = std::max(gen_max, other.gen_max);
    }
   
    void print_statistics(){
        std::cout << "Statistics for indecomposable summands: " << std::endl;
        std::cout << "  # indecomposables: " << num_of_summands << std::endl;
        std::cout << "  # free summands: " << num_of_free << std::endl;
        std::cout << "  # non-free cyclic summands: " << num_of_cyclic << std::endl;
        std::cout << "  # non-cyclic intervals: " << num_of_intervals << std::endl;
        std::cout << "  # non-intervals: " << num_of_summands - (num_of_intervals + num_of_cyclic + num_of_free) << std::endl;
        std::cout << "  # generators of non-cyclic intervals: " << size_of_intervals << std::endl;
        std::cout << "  # generators of non-intervals: " << size_of_non_intervals << std::endl;
        std::cout << "  # generators of largest (non-cyclic) indecomoposable: " << gen_max << std::endl;
        std::cout << "  Ratio of intervals: " << interval_ratio << std::endl;
        std::cout << "  Ratio of generators belonging to intervals : " << interval_size_ratio << std::endl;
    }
};

/**
 * @brief Configures the AIDA Functor.
 * 
 */


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
    std::string base_path = getExecutableDir();
    std::string relative_path_1 = "/../lists_of_decompositions";
    std::string relative_path_2 = "/lists_of_decompositions";

    std::string full_path_1 = base_path + relative_path_1;
    std::string full_path_2 = base_path + relative_path_2;

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

/**
 * @brief Applies AIDA to streams.
 * 
 */
struct AIDA_functor {

    AIDA_config config;
    vec<AIDA_statistics> statistics_vec;
    AIDA_statistics cumulative_statistics;
    vec<AIDA_runtime_statistics> runtime_statistics;
    vec<vec<transition>> vector_space_decompositions;
    AIDA_runtime_statistics cumulative_runtime_statistics;
    vec<Full_merge_info> merge_data_vec;
    vec<std::shared_ptr<Base_change_virtual>> base_changes;

    AIDA_functor() : base_changes(), config(), statistics_vec(), runtime_statistics(), merge_data_vec(), cumulative_statistics(), cumulative_runtime_statistics() {}

    /** This stores all vector space decompositions
    / DecompTree: bitset-> vector(vector( Pair of Matrices)), 
    / The first vector indicates the pivots, the second the Plücker Coordinates.
    */
    void load_vector_space_decompositions(int max_dim, std::string decomp_path){
        std::string tran_path = decomp_path + "/transitions_reduced_";
        int start = vector_space_decompositions.size();
        for(int k = start + 2; k <= max_dim; k++) {
            try { 
                vector_space_decompositions.emplace_back(load_transition_list(tran_path + std::to_string(k) + ".bin"));
            } catch (std::exception& e) {
                std::cout << "Could not load transitions_reduced_" << k << ".bin " << std::endl;
                abort();
            }
        }
    }

    void load_existing_decompositions(int& k_max){
        if(k_max > vector_space_decompositions.size() + 1){
            std::string decomp_path = findDecompositionsDir();
            int largest_local_decomposition_list = findLargestNumberInFilenames(decomp_path);
            if(k_max <= largest_local_decomposition_list){
                if(config.show_info){
                    std::cout << "Loading vector space decompositions up to dim " << k_max << std::endl;
                }
                load_vector_space_decompositions(k_max, decomp_path);
            } else {
                load_vector_space_decompositions(largest_local_decomposition_list, decomp_path);
                if(config.show_info){
                    std::cout << "k_max is " << k_max << " but only found decompositions up to dim " << largest_local_decomposition_list << 
                ". \n It is possible that the computation will produce an error if we need to decompose more relations at the same time." << std::endl;
                }
            }
        }
    }

    void compute_vector_space_decompositions_faulty(const std::string& path){
        // Rewrite this.
        const std::string command = "../generate_decompositions -at -cover -transitions ";
        int result = system( (command + std::to_string(9)).c_str() );
    }

    void clear_decompositions(){
        vector_space_decompositions.clear();
    }

    template<typename InputStream, typename OutputStream>
    void operator()(InputStream& ifstr, OutputStream& ofstr) {
        Block_list B_list_cumulative;
        // Decompose the stream into matrices and process each one
        vec<GradedMatrix> matrices;
        #if TIMERS
            aida::load_matrices_timer.start();
        #endif
        construct_matrices_from_stream(matrices, ifstr, config.sort, true);
        #if TIMERS
            aida::load_matrices_timer.stop();
            double load_matrices = aida::load_matrices_timer.elapsed().wall/1e9;
            std::cout << "Time to load matrices: " << load_matrices << std::endl;
        #endif
        
        int k_max = 0;
        for(auto& A : matrices){
            if(A.k_max > k_max){
                k_max = A.k_max;
            }
        }

        load_existing_decompositions(k_max);

        for (GradedMatrix& A : matrices) {

            if(config.show_info && matrices.size() == 1){
                std::cout << " Matrix has " << A.get_num_rows() << " rows and " << A.get_num_cols() << 
                " columns, k_max is " << A.k_max << ", and there are " << A.col_batches.size() << " batches." << std::endl;
            }
            std::shared_ptr<Base_change_virtual> base_change;
            if(config.save_base_change){
                base_change = std::make_shared<Base_change>();
            } else {
                base_change = std::make_shared<Null_base_change>();
            }
            base_changes.push_back(base_change);
            statistics_vec.push_back(AIDA_statistics());
            runtime_statistics.push_back(AIDA_runtime_statistics());
            Block_list B_list;
            Full_merge_info merge_info;

            #if TIMERS
                runtime_statistics.back().initialise_timers();
            #endif

            AIDA(A, B_list, vector_space_decompositions, base_changes.back(), runtime_statistics.back(), config, merge_info);
            #if TIMERS
                runtime_statistics.back().evaluate_timers();
            #endif

            merge_data_vec.push_back(merge_info);
            statistics_vec.back().compute_statistics(B_list);
            B_list_cumulative.splice(B_list_cumulative.end(), B_list);
        }
        
        std::cout << B_list_cumulative.size() << " indecomposable summands." << std::endl;

        cumulative_statistics = AIDA_statistics();
        cumulative_statistics.compute_statistics(B_list_cumulative);
        cumulative_runtime_statistics = AIDA_runtime_statistics();
        for(auto& runtime_stat : runtime_statistics){
            cumulative_runtime_statistics += runtime_stat;
        }

        if(config.sort_output){
            B_list_cumulative.sort(Compare_by_degrees<r2degree, index>());
        }

        for(auto& indecomposable : B_list_cumulative){
            indecomposable.to_stream(ofstr);
        }
    }

    template<typename GradedMatrix>
    void operator()(GradedMatrix& input, Block_list& B_list) {
        
        int k_max = input.k_max;
        load_existing_decompositions(k_max);

        if(config.show_info && matrices.size() == 1){
            std::cout << " Matrix has " << A.get_num_rows() << " rows and " << A.get_num_cols() << 
            " columns, k_max is " << A.k_max << ", and there are " << A.col_batches.size() << " batches." << std::endl;
        }
        std::shared_ptr<Base_change_virtual> base_change;
        if(config.save_base_change){
            base_change = std::make_shared<Base_change>();
        } else {
            base_change = std::make_shared<Null_base_change>();
        }
        base_changes.push_back(base_change);
        statistics_vec.push_back(AIDA_statistics());
        runtime_statistics.push_back(AIDA_runtime_statistics());
        Full_merge_info merge_info;

        #if TIMERS
            runtime_statistics.back().initialise_timers();
        #endif

        AIDA(input, B_list, vector_space_decompositions, base_changes.back(), runtime_statistics.back(), config, merge_info);
        #if TIMERS
            runtime_statistics.back().evaluate_timers();
        #endif

        merge_data_vec.push_back(merge_info);
        statistics_vec.back().compute_statistics(B_list);
        
        
        std::cout << B_list.size() << " indecomposable summands." << std::endl;

        cumulative_statistics = AIDA_statistics();
        cumulative_statistics.compute_statistics(B_list);
        cumulative_runtime_statistics = AIDA_runtime_statistics();
        for(auto& runtime_stat : runtime_statistics){
            cumulative_runtime_statistics += runtime_stat;
        }

        if(config.sort_output){
            B_list.sort(Compare_by_degrees<r2degree, index>());
        }

    }
    
};

} // namespace aida

#endif // AIDA_INTERFACE_HPP

