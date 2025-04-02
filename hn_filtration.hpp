
#include "aida_interface.hpp"
#include <unistd.h> 
#include <getopt.h>

namespace fs = std::filesystem;

using namespace graded_linalg;

namespace hnf{

using Block = aida::Block;
using Module_w_slope = std::pair<Block, double>;
using Block_list = aida::Block_list;
using HN_factors = std::list<Module_w_slope>;

HN_factors skyscraper_invariant(Block_list& summands, vec<vec<SparseMatrix<int>>>& subspaces){
    for(Block X : summands){
        int k = X.get_num_rows();
        if(subspaces.size() < k){
            std::cerr << "Have not loaded enough subspaces" << std::endl;
            std::exit(1);
        }
        double max_slope = 0;
        R2Resolution<int> scss;
        for(auto ungraded_subspace : subspaces[k-1]){
            int num_gens = ungraded_subspace.get_num_cols();
            R2GradedSparseMatrix<int> subspace = R2GradedSparseMatrix<int>(ungraded_subspace);
            subspace.row_degrees = X.row_degrees;
            subspace.col_degrees = vec<r2degree>(num_gens, X.row_degrees[0]);
            R2GradedSparseMatrix<int> submodule_pres = X.submodule_generated_by(subspace);
            R2Resolution<int> res(submodule_pres);
            double slope = res.slope(); 
            if(slope > max_slope){
                max_slope = slope;
                scss = res;
            }
        }
    }
}


void calculate_stats(const std::vector<int>& all_dimensions) {
    if (all_dimensions.empty()) {
        std::cout << "The vector is empty!" << std::endl;
        return;
    }

    int max_value = *std::max_element(all_dimensions.begin(), all_dimensions.end());


    double sum = std::accumulate(all_dimensions.begin(), all_dimensions.end(), 0);
    double average = sum / all_dimensions.size();

    double squared_diff_sum = 0;
    for (int val : all_dimensions) {
        squared_diff_sum += (val - average) * (val - average);
    }
    double variance = squared_diff_sum / all_dimensions.size();
    double standard_deviation = std::sqrt(variance);

    std::cout << "Maximum: " << max_value << std::endl;
    std::cout << "Average: " << average << std::endl;
    std::cout << "Standard Deviation: " << standard_deviation << std::endl;
}

vec<r2degree> get_grid_points( pair<r2degree> bounds, int grid_size) {
    vec<r2degree> grid_points;
    double x_min = bounds.first.first;
    double x_max = bounds.second.first;
    double y_min = bounds.first.second;
    double y_max = bounds.second.second;

    double x_step = (x_max - x_min) / (grid_size - 1);
    double y_step = (y_max - y_min) / (grid_size - 1);

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            grid_points.push_back({x_min + i * x_step, y_min + j * y_step});
        }
    }
    return grid_points;
}

 

template<typename Container>
void process_list_of_summands(aida::AIDA_functor& decomposer, std::ifstream& istream, const Container& indecomps) {
    
    int grid_size = 50;
    
    bool progress_bar = false;
    if (decomposer.config.progress){
        progress_bar = true;
        decomposer.config.progress = false;
    } 
    bool show_info = false;
    if (decomposer.config.show_info) {
        decomposer.config.show_info = false;
    }
    int num_of_summands = indecomps.size();
    if (show_info) {
        std::cout << "The first decomposition has " << num_of_summands << " indecomposable summands." << std::endl;
    }
    vec<int> all_dimensions;
    int current_block = 0;
    for(auto& B : indecomps){
        current_block++;

        if (progress_bar) {
            static int last_percent = -1;
            // (-)^{1.5} progress bar for now, but not clear that computational time increases with this exponent.
            int percent = static_cast<int>(pow(static_cast<double>(current_block) / num_of_summands, 1.5) * 100);
            if (percent != last_percent) {
                // Calculate the number of symbols to display in the progress bar
                int num_symbols = percent / 2;
                std::cout << "\r" << current_block << " summands : [";
                // Print the progress bar
                for (int i = 0; i < 50; ++i) {
                    if (i < num_symbols) {
                        std::cout << "#";
                    } else {
                        std::cout << " ";
                    }
                }
                std::cout << "] " << percent << "%";
                std::flush(std::cout);
                last_percent = percent;
            }
            if (current_block == num_of_summands) {
                std::cout << std::endl;
            }
        }

        if(B.get_num_rows() == 1){
            all_dimensions.push_back(1);
        }
        pair<r2degree> bounds = B.bounding_box();
        vec<r2degree> support = B.discrete_support();
        if(support.size() <= grid_size){
            grid_size = support.size();
        }
        vec<r2degree> grid_points = get_grid_points(bounds, grid_size);

        for(auto& degree : support){           
            auto B_induced = B.submodule_generated_at(degree);
            if(B_induced.get_num_rows() == 1){
                all_dimensions.push_back(1);
            } else if ( B_induced.get_num_rows() == 0){

            } else {
                aida::Block_list sub_B_list;
                B_induced.compute_col_batches();
                decomposer(B_induced, sub_B_list);
                // HN of sub_B_list

                for(auto& sub_B : sub_B_list){
                    all_dimensions.push_back(sub_B.get_num_rows());
                }
            }
        }
    }
    std::cout << " tracked the dimension of " << all_dimensions.size() << " indecomposable summands." << std::endl;
    calculate_stats(all_dimensions);
}


void full_grid_induced_decomposition(aida::AIDA_functor& decomposer, std::ifstream& istream, bool show_indecomp_statistics, bool show_runtime_statistics, bool is_decomposed = false){
    
    if(is_decomposed){
        vec<R2GradedSparseMatrix<int>> matrices;
        graded_linalg::construct_matrices_from_stream(matrices, istream);
        process_list_of_summands(decomposer, istream, matrices);
    } else {
        aida::Block_list B_list;
        decomposer(istream, B_list);
        if(show_indecomp_statistics){
            decomposer.cumulative_statistics.print_statistics();
        }
        if(show_runtime_statistics){
            decomposer.cumulative_runtime_statistics.print();
            #if TIMERS
                decomposer.cumulative_runtime_statistics.print_timers();
            #endif
        }
        process_list_of_summands(decomposer, istream, B_list);
    }
    
} // full_grid_induced_decomposition

} // namespace hnf

