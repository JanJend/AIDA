
#include "aida_interface.hpp"
#include <unistd.h> 
#include <getopt.h>

namespace fs = std::filesystem;

using namespace graded_linalg;

namespace hnf{

using Block = aida::Block;
using Module_w_slope = std::pair<R2Resolution<int>, double>;
using Block_list = aida::Block_list;
using HN_factors = vec<Module_w_slope>;

struct slope_comparator{
    bool operator()(const Module_w_slope& X, const Module_w_slope& Y) const noexcept {
        return X.second > Y.second;
    }
};


/**
 * @brief Computes the submodule with the highest slope in the given module if X is generated at a single degree.
 * 
 * @param X 
 * @param subspaces 
 * @return Module_w_slope 
 */
Module_w_slope find_scss_bruteforce(const R2GradedSparseMatrix<int>& X, 
        vec<vec<SparseMatrix<int>>>& subspaces, 
        R2GradedSparseMatrix<int>& max_subspace,
        const r2degree& bound){
    int k = X.get_num_rows();
    double max_slope = 0;
    R2Resolution<int> scss;
    if(k == 1){
        scss = R2Resolution<int>(X);
        max_slope = scss.slope(bound);
    } else {
        assert(k < 6);
        if(subspaces.size() < k){
            std::cerr << "Have not loaded enough subspaces" << std::endl;
            std::exit(1);
        }
        
        for(auto ungraded_subspace : subspaces[k-1]){
            int num_gens = ungraded_subspace.get_num_cols();
            R2GradedSparseMatrix<int> subspace = R2GradedSparseMatrix<int>(ungraded_subspace);
            subspace.row_degrees = X.row_degrees;
            subspace.col_degrees = vec<r2degree>(num_gens, X.row_degrees[0]);
            assert(subspace.get_num_rows() == X.get_num_rows());
            assert(subspace.get_num_cols() == num_gens);
            R2GradedSparseMatrix<int> submodule_pres = X.submodule_generated_by(subspace);
            R2Resolution<int> res(submodule_pres);
            double slope = res.slope(bound); 
            if(slope > max_slope){
                max_slope = slope;
                scss = std::move(res);
                max_subspace = subspace;
            }
        }
    }
    return std::make_pair(scss, max_slope); 
}



HN_factors skyscraper_invariant(Block_list& summands, 
        vec<vec<SparseMatrix<int>>>& subspaces, 
        const r2degree& bound){
    HN_factors result;
    for(Block X : summands){
        int dim_at_degree = X.get_num_rows();
        if(dim_at_degree > 4){
            assert(false);
        }
        while(X.get_num_rows() > 1){
            R2GradedSparseMatrix<int> subspace;
            result.emplace_back(find_scss_bruteforce(X, subspaces, subspace, bound));
            if(result.back().first.d1.get_num_rows() == X.get_num_rows()){
                break;
            } else {
                X.quotient_by(subspace);
            }
        }
    }
    return result;
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

template< typename Outputstream>
void to_stream(Outputstream& ostream, Module_w_slope& scss){
    ostream << "Slope: " << scss.second << std::endl;
    scss.first.d1.to_stream_r2(ostream);
}

void write_slopes_to_csv(const vec<vec<double>>& slopes,
        const vec<r2degree>& grid_points,
        const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    for (int i = 0; i < slopes.size(); ++i) {
        file << grid_points[i] << ";";

        const auto& slopes_at_degree = slopes[i];
        for (int j = 0; j < slopes_at_degree.size(); ++j) {
            file << slopes_at_degree[j];
            if (j < slopes_at_degree.size() - 1) {
                file << ",";
            }
        }    
        file << "\n";
    }
    file.close();
}

template<typename Container, typename Outputstream>
void process_list_of_summands(aida::AIDA_functor& decomposer, std::ifstream& istream, Outputstream& ostream, const Container& indecomps) {
    
    int grid_size = 50;
    
    bool progress_bar = false;
    if (decomposer.config.progress){
        progress_bar = true;
        decomposer.config.progress = false;
    } 
    bool show_info = false;
    if (decomposer.config.show_info) {
        decomposer.config.show_info = false;
        show_info = true;
    }
    int num_of_summands = indecomps.size();
    if (show_info) {
        std::cout << "The first decomposition has " << num_of_summands << " indecomposable summands." << std::endl;
    }
    int total_generators = 0;
    
    vec<int> all_scss_dimensions;
    vec<int> first_ind_dimensions;
    vec<int> grid_ind_dimensions;
    int processed_generators = 0;

    pair<r2degree> bounds = indecomps.front().bounding_box();

    for (auto& B : indecomps) {
        total_generators += B.get_num_rows();
        pair<r2degree> B_bounds = B.bounding_box();
        first_ind_dimensions.push_back(B.get_num_rows());
        r2degree lower_bound = Degree_traits<r2degree>::meet(bounds.first, B_bounds.first);
        r2degree upper_bound = Degree_traits<r2degree>::join(bounds.second, B_bounds.second);
        bounds = {lower_bound, upper_bound};
    }

    // TO-DO: FIX THIS, if bounds are negative this cannot work!
    r2degree slope_bound = {2*bounds.second.first, 2*bounds.second.second};
    vec<r2degree> grid_points = get_grid_points(bounds, grid_size);
        
    vec<vec<double>> slopes = vec<vec<double>>(grid_points.size(), vec<double>());  
    for(int i = 0; i< grid_points.size(); i++){ 
        r2degree degree = grid_points[i];
        if (progress_bar) {
            static int last_percent = -1;
            // (-)^{1.5} progress bar for now, but not clear that computational time increases with this exponent.
            int percent = static_cast<int>(static_cast<double>(i) / grid_points.size() * 100);
            if (percent != last_percent) {
                // Calculate the number of symbols to display in the progress bar
                int num_symbols = percent / 2;
                std::cout << "\r" << i << " grid points : [";
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
            if (processed_generators >= total_generators) {
                std::cout << std::endl;
            }
        }
        ostream << degree << std::endl;
        for(auto& B : indecomps){
 
            auto B_induced = B.submodule_generated_at(degree);
            if(B_induced.get_num_rows() == 1){
                grid_ind_dimensions.push_back(1);
                all_scss_dimensions.push_back(1);
                R2Resolution<int> res(B_induced);
                double slope = res.slope(slope_bound);
                if(slope == INFINITY){
                    assert(false);
                    std::cerr << "Slope is infinite, consider passing a bound." << std::endl;
                }

                slopes[i].push_back(slope);
                Module_w_slope single_stable = std::make_pair(res, slope);
                to_stream(ostream, single_stable);

            } else if ( B_induced.get_num_rows() == 0){
                ostream << "0" << std::endl;
            } else {
                aida::Block_list sub_B_list;
                B_induced.compute_col_batches();
                decomposer(B_induced, sub_B_list);
                int max_dim = 0;
                for(Block sub_B : sub_B_list){
                    if(sub_B.get_num_rows() > max_dim){
                        max_dim = sub_B.get_num_rows();
                    }
                    grid_ind_dimensions.push_back(sub_B.get_num_rows());
                }
                auto subspaces = all_sparse_proper_subspaces(max_dim);
                auto skyscraper_degree = skyscraper_invariant(sub_B_list, subspaces, slope_bound);

                for(auto& hn_factor : skyscraper_degree){
                    all_scss_dimensions.push_back(hn_factor.first.d1.get_num_rows());
                    if(hn_factor.second == INFINITY){
                        assert(false);
                    }
                    slopes[i].push_back(hn_factor.second);
                    to_stream(ostream, hn_factor);
                }
            }
        }

        std::sort(slopes[i].begin(), slopes[i].end());
    }



    std::cout << " tracked the dimensions of " << grid_ind_dimensions.size() << " indecomposable summands." << std::endl;

    calculate_stats(grid_ind_dimensions);
    calculate_stats(all_scss_dimensions);

    write_slopes_to_csv(slopes, grid_points, "slopes.csv");

}

template <typename Outputstream>
void full_grid_induced_decomposition(aida::AIDA_functor& decomposer, std::ifstream& istream, Outputstream& ostream, bool show_indecomp_statistics, bool show_runtime_statistics, bool is_decomposed = false){
    
    if(is_decomposed){
        vec<R2GradedSparseMatrix<int>> matrices;
        graded_linalg::construct_matrices_from_stream(matrices, istream);
        process_list_of_summands(decomposer, istream, ostream, matrices);
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
        process_list_of_summands(decomposer, istream, ostream, B_list);
    }
    
} // full_grid_induced_decomposition

} // namespace hnf

