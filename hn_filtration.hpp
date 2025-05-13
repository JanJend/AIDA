
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

double evaluate_slope_polynomial( std::array<double, 4>& coeffs, r2degree d){
    double& x = d.first;
    double& y = d.second;
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*x*y;
}

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
        const pair<r2degree>& bounds){
    int k = X.get_num_rows();
    double max_slope = 0;
    R2Resolution<int> scss;
    if(k == 1){
        scss = R2Resolution<int>(X);
        max_slope = scss.slope(bounds);
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
            double slope = res.slope(bounds); 
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
        const pair<r2degree>& bounds){
    HN_factors result;
    for(Block X : summands){
        int dim_at_degree = X.get_num_rows();
        if(dim_at_degree > 4){
            assert(false);
        }
        while(X.get_num_rows() > 1){
            R2GradedSparseMatrix<int> subspace;
            result.emplace_back(find_scss_bruteforce(X, subspaces, subspace, bounds));
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

vec<r2degree> get_grid_diagonal( pair<r2degree> bounds, int grid_length) {
    vec<r2degree> grid_diagonal = vec<r2degree>();
    double x_min = bounds.first.first;
    double x_max = bounds.second.first;
    double y_min = bounds.first.second;
    double y_max = bounds.second.second;

    double x_step = (x_max - x_min) / (grid_length - 1);
    double y_step = (y_max - y_min) / (grid_length - 1);

    for (int i = 0; i < grid_length; ++i) {
        grid_diagonal.push_back({x_min + i * x_step, y_min + i * y_step});
    }
    return grid_diagonal;
}

template< typename Outputstream>
void to_stream(Outputstream& ostream, Module_w_slope& scss){
    if(scss.first.d1.get_num_rows() == 1){
        ostream << scss.second;
        for(r2degree d : scss.first.d1.col_degrees){
            ostream << "," << "(" << d.first << ";" << d.second << ")";
        }
        ostream << std::endl;
    } else {
        std::cerr << "  Passing a submodule of dimension " << scss.first.d1.get_num_rows() << std::endl;
        ostream << scss.second << std::endl;
        scss.first.d1.to_stream_r2(ostream);
    }
    
    
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

void show_progress_bar(int& i, int& total, std::string& name) {

    static int last_percent = -1;
    int percent = static_cast<int>(static_cast<double>(i) / total * 100);
    if (percent != last_percent) {
        // Calculate the number of symbols to display in the progress bar
        int num_symbols = percent / 2;
        std::cout << "\r" << i << " " << name << " : [";
        // Print the progress bar
        for (int j = 0; j < 50; ++j) {
            if (j < num_symbols) {
                std::cout << "#";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << percent << "%";
        std::flush(std::cout);
        last_percent = percent;
    }
    if (i >= total) {
        std::cout << std::endl;
    }
}

template<typename Container, typename Outputstream>
void process_list_of_summands(aida::AIDA_functor& decomposer, 
    std::ifstream& istream, Outputstream& ostream, 
    vec<r2degree>& grid_points, const Container& indecomps) {
    
    int grid_length = 50;
    
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

    // This is super ineffcient and should be computed earlier directly from the module before decomposition
    // TO-DO: There is also a computation error here, the upper bound is not computed correctly.
    for (auto& B : indecomps) {
        total_generators += B.get_num_rows();
        pair<r2degree> B_bounds = B.bounding_box();
        first_ind_dimensions.push_back(B.get_num_rows());
        r2degree lower_bound = Degree_traits<r2degree>::meet(bounds.first, B_bounds.first);
        r2degree upper_bound = Degree_traits<r2degree>::join(bounds.second, B_bounds.second);
        bounds = {lower_bound, upper_bound};
    }

    vec<r2degree> grid_diagonal = get_grid_diagonal(bounds, grid_length);
    ostream << "HNF" << std::endl;
    assert(grid_diagonal.size() == grid_length);
    ostream << grid_length << " " << grid_length << std::endl;
    ostream << grid_diagonal << std::endl;
    int grid_size = grid_diagonal.size()*grid_diagonal.size();

    // Since a lot of applications will create unbounded modules, we need to set a bound where to cut off
    // OR use a measure where the dimension function is still integrable
    // Here I am trying to cut off the density parameter (which should be the second one!) 
    // of a density-rips bifiltration quite early  after stabilisation
    // so that features which are already visible in low density regions are preferred.
    // For the scale parameter (first parameter) instead, it should not matter,
    // because for reduced homology the modules are bounded in this direction.
    double overlap = 0.1;
    r2degree range = bounds.second - bounds.first;
    pair<r2degree> slope_bounds = {bounds.first, bounds.second + overlap * range};

    std::cout << "  Presentation is bounded by " << bounds.first << " and " << bounds.second << std::endl;
    std::cout << "  Modules are cut off at " << slope_bounds.second << std::endl;
    
    // array<vec<double>> slopes;  

    vec<HN_factors> composition_factors(grid_size);

    for(int i = 0; i < grid_length; i++){ 
      for(int j = 0; j < grid_length; j++){
        r2degree degree = {grid_diagonal[i].first, grid_diagonal[j].second};
        ostream << "G," << i << "," << j << ", " << degree << std::endl;

        if (progress_bar) {
            int current_index = i * grid_length + j;
            std::string name = "Grid point";
            show_progress_bar(current_index, grid_size, name);
        }
        for(auto& B : indecomps){
 
            auto B_induced = B.submodule_generated_at(degree);
            if(B_induced.get_num_rows() == 1){
                grid_ind_dimensions.push_back(1);
                all_scss_dimensions.push_back(1);
                R2Resolution<int> res(B_induced);
                double slope = res.slope(slope_bounds);
                if(slope == INFINITY){
                    assert(false);
                    std::cerr << "Slope is infinite, consider passing a bound." << std::endl;
                }

                // slopes[i].push_back(slope);
                Module_w_slope single_stable = std::make_pair(res, slope);
                to_stream(ostream, single_stable);

            } else if ( B_induced.get_num_rows() == 0){
                // Do nothing.
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
                composition_factors[i] = skyscraper_invariant(sub_B_list, subspaces, slope_bounds);

                for(auto& hn_factor : composition_factors[i]){
                    all_scss_dimensions.push_back(hn_factor.first.d1.get_num_rows());
                    if(hn_factor.second == INFINITY){
                        std::cout << "  There are unbounded modules in the decomposition." << std::endl;
                        std::cout << "  Consider passing a bound." << std::endl;
                        assert(false);
                    }
                    // slopes[i].push_back(hn_factor.second);
                    to_stream(ostream, hn_factor);
                }
            }
        }

        // std::sort(slopes[i].begin(), slopes[i].end());
      }
    }

    std::cout << std::endl;
    std::cout << "  Tracked the dimensions of " << grid_ind_dimensions.size() << " indecomposable summands." << std::endl;
    
    std::cout << "  The dimensions of indecomposable summands at the grid points are distributed as:" << std::endl;
    calculate_stats(grid_ind_dimensions);

    std::cout << "  The dimensions of the composition factors at the grid points are distributed as:" << std::endl;
    calculate_stats(all_scss_dimensions);

    // write_slopes_to_csv(slopes, grid_points, "slopes.csv");

}

template <typename Outputstream>
void full_grid_induced_decomposition(aida::AIDA_functor& decomposer, std::ifstream& istream, Outputstream& ostream, bool show_indecomp_statistics, bool show_runtime_statistics, bool is_decomposed = false){
    
    if(is_decomposed){
        vec<R2GradedSparseMatrix<int>> matrices;
        graded_linalg::construct_matrices_from_stream(matrices, istream);
        vec<r2degree> grid_points;
        process_list_of_summands(decomposer, istream, ostream, grid_points, matrices);
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
        vec<r2degree> grid_points;
        process_list_of_summands(decomposer, istream, ostream, grid_points, B_list);
    }
    
} // full_grid_induced_decomposition

} // namespace hnf

