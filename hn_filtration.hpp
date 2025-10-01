
#include "aida_interface.hpp"
#include "grlina/graded_linalg.hpp"
#include "grlina/orders_and_graphs.hpp"
#include "grlina/r2graded_matrix.hpp"
#include <iostream>
#include <unistd.h> 
#include <getopt.h>

using namespace graded_linalg;

namespace hnf{



template<typename index>
struct Uni_B1{

    R2GradedSparseMatrix<index> d1;
    R2GradedSparseMatrix<index> d2;

    std::array<double, 4> area_polynomial;

    // Default constructor
    Uni_B1() = default;

    // Constructor with move-from reference
    Uni_B1(R2GradedSparseMatrix<index>&& d1_, bool is_minimal = false)
        : d1(std::move(d1_)) {
        // Kernel computation is easy if the presentation is minimal, sorted, and has one generator.
        //TO-DO: Test this:
        if(is_minimal && d1.get_num_rows() == 1){
            assert(std::is_sorted(d1.col_degrees.begin(), d1.col_degrees.end(), Degree_traits<r2degree>::lex_lambda));
            d2 = R2GradedSparseMatrix<index>(d1.get_num_cols()-1, d1.get_num_cols());
            d2.data = vec< vec<index> >(d1.get_num_cols()-1);
            d2.row_degrees = d1.col_degrees;
            d2.col_degrees = vec<r2degree>(d1.get_num_cols()-1);
            r2degree last_degree = d1.col_degrees[0];
            for(index i = 1; i < d1.get_num_cols(); i++){
                r2degree join = Degree_traits<r2degree>::join(last_degree, d1.col_degrees[i]);
                d2.data[i] = {i -1, i};
                d2.col_degrees[i] = join;
                last_degree = d1.col_degrees[i];
            }
        } else {
            auto d1_copy = d1;
            d2 = d1_copy.graded_kernel();
        }
    }

    // Copy constructor 
    Uni_B1(const R2GradedSparseMatrix<index>& d1_, bool is_minimal = false)
        : d1(d1_) {
        // Kernel computation is easy if the presentation is minimal, sorted, and has one generator.
        //TO-DO: Test this:
        if(is_minimal && d1.get_num_rows() == 1){
            assert(std::is_sorted(d1.col_degrees.begin(), d1.col_degrees.end(), Degree_traits<r2degree>::lex_lambda));
            d2 = R2GradedSparseMatrix<index>(d1.get_num_cols()-1, d1.get_num_cols());
            d2.data = vec< vec<index> >(d1.get_num_cols()-1);
            d2.row_degrees = d1.col_degrees;
            d2.col_degrees = vec<r2degree>(d1.get_num_cols()-1);
            r2degree last_degree = d1.col_degrees[0];
            for(index i = 1; i < d1.get_num_cols(); i++){
                r2degree join = Degree_traits<r2degree>::join(last_degree, d1.col_degrees[i]);
                d2.data[i] = {i -1, i};
                d2.col_degrees[i] = join;
                last_degree = d1.col_degrees[i];
            }
        } else {
            auto d1_copy = d1;
            d2 = d1_copy.graded_kernel();
        }
    }

    /**
     * @brief Computes the dimension at any point in R^2
     * 
     * @param alpha 
     * @return index 
     */
    index dim_at (r2degree alpha) const {
        index num_chains_0 = d1.num_rows_before(alpha);
        index num_chains_1 = d1.num_cols_before(alpha);
        index num_chains_2 = d2.num_cols_before(alpha);
        return num_chains_0 - num_chains_1 + num_chains_2;
    }

    /**
     * @brief Only works for modules with all generators at a single degree
     *  TO-DO: Log transform on the scale parameter. What to do on density?
     * @return double 
     */
    double area () const {
        auto [min, max] = d1.bounding_box();
        double base_area = d1.get_num_rows()*(max.first - min.first) * (max.second - min.second);
        for(const auto& degree : d1.col_degrees){
            base_area -= (max.first - degree.first) * (max.second - degree.second);
        }
        for(const auto& degree : d2.col_degrees){
            base_area += (max.first - degree.first) * (max.second - degree.second);
        }
        return base_area;
    }


    /**
     * @brief Only works for modules with all generators at a single degree
     * @return double 
     */
    double area (const r2degree& bound) const {
        auto [min, max] = d1.bounding_box();
        max = bound;
        double base_area = d1.get_num_rows()*(max.first - min.first) * (max.second - min.second);
        for(const auto& degree : d1.col_degrees){
            base_area -= (max.first - degree.first) * (max.second - degree.second);
        }
        for(const auto& degree : d2.col_degrees){
            base_area += (max.first - degree.first) * (max.second - degree.second);
        }
        return base_area;
    }

    /**
     * @brief Only works for modules with all generators at a single degree
     * Normalised so that the area of the free module is 1.
     *  TO-DO: Log transform on the scale parameter. What to do on density?
     * @return double 
     */
    double area (const pair<r2degree>& bounds) const {
        //TO-DO: Actually only need to compute the minimal element here.
        auto [min, max] = d1.bounding_box();
        max = bounds.second;
        assert(max.first >= min.first);
        assert(max.second >= min.second);
        assert(min == d1.row_degrees[0]);
        r2degree range = bounds.second-bounds.first;
        double range_area = range.first * range.second;
        double base_area = d1.get_num_rows() * (max.first - min.first) * (max.second - min.second);
        for(const auto& degree : d1.col_degrees){
            assert(degree.first <= max.first);
            assert(degree.second <= max.second);
            assert(degree.first >= min.first);
            assert(degree.second >= min.second);
            base_area -= (max.first - degree.first) * (max.second - degree.second);
        }
        for(const auto& degree : d2.col_degrees){
            assert(degree.first <= max.first);
            assert(degree.second <= max.second);
            assert(degree.first >= min.first);
            assert(degree.second >= min.second);
            base_area += (max.first - degree.first) * (max.second - degree.second);
        }
        double normalised_area = base_area/range_area;
        return normalised_area;
    }

    /**
     * @brief Returns a polynomial a[0] + a[1]*x + a[2]*y + a[3]*x*y
     * which gives the area of the module after shifting the (assumed to be) single generator by (x,y)
     * 
     * @param bounds 
     * @return std::array<double, 4> 
     */
    void compute_area_polynomial (const pair<r2degree>& bounds) {
        int num_rows = d1.get_num_rows();
   
        assert(std::is_sorted(d1.col_degrees.begin(), d1.col_degrees.end(), Degree_traits<r2degree>::lex_lambda));
    
        bool test = true;

        // For normalisation, so that area is always <=1
        r2degree range = bounds.second-bounds.first;
        double range_area = range.first * range.second;

        const r2degree& gen_degree = this->d1.row_degrees[0]; 
        
        // Make sure this is properly initialised
        area_polynomial.fill(0.0);

        // vector from generators to the max bound
        r2degree gen_vector = bounds.second - gen_degree; 
        // Generators are at gen_degree + (x,y), so the base value they add to the area is spanned by
        area_polynomial[0] += num_rows * gen_vector.first * gen_vector.second; 
        // subtract what happens when shifted to (x,y)
        area_polynomial[1] -= num_rows * gen_vector.second;
        area_polynomial[2] -= num_rows * gen_vector.first;
        area_polynomial[3] += num_rows;

        // Relations subtract from the area as before

        for(const auto& degree : d1.col_degrees){
            assert( Degree_traits<r2degree>::smaller_equal(degree, bounds.second));
            assert( Degree_traits<r2degree>::smaller_equal(bounds.first, degree) );
            r2degree rel_vector = bounds.second - degree;
            area_polynomial[0] -= rel_vector.first* rel_vector.second;
        }

        // The first and last relation could be affected by the shift, too
        for(const auto& degree :  d1.col_degrees){
            if(degree.first == gen_degree.first){
                assert(bounds.second.second - degree.second >= 0);
                area_polynomial[1] += (bounds.second.second - degree.second);
                if( area_polynomial[1] > 0){
                    std::cout << "Area polynomial[1] is positive: " << area_polynomial[1] << std::endl;
                    this->d1.print_graded();
                    std::cout << "Bounds: " << bounds.first.first << " " << bounds.first.second << " to " 
                        << bounds.second.first << " " << bounds.second.second << std::endl;
                    std::cout << "Range area: " << range_area << std::endl;
                    assert(false);
                }
            }
            if(degree.second == gen_degree.second){
                assert(bounds.second.first - degree.first >= 0);
                area_polynomial[2] += (bounds.second.first - degree.first);
                if( test && area_polynomial[2] > 0){
                    std::cout << "Area polynomial[2] is positive: " << area_polynomial[1] << std::endl;
                    this->d1.print_graded();
                    std::cout << "Bounds: " << bounds.first.first << " " << bounds.first.second << " to " 
                        << bounds.second.first << " " << bounds.second.second << std::endl;
                    std::cout << "Range area: " << range_area << std::endl;
                    assert(false);
                }
            }
        }

        // Syzygies add again
        for(const auto& degree : d2.col_degrees){
            assert( Degree_traits<r2degree>::smaller_equal(degree, bounds.second));
            assert( Degree_traits<r2degree>::smaller_equal(bounds.first, degree) );
            r2degree rel_vector = bounds.second - degree;
            area_polynomial[0] += rel_vector.first* rel_vector.second;
        }
        for(index i = 0; i < 4; i++){
            area_polynomial[i] /= range_area;
        }
    }

    double evaluate_area_polynomial( r2degree d){
        double& x = d.first;
        double& y = d.second;
        return area_polynomial[0] + area_polynomial[1]*x + area_polynomial[2]*y + area_polynomial[3]*x*y;
    }

    double evaluate_slope_polynomial( r2degree d){
        double& x = d.first;
        double& y = d.second;
        return 1/ (area_polynomial[0] + area_polynomial[1]*x + area_polynomial[2]*y + area_polynomial[3]*x*y);
    }

    double slope () const {
        double area = this->area();
        if(area == 0){
            std::cerr << "Area is zero, slope will be infinite. Consider passing a bound." << std::endl;
        }
        double slope_value = (static_cast<double>(d1.get_num_rows())/area);
        return slope_value;
    }

    double slope (const r2degree& bound) const {
        double area = this->area(bound);
        if(area == 0){
            std::cerr << "Area is zero, slope will be infinite. The bound you passed is insufficient." << std::endl;
        }
        double slope_value = (static_cast<double>(d1.get_num_rows())/area);
        return slope_value;
    }

    double slope (const pair<r2degree>& bounds) const {
        double area = this->area(bounds);
        if(area == 0){
            std::cerr << "Area is zero, slope will be infinite. The bound you passed is insufficient." << std::endl;
        }
        double slope_value = (static_cast<double>(d1.get_num_rows())/area);
        return slope_value;
    }

    

    /**
     * @brief Computes the dimension at every point in the grid. Unoptimised. Not finished.
     * 
     * @return array<index> 
     */
    array<index> dimension_vector(bool sort = true) const {
        if(sort){   
            d1.sort_rows_colexicographically();
            d1.sort_columns_colexicographically();
            d2.sort_rows_colexicographically();
            d2.sort_columns_colexicographically();
        }

        vec<index> x_grid = d1.x_grid;
        vec<index> y_grid = d1.y_grid;
        index num_x = x_grid.size();
        index num_y = y_grid.size();

        auto itc1 = d1.row_degrees.begin();
        auto itc2 = d2.row_degrees.begin();
        auto itc3 = d2.col_degrees.begin();

        for(index i = 0; i < num_y; i++){
             //TO-DO finish.
        }
    }

};


using Block = aida::Block;
using Module_w_slope = std::pair<Uni_B1<int>, double>;
using Block_list = aida::Block_list;
using HN_factors = vec<Module_w_slope>;
using R2Mat = R2GradedSparseMatrix<int>;

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
Module_w_slope find_scss_bruteforce(const R2Mat& X, 
        vec<vec<SparseMatrix<int>>>& subspaces, 
        R2Mat& max_subspace,
        const pair<r2degree>& bounds){
    int k = X.get_num_rows();
    Uni_B1<int> scss = Uni_B1<int>(X);
    double max_slope = scss.slope(bounds);
    if(k == 1){
        // Nothing to do?
    } else {
        assert(k < 6);
        if(subspaces.size() < k){
            std::cerr << "Have not loaded enough subspaces" << std::endl;
            std::exit(1);
        }
        
        for(auto ungraded_subspace : subspaces[k-1]){
            int num_gens = ungraded_subspace.get_num_cols();
            R2Mat subspace = R2Mat(ungraded_subspace);
            subspace.row_degrees = X.row_degrees;
            subspace.col_degrees = vec<r2degree>(num_gens, X.row_degrees[0]);
            assert(subspace.get_num_rows() == X.get_num_rows());
            assert(subspace.get_num_cols() == num_gens);
            R2Mat submodule_pres = X.submodule_generated_by(subspace);
            Uni_B1<int> res(submodule_pres);
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

        if(X.get_num_rows() >= 4){
            std::cout << "  Warning: Computing HNF for a module of dimension " 
                << X.get_num_rows() << std::endl;
        }

        while(X.get_num_rows() > 1){
            R2Mat max_subspace;
            result.emplace_back(find_scss_bruteforce(X, subspaces, max_subspace, bounds));
            if(result.back().first.d1.get_num_rows() == X.get_num_rows()){
                break;
            } else {
                X.quotient_by(max_subspace);
            }
        }
        if(X.get_num_rows() == 1){
            Uni_B1<int> res(X);
            double slope = res.slope(bounds);
            result.emplace_back(std::make_pair(res, slope));
        } else if (X.get_num_rows() == 0){
            // Nothing to do.
        }
            
    }
    return result;
}

void skyscraper_invariant(const R2Mat& input, 
    HN_factors& result,
    vec<vec<SparseMatrix<int>>> &subspaces, const pair<r2degree> &bounds){
    R2Mat X = input;
    if(X.get_num_rows() >= 4){
            std::cout << "  Warning: Computing HNF for a module of dimension " 
                << X.get_num_rows() << std::endl;
        }
    while(X.get_num_rows() > 1){
        R2Mat subspace;
        result.emplace_back(find_scss_bruteforce(X, subspaces, subspace, bounds));
        if(result.back().first.d1.get_num_rows() == X.get_num_rows()){
            break;
        } else {
            X.quotient_by(subspace);
        }
    }
    if(X.get_num_rows() == 1){
        Uni_B1<int> res(X);
        double slope = res.slope(bounds);
        result.emplace_back(std::make_pair(res, slope));
    } else if (X.get_num_rows() == 0){
        // Nothing to do.
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

r2degree get_grid_step(const r2degree& lower_bound, const r2degree& upper_bound, const int& grid_length_x, const int& grid_length_y) {
    const double& x_min = lower_bound.first;
    const double& x_max = upper_bound.first;
    const double& y_min = lower_bound.second;
    const double& y_max = upper_bound.second;

    double x_step = (x_max - x_min) / (grid_length_x - 1);
    double y_step = (y_max - y_min) / (grid_length_y - 1);

    return {x_step, y_step};
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

    for (size_t i = 0; i < slopes.size(); ++i) {
        file << grid_points[i] << ";";

        const auto& slopes_at_degree = slopes[i];
        for (size_t j = 0; j < slopes_at_degree.size(); ++j) {
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
    int percent = static_cast<int>(static_cast<double>(i+1) / total * 100);
    if (percent != last_percent) {
        // Calculate the number of symbols to display in the progress bar
        int num_symbols = percent / 2;
        std::cout << "\r" << i + 1 << " " << name << " : [";
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
void process_summands_fixed_grid(aida::AIDA_functor& decomposer, 
    Outputstream& ostream, 
    const int& grid_length_x, const int& grid_length_y, 
    Container& indecomps) {
    
    
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

    
    vec<int> all_scss_dimensions;
    vec<int> first_ind_dimensions;
    vec<int> grid_ind_dimensions;

    pair<r2degree> bounds = indecomps.front().bounding_box();
    

    for (auto& B : indecomps) {
        
        first_ind_dimensions.push_back(B.get_num_rows());

        auto [min, max] = B.bounding_box();
        // This is a bit ineffcient maybe and should be computed earlier directly from the module before decomposition
        // TO-DO: There is also a computation error here, the upper bound is not computed correctly?
        
        r2degree lower_bound = Degree_traits<r2degree>::meet(bounds.first, min);
        r2degree upper_bound = Degree_traits<r2degree>::join(bounds.second, max);
        bounds = {lower_bound, upper_bound};

    }
    r2degree range = bounds.second - bounds.first;
    bounds = {bounds.first, bounds.second + range * 0.3}; // Add a bit of space to the upper bound
    r2degree grid_step = get_grid_step(bounds.first, bounds.second, grid_length_x, grid_length_y);


    ostream << "HNF" << std::endl;

    ostream << grid_length_x << "," << grid_length_y << std::endl;
    ostream  << bounds.first << "," << bounds.second << "," << grid_step << std::endl;
    int grid_size = grid_length_x * grid_length_y;

    // Since a lot of applications will create unbounded modules, we need to set a bound where to cut off
    // OR use a measure where the dimension function is still integrable
    // Here I am trying to cut off the density parameter (which should be the second one!) 
    // of a density-rips bifiltration quite early  after stabilisation
    // so that features which are already visible in low density regions are preferred.
    // For the scale parameter (first parameter) instead, it should not matter,
    // because for reduced homology the modules are bounded in this direction.
    double slope_overlap = 0.1;
    pair<r2degree> slope_bounds = {bounds.first, bounds.second + slope_overlap * range};

    std::cout << "  Presentation is bounded by " << bounds.first << " and " << bounds.second << std::endl;
    std::cout << "  Modules are cut off at " << slope_bounds.second << std::endl;
    

    array<HN_factors> composition_factors(grid_length_x, vec<HN_factors>(grid_length_y));

    r2degree current_grid_degree = bounds.first;
    for(int i = 0; i < grid_length_x; i++){ 
      current_grid_degree.second = bounds.first.second; // Reset y-coordinate for each x-coordinate
      for(int j = 0; j < grid_length_y; j++){
        
        ostream << "G," << i << "," << j << ", " << current_grid_degree << std::endl;

        if (progress_bar) {
            int current_index = i * grid_length_y + j;
            std::string name = "Grid point";
            show_progress_bar(current_index, grid_size, name);
        }
        int indecomp_index = -1;
        for(auto& B : indecomps){
            indecomp_index++;

            auto B_induced = B.submodule_generated_at(current_grid_degree);
            if(B_induced.get_num_rows() == 1){
                grid_ind_dimensions.push_back(1);
                all_scss_dimensions.push_back(1);
                Uni_B1<int> res(B_induced);
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
                for(Block& sub_B : sub_B_list){
                    if(sub_B.get_num_rows() > max_dim){
                        max_dim = sub_B.get_num_rows();
                    }
                    grid_ind_dimensions.push_back(sub_B.get_num_rows());
                }
                auto subspaces = all_sparse_proper_subspaces(max_dim);
                composition_factors[i][j] = skyscraper_invariant(sub_B_list, subspaces, slope_bounds);

                for(auto& hn_factor : composition_factors[i][j]){
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
        current_grid_degree.second += grid_step.second;
      }
      current_grid_degree.first += grid_step.first;
    }

    std::cout << std::endl;
    std::cout << "  Tracked the dimensions of " << grid_ind_dimensions.size() << " indecomposable summands." << std::endl;
    
    std::cout << "  The dimensions of indecomposable summands at the grid points are distributed as:" << std::endl;
    calculate_stats(grid_ind_dimensions);

    std::cout << "  The dimensions of the composition factors at the grid points are distributed as:" << std::endl;
    calculate_stats(all_scss_dimensions);

    // write_slopes_to_csv(slopes, grid_points, "slopes.csv");

}


struct Dynamic_HNF {
    // Stores the decomposition at every local grid point in the current row
    vec< vec<Uni_B1<int>> > indecomposable_summands;
    vec<int> grid_ind_dimensions;

    Dynamic_HNF(){
        indecomposable_summands = vec< vec<Uni_B1<int>> >();
        grid_ind_dimensions = vec<int>();
    }
    
    // We should store all possible HN_filtrations in that quadrant. But for now, I will recompute it everytime and see how slow that actually is.

    void compute_HNF_row(aida::AIDA_functor& decomposer,
        R2Mat& M, 
        int& y_index,
        pair<r2degree> slope_bounds) {

        assert(y_index > -1);
        double y_coordinate = M.y_grid[y_index];
        int x_length = M.x_grid.size();

        indecomposable_summands = vec< vec<Uni_B1<int>> >(x_length, vec<Uni_B1<int>>());
        
        int max_dim = 0;

        for(int x_index = 0; x_index < x_length; x_index++){
            r2degree grid_point = {M.x_grid[x_index], y_coordinate};
            auto M_induced = M.submodule_generated_at(grid_point);
            if(M_induced.get_num_rows() == 0){
                // Nothing to do?
            } else if (M_induced.get_num_rows() == 1){
                Uni_B1<int> res(std::move(M_induced));
                indecomposable_summands[x_index].push_back(res);
                // resolutions[x_index].push_back(res);
                indecomposable_summands[x_index].back().compute_area_polynomial(slope_bounds);
                grid_ind_dimensions.push_back(1);
            } else {
                aida::Block_list sub_M_list;
                M_induced.compute_col_batches();
                decomposer(M_induced, sub_M_list);
                
                for(Block sub_M : sub_M_list){
                    if(sub_M.get_num_rows() > max_dim){
                        max_dim = sub_M.get_num_rows();
                    }
                    indecomposable_summands[x_index].emplace_back(Uni_B1<int>(std::move(sub_M)));
                    indecomposable_summands[x_index].back().compute_area_polynomial(slope_bounds);
                    grid_ind_dimensions.push_back(sub_M.get_num_rows());
                }
            }
        }
        if (max_dim >= 4){
            std::cout << " Careful, there are high-dimensional summands which might slow down HNF computation excessively." << std::endl;
        }  
    }


};

template <typename Container>
std::tuple<r2degree, r2degree, r2degree, pair<r2degree>> compute_bounds_and_grid(
    Container& indecomps, 
    vec<int>& first_ind_dimensions,
    const int& grid_length_x,
    const int& grid_length_y) {

    const double range_extension = 0.3;
    const double slope_overlap = 0.1;

    auto [lower_bound, upper_bound] = indecomps.front().bounding_box();
    for (R2Mat& M : indecomps) {
        // Get a Grid for each component, so that we do not have to recompoute the submodules and their decomposition
        M.compute_grid_representation();
        first_ind_dimensions.push_back(M.get_num_rows());

        // TO-DO: There was a computation error here, the upper bound is not computed correctly?
        
        lower_bound = Degree_traits<r2degree>::meet(lower_bound, {M.x_grid[0], M.y_grid[0]});
        upper_bound = Degree_traits<r2degree>::join(upper_bound, {M.x_grid.back(), M.y_grid.back()});
    }

    upper_bound = upper_bound + range_extension*(upper_bound - lower_bound);
    r2degree grid_step = get_grid_step(lower_bound, upper_bound, grid_length_x, grid_length_y);
    pair<r2degree> slope_bounds = {lower_bound, upper_bound + slope_overlap * ( upper_bound - lower_bound) };
    return {lower_bound, upper_bound, grid_step, slope_bounds};
};

template<typename Container>
void update_HNF_rows_at_y_level(
    r2degree current_grid_degree,
    Container& indecomps,
    vec<pair<int>>& grid_locations,
    vec<Dynamic_HNF>& local_grid_row_data,
    aida::AIDA_functor& decomposer,
    const pair<r2degree>& slope_bounds){

    int k = -1;
    for(R2Mat& M : indecomps){
        k++;
        bool recompute = false;
        grid_locations[k].first = -1; // Reset x-coordinate
        int& local_y = grid_locations[k].second;
        if(local_y + 1 == static_cast<int>(M.y_grid.size()) ){
            // Do nothing for now
        } else {
            while(local_y + 1 < static_cast<int>(M.y_grid.size())){
                if(current_grid_degree.second >= M.y_grid[local_y + 1]){
                    local_y++;
                    recompute = true;
                } else {
                    break;
                }
            }
        }

        if(recompute){
            local_grid_row_data[k].compute_HNF_row(decomposer, M, local_y, slope_bounds);
        }
        if(local_y != -1){
            assert(current_grid_degree.second >= M.y_grid[local_y] );
            if( local_y +1 < static_cast<int>(M.y_grid.size())){
                assert(current_grid_degree.second < M.y_grid[local_y + 1]);
            } 
        }
    }
}

template< typename Container>
void update_grid_locations_x(
    r2degree current_grid_degree,
    Container& indecomps,
    vec<pair<int>>& grid_locations){

    int k = -1;
    for(R2Mat& M : indecomps){
        k++;
        int& local_x = grid_locations[k].first;
        if(local_x + 1 == static_cast<int>(M.x_grid.size()) ){
            
        } else {
            while(local_x + 1 < static_cast<int>(M.x_grid.size()) ){
                if(current_grid_degree.first >= M.x_grid[local_x + 1]){
                    local_x++;
                } else {
                    break;
                }
            }
        }
        if(local_x != -1){
            assert(current_grid_degree.first >= M.x_grid[local_x]);
            if( local_x + 1 < static_cast<int>(M.x_grid.size()) ){
                assert(current_grid_degree.first < M.x_grid[local_x + 1]);
            } 
        }
    }
};

template<typename Outputstream>
void write_grid_metadata(Outputstream& ostream,
    int grid_length_x, int grid_length_y,
    const r2degree& lower_bound,
    const r2degree& upper_bound,
    const r2degree& grid_step,
    const pair<r2degree>& slope_bounds,
    bool show_info = false) {
    
    ostream << "HNF" << std::endl;
    ostream << grid_length_x << "," << grid_length_y << std::endl;
    ostream  << lower_bound << "," << upper_bound << "," << grid_step << std::endl;
    
    // Since a lot of applications will create unbounded modules, we need to set a bound where to cut off
    // OR use a measure where the dimension function is still integrable
    // Here I am trying to cut off the density parameter (which should be the second one!) 
    // of a density-rips bifiltration quite early  after stabilisation
    // so that features which are already visible in low density regions are preferred.
    // For the scale parameter (first parameter) instead, it should not matter,
    // because for reduced homology the modules are bounded in this direction.

    // TO-DO: actually implement what is described above.
    if(show_info) {
        std::cout << "  Presentation is bounded by " << lower_bound << " and " << upper_bound << std::endl;
        std::cout << "  Modules are cut off at " << slope_bounds.second << std::endl;
    }
};

// Just for comparing doubles with a tolerance
bool essentially_equal(double a, double b, double relTol = 1e-9, double absTol = 1e-12) {
    return std::fabs(a - b) <= std::max(relTol * std::max(std::fabs(a), std::fabs(b)), absTol);
}

void compare_slopes_test(
    r2degree current_grid_degree,
    r2degree local_grid_degree,
    const HN_factors& composition_factors,
    const HN_factors& test_factors,
    int i, int j, int k) {

    vec<double> slopes;
    vec<double> test_slopes;
    assert(test_factors.size() == composition_factors.size());
    auto it = test_factors.begin();
    for(auto& hn_factor : composition_factors){
        slopes.push_back(hn_factor.second);
        test_slopes.push_back(it->second);
        it++;
    }
    
    // std::sort(slopes.begin(), slopes.end());
    // std::sort(test_slopes.begin(), test_slopes.end());
    for(size_t l = 0; l < slopes.size(); l++){
        if(!essentially_equal(slopes[l], test_slopes[l], 1e-6, 1e-8)){
            std::cout << std::fixed << std::setprecision(12);
            std::cout << "  Slope mismatch in compare_slopes: " << slopes[l] << " vs. " << test_slopes[l] << std::endl;
            std::cout << "  Difference: " << slopes[l] - test_slopes[l] << std::endl;
            std::cout << "  Current grid degree: " << current_grid_degree << std::endl;
            std::cout << "  Local grid degree: " << local_grid_degree << std::endl;
            std::cout << "  i: " << i << ", j: " << j << ", k: " << k << std::endl;
            std::cout << "  Summand: " << std::endl;
            composition_factors[l].first.d1.print_graded();
            std::cout << "  Test_summand: " << std::endl;
            test_factors[l].first.d1.print_graded();
            assert(false);
        }
    }
}

template <typename Container, typename Outputstream>
void process_grid_cell(
    int i, int j,
    r2degree current_grid_degree,
    Container& indecomps,
    vec<pair<int>>& grid_locations,
    vec<Dynamic_HNF>& local_grid_row_data,
    array<HN_factors>& composition_factors,
    vec<int>& grid_ind_dimensions,
    vec<int>& all_scss_dimensions,
    Outputstream& ostream,
    vec<vec<SparseMatrix<int>>>& subspaces,
    const pair<r2degree>& slope_bounds,
    aida::AIDA_functor& decomposer) {
    
    bool test = true;
    

    int k = -1;
    for(auto & M : indecomps){
        HN_factors test_factors = HN_factors();
        HN_factors copy_factors = HN_factors();
        k++;
        auto& local_grid_index = grid_locations[k];
        auto& local_x = local_grid_index.first;
        r2degree local_grid_degree;
        if(local_x == -1 || local_grid_index.second == -1){
            // We're not in the local grid yet, so can skip this summand.
            continue;
        } else {
            local_grid_degree = std::make_pair(M.x_grid[local_grid_index.first], M.y_grid[local_grid_index.second]);
        }

        
        
        Dynamic_HNF& local_dhnf =  local_grid_row_data[k];
        auto& local_summands = local_dhnf.indecomposable_summands[local_x];

        
        if(test){
            auto B_induced = M.submodule_generated_at(current_grid_degree);
            if(B_induced.get_num_rows() != 0){
                Block_list sub_B_list;
                B_induced.compute_col_batches();
                decomposer(B_induced, sub_B_list);
                assert(local_summands.size() == sub_B_list.size());
                int k = 0;
                for(auto& sub_B : sub_B_list){
                    if (sub_B.get_num_rows() > subspaces.size()) {
                        fill_up_subspaces(subspaces, sub_B.get_num_rows());
                    }
                    k++;
                }
                test_factors = skyscraper_invariant(sub_B_list, subspaces, slope_bounds);
            } else {
                assert(local_summands.size() == 0);
            }
        }

        
        for( Uni_B1<int>& summand : local_summands){
            r2degree verschiebung = current_grid_degree - local_grid_degree;
            if( summand.d1.get_num_rows() == 0){
                std::cout << " Empty summands should have been filtered out." << std::endl;
                assert(false);
            } else if(summand.d1.get_num_rows() == 1){
                double slope = summand.evaluate_slope_polynomial(verschiebung);
                if(test){
                    double area = summand.evaluate_area_polynomial(verschiebung);
                    R2Mat test_cutoff = summand.d1.submodule_generated_at(current_grid_degree);
                    if(test_cutoff.get_num_rows() != 0){
                        Uni_B1<int> test_summand(test_cutoff);
                        // double test_slope = test_summand.slope(slope_bounds);
                        double test_area = test_summand.area(slope_bounds);
                        if(essentially_equal(area, test_area, 1e-7, 1e-9) == false){
                            std::cout << std::fixed << std::setprecision(12);
                            std::cout << "  Area mismatch at direct cutting off: " << area << " vs. " << test_area << std::endl;
                            std::cout << "  Difference: " << area - test_area << std::endl;
                            std::cout << "  Current grid degree: " << current_grid_degree << std::endl;
                            std::cout << "  Local grid degree: " << local_grid_degree << std::endl;
                            std::cout << "  i: " << i << ", j: " << j << ", k: " << k << std::endl;
                            std::cout << "  Summand: " << std::endl;
                            summand.d1.print_graded();
                            std::cout << "  area polynomial: " << 
                                summand.area_polynomial[0] << "  " << summand.area_polynomial[1] << "  " << 
                                summand.area_polynomial[2] << "  " << summand.area_polynomial[3] << std::endl;
                            std::cout << "  Verschiebung: " << verschiebung << std::endl;
                            std::cout << "  Slope bounds: " << slope_bounds.first << " " << slope_bounds.second << std::endl;
                            auto normalisation = slope_bounds.second - slope_bounds.first;
                            std::cout << "  Normalisation area: " << normalisation.first * normalisation.second << std::endl;
                            std::cout << "  Cut off summand: " << std::endl;
                            test_summand.d1.print_graded();
                            assert(false);
                        }
                    }
                }
                // TO-DO: summand is the original local summand, we need to cut it off at the current grid degree,
                // Even if the slope is already correctly computed by the polynomial.
                summand.d1.set_all_generator_degrees(current_grid_degree);
                summand.d1.column_reduction_graded();
                composition_factors[i][j].emplace_back(std::make_pair(summand, slope));
                if(test){
                    copy_factors.emplace_back(std::make_pair(summand, slope));
                }
                grid_ind_dimensions.push_back(1);
            } else {
                if (summand.d1.get_num_rows() > subspaces.size()) {
                    fill_up_subspaces(subspaces, summand.d1.get_num_rows());
                }
                grid_ind_dimensions.push_back(summand.d1.get_num_rows());
                auto cut_off = summand.d1;
                cut_off.set_all_generator_degrees(current_grid_degree);
                if(test){
                    skyscraper_invariant(cut_off, copy_factors, subspaces, slope_bounds);
                }
                skyscraper_invariant(cut_off, composition_factors[i][j], subspaces, slope_bounds);
            }
            
            for(auto& hn_factor : composition_factors[i][j]){
                all_scss_dimensions.push_back(hn_factor.first.d1.get_num_rows());
                if(hn_factor.second == INFINITY){
                    std::cout << "  There are unbounded modules in the decomposition." << std::endl;
                    std::cout << "  Consider passing a bound." << std::endl;
                    assert(false);
                }
                to_stream(ostream, hn_factor);
            }
        }
        if(test){
            compare_slopes_test(current_grid_degree, local_grid_degree, 
                    copy_factors, test_factors, i, j, k);
        }
    }
};

template<typename Container, typename Outputstream>
void process_summands_smart_grid(aida::AIDA_functor& decomposer, 
    Outputstream& ostream, 
    const int& grid_length_x, const int& grid_length_y, 
    Container& indecomps) {
    
    vec<Dynamic_HNF> local_grid_row_data;
    vec<vec<SparseMatrix<int>>> subspaces = all_sparse_proper_subspaces(3);
    int grid_size = grid_length_x * grid_length_y;
    
    bool progress_bar = decomposer.config.progress;
    decomposer.config.progress = false;

    bool show_info = decomposer.config.show_info;
    decomposer.config.show_info = false;

    if (show_info) {
        std::cout << "The first decomposition has " << indecomps.size() 
                  << " indecomposable summands." << std::endl;
    }
    
    // Only for statistics:
    vec<int> all_scss_dimensions;
    vec<int> grid_ind_dimensions;
    vec<int> first_ind_dimensions;

    auto [lower_bound, upper_bound, grid_step, slope_bounds] = compute_bounds_and_grid(indecomps, first_ind_dimensions, grid_length_x, grid_length_y);
    write_grid_metadata(ostream, grid_length_x, grid_length_y, lower_bound, upper_bound, grid_step, slope_bounds, show_info);
    
    // Will store the actuall composition factors of the HNF at the grid points:
    array<HN_factors> composition_factors(grid_length_x, vec<HN_factors>(grid_length_y));
    // Will store where we are in the local grids:
    vec<pair<int>> grid_locations = vec<pair<int>>(indecomps.size(), {-1,-1});
    // Will store the decomposed modules generated at the local grid points:
    local_grid_row_data = vec<Dynamic_HNF>(indecomps.size(), Dynamic_HNF());
    
    // Start one step too early for better indexing:
    r2degree current_grid_degree = lower_bound - grid_step;

    for(int i = 0; i < grid_length_y; i++){ 
        current_grid_degree.first = lower_bound.first - grid_step.first; // Reset x-coordinate for each y-coordinate
        current_grid_degree.second = lower_bound.second + i*grid_step.second;
        // First in y direction, we recompute all local decompositions whenever necessary.
        update_HNF_rows_at_y_level(current_grid_degree, indecomps, grid_locations, local_grid_row_data, decomposer, slope_bounds);
        
        for(int j = 0; j < grid_length_x; j++){
            current_grid_degree.first += grid_step.first; 
            // Then we need to check if we have crossed into a new grid-square in any local grid.    
            update_grid_locations_x(current_grid_degree, indecomps, grid_locations);

            ostream << "G," << i << "," << j << ", " << current_grid_degree << std::endl;
            if (progress_bar) {
                int points_processed = i * grid_length_y + j;
                std::string name = "Grid point";
                show_progress_bar(points_processed, grid_size, name);
            }
            // Now actually compute the HNF, but use the data previously computed 
            process_grid_cell(i, j, current_grid_degree, indecomps, grid_locations, local_grid_row_data, 
                composition_factors, grid_ind_dimensions, all_scss_dimensions, ostream, subspaces, slope_bounds, decomposer);
        }
    }

    std::cout << std::endl;
    std::cout << "  Tracked the dimensions of " << grid_ind_dimensions.size() << " indecomposable summands." << std::endl;
    std::cout << "  The dimensions of indecomposable summands at the grid points are distributed as:" << std::endl;
    calculate_stats(grid_ind_dimensions);
    std::cout << "  The dimensions of the composition factors at the grid points are distributed as:" << std::endl;
    calculate_stats(all_scss_dimensions);
}

template <typename Outputstream>
void full_grid_induced_decomposition(aida::AIDA_functor& decomposer, 
    std::ifstream& istream, Outputstream& ostream, 
    bool show_indecomp_statistics, bool show_runtime_statistics, 
    bool dynamic_grid = true,
    bool is_decomposed = false,
    const int& grid_length_x = 30, const int& grid_length_y = 30) {
    
    

    if(is_decomposed){
        vec<R2Mat> matrices;
        graded_linalg::construct_matrices_from_stream(matrices, istream);
        vec<r2degree> grid_points;
        if(dynamic_grid){
            process_summands_smart_grid(decomposer, ostream, grid_length_x, grid_length_y, matrices);
        } else {
            process_summands_fixed_grid(decomposer, ostream, grid_length_x, grid_length_y, matrices);
        }
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
        if(dynamic_grid){
            process_summands_smart_grid(decomposer, ostream, grid_length_x, grid_length_y, B_list);
        } else {
            process_summands_fixed_grid(decomposer, ostream, grid_length_x, grid_length_y, B_list);
        }
    }
    
} // full_grid_induced_decomposition

} // namespace hnf

