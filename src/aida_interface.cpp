/**
 * @file aida_interface.cpp
 * @author Jan Jendrysiak
 */



#include "aida_interface.hpp"
#include <regex>
#include <filesystem>

namespace aida {

namespace fs = std::filesystem;

AIDA_functor::AIDA_functor() 
    : base_changes(), config(), statistics_vec(), runtime_statistics(), 
      merge_data_vec(), cumulative_statistics(), cumulative_runtime_statistics() {}

void AIDA_functor::load_vector_space_decompositions(int max_dim, std::string decomp_path) {
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

void AIDA_functor::load_existing_decompositions(int& k_max) {
    if(k_max > vector_space_decompositions.size() + 1) {
        std::string decomp_path = findDecompositionsDir();
        int largest_local_decomposition_list = findLargestNumberInFilenames(decomp_path);
        if(k_max <= largest_local_decomposition_list) {
            if(config.show_info) {
                std::cout << "Loading vector space decompositions up to dim " << k_max << std::endl;
            }
            load_vector_space_decompositions(k_max, decomp_path);
        } else {
            load_vector_space_decompositions(largest_local_decomposition_list, decomp_path);
            if(config.show_info) {
                std::cout << "k_max is " << k_max << " but only found decompositions up to dim " 
                          << largest_local_decomposition_list 
                          << ". \n It is possible that the computation will produce an error if we need to decompose more relations at the same time." 
                          << std::endl;
            }
        }
    }
}

void AIDA_functor::compute_vector_space_decompositions_faulty(const std::string& path) {
    const std::string command = "../generate_decompositions -at -cover -transitions ";
    int result = system((command + std::to_string(9)).c_str());
}

void AIDA_functor::clear_decompositions() {
    vector_space_decompositions.clear();
}

void AIDA_functor::operator()(R2GradedSparseMatrix<index>& A, Block_list& B_list) {
    int k_max = A.k_max;
    load_existing_decompositions(k_max);
    if (config.sort){
        A.sort_columns_lexicographically();
        A.sort_rows_lexicographically();
    }
    A.compute_col_batches();

    if(config.show_info) {
        std::cout << " Matrix has " << A.get_num_rows() << " rows and " << A.get_num_cols()
                  << " columns, k_max is " << A.k_max << ", and there are " << A.col_batches.size() << " batches." << std::endl;
    }
    
    std::shared_ptr<Base_change_virtual> base_change;
    if(config.save_base_change) {
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

    AIDA(A, B_list, vector_space_decompositions, base_changes.back(), runtime_statistics.back(), config, merge_info);
    
    #if TIMERS
        runtime_statistics.back().evaluate_timers();
    #endif

    merge_data_vec.push_back(merge_info);
    statistics_vec.back().compute_statistics(B_list);
    
    if(config.show_info) {
        std::cout << B_list.size() << " indecomposable summands." << std::endl;
    }

    cumulative_statistics = AIDA_statistics();
    cumulative_statistics.compute_statistics(B_list);
    cumulative_runtime_statistics = AIDA_runtime_statistics();
    for(auto& runtime_stat : runtime_statistics) {
        cumulative_runtime_statistics += runtime_stat;
    }


    if(config.sort_output) {
        B_list.sort(Compare_by_degrees<r2degree, index, R2GradedSparseMatrix<index>>());
    }
}

Sparse_Matrix AIDA_functor::get_row_basis(index i, index m) {
    vec<pair>& performed_row_ops = base_changes[i]->performed_row_ops;
    Sparse_Matrix result = Sparse_Matrix(m, m, "Identity");
    for(index j = performed_row_ops.size()-1; j > -1; j--) {
        result.row_op_on_cols(performed_row_ops[j].first, performed_row_ops[j].second);
    }
    return result;
}

Sparse_Matrix AIDA_functor::get_row_basis(index m) {
    vec<pair>& performed_row_ops = base_changes.back()->performed_row_ops;
    Sparse_Matrix result = Sparse_Matrix(m, m, "Identity");
    for(index j = performed_row_ops.size()-1; j > -1; j--) {
        result.row_op_on_cols(performed_row_ops[j].first, performed_row_ops[j].second);
    }
    return result;
}

} // namespace aida