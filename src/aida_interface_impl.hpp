#pragma once

namespace aida {

template<typename Index>
void AIDA_functor<Index>::load_vector_space_decompositions(std::size_t max_dim, const std::string& decomp_path) {
    std::string tran_path = decomp_path + "/transitions_reduced_";
    std::size_t start = vector_space_decompositions.size();
    for(std::size_t k = start + 2; k <= max_dim; k++) {
        try {
            vector_space_decompositions.emplace_back(load_transition_list(tran_path + std::to_string(k) + ".bin"));
        } catch (std::exception& e) {
            std::cout << "Could not load transitions_reduced_" << k << ".bin " << std::endl;
            abort();
        }
    }
}

template<typename Index>
void AIDA_functor<Index>::load_existing_decompositions(std::size_t k_max) {
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

template<typename Index>
void AIDA_functor<Index>::compute_vector_space_decompositions_faulty(const std::string& path) {
    const std::string command = "../generate_decompositions -at -cover -transitions ";
    int result = system((command + std::to_string(9)).c_str());
}

template<typename Index>
template<class Matrix>
void AIDA_functor<Index>::decompose(Matrix& A, Block_list_t<Matrix>& B_list) {
    if (config.sort){
        A.sort_compatibly();
    }
    if (!A.compatibly_sorted) {
        const auto order = Degree_traits<base_degree<Matrix>>::lex_lambda();
        A.compatibly_sorted = std::is_sorted(A.row_degrees.begin(), A.row_degrees.end(), order) &&
                              std::is_sorted(A.col_degrees.begin(), A.col_degrees.end(), order);
    }
    GRLINA_DEBUG_CHECK(if (!A.compatibly_sorted)
        throw std::invalid_argument("AIDA requires compatibly sorted rows and columns"));
    const Column_batches_t<Index> batch_info(A);
    std::size_t k_max = batch_info.k_max;
    load_existing_decompositions(k_max);

    if(config.show_info) {
        std::cout << " Matrix has " << A.get_num_rows() << " rows and " << A.get_num_cols()
                  << " columns, k_max is " << batch_info.k_max << ", and there are " << batch_info.col_batches.size() << " batches." << std::endl;
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

    AIDA(A, B_list, vector_space_decompositions, base_changes.back(), runtime_statistics.back(), config, merge_info, batch_info);

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
        B_list.sort(CompareBlocksByDegrees{});
    }
}


} // namespace aida
