/**
 * @file aida_decompose.cpp
    * @author Jan Jendrysiak
    * @brief Decomposes a presentation matrix using AIDA
    * @version 0.2
    * @date 2025-10-21
    *   This file is part of the AIDA library.
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/
#include "aida_decompose.hpp"


namespace aida {

/**
 * @brief Decomposes the matrix A into a direct sum of indecomposable submatrices.
 * 
 * @param A 
 * @param B_list 
 * @param base_change 
 * @param statistics 
 * @param config
 * @param merge_info 
 */
void AIDA(GradedMatrix& A, Block_list& B_list, vec<vec<transition>>& vector_space_decompositions, std::shared_ptr<Base_change_virtual>& base_change, AIDA_runtime_statistics& statistics,
    AIDA_config& config, Full_merge_info& merge_info) {
    
    #if TIMERS
        aida::full_timer.start();
        aida::misc_timer.start();
    #endif

    index batches = A.col_batches.size();
    

    // Only for analysis and optimisation ->
    

    #if OBSERVE
        // Continuous monitoring of the content of a batch
        observe_row_indices = vec<index>();

        std::cout << "Observing batch " << observe_batch_index << " with columns " << A.col_batches[observe_batch_index] << " at " << std::endl;
        Degree_traits<r2degree>::print_degree(A.col_degrees[A.col_batches[observe_batch_index][0]]);
        std::cout << " and with content:" << std::endl;
        Sparse_Matrix observed_batch_comparison = A.restricted_domain_copy(A.col_batches[observe_batch_index]);
        observed_batch_comparison.print();
        // Save indices to see where the first change to this batch occurs.
        for(vec<index> column : observed_batch_comparison.data){
            observe_row_indices.insert(observe_row_indices.end(), column.begin(), column.end());
        }
    #endif

    vec<index> row_map(A.get_num_rows(), 0);
    vec<Block_iterator> block_map;
    #if TIMERS
        misc_timer.stop();
        update_block_timer.resume();
    #endif
    initialise_block_list(A, B_list, block_map);
    if(batches == 0){
        if(config.show_info){
            std::cout << "The entered matrix has no columns and is thus trivially decomposable w.r.t. any basis." << std::endl;   
        }
        return;
    } 
    #if TIMERS
        misc_timer.resume();
        update_block_timer.stop();
    #endif

    Sub_batch N_map;
    Hom_map hom_spaces; // Stores the hom-spaces for each pair of initials (c, b) where necessary
    std::unordered_map< index, vec<index>> domain_keys; // For some c, stores the indices b for which hom_spaces has a key (c, b).
    std::unordered_map< index, vec<index>> codomain_keys; // For some b, stores the indices c for which hom_spaces has a key (c, b).    
    vec<bitset> e_vec = compute_standard_vectors(A.k_max);
    vec<bitset> count_vector = compute_sum_of_standard_vectors(A.k_max);

    #if TIMERS
        misc_timer.stop();
        compute_rows_timer.resume();
    #endif
    A.compute_revrows();
    #if TIMERS
        misc_timer.resume();
        compute_rows_timer.stop();
    #endif

    for(index t = 0; t < batches; t++){
        #if !DETAILS
            if (config.progress) {
                static index last_percent = -1;
                // (-)^{1.5} progress bar for now, but not clear that computational time increases with this exponent.
                index percent = static_cast<index>(pow(static_cast<double>(t + 1) / batches, 1.5) * 100);
                if (percent != last_percent) {
                    // Calculate the number of symbols to display in the progress bar
                    int num_symbols = percent / 2;
                    std::cout << "\r" << t + 1 << " batches : [";
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
                if (t == batches - 1) {
                    std::cout << std::endl;
                }
            }
        #endif

        bool one_block_left = false;
        vec<index> batch_indices = A.col_batches[t]; // Indices of the columns in the batch
        int k_ = batch_indices.size(); // Number of columns in the batch
        // TO-DO: Have seen rounding errors here, investigate.
        r2degree alpha = A.col_degrees[batch_indices[0]]; // r2degree of the batch
        
        #if DETAILS
            std::cout << "Processing batch " << t << " with " << k_ << " columns at the indices " << batch_indices <<  std::endl;
        #endif
        N_map.clear(); bool no_further_comp = false; vec<Merge_data> block_partition = {}; indtree active_blocks;
        
        // Get the batch as a set of columns from the rows and identify the blocks which need to be processed
        #if TIMERS
            misc_timer.stop();
            compute_N_timer.resume();
        #endif
        compute_touched_blocks(active_blocks, block_map, A, batch_indices, N_map); 

        #if OBSERVE
        if( t == observe_batch_index ){
            std::cout << "Analysing batch " << t << " at " << batch_indices << " - Printing B and N:" << std::endl;
            for(index b : active_blocks){
                std::cout << "Block " << b << ": " << std::endl;
                block_map[b]->print();
                std::cout << "N[" << b << "]: " << std::endl;
                N_map[b].print();
            }
        }
        #endif
        #if TIMERS
            misc_timer.resume();
            compute_N_timer.stop();
        #endif
        #if DETAILS
             std::cout << "  !! There are "  << active_blocks.size() << " touched blocks with the following indices: ";
            for(index i : active_blocks){
                std::cout << i << " ";
            }
            std::cout << std::endl;
        #endif
        // First try to delete every whole sub-batch only with column operations. That is, compute the *affected* blocks.
        if(active_blocks.size() != 1) {
            for(auto it = active_blocks.begin(); it != active_blocks.end();){
                index j = *it;
                Block& B = *block_map[j];
                // No need to do anything here if the block is empty. 
                    if(B.columns.size() == 0){
                        it++;
                        B.local_data = std::make_shared<Sparse_Matrix>(0,0);
                        N_map[j].compute_rows_forward_map(row_map, B.rows.size());
                        continue;}
                auto& N = N_map[j];
                #if TIMERS
                    delete_with_col_timer.resume();
                    misc_timer.stop();
                #endif 
                bool only_col_ops = B.delete_with_col_ops(alpha, N, config.supress_col_sweep);
                #if TIMERS
                    delete_with_col_timer.stop();
                    misc_timer.resume();
                #endif 
                if(only_col_ops){
                    #if DETAILS
                        std::cout << "      Deleted N at index " << j << " with column ops." << std::endl;
                    #endif
                    statistics.counter_col_deletion++;
                    B.delete_local_data();
                    N_map.erase(j);
                    it = active_blocks.erase(it);
                } else {
                    it++;
                    // If the block could not be deleted with column operations, then it is still active.
                    // We will need its row-information later.
                    #if TIMERS
                    misc_timer.stop();
                    compute_rows_timer.resume();
                    #endif 
                    B.compute_rows(row_map); 
                    N_map[j].compute_rows_forward_map(row_map, B.rows.size());
                    #if TIMERS
                    compute_rows_timer.stop();
                    misc_timer.resume();
                    #endif
                }
            }
        } else {
            one_block_left = true;
            statistics.counter_no_comp++;
        }
        
        // Next try to delete every whole sub-batch also with the help of row operations.
        // To find all row-operations needed, we need to compute the hom-spaces between the blocks.

        if(active_blocks.size() != 1){ 
            assert(active_blocks.size() > 0);
            #if OBSERVE
                for(index r : observe_row_indices){
                    if( active_blocks.find(block_map[r]->rows[0]) != active_blocks.end() ){
                        std::cout << "Row index " << r << " belongs to block " << block_map[r]->rows[0] << std::endl;
                    }
                }
            #endif
            #if DETAILS
                std::cout << "   !! There are "  << active_blocks.size() << " affected blocks with the following row indices: ";
                for(index i : active_blocks){
                    std::cout << " " << block_map[i]->get_type() << " ";
                    std::cout << (*block_map[i]).rows << " - ";
                }
                std::cout << std::endl;
                for(index i : active_blocks){
                    std::cout << " N[" << i << "]: ";
                    N_map[i].print_rows();
                }
            #endif

            
            // Then delete with previously computed hom-spaces.
            // We want to start with the blocks of lowest r2degree, because the spaces of homomorphisms with these codomains are smaller.

            for(auto itr = active_blocks.rbegin(); itr != active_blocks.rend();) {
                index b = *itr; vec<index> b_vec = {b}; 
                // This is ugly. Need to find a solution for the set / vector problem.
                vec<index> active_blocks_vec(active_blocks.begin(), active_blocks.end());

                if( !config.brute_force ){
                    compute_hom_to_b(A, b, block_map, active_blocks, hom_spaces, domain_keys, codomain_keys, alpha, statistics, config);
                }

                #if TIMERS
                    full_block_reduce_timer.start();
                #endif
                bool deleted_N_b = use_either_block_reduce_full_support(A, b_vec, N_map, batch_indices, false, 
                    active_blocks_vec, block_map, base_change, row_map, hom_spaces, config.brute_force, config.compare_both);

                #if TIMERS
                    full_block_reduce_timer.stop();
                    dispose_S_timer.stop();
                    misc_timer.resume();
                #endif
                
                if( deleted_N_b){
                    #if DETAILS
                        std::cout << "      Deleted N at index " << b << " with row ops." << std::endl;
                    #endif
                    statistics.counter_row_deletion++;
                    block_map[b]->delete_local_data();
                    N_map.erase(b);
                    auto it = active_blocks.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it);
                } else {
                    // It is imporant to have the local data be completely reduced 
                    // ( that is, have a canonical representation in the quotient space )
                    bool further_reduce = block_map[b] -> reduce_N_fully(N_map[b], false);
                    
                    #if TIMERS
                        misc_timer.stop();
                        compute_rows_timer.resume();
                    #endif               
                    N_map[b].compute_rows_forward_map(row_map, block_map[b]->rows.size());
                    #if TIMERS
                        misc_timer.resume();
                        compute_rows_timer.stop();
                    #endif
                    assert(further_reduce == false);
                    itr++;
                }
            }
        } else {
            if(!one_block_left){
                statistics.counter_only_col++;
                one_block_left = true;
            }
        }

        if( k_ != 1 &&  active_blocks.size() != 1) {
            assert(active_blocks.size() > 0);
            statistics.num_of_pierced_blocks.push_back(active_blocks.size());
            #if DETAILS
                std::cout << "There are "  << active_blocks.size() << " pierced blocks with the following row indices: ";
                for(index i : active_blocks){
                    std::cout << " " << block_map[i]->get_type() << " ";
                    std::cout << (*block_map[i]).rows << " - ";
                }
                std::cout << std::endl;
                for(index i : active_blocks){
                    std::cout << " N[" << i << "]: ";
                    N_map[i].print_rows();
                }
                std::cout << std::endl;
            #endif
            vec<index> pierced_blocks(active_blocks.begin(), active_blocks.end());

            #if TIMERS
                pre_alpha_decomp_optimisation_timer.resume();
                misc_timer.stop();
            #endif
            bitset non_zero_indices = simultaneous_column_reduction_full_support(N_map, pierced_blocks, pierced_blocks);
            if(non_zero_indices.count() < k_){
                std::cout << "Either the input file was not minimised correctly or there is a bug in the algorithm." << std::endl;
                #if DETAILS
                for(index b: pierced_blocks){
                    std::cout << "Block and N for " << b << std::endl;
                    block_map[b]->print();
                    N_map[b].print_rows();
                }
                #endif

                assert(false);
            }
            // TO-DO: Check how decomposed the matrix already is and pass the pieces to the next algorithm:
            
            vec<Merge_data> prelim_decomposition = find_prelim_decomposition(N_map, pierced_blocks, count_vector[k_-1]);
            // Ordering is not strictly necessary, but some parts might (will!) not work because they have been coded in a way which assumes it.
            for(Merge_data& merge : prelim_decomposition){
                std::sort(merge.first.begin(), merge.first.end());
            }
            #if DETAILS
                std::cout << "Prelim decomposition: " << std::endl;
                for(auto& pair : prelim_decomposition){
                    std::cout << "  " << pair.first << " -> " << pair.second << std::endl;
                }
            #endif

            #if TIMERS
                pre_alpha_decomp_optimisation_timer.stop();
                alpha_decomp_timer.resume();
            #endif
            for( Merge_data& pair : prelim_decomposition){
                vec<index>& local_pierced_blocks = pair.first;
                bitset& N_column_indices = pair.second;
                if(N_column_indices.count() > statistics.local_k_max){
                    statistics.local_k_max = N_column_indices.count();
                }
                #if OBSERVE
                    if(std::find(local_pierced_blocks.begin(), local_pierced_blocks.end(), observe_block_index) != local_pierced_blocks.end()){
                        
                        std::cout << " Found " << observe_block_index << " in local_pierced_blocks." << std::endl;
                        std::cout << "  Pierced blocks: " << active_blocks << std::endl;
                        std::cout << "  Local pierced blocks: " << local_pierced_blocks << std::endl;
                        std::cout << "  Batch indices: " << batch_indices << std::endl;
                    }
                #endif

                if(local_pierced_blocks.size() == 1 || N_column_indices.count() == 1){
                    block_partition.push_back(pair);
                } else {
                    Graph hom_digraph; std::vector<index> component; vec<vec<index>> scc; Graph condensation;
                    vec<bool> is_resolvable_cycle;
                    vec<index> computation_order;
                    
                    bool is_resolvable = construct_graphs_from_hom(hom_digraph, component, scc, is_resolvable_cycle, condensation, computation_order, 
                            local_pierced_blocks, hom_spaces, statistics, config, t, block_map, alpha);

                    vec<Merge_data> result;
                    if(is_resolvable && !config.exhaustive && !config.brute_force && !config.compare_both){
                        #if TIMERS
                            full_aida_timer.resume();
                        #endif
                        result = automorphism_sensitive_alpha_decomp(A, B_list, block_map, local_pierced_blocks, batch_indices, N_column_indices, 
                            e_vec, N_map, vector_space_decompositions, config, base_change, hom_spaces, row_map, hom_digraph,
                            computation_order, scc, condensation, is_resolvable_cycle);
                        #if TIMERS
                            full_aida_timer.stop();
                        #endif
                    } else {
                        #if TIMERS
                            full_exhaustive_timer.resume();
                        #endif
                        result = exhaustive_alpha_decomp(A, B_list, block_map, local_pierced_blocks, batch_indices, N_column_indices, 
                            e_vec, N_map, vector_space_decompositions, config, base_change,
                            hom_spaces, row_map, config.brute_force, config.compare_both);
                        #if TIMERS
                            full_exhaustive_timer.stop();
                        #endif
                        if(result.size() == 1){
                            statistics.counter_naive_full_iteration++;
                        }
                        for(auto& merge : result){
                            if(merge.second.count() > 1){
                                statistics.counter_extra_iterations += statistics.num_subspace_iterations[merge.second.count()-1];
                            }
                        }
                        statistics.counter_naive_deletion += result.size()-1;
                    }
                    block_partition.insert(block_partition.end(), std::make_move_iterator(result.begin()), std::make_move_iterator(result.end()));
                }
            }
            #if TIMERS
                alpha_decomp_timer.stop();
                misc_timer.resume();
            #endif
        } else {
            if(!one_block_left){
                statistics.counter_only_row++;
                one_block_left = true;
            }
            // If at some point active_blocks contains only one block OR k == 1, we should eventually land here, thus:
            block_partition = {{vec<index>(active_blocks.begin(), active_blocks.end()) , count_vector[k_-1]}};
        }
        // Deleting remaining local data.
        for(index i : active_blocks){
            block_map[i]->delete_local_data();
        }
        #if TIMERS
            update_block_timer.resume();
            misc_timer.stop();
        #endif
        

        merge_blocks(B_list, N_map, block_map, block_partition, batch_indices, row_map, alpha);
        #if TIMERS
            update_block_timer.stop();
            misc_timer.resume();
        #endif
        // If a block has changed, we delete its hom space. TO-DO: if the change is small, we could update them instead.
        if(!config.brute_force){
            #if TIMERS
                misc_timer.stop();  
                update_hom_timer.resume();
            #endif
            update_hom_spaces(block_partition, hom_spaces, domain_keys, codomain_keys);
            #if TIMERS
                update_hom_timer.stop();
                misc_timer.resume();  
            #endif
        }
        merge_info.push_back(block_partition);

        #if OBSERVE
        Sparse_Matrix observed_batch = A.restricted_domain_copy(A.col_batches[observe_batch_index]);
        index changed_relation = observed_batch.equals_with_entry_check(observed_batch_comparison, true);
        if( changed_relation > -1 ){
            std::cout << " Observed batch was altered in batch " << t << " at " << A.col_batches[t][changed_relation] << std::endl;
            vec<index> sorted_current = observed_batch.data[changed_relation];
            vec<index> sorted_old = observed_batch_comparison.data[changed_relation];
            std::sort(sorted_current.begin(), sorted_current.end());   
            std::sort(sorted_old.begin(), sorted_old.end());
            convert_mod_2(sorted_current);
            convert_mod_2(sorted_old);
            vec<index> diff = sorted_current + sorted_old; 
            std::cout << "  First difference: " << diff << std::endl;
            std::cout << "  Before: " << std::endl;
            observed_batch_comparison.print();
            std::cout << "  After: " << std::endl;
            observed_batch.print();
            std::cout << " Merge data: " << std::endl;
            for(auto& merge : block_partition){
                std::cout << "  " << merge.first << " -> " << merge.second << std::endl;
            }
            observed_batch_comparison = SparseMatrix(observed_batch);
        }
        #endif

    } 

    // Normalise the blocks

    for(auto it = B_list.begin(); it != B_list.end(); it++){
        Block& B = *it;
        B.transform_data(row_map);
    }

    #if TIMERS
        full_timer.stop();
        misc_timer.stop();
    #endif
    
    /** 
    #if DETAILS
        std::cout << "Full merge details: " << std::endl;
        for(index i = 0; i < merge_info.size(); i++){
            std::cout << "  #Merges at batch " << i << ": ";
            for(auto& merge : merge_info[i]){
                std::cout << merge.first << " ";
            }
            std::cout << std::endl;
        }
    #endif
    */
} //AIDA

} //namespace aida