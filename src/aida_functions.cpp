/**
 * @file aida.cpp
 * @author Jan Jendrysiak
 * @version 0.2
 * @date 2025-10-21
 * @brief  How to use the AIDA program
 * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "aida_functions.hpp"


namespace aida {




/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @param sub_batch_indices
 * @return vec<index> 
 */
vec<index> hom_action(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N, bitset& sub_batch_indices){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].rbegin(); it != N._rows[i].rend(); it++){
            if( sub_batch_indices.test(*it)){
                result.push_back( linearise_position_reverse_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
            } 
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}

/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @return vec<index> 
 */
vec<index> hom_action_full_support(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].rbegin(); it != N._rows[i].rend(); it++){
            result.push_back( linearise_position_reverse_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}

/**
 * @brief Given a homomorphism from a block C to another, presented by a matrix Q, which is given as a list hom of row-operations,
 *         this function computes the/a linearisation of Q*N_C.
 * 
 * @param row_glueing
 * @param total_num_rows
 * @param hom
 * @param row_ops
 * @param N
 * @return vec<index> 
 */
vec<index> hom_action_extension(index& row_glueing, index& total_num_rows, vec<index>& hom, vec<pair>& row_ops, Sparse_Matrix& N){    
    vec<index> result = vec<index>(0);
    for( index q : hom){
        auto [i,j] = row_ops[q];
        for(auto it = N._rows[i].begin(); it != N._rows[i].end(); it++){
            result.push_back( linearise_position_ext<index>(*it, (j + row_glueing), N.get_num_cols(), total_num_rows));
        }
    }
    std::sort(result.begin(), result.end());
    convert_mod_2(result);
    return result;
}


/**
 * @brief Apply a homomorphism from c to b to all of A
 * 
 * @param A 
 * @param B 
 * @param C 
 * @param hom 
 * @param row_ops 
 */
void hom_action_A(GradedMatrix& A, vec<index>& source_rows, vec<index>& target_rows, vec<index>& hom, vec<pair>& row_ops, std::shared_ptr<Base_change_virtual>& base_change){
    for( index q : hom){
        auto [i,j] = row_ops[q];
        i = source_rows[i];
        j = target_rows[j];
        #if OBSERVE
            if( std::find(observe_row_indices.begin(), observe_row_indices.end(), i) != observe_row_indices.end() ){
                std::cout << "Row operation: " << i << " -> " << j << std::endl;
            }
        #endif
        base_change->add_row_op(i, j);
        assert(A.is_admissible_row_operation(i, j));
        A.fast_rev_row_op(i, j);
    }
}

/**
 * @brief Apply a homomorphism from c to b to N
 *  TO-DO: At the moment we change N everywhere, is that a problem?
 * 
 */
void hom_action_N(Block& B_target, Sparse_Matrix& N_source, Sparse_Matrix& N_target, vec<index>& hom, vec<pair>& row_ops){
    for( index q : hom){
        auto [i, j] = row_ops[q];
        CT::add_to(N_source._rows[i], N_target._rows[j]);
    }
    N_target.compute_columns_from_rows(B_target.rows);
    bool reduction = B_target.reduce_N_fully(N_target, true);
}

/**
 * @brief Deletes all hom-spaces whose (co)domain is merged or extended.
 *          TO-DO: We could, at this point, also compute which of the homomorphisms factor through the new block, 
 *          but since it is not always clear that we will need this information again, this might increase the total running time,
 *          instead of decreasing it.
 * 
 * @param block_partition 
 * @param hom_spaces 
 * @param domain_keys 
 * @param codomain_keys
 */
void update_hom_spaces( vec<Merge_data>& block_partition, Hom_map& hom_spaces, 
    std::unordered_map<index, vec<index>>& domain_keys, std::unordered_map<index, vec<index>>& codomain_keys){
    for(auto& partition : block_partition){
        vec<index>& block_indices = partition.first;
        for(index c : block_indices){
            for(index b : domain_keys[c]){
                hom_spaces.erase({c, b});
            }
            domain_keys.erase(c);
            for(index b : codomain_keys[c]){
                hom_spaces.erase({b, c});
            }
            codomain_keys.erase(c);
        }
    }
}

/**
 * @brief Constructs the digraph of non-zero homomorphisms between the blocks in vertex_labels.
 *        Not that the way hom_spaces is computed, meanse that there might be extra edges in the digraph, 
 *        where the corresponding homomorphisms are actually zero or zero at the r2degree of the current batch.
 * 
 * @param hom_spaces 
 * @param vertex_labels 
 * @return edge_list 
 */
Graph construct_hom_digraph( Hom_map& hom_spaces, vec<index>& vertex_labels){
    auto edge_checker = [&hom_spaces](const index& c, const index& b) -> bool {
        return hom_spaces[{c,b}].first.data.size(); 
    };
    return construct_boost_graph(vertex_labels, edge_checker);
}

Graph construct_batch_transform_graph(Transform_Map& batch_transforms, vec<Merge_data>& virtual_blocks){
    auto edge_checker = [&batch_transforms, virtual_blocks](const index& c, const index& b) -> bool {
        return ! batch_transforms[std::make_pair(virtual_blocks[c], virtual_blocks[b])].empty();
    };
    return construct_boost_graph(virtual_blocks.size(), edge_checker);
}


/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns, vec<index>& batch_indices, vec<long>& result, bitset& sub_batch_indices) {
    
    assert(batch_indices.size() == sub_batch_indices.size());
    for(auto& ref : Ns){
        Sparse_Matrix& N = ref.get();
        for(index i = sub_batch_indices.size()-1; i >= 0; i--){
            if(sub_batch_indices.test(i)){
                for(index j : N.data[i]){
                    result.push_back(A.linearise_position_reverse(batch_indices[i], j));
                }
            }
        }
    }
    std::sort(result.begin(), result.end());
}

/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior_full_support( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns, vec<index>& batch_indices, vec<long>& result) {
    for(auto& ref : Ns){
        Sparse_Matrix& N = ref.get();
        for(index i = batch_indices.size()-1; i >= 0; i--){
                for(index j : N.data[i]){
                    result.push_back(A.linearise_position_reverse(batch_indices[i], j));
                }
        }
    }
}

/**
 * @brief Constructs the linear system to delete the block N=(A_t)_B with row and column operations. Concretly:
 *          Solve for matrices: P, P_i, Q_i, (i != j in relevant blocks)
 *          B * P_i = Q_i * B_i 
 *          B * P + Q_i * N_i = N
 * 
 * @param A 
 * @param Ns
 * @param sub_batch_indices
 * @param restricted_batch
 * @param relevant_blocks
 * @param block_map
 * @param S
 * @param ops
 * @param b_vec
 * @param N_map
 */
void construct_linear_system(GradedMatrix& A, vec<index>& batch_indices, bitset& sub_batch_indices, bool restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            SparseMatrix<long>& S, vec<op_info>& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            const bitset& extra_columns){
    //TO-DO: Parallelise this whole subroutine.

    Block& B_first = *block_map[*b_vec.begin()];  
    Block& B_probe = *block_map[*relevant_blocks.begin()];
    size_t buffer = 0;
    size_t max_buffer_size = 200000;
    if(B_first.local_data != nullptr){
        buffer = B_first.local_data->data.size();
    }
    buffer += b_vec.size()*relevant_blocks.size()*(B_first.rows.size()*B_probe.rows.size() 
    + B_probe.columns.size()*B_first.columns.size());
    buffer = std::min(buffer, max_buffer_size);
    S.data.reserve(buffer);
    
    // First find all blocks which can actually contribute by having a non-zero admissible row operation to any row of B:
    // While doing that, construct the associated columns of S belonging to these row operations.
    index S_index = 0; indtree admissible_relevant_blocks;

    bool no_new_inserts = false;
    for(index b: b_vec){
        Block& B = *block_map[b];
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            for(index i = 0; i < C.rows.size(); i++){
                auto source_index = C.rows[i];
                for(index j = 0; j < B.rows.size(); j++){
                    auto target_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_index, target_index)){
                        S.data.push_back(vec<long>());
                        ops.emplace_back( std::make_pair(std::make_pair(i , j), std::make_pair(c, b)) );
                        // Fill the column of S belonging to the operation first with the row in N_C, 
                        // then with the row in C, so that no sorting is needed.
                        for(auto row_it = N_C._rows[i].rbegin(); row_it != N_C._rows[i].rend(); row_it++){
                            if(!restricted_batch || sub_batch_indices.test(*row_it)){
                                S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[*row_it], target_index));
                            }
                        }    

                        if(!C._rows[i].empty()){
                        for(auto row_it2 = C._rows[i].rbegin(); row_it2 != C._rows[i].rend(); row_it2++){
                            // only insert if the row operation has an effect on B.rows*C.columns.
                            if(!no_new_inserts){
                                auto result = admissible_relevant_blocks.insert(c);
                                no_new_inserts = result.second;
                            }
                            auto effect_position = A.linearise_position_reverse(C.columns[*row_it2], target_index);
                            S.data[S_index].emplace_back(effect_position);
                        }
                        }
                        S_index++;
                    } 
                }
            }
            no_new_inserts = false;
        }
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        // Next add all col ops from all blocks in b_vec to the columns of the blocks which could contribute.

        for(index c : admissible_relevant_blocks){
            auto it = block_map[c];
            Block& C = *it;
            for(index i = 0; i < C.columns.size(); i++){
                for(index j = 0; j < B.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[j], C.columns[i])){
                        S.data.push_back(vec<long>());
                        for(index row_index : B.data[j]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[i], row_index));
                        }
                        S_index++;
                    }
                }
            }
        }
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This doesnt work yet, somehow local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = sub_batch_indices.find_first(); i != bitset::npos ; i = sub_batch_indices.find_next(i)){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<long>());
                    for(index j : column){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[i], j)); 
                    }
                    S_index++;
                }
            }
        }
    }

    // If we're in the last step of naive decomposition, 
    // need to the additional column operation from extra columns to sub_batch_indices
    if(extra_columns.any()){
        for(index i = extra_columns.find_first(); i != bitset::npos; i = extra_columns.find_next(i)){
            for(index j = sub_batch_indices.find_first(); j != bitset::npos; j = sub_batch_indices.find_next(j)){
                S.data.push_back(vec<long>());
                for(auto b : b_vec){
                    for(index row_index : N_map[b].data[i]){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[j], row_index));
                    }
                }
                std::sort(S.data[S_index].begin(), S.data[S_index].end());
                S_index++;
            }          
        }
    }

} //construct_linear_system

/**
 * @brief Constructs the linear system to delete the block N=(A_t)_B with row and column operations. Concretly:
 *          Solve for matrices: P, P_i, Q_i, (i != j in relevant blocks)
 *          B * P_i = Q_i * B_i 
 *          B * P + Q_i * N_i = N
 * 
 * @param A 
 * @param Ns
 * @param sub_batch_indices
 * @param restricted_batch
 * @param relevant_blocks
 * @param block_map
 * @param S
 * @param ops
 * @param b_vec
 * @param N_map
 */
void construct_linear_system_full_support(GradedMatrix& A, vec<index>& batch_indices, bool restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            SparseMatrix<long>& S, vec<op_info>& ops, 
                            vec<index>& b_vec, Sub_batch& N_map){
    //TO-DO: Parallelise this whole subroutine.

    Block& B_first = *block_map[*b_vec.begin()];  
    Block& B_probe = *block_map[*relevant_blocks.begin()];
    size_t buffer = 0;
    size_t max_buffer_size = 200000;
    if(B_first.local_data != nullptr){
        buffer = B_first.local_data->data.size();
    }
    buffer += b_vec.size()*relevant_blocks.size()*(B_first.rows.size()*B_probe.rows.size() 
    + B_probe.columns.size()*B_first.columns.size());
    buffer = std::min(buffer, max_buffer_size);
    S.data.reserve(buffer);
    
    // First find all blocks which can actually contribute by having a non-zero admissible row operation to any row of B:
    // While doing that, construct the associated columns of S belonging to these row operations.
    index S_index = 0; indtree admissible_relevant_blocks;

    bool no_new_inserts = false;
    for(index b: b_vec){
        Block& B = *block_map[b];
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            for(index i = 0; i < C.rows.size(); i++){
                auto source_index = C.rows[i];
                for(index j = 0; j < B.rows.size(); j++){
                    auto target_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_index, target_index)){
                        S.data.push_back(vec<long>());
                        ops.emplace_back( std::make_pair(std::make_pair(i , j), std::make_pair(c, b)) );
                        // Fill the column of S belonging to the operation first with the row in N_C, 
                        // then with the row in C, so that no sorting is needed.
                        for(auto row_it = N_C._rows[i].rbegin(); row_it != N_C._rows[i].rend(); row_it++){
                            
                            S.data[S_index].emplace_back(A.linearise_position_reverse(batch_indices[*row_it], target_index));

                        }    

                        if(!C._rows[i].empty()){
                        for(auto row_it2 = C._rows[i].rbegin(); row_it2 != C._rows[i].rend(); row_it2++){
                            // only insert if the row operation has an effect on B.rows*C.columns.
                            if(!no_new_inserts){
                                auto result = admissible_relevant_blocks.insert(c);
                                no_new_inserts = result.second;
                            }
                            auto effect_position = A.linearise_position_reverse(C.columns[*row_it2], target_index);
                            S.data[S_index].emplace_back(effect_position);
                        }
                        }
                        S_index++;
                    } 
                }
            }
            no_new_inserts = false;
        }
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        // Next add all col ops from all blocks in b_vec to the columns of the blocks which could contribute.

        for(index c : admissible_relevant_blocks){
            auto it = block_map[c];
            Block& C = *it;
            for(index i = 0; i < C.columns.size(); i++){
                for(index j = 0; j < B.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[j], C.columns[i])){
                        S.data.push_back(vec<long>());
                        for(index row_index : B.data[j]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[i], row_index));
                        }
                        S_index++;
                    }
                }
            }
        }
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This doesnt work yet, somehow local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i_b : batch_indices){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<long>());
                    for(index j : column){
                        S.data[S_index].emplace_back(A.linearise_position_reverse(i_b, j)); 
                    }
                    S_index++;
                }
            }
        }
    }


} //construct_linear_system_full_support

/**
 * @brief  
 * 
 * @param batch_indices 
 * @param sub_batch_indices 
 * @param restricted_batch 
 * @param relevant_blocks 
 * @param block_map 
 * @param S 
 * @param ops 
 * @param b_vec 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @param y 
 * @param extra_columns 
 */
void construct_linear_system_hom(vec<index>& batch_indices, bitset& sub_batch_indices, bool& restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            Sparse_Matrix& S, vec< hom_info >& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            Hom_map& hom_spaces,
                            vec<index>& row_map,
                            vec<index>& y,
                            const bitset& extra_columns){
    
    //TO-DO: Parallelise this
    index row_glueing = 0;
    index total_num_rows = 0;
    index S_index = 0;
    for(index b : b_vec){
        total_num_rows += block_map[b]->get_num_rows();
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        Sparse_Matrix& N_B = N_map[b];
        // Populate y
        for(index row_index = 0; row_index < B.rows.size(); row_index++){
            for(auto int_col_it = N_B._rows[row_index].rbegin(); int_col_it != N_B._rows[row_index].rend(); int_col_it++){
                if(sub_batch_indices.test(*int_col_it)){
                    y.emplace_back(linearise_position_reverse_ext<index>(*int_col_it, row_index + row_glueing, N_B.get_num_cols(), total_num_rows));
                }
            }
        }
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            Hom_space& hom_cb = hom_spaces[{c,b}];
            for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
                ops.emplace_back( i_B , std::make_pair(c, b) );
                S.data.emplace_back(hom_action(row_glueing, total_num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C, sub_batch_indices) );
            }
        }
        row_glueing += B.get_num_rows();
    }
    std::sort(y.begin(), y.end());
    S_index = S.data.size();
    row_glueing = 0;
    for(index b: b_vec){
        Block& B = *block_map[b];
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This isnt fully optimised yet, local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<index>());
                    for(index j : column){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( i, row_map[j]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    S_index++;
                }
            }
        }
        row_glueing += B._rows.size();
    }
    // If we're in the last step of naive decomposition, use also column-operations internal to the batch:
    
    if(extra_columns.any()){
        for(index i = extra_columns.find_first(); i != bitset::npos; i = extra_columns.find_next(i)){
            for(index j = sub_batch_indices.find_first(); j != bitset::npos; j = sub_batch_indices.find_next(j)){
                S.data.push_back(vec<index>());
                row_glueing = 0;
                for(auto b : b_vec){
                    for(index row_index : N_map[b].data[i]){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( j, row_map[row_index]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    row_glueing += N_map[b]._rows.size();
                }
                //TO-DO: with smarter book-keeping this could be avoided:
                std::sort(S.data[S_index].begin(), S.data[S_index].end());
                S_index++;
            }
        }
    }

} //construct_linear_system_hom


/**
 * @brief  
 * 
 * @param batch_indices 
 * @param sub_batch_indices 
 * @param restricted_batch 
 * @param relevant_blocks 
 * @param block_map 
 * @param S 
 * @param ops 
 * @param b_vec 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @param y 
 * @param extra_columns 
 */
void construct_linear_system_hom_full_support(vec<index>& batch_indices, bool& restricted_batch,
                            vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            Sparse_Matrix& S, vec< hom_info >& ops, 
                            vec<index>& b_vec, Sub_batch& N_map,
                            Hom_map& hom_spaces,
                            vec<index>& row_map,
                            vec<index>& y){
    
    //TO-DO: Parallelise this
    index row_glueing = 0;
    index total_num_rows = 0;
    index S_index = 0;
    for(index b : b_vec){
        total_num_rows += block_map[b]->get_num_rows();
    }
    for(index b: b_vec){
        Block& B = *block_map[b];
        Sparse_Matrix& N_B = N_map[b];
        // Populate y
        for(index row_index = 0; row_index < B.rows.size(); row_index++){
            for(auto int_col_it = N_B._rows[row_index].rbegin(); int_col_it != N_B._rows[row_index].rend(); int_col_it++){

                y.emplace_back(linearise_position_reverse_ext<index>(*int_col_it, row_index + row_glueing, N_B.get_num_cols(), total_num_rows));

            }
        }
        auto b_it = b_vec.begin();
        for(auto c_it = relevant_blocks.begin(); c_it != relevant_blocks.end(); c_it++){
            index c = *c_it;
            // Only consider operations from outside of b_vec!
            if(b_it != b_vec.end()){
                if(c == *b_it){
                    b_it++; continue;}
            }
            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            Hom_space& hom_cb = hom_spaces[{c,b}];
            for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
                ops.emplace_back( i_B , std::make_pair(c, b) );
                S.data.emplace_back(hom_action_full_support(row_glueing, total_num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C) );
            }
        }
        row_glueing += B.get_num_rows();
    }
    std::sort(y.begin(), y.end());
    S_index = S.data.size();
    row_glueing = 0;
    for(index b: b_vec){
        Block& B = *block_map[b];
        // At last, add the basic column-operations from B to N which have already been computed
        // TO-DO: This isnt fully optimised yet, local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i = 0; i < batch_indices.size(); i++){
                for(vec<index>& column : (*B.local_data).data){
                    S.data.push_back(vec<index>());
                    for(index j : column){
                        S.data[S_index].emplace_back( linearise_position_reverse_ext<index>( i, row_map[j]+row_glueing, batch_indices.size(), total_num_rows));
                    }
                    S_index++;
                }
            }
        }
        row_glueing += B._rows.size();
    }

} //construct_linear_system_hom_full_support

/**
 * @brief Stores all entries of N[b] at the column_indices given in a single vector of size N[b].rows*N[b].columns 
 * 
 * @param b 
 * @param batch_column_indices 
 * @param N_map 
 * @param row_map 
 * @return vec<index> 
 */
void linearise_sub_batch_entries(vec<index>& result, Sparse_Matrix& N, bitset& batch_column_indices, vec<index>& row_map){
    for(index i = batch_column_indices.find_first(); i != bitset::npos; i = batch_column_indices.find_next(i) ){
        for(index r : N.data[i]){
            result.emplace_back( linearise_position_ext<index>(i, row_map[r], batch_column_indices.size(), N.get_num_rows()) );
        }
    }
}


void construct_linear_system_extension(Sparse_Matrix& S, vec<hom_info>& hom_storage, index& E_threshold,
    index& N_threshold, index& M_threshold, index& b, bitset& b_non_zero_columns, 
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){

    vec<index>& pro_block_blocks = pro_block.first;
    bitset& target = pro_block.second;
    index num_rows = block_map[b]->get_num_rows();
    assert( block_map[b]->get_num_rows() == N_map[b].get_num_rows());
    assert( N_map[b].get_num_rows() == N_map[b]._rows.size());


    // First add row-operations from the virtual processed block.

    for( index c : pro_block_blocks){
        Sparse_Matrix& N_C = N_map[c];
        Hom_space& hom_cb = hom_spaces[{c,b}];
        for(index i_B = 0; i_B < hom_cb.first.data.size(); i_B++){
            hom_storage.emplace_back( i_B , std::make_pair(c, b) );
            index row_glue = 0;
            S.data.emplace_back(hom_action_extension(row_glue, num_rows, hom_cb.first.data[i_B], hom_cb.second, N_C) );
            assert(is_sorted(S.data.back()));
        }
    }

    E_threshold = hom_storage.size();
    // Then the internal column-operations

    for(index i : incoming_vertices){
        // Do not need to consider this, if the respective cocycle has been deleted
        if(deleted_cocycles_b.test(i)){
            Merge_data& E = pro_blocks[i];
            vec<Batch_transform>& internal_col_ops = batch_transforms[{E, pro_block}];
            assert( !internal_col_ops.empty());
            for( index j = 0; j < internal_col_ops.size(); j++){
                Batch_transform col_ops = internal_col_ops[j];
                DenseMatrix& T = col_ops.first;
                // The following is bloaty, could be fixed by not having T as a Dense_Matrix or directly reading of the result.
                Sparse_Matrix N_b = N_map[b];
                N_b.multiply_dense(T);
                S.data.push_back(vec<index>());
                linearise_sub_batch_entries(S.data.back(), N_b, target, row_map);
                assert(is_sorted(S.data.back()));
                hom_storage.push_back({j, {i, b}});
            }
        } else {
            // Nothing to do. might count how often this happens.
        }
    }

    N_threshold = hom_storage.size();

    // Column-operations from the support of B in the batch:
    
    for(index i = b_non_zero_columns.find_first(); i != bitset::npos; i = b_non_zero_columns.find_next(i)){
        for(index j = target.find_first(); j != bitset::npos; j = target.find_next(j)){
            S.data.push_back(vec<index>());
            for(index row_index : N_map[b].data[i]){
                S.data.back().emplace_back( linearise_position_ext<index>( j, row_map[row_index], target.size(), num_rows));
            }
            assert(is_sorted(S.data.back()));
            hom_storage.push_back({i, {j, b}});
        }
    }

    M_threshold = hom_storage.size();
    // Add the basic column-operations from B to N which have already been computed
    
    if(block_map[b]->local_data != nullptr){
    for(index i = target.find_first(); i != bitset::npos; i = target.find_next(i)){
        for(vec<index>& column : (block_map[b]->local_data)->data){
            S.data.push_back(vec<index>());
            for(index j : column){
                S.data.back().emplace_back( linearise_position_ext<index>( i, row_map[j], target.size(), num_rows));
                assert(is_sorted(S.data.back()));
            }
        }
    }
    }

    

} //construct_linear_system_extension

}



