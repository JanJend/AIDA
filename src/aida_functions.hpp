/**
 * @file aida_functions.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.2
 * @date 2025-10-21
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */


#pragma once

#ifndef AIDA_FUNCTIONS_HPP
#define AIDA_FUNCTIONS_HPP

#include "config.hpp"
#include "aida_helpers.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <cmath>
#include <utility>

namespace aida{

/**
 * @brief Comparator for merge_data
 * 
 */
inline auto merge_comparator = [](const Merge_data& a, const Merge_data& b) {
    return a.first[0] < b.first[0];
};

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
vec<index> hom_action(index& row_glueing, index& total_num_rows, 
    vec<index>& hom, vec<pair>& row_ops, 
    Sparse_Matrix& N, bitset& sub_batch_indices);


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
vec<index> hom_action_full_support(index& row_glueing, index& total_num_rows, 
    vec<index>& hom, vec<pair>& row_ops, 
    Sparse_Matrix& N);

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
vec<index> hom_action_extension(index& row_glueing, index& total_num_rows, 
    vec<index>& hom, vec<pair>& row_ops, 
    Sparse_Matrix& N);

/**
 * @brief Apply a homomorphism from c to b to all of A
 * 
 * @param A 
 * @param B 
 * @param C 
 * @param hom 
 * @param row_ops 
 */
void hom_action_A(GradedMatrix& A, vec<index>& source_rows, 
    vec<index>& target_rows, vec<index>& hom, 
    vec<pair>& row_ops, std::shared_ptr<Base_change_virtual>& base_change);


/**
 * @brief Apply a homomorphism from c to b to N
 *  TO-DO: At the moment we change N everywhere, is that a problem?
 * 
 */
void hom_action_N(Block& B_target, Sparse_Matrix& N_source, 
    Sparse_Matrix& N_target, vec<index>& hom, 
    vec<pair>& row_ops);


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
    std::unordered_map<index, vec<index>>& domain_keys, 
    std::unordered_map<index, vec<index>>& codomain_keys);


/**
 * @brief Constructs the digraph of non-zero homomorphisms between the blocks in vertex_labels.
 *        Not that the way hom_spaces is computed, meanse that there might be extra edges in the digraph, 
 *        where the corresponding homomorphisms are actually zero or zero at the r2degree of the current batch.
 * 
 * @param hom_spaces 
 * @param vertex_labels 
 * @return edge_list 
 */
Graph construct_hom_digraph( Hom_map& hom_spaces, vec<index>& vertex_labels);

Graph construct_batch_transform_graph(Transform_Map& batch_transforms, vec<Merge_data>& virtual_blocks);



/**
 * @brief Adds the content of source to target, thereby merging virtual blocks.
 * 
 * @param target 
 * @param source 
 */
inline void merge_virtual_blocks(Merge_data& target, Merge_data& source ){
    assert( (target.second & source.second).none() );
    target.first.insert(target.first.end(), source.first.begin(), source.first.end());
    target.second |= source.second;
}

/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns, vec<index>& batch_indices, vec<long>& result, bitset& sub_batch_indices);


/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior_full_support( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>>& Ns,
     vec<index>& batch_indices, vec<long>& result);



inline void test_rows_of_A(GradedMatrix& A, index batch){
    for(index i = 0; i < A.get_num_rows(); i++){
        if(!A._rows[i].empty()){
            for(index j : A._rows[i]){
                if(j < batch){
                    std::cout << "Warning: The row " << i << " has an entry smaller than the current batch: " << j << std::endl;
                }
                if(j > A.get_num_cols()){
                    std::cout << "Warning: The row " << i << " has an entry larger than the number of columns: " << j << std::endl;
                }
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
void construct_linear_system(GradedMatrix& A, vec<index>& batch_indices, 
    bitset& sub_batch_indices, bool restricted_batch,
    vec<index>& relevant_blocks, vec<Block_list::iterator>& block_map, 
    SparseMatrix<long>& S, vec<op_info>& ops, 
    vec<index>& b_vec, Sub_batch& N_map,
    const bitset& extra_columns = bitset(0));


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
                            vec<index>& b_vec, Sub_batch& N_map);

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
                            const bitset& extra_columns = bitset(0));

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
                            vec<index>& y);

/**
 * @brief Stores all entries of N[b] at the column_indices given in a single vector of size N[b].rows*N[b].columns 
 * 
 * @param b 
 * @param batch_column_indices 
 * @param N_map 
 * @param row_map 
 * @return vec<index> 
 */
void linearise_sub_batch_entries(vec<index>& result, Sparse_Matrix& N, 
    bitset& batch_column_indices, vec<index>& row_map);

void construct_linear_system_extension(Sparse_Matrix& S, vec<hom_info>& hom_storage, index& E_threshold,
    index& N_threshold, index& M_threshold, index& b, bitset& b_non_zero_columns, 
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map);

/**
 * @brief Computes a basis for the hom-space Hom(C, B).
 * 
 * @param A 
 * @param C
 * @param B 
 * @return vec<Sparse_Matrix> 
 */
inline Hom_space compute_hom_space(GradedMatrix& A, Block& C, Block& B, 
    r2degree& alpha, const bool& optimised = false, 
    const bool& alpha_hom = false){

    vec< pair > row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.

    Sparse_Matrix K(0,0);

    switch (C.type){
        case BlockType::FREE : {
            index counter = 0;
            for( index i = 0; i < C.get_num_rows(); i++){
                // indices in rows_alpha are internal to C. For external change to .., true) 
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(C.row_degrees[i], false);
                vec<index> basislift = B_alpha.coKernel_basis(rows_alpha, B.rows);
                for( index j : basislift){
                    row_ops.push_back( {i, j} );
                    K.data.push_back( {counter} );
                    counter++;
                }
            }
            K.compute_num_cols();
            return {K, row_ops};
            break;
        }

        case BlockType::CYC : {
            //TO-DO: Implement
            if(optimised){
                return hom_space_optimised(C, B, C.rows, B.rows);
            } else {
                return hom_space_no_opt(C, B, true, C.rows, B.rows);
            }
            
            break;
        }

        case BlockType::INT : {
            //TO-DO: Implement
            if(optimised){
                return hom_space_optimised(C, B, C.rows, B.rows);
            } else {
                return hom_space_no_opt(C, B, true, C.rows, B.rows);
            }
            break;
            #if CHECK_INT
            if( B.type == BlockType::INT ){
                degree_list endpoints_C = C.endpoints();
                degree_list endpoints_B = B.endpoints();
                // Assuming rows are already lexicographically sorted, while columns are not because of the merging.
                std::sort(endpoints_C.begin(), endpoints_C.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                std::sort(endpoints_B.begin(), endpoints_B.end(),  [ ]( const auto& lhs, const auto& rhs )
                    {
                    return lex_order( lhs, rhs);
                    });
                
                vec<vec<r2degree>> intersection;
                index segment_counter;

                //TO-DO: finish

                auto B_it = B.row_degrees.begin();
                for(auto C_it = C.row_degrees.begin(); C_it != C.row_degrees.end();  ){
                    if( lex_order(*C_it, *B_it) ){
                        C_it++;
                    } else if ( (*C_it).second < (*B_it).second ) {
                        B_it++;
                    } else {
                        intersection[segment_counter].push_back(*C_it);
                    }
                }

            } else {
                // Find a fast algorithm to compute Hom_alpha(M, -) for M an interval? Do I need the codomain to be an intervall too if i want it fast?
            }
            #endif

        }

        case BlockType::NON_INT : {
            if(optimised){
                return hom_space_optimised(C, B, C.rows, B.rows);
            } else {
                return hom_space_no_opt(C, B, true, C.rows, B.rows);
            }
            break;
            /*
            Non-optimised version
            Sparse_Matrix S(0,0);
            S.data.reserve( C.rows.size() + B.rows.size() + 1);
            index S_index = 0;
            // First add all row-operations from C to B
            for(index i = 0; i < C.rows.size(); i++){
                for(index j = 0; j < B.rows.size(); j++){
                    auto source_row_index = C.rows[i];
                    auto target_row_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                        row_ops.push_back({source_row_index, target_row_index});
                        S.data.push_back(vec<index>());
                        for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                            auto& internal_column_index = *rit;
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[internal_column_index], target_row_index));
                        }
                        S_index++;
                    }
                }
            }

            index row_op_threshold = S_index;
            assert( row_ops.size() == S_index );

            if(row_op_threshold == 0){
                // If there are no row-operations, then the hom-space is zero.
                return {Sparse_Matrix(0, 0), row_ops};
            }

            

            // Then all column-operations from B to C
            for(index i = 0; i < B.columns.size(); i++){
                for(index j = 0; j < C.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[i], C.columns[j])){
                        S.data.push_back(vec<index>());
                        for(index row_index : B.data[i]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[j], row_index));
                        }
                        S_index++;
                    }
                }
            }
 
            S.compute_get_num_cols()();
            
            // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
            K = S.get_kernel();
            // To see how much the following reduces K: index K_size = K.data.size();
            // Now we need to delete the entries of K which correspond to the row-operations.
            K.cull_columns(row_op_threshold, false);
            
            // Last we need to quotient out those Q where for every i the column Q_i - with its r2degree alpha_i -
            // lies in the image of N, that is, it lies in the image of N|alpha_i.
            // That is equivalent to locally reducing every column of Q.
            
            // Given a row-op, this gives the position in a vector corresponding to it.
            std::unordered_map< std::pair<index, index>, index, pair_hash<index> > pair_to_index = pair_to_index_map(row_ops);

            for(index i = 0; i < C.rows.size(); i++){
                r2degree alpha = C.row_degrees[i];
                vec<index> local_admissible_columns;
                auto [B_alpha, rows_alpha] = B.map_at_degree_pair(alpha);
                if(rows_alpha.empty() || B_alpha.get_num_cols() == 0){
                    continue;
                }
                std::unordered_map<index, index> reIndexMap;

                for(index j : rows_alpha){
                    reIndexMap[B.rows[j]] = pair_to_index[{C.rows[i], B.rows[j]}];
                }
                B_alpha.transform_data(reIndexMap);
                B_alpha.reduce_fully(K);
            }
            
            //  delete possible linear dependencies.
            K.column_reduction_triangular(true);
            */
        }

        default: {
            std::cerr << "Error: Each Block needs a type." << std::endl;
            exit(1);
            return {Sparse_Matrix(0,0), {}};
        }
    }
    return {Sparse_Matrix(0,0), {}};
}



/**
 * @brief This computes all hom-spaces (possibly only the alpha-homs) between active blocks and stores them in hom_spaces.
 *      It also keeps track of which hom-spaces have been computed in domain_keys and codomain_keys.
 * 
 * @param A 
 * @param block_map 
 * @param active_blocks 
 * @param hom_spaces 
 * @param domain_keys 
 * @param codomain_keys 
 * @param alpha 
 * @param compare_hom_space_computation 
 */
inline void compute_hom_to_b (GradedMatrix& A, index& b, vec<Block_list::iterator>& block_map, indtree& active_blocks, 
                                Hom_map& hom_spaces, std::unordered_map<index, vec<index>>& domain_keys, 
                                std::unordered_map<index, vec<index>>& codomain_keys, r2degree& alpha, AIDA_runtime_statistics& statistics, 
                                AIDA_config& config){ //turn_off_hom_optimisation, bool compare_hom_space_computation = false, bool compute_alpha_hom = true
    Block& B = *block_map[b];
    for(index c : active_blocks){
        Block& C = *block_map[c];
        if(b != c){
            // Do not compute again unless needed; This should save a lot of time hopefully.
            // TO-DO: Run some tests, cannot use if hom_alpha is used
            if( (hom_spaces.find({c,b}) == hom_spaces.end() || false) || config.alpha_hom){
                #if TIMERS
                    hom_space_timer.resume();
                    misc_timer.stop();
                #endif
                
                
                hom_spaces.emplace(std::make_pair(c,b), compute_hom_space(A, C, B, alpha, !config.turn_off_hom_optimisation, config.alpha_hom));    
                
                
                #if TIMERS
                    hom_space_timer.stop();
                    misc_timer.resume();
                #endif
                
                
                int dim = hom_spaces[{c,b}].first.get_num_cols();
                statistics.dim_hom_vec.push_back(dim);
                if(dim > statistics.dim_hom_max){
                    statistics.dim_hom_max = dim;
                }

                #if DETAILS
                    std::cout << "      Hom-space " << c << " -> " << b << " computed, dim " << dim << std::endl;
                #endif
                 
                
                // We need to know for which blocks b we have computed a hom-space from c to b.
                // When c merges/extends, then these hom-spaces become void.
                // TO-DO: Instead of recomputing, one can check for each computed morphism if it factors/extend.
                if(domain_keys.find(c) == domain_keys.end()){
                    domain_keys[c] = vec<index>();
                }
                if(codomain_keys.find(b) == codomain_keys.end()){
                    codomain_keys[b] = vec<index>();
                }
                domain_keys[c].push_back(b);	
                codomain_keys[b].push_back(c);

                
                if(config.compare_hom){
                    #if TIMERS
                        hom_space_test_timer.resume();
                        misc_timer.stop();
                    #endif
                    Sparse_Matrix S(0,0);
                    auto non_optimised_hom = compute_hom_space(A, C, B, alpha, false, config.alpha_hom);
                    #if TIMERS
                        hom_space_test_timer.stop();
                        misc_timer.resume();
                    #endif
                    if(hom_spaces[{c,b}].first.get_num_cols() != non_optimised_hom.first.get_num_cols()){
                        std::cout << "Error: Hom-spaces " << c << " -> " << b << " do not match: " << dim << " optimised, vs " << non_optimised_hom.first.get_num_cols() << "brute_force" << std::endl;
                    }
                }
            } else {
                #if DETAILS
                    std::cout << "      Hom-space " << c << " -> " << b << " already computed, dim " << hom_spaces[{c,b}].first.get_num_cols() << std::endl;
                #endif  
            }
        }
    }
}

/**
 * @brief Changes A and N according to the row_operations computed by block_reduce
 * 
 */
inline void update_matrix(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, vec<index>& batch_indices, 
            vec<index>& solution, index& row_op_limit, vec<op_info>& ops, bool& restricted_batch, bool& delete_N){

    for(index operation_index : solution){

        if(operation_index >= row_op_limit){

        } else {
            auto op = ops[operation_index];
            auto& B_source = *block_map[op.second.first];
            auto& B_target = *block_map[op.second.second];
            

            #if OBSERVE
                if( std::find(observe_row_indices.begin(), observe_row_indices.end(), B_source.rows[op.first.first]) != observe_row_indices.end() ){
                    std::cout << "Row operation: " << B_source.rows[op.first.first] << " -> " << B_target.rows[op.first.second] << std::endl;
                }
            #endif
            A.fast_rev_row_op(B_source.rows[op.first.first], B_target.rows[op.first.second]);
            if(restricted_batch && !delete_N){
                auto& N_source = N_map[op.second.first];
                auto& N_target = N_map[op.second.second];
                // TO-DO: Here we only change _rows of N. We should also change the columns/data. 
                // We also dont need to change the part of N which we are looking at, because it can be reduced to zero with the column operations.
                auto& source_row = N_source._rows[op.first.first];
                CT::add_to(source_row, N_target._rows[op.first.second]);     
            }
        }
    }
}

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param extra_columns 
 */                     
inline bool block_reduce(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, bitset& sub_batch_indices, std::shared_ptr<Base_change_virtual>& base_change,   
                vec<index>& row_map, 
                bool compare_both = false, const bitset& extra_columns = bitset(0), bool delete_N = false) {
    vec<long> solution_long; vec<index> solution; index row_op_limit; vec<op_info> ops; bool reduced_to_zero = false;
    vec<long> c;  std::vector<std::reference_wrapper<Sparse_Matrix>> Ns; SparseMatrix<long> S(0,0);

    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }

    #if DETAILS
        std::cout << "  Block_reduce called on (b_vec) ";
        for(index b : b_vec){ std::cout << b << " ";}
            std::cout << "  Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }

    #endif  
    

    #if TIMERS
        misc_timer.stop();      
        constructing_linear_system_timer.resume();
    #endif
    linearise_prior(A, Ns, batch_indices, c, sub_batch_indices);
    construct_linear_system(A, batch_indices, sub_batch_indices, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map, extra_columns);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    
    #if SYSTEM_SIZE
        std::cout << "Solving linear system of size: " << S.get_num_cols() << std::endl;
        // S.print();
        // std::cout << "c: " <<  c << std::endl;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution_long);
    for(long i : solution_long){
        solution.push_back(i);
    }
    #if TIMERS
        solve_linear_system_timer.stop();
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero && !compare_both){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << "x (" << sub_batch_indices << ")" << std::endl;
        #endif
        update_matrix(A, N_map, block_map, batch_indices, solution, row_op_limit, ops, restricted_batch, delete_N);
        if(restricted_batch){
            // TO-DO: This could be done faster.
            for(index b : b_vec){
                N_map[b].compute_columns_from_rows((*block_map[b]).rows);
                for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                    N_map[b].data[i].clear();
                }
                N_map[b].compute_rows_forward_map(row_map);
            }
        }
    }
    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param extra_columns 
 */                     
inline  bool block_reduce_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, std::shared_ptr<Base_change_virtual>& base_change,   
                vec<index>& row_map, 
                bool compare_both = false, bool delete_N = false) {
    vec<long> solution_long; vec<index> solution; index row_op_limit; vec<op_info> ops; bool reduced_to_zero = false;
    vec<long> c;  std::vector<std::reference_wrapper<Sparse_Matrix>> Ns; SparseMatrix<long> S(0,0);

    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }

    #if DETAILS
        std::cout << "  Block_reduce called on (b_vec) ";
        for(index b : b_vec){ std::cout << b << " ";}
            std::cout << "  Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
    #endif  
    

    #if TIMERS
        misc_timer.stop();      
        constructing_linear_system_timer.resume();
    #endif
    linearise_prior_full_support(A, Ns, batch_indices, c);
    construct_linear_system_full_support(A, batch_indices, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    
    #if SYSTEM_SIZE
        std::cout << "Solving linear system of size: " << S.get_num_cols() << std::endl;
        // S.print();
        // std::cout << "c: " <<  c << std::endl;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution_long);
    for(long i : solution_long){
        solution.push_back(i);
    }

    #if TIMERS
        solve_linear_system_timer.stop();
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero && !compare_both){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << "x (" << batch_indices << ")" << std::endl;
        #endif
        update_matrix(A, N_map, block_map, batch_indices, solution, row_op_limit, ops, restricted_batch, delete_N);
    }
    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_full_support


/**
 * @brief Updates A and N according to the hom-operations computed by block_reduce_hom
 * 
 * @param A 
 * @param N_map 
 * @param block_map 
 * @param solution 
 * @param row_op_limit 
 * @param ops 
 * @param restricted_batch 
 * @param naive_first 
 */
inline void update_matrix_hom(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, 
            vec<index>& batch_indices, Hom_map& hom_spaces, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map,
            vec<index>& solution, index& row_op_limit, vec<hom_info>& ops,
            bool restricted_batch = false, bool delete_N = false){
    

    for(index operation_index : solution){

        if(operation_index >= row_op_limit){
           
        } else {
            auto op = ops[operation_index];
            auto& B_source = *block_map[op.second.first];
            auto& B_target = *block_map[op.second.second];
            
                
            auto& C = B_source;
            auto& B = B_target;
            Hom_space& hom_cb = hom_spaces[{op.second.first, op.second.second}];

            hom_action_A(A, B_source.rows, B_target.rows, hom_cb.first.data[op.first], hom_cb.second, base_change);

            if(restricted_batch && !delete_N){
                auto& N_source = N_map[op.second.first];
                auto& N_target = N_map[op.second.second];
                hom_action_N(B, N_source, N_target, hom_cb.first.data[op.first], hom_cb.second);
            } 
        }
    }
}

inline void update_matrix_extension(GradedMatrix& A, Sub_batch& N_map, vec<Block_iterator>& block_map, 
    Hom_map& hom_spaces, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& non_processed_blocks, 
    vec<index>& row_map, vec<index>& solution, index& E_threshold, index& N_threshold, index& M_threshold, vec<hom_info>& hom_storage,
    Transform_Map& batch_transforms, vec<Merge_data>& pro_blocks, Merge_data& pro_block){

    for(index operation_index : solution){

        if(operation_index < E_threshold){
            auto op = hom_storage[operation_index];
            auto& C = *block_map[op.second.first];
            auto& B = *block_map[op.second.second];
           
            Hom_space& hom_cb = hom_spaces[{op.second.first, op.second.second}];
            hom_action_A(A, C.rows, B.rows, hom_cb.first.data[op.first], hom_cb.second, base_change);
        } else if (operation_index < N_threshold){
            auto op = hom_storage[operation_index];
            Batch_transform& transform_space = batch_transforms[std::make_pair(pro_blocks[op.second.first], pro_block)][op.first];
            auto& hom_infos = transform_space.second;
            auto& T = transform_space.first;
            for(index c : non_processed_blocks){
                N_map[c].multiply_id_triangular(T);
            }
            for(auto hom_info : hom_infos){
                auto& C = *block_map[std::get<0>(hom_info)];
                auto& B = *block_map[std::get<1>(hom_info)];
                Hom_space& hom_cb = hom_spaces[{std::get<0>(hom_info), std::get<1>(hom_info)}];
                hom_action_A(A, C.rows, B.rows, hom_cb.first.data[std::get<2>(hom_info)], hom_cb.second, base_change);
            }
        } else if (operation_index < M_threshold){
            auto& op = hom_storage[operation_index];
            index& source = op.first;
            index& target = op.second.first;
            for(index c : non_processed_blocks){
                N_map[c].col_op(source, target);
            }
        }
    }
}

/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param morphisms The morphisms between blocks.
 * @param extra_columns if naive_first is true, this is the set of columns-indices of the batch which belong to the second subspace tested in naive decomposition.
 *                      If naive_first is false, this is the set of columns-indices of the batch which belong to the first subspace tested in naive decomposition.
 *                      It should be empty if block reduce is not called from naive decomposition.
 */                     
inline bool block_reduce_hom(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map, bitset& sub_batch_indices,
                std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map, Hom_map& hom_spaces,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){ 
    vec<index> solution; index row_op_limit; vec<hom_info> ops; bool reduced_to_zero = false;
    vec<index> c;  Sparse_Matrix S(0,0); 
    std::vector<std::reference_wrapper<Sparse_Matrix>> Ns;
    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }
    #if DETAILS
        std::cout << "  block_reduce_hom called on blocks ";
        for(index b : b_vec){ std::cout << b << " ";}
        std::cout << " - Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
        // std::cout << "      batch_indices: " << batch_indices << std::endl;
        if(restricted_batch){
            // std::cout << "      sub_batch_indices: " << sub_batch_indices << std::endl;
        }
    #endif

    #if TIMERS 
        misc_timer.stop();     
        constructing_linear_system_timer.resume();
    #endif
    construct_linear_system_hom(batch_indices, sub_batch_indices, restricted_batch, 
            relevant_blocks, block_map, S, ops, b_vec, N_map, hom_spaces, row_map, c, extra_columns);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
        // S.print(true, true);
        // std::cout << "  c: " <<  c;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution);

    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << " and " << sub_batch_indices << std::endl;
        #endif
        update_matrix_hom(A, N_map, block_map, batch_indices, hom_spaces, base_change, row_map, solution, row_op_limit, ops, restricted_batch, delete_N);
        for(index b : b_vec){
            if(restricted_batch && !delete_N){
                N_map[b].compute_columns_from_rows((*block_map[b]).rows);
            }
            for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
            }
            N_map[b].compute_rows_forward_map(row_map);
        }
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_hom


/**
 * @brief Tries to delete the columns of N_B given by sub_batch_indices
 *  with all admissible operations without changing A up to the current batch.
 * 
 * @param A The Graded Matrix where the block lives.
 * @param b_vec The indices of the blocks we want to delete.
 * @param N_map The map to all sub-batches.
 * @param batch_indices The indices of the current batch.
 * @param restricted_batch If the batch is restricted to a subset of the batch_indices.
 * @param relevant_blocks The blocks for which N_map contains relevant information.
 * @param block_map The map from indices (relevant blocks) to block iterators.
 * @param sub_batch_indices The indices of the columns of N_B which are to be deleted.
 * @param morphisms The morphisms between blocks.
 * @param extra_columns if naive_first is true, this is the set of columns-indices of the batch which belong to the second subspace tested in naive decomposition.
 *                      If naive_first is false, this is the set of columns-indices of the batch which belong to the first subspace tested in naive decomposition.
 *                      It should be empty if block reduce is not called from naive decomposition.
 */                     
inline bool block_reduce_hom_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& relevant_blocks, vec<Block_iterator>& block_map,
                std::shared_ptr<Base_change_virtual>& base_change, vec<index>& row_map, Hom_map& hom_spaces,  
                bool delete_N = false){ 
    vec<index> solution; index row_op_limit; vec<hom_info> ops; bool reduced_to_zero = false;
    vec<index> c;  Sparse_Matrix S(0,0); 
    std::vector<std::reference_wrapper<Sparse_Matrix>> Ns;
    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }
    #if DETAILS
        std::cout << "  block_reduce_hom called on blocks ";
        for(index b : b_vec){ std::cout << b << " ";}
        std::cout << " - Ns:" << std::endl;
        for(auto& ref : Ns){
            ref.get().print_rows();
        }
        // std::cout << "      batch_indices: " << batch_indices << std::endl;
        if(restricted_batch){
            // std::cout << "      sub_batch_indices: " << sub_batch_indices << std::endl;
        }
    #endif

    #if TIMERS 
        misc_timer.stop();     
        constructing_linear_system_timer.resume();
    #endif
    construct_linear_system_hom_full_support(batch_indices, restricted_batch, 
            relevant_blocks, block_map, S, ops, b_vec, N_map, hom_spaces, row_map, c);
    row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
        // S.print(true, true);
        // std::cout << "  c: " <<  c;
    #endif
    S.compute_num_cols();
    reduced_to_zero = S.solve_col_reduction(c, solution);

    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b_vec << " and " << batch_indices << std::endl;
        #endif
        update_matrix_hom(A, N_map, block_map, batch_indices, hom_spaces, base_change, row_map, solution, row_op_limit, ops, restricted_batch, delete_N);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    return reduced_to_zero;
} //Block_reduce_hom_full_support

/**
 * @brief This function is used to test both block_reduce and block_reduce_hom and compare the results.
 * 
 * @param A 
 * @param b_vec 
 * @param N_map 
 * @param batch_indices 
 * @param restricted_batch 
 * @param blocks 
 * @param block_map 
 * @param support 
 * @param base_change 
 * @param row_map 
 * @param hom_spaces 
 * @param brute_force 
 * @param compare_both 
 * @param extra_columns 
 * @param delete_N If true, then we will not perform row operations on N, 
 *                 but only delete the part of N belonging to b_vec and sub_batch_indices.
 * @return true 
 * @return false 
 */
inline bool use_either_block_reduce(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& blocks, vec<Block_iterator>& block_map, bitset& support, std::shared_ptr<Base_change_virtual>& base_change, 
                vec<index>& row_map, Hom_map& hom_spaces, bool brute_force = false, bool compare_both = false,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){

    bool block_reduce_result = false;
    bool block_reduce_result_hom = false;

    if (brute_force || compare_both) {
        block_reduce_result = block_reduce(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, support, base_change, row_map, compare_both, extra_columns, delete_N);
    }

    if (!brute_force || compare_both){
        block_reduce_result_hom = block_reduce_hom(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, support, base_change, row_map, hom_spaces, extra_columns, delete_N);
    }
    
    if(compare_both){
        if(block_reduce_result != block_reduce_result_hom){
            std::cout << "Error: Block reduce and block reduce hom do not agree." << std::endl;
            std::cout << "Blocks to delete: " << b_vec << std::endl;
            std::cout << "All blocks: " << blocks << std::endl;
            std::cout << "Block reduce result: " << block_reduce_result << " Block reduce hom result: " << block_reduce_result_hom << std::endl;
        }
        assert(block_reduce_result == block_reduce_result_hom);
    }
    return block_reduce_result || block_reduce_result_hom;
}

/**
 * @brief This function is used to test both block_reduce and block_reduce_hom and compare the results.
 * 
 * @param A 
 * @param b_vec 
 * @param N_map 
 * @param batch_indices 
 * @param restricted_batch 
 * @param blocks 
 * @param block_map 
 * @param support 
 * @param base_change 
 * @param row_map 
 * @param hom_spaces 
 * @param brute_force 
 * @param compare_both 
 * @param extra_columns 
 * @param delete_N If true, then we will not perform row operations on N, 
 *                 but only delete the part of N belonging to b_vec and sub_batch_indices.
 * @return true 
 * @return false 
 */
inline bool use_either_block_reduce_full_support(GradedMatrix& A, vec<index>& b_vec, Sub_batch& N_map, vec<index>& batch_indices,
                bool restricted_batch, vec<index>& blocks, vec<Block_iterator>& block_map, std::shared_ptr<Base_change_virtual>& base_change, 
                vec<index>& row_map, Hom_map& hom_spaces, bool brute_force = false, bool compare_both = false,  
                const bitset& extra_columns = bitset(0), bool delete_N = false){

    bool block_reduce_result = false;
    bool block_reduce_result_hom = false;

    if (brute_force || compare_both) {
        block_reduce_result = block_reduce_full_support(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, base_change, row_map, compare_both, delete_N);
    }

    if (!brute_force || compare_both){
        block_reduce_result_hom = block_reduce_hom_full_support(A, b_vec, N_map, batch_indices, restricted_batch, 
        blocks, block_map, base_change, row_map, hom_spaces, delete_N);
    }
    
    if(compare_both){
        if(block_reduce_result != block_reduce_result_hom){
            std::cout << "Error: Block reduce and block reduce hom do not agree." << std::endl;
            std::cout << "Blocks to delete: " << b_vec << std::endl;
            std::cout << "All blocks: " << blocks << std::endl;
            std::cout << "Block reduce result: " << block_reduce_result << " Block reduce hom result: " << block_reduce_result_hom << std::endl;
        }
        assert(block_reduce_result == block_reduce_result_hom);
    }
    return block_reduce_result || block_reduce_result_hom;
}

/**
 * @brief Considers two virtual blocks (blocks + columns in batch) and computes all batch_internal 
 *          column operations from source to target which are part of a homomorphism (which has to go in the opposite direction).
 * 
 * @param virtual_source_block 
 * @param virtual_target_block 
 * @param N_map 
 * @param hom_spaces 
 * @param row_map 
 * @return array<index> 
 */
inline vec< Batch_transform > compute_internal_col_ops(Merge_data& virtual_source_block, Merge_data& virtual_target_block, 
    Sub_batch& N_map, Hom_map& hom_spaces, vec<index>& row_map, index& k, vec<Block_iterator>& block_map){

    bitset& source_batch_indices = virtual_source_block.second;
    vec<index>& source_block_indices = virtual_source_block.first;
    bitset& target_batch_indices = virtual_target_block.second;
    vec<index>& target_block_indices = virtual_target_block.first;
    // Let N_source and N_target be the sub-batches of N corresponding to the virtual blocks.
    // Then a set of batch-internal column operations from source to target is a matrix P such that 
    // N_source * P = Q * N_target for some Q in Hom(virtual_target_block, virtual_source_block).

    // First add the column operations to the linear system.
    
    index current_row = 0;
    vec<index> source_block_row_map; // Maps the rows of the blocks in the virtual source block to the rows of the linear system.
    for(index i : source_block_indices){
        source_block_row_map.push_back( current_row );
        current_row += N_map[i].get_num_rows();
    }
    
    // Need a linearisation scheme for the entries which can be touched by Q N_target.
    auto position = [source_block_row_map, target_batch_indices, current_row, k](index& column_index, index& row_index, index& target_block_number) -> index {
        return linearise_position_reverse_ext<index>(column_index, source_block_row_map[target_block_number] + row_index, k, current_row);
    };

    Sparse_Matrix S(0, target_batch_indices.count()*current_row);
    index col_op_threshold = 0;
    index hom_threshold = 0;
    vec<pair> col_op_keys;
    vec< std::tuple<index, index, index> > hom_keys;

    // First the homomorphisms aka row-ops from blocks in the target to blocks in the source.

    for(index c : target_block_indices){
        for(index i = 0; i< source_block_indices.size(); i++){
            index& b = source_block_indices[i];
            Hom_space& hom = hom_spaces[{c, b}];
            Sparse_Matrix& hom_matrices = hom.first;
            vec<pair>& row_op_keys = hom.second;
            for(index hom_counter = 0; hom_counter < hom_matrices.data.size(); hom_counter++){
                vec<index>& Q = hom_matrices.data[hom_counter];
                S.data.emplace_back(hom_action_full_support(source_block_row_map[i], current_row, Q, row_op_keys, N_map[c]));
                hom_keys.push_back({c, b, hom_counter});
            }
        }
    }

    hom_threshold = S.data.size();

    // Then add the sets of internal column operations from source to target.
    for(index i = source_batch_indices.find_first(); i != bitset::npos; i = source_batch_indices.find_next(i)){
        for(index j = target_batch_indices.find_first(); j != bitset::npos; j = target_batch_indices.find_next(j)){
            col_op_keys.push_back({i,j});
            S.data.push_back(vec<index>());
            for(index block_counter = 0; block_counter < source_block_indices.size(); block_counter++){
                for(index row_index : N_map[source_block_indices[block_counter]].data[i]){
                    S.data.back().push_back(position(j, row_map[row_index], block_counter));
                }
            }
        }
    }
    col_op_threshold = S.data.size();

    // At last, local column operations from the source.

    for(index i = 0; i< source_block_indices.size(); i++){
        Block& B = *block_map[source_block_indices[i]];
        for(vec<index> column : B.local_data -> data){
            for(index j = target_batch_indices.find_first(); j != bitset::npos; j = target_batch_indices.find_next(j)){
                S.data.push_back(vec<index>());
                for(index row_index : column){
                    S.data.back().push_back(position(j, row_map[row_index], i));
                }
            }
        }
    }

    // Reduction to independent column operations:
    S.compute_num_cols();
    auto K = S.kernel();
    K.cull_columns(col_op_threshold, false);
    K.column_reduction_triangular(hom_threshold, true);

    // Build corresponding matrices.

    vec< std::pair<DenseMatrix, vec<std::tuple<index, index, index>> > > result;
    for(vec<index> column : K.data){
        result.push_back({DenseMatrix(k, k), {}});
        for(index i : column){
            if(i < hom_threshold){
                result.back().second.push_back(hom_keys[i]);
            } else {
                index j = i - hom_threshold;
                auto& [source_col, target_col] = col_op_keys[j];
                result.back().first.data[target_col].set(source_col);
            }
        }
    }

    return result;  
}

/**
 * @brief This computes a column-form of the entries in the batch, splits it up into the blocks which are touched by the batch, and stores the information.
 * 
 * @param active_blocks Stores the touched blocks.
 * @param block_map 
 * @param A the matrix
 * @param batch 
 */
inline void compute_touched_blocks(indtree& active_blocks, vec<Block_iterator>& block_map, 
                            GradedMatrix& A, vec<index>& batch, Sub_batch& N_map) {

    for(index j = 0; j < batch.size(); j++){
        index bj = batch[j];
        // Q: Is this the fastest way to do it? It is possible we want a specific sorting function.
        A.sort_column(bj);
        convert_mod_2(A.data[bj]); 
        #if DETAILS
            std::cout << "  Batch-col " << j << " after sorting: " << A.data[bj] << std::endl;
        #endif
        for(index i : A.data[bj]){
            Block& B = *block_map[i];
            index initial = B.rows.front();
	        auto new_touched = active_blocks.insert(initial); 
            if(new_touched.second){
                N_map.emplace(initial, Sparse_Matrix(batch.size(), B.rows.size()));
                N_map[initial].data = vec<vec<index>>(batch.size(), vec<index>());
            }
            N_map[initial].data[j].push_back(i);
            // Maybe we should also store the rows, but not sure right now.
            assert( A._rows[i].back() == bj);
            A._rows[i].pop_back();
        }
    }
} // compute_touched_blocks

/**
 * @brief Changes N to achieve a partial decomposition of (A_B | N) by recursively finding a decomposition into two components.
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 * @param pierced blocks
 * @param batch_indices
 * @param N_column_indices 
 * @param e_vec 
 * @param N_map 
 * @param vector_space_decompositions 
 */
inline vec< Merge_data > exhaustive_alpha_decomp(
                        GradedMatrix& A, Block_list& B_list, vec<Block_iterator>& block_map, 
                        vec<index>& pierced_blocks, vec<index>& batch_indices, const bitset& N_column_indices, 
                        vec<bitset>& e_vec, Sub_batch& N_map, vec<vec<transition>>& vector_space_decompositions,
                        AIDA_config& config, std::shared_ptr<Base_change_virtual>& base_change,
                        Hom_map& hom_spaces, vec<index> row_map, bool brute_force = false, bool compare_both = false) {
                           
    //TO-DO: Implement a branch and bound strategy such that we do not need to iterate over those decompositions
    //TO-DO: Need to first decompose/reduce the left-hand columns, create an updated temporary block-merge and then decompose the right-hand columns.
    int k = N_column_indices.count();    
    int num_b = pierced_blocks.size();
    assert(batch_indices.size() == N_column_indices.size());
    #if DETAILS
        std::cout << "  Calling exhaustive_alpha_decomp with " << k << " columns and " << 
        num_b << " blocks at N_column_indices: " << N_column_indices << std::endl;
    #endif
    if( k == 1 ){
        vec< Merge_data > result;
        result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
        return result;
    }
    if( k > vector_space_decompositions.size() + 1 ){
        config.decomp_failure.push_back(batch_indices);
        if(true){
            std::cout << "  No vector space decompositions for the local (!) dimension " << k << 
            " provided. \n Warning: Decomposition is almost surely only partial from here on." << std::endl;
        }
        // TO-DO: We could call methods from generate decompositions here to try some column-operations 
        // to find a decomposition, then break if we dont find one after a certain time.
        vec< Merge_data > result;
        result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
        return result;
    }

    // Iterate over all decompositions of GF(2)^k into two subspaces.
    for(auto transition : vector_space_decompositions[k-2]){
        auto& basechange = transition.first;
        auto& partition_indices = transition.second;
        bitset indices_1 = glue(N_column_indices, partition_indices);
        bitset indices_2 = glue(N_column_indices, partition_indices.flip());
        #if DETAILS
            std::cout << "  Indices 1: " << indices_1 << " Indices 2: " << indices_2 << std::endl;
        #endif
        vec<index> blocks_1;
        vec<index> blocks_2;
        indtree blocks_conflict;
        for(index b : pierced_blocks){
            N_map[b].multiply_dense_with_e_check(basechange, e_vec, N_column_indices);
            Block& B = *block_map[b];
            N_map[b].compute_rows_forward_map(row_map);
        }

        for(index b : pierced_blocks){
            if(N_map[b].is_zero(indices_1)){
                blocks_2.push_back(b);
            } else if (N_map[b].is_zero(indices_2)){
                blocks_1.push_back(b);
            } else {
                blocks_conflict.insert(b);
                blocks_1.push_back(b);
                blocks_2.push_back(b);
            }
        }

        #if DETAILS
            for(index b : pierced_blocks){
                std::cout << "  Block " << b << ": ";
                N_map[b].print_rows();
            }
            std::cout << "B_1: " << blocks_1 << " B_2: " << blocks_2 << " B_conflict: " << blocks_conflict << std::endl;
        #endif
        bool conflict_resolution = true;

        // Optimisation step: If we could not delete the lhs of a block in conflict, then try to delete the rhs without all operations. 
        // Only stop doing this if this does not work for one block because in this case we will have to try to delete all of these blcoks at once anyways.
        #if DETAILS
            vec<pair> deletion_tracker;
        #endif
        if (blocks_conflict.size() > 0){

            for(auto itr = blocks_conflict.rbegin(); itr != blocks_conflict.rend();){
                index b = *itr;
                vec<index> b_vec = {b};

                #if TIMERS
                    alpha_decomp_timer.stop();
                    misc_timer.resume();  
                #endif

                bool deleted_N1 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                    blocks_1, block_map, indices_1, base_change, row_map, hom_spaces, brute_force, compare_both);

                #if TIMERS
                    dispose_S_timer.stop();
                    alpha_decomp_timer.resume();
                #endif
                if(deleted_N1){
                    assert(N_map[b].is_zero(indices_1));
                    erase_from_sorted_vector(blocks_1, b);
                    #if DETAILS
                        deletion_tracker.push_back({b, 1});
                    #endif
                    auto it = blocks_conflict.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it); 
                } else if ( conflict_resolution ) {
                    
                    #if TIMERS
                        alpha_decomp_timer.stop();
                        misc_timer.resume();
                    #endif
                    bool deleted_N2 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                        blocks_2, block_map, indices_2, base_change, row_map, hom_spaces, brute_force, compare_both);

                    #if TIMERS
                        dispose_S_timer.stop();
                        alpha_decomp_timer.resume();
                    #endif
                    if(deleted_N2){
                        assert(N_map[b].is_zero(indices_2));
                        erase_from_sorted_vector(blocks_2, b);
                        #if DETAILS
                            deletion_tracker.push_back({b, 2});
                        #endif
                        auto it = blocks_conflict.erase(--itr.base());
                        itr = std::reverse_iterator<decltype(it)>(it);
                    } else {
                        conflict_resolution = false;
                        itr++;
                    }
                } else {
                    itr++;
                
                }
            }
        }
        #if DETAILS
            for(auto [b, side] : deletion_tracker){
                    std::cout << "    Deleted " << b << " from side " << side << std::endl;
                }
        #endif
        if(conflict_resolution){

            #if DETAILS
                std::cout << "  Conflict resolved directly." << std::endl;
            #endif
        } else {
            #if DETAILS
                std::cout << "  Conflict could not be resolved. First reducing N_1 as much as possible." << std::endl;
            #endif
            
            indtree blocks_excl_1;
            std::set_difference(blocks_1.begin(), blocks_1.end(), blocks_conflict.begin(), blocks_conflict.end(), std::inserter(blocks_excl_1, blocks_excl_1.begin()));
            // Only need to further reduce those blocks which are not in conflict, because they would have already been deleted in the first step.
            for(auto itr = blocks_excl_1.rbegin(); itr != blocks_excl_1.rend();){
                index b = *itr;
                vec<index> b_vec = {b};

                #if TIMERS
                    alpha_decomp_timer.stop();
                    misc_timer.resume();
                #endif
                bool deleted_more_1 = use_either_block_reduce(A, b_vec, N_map, batch_indices, true, 
                    blocks_1, block_map, indices_1, base_change, row_map, hom_spaces, brute_force, compare_both);
                #if TIMERS
                    dispose_S_timer.stop();
                    alpha_decomp_timer.resume();
                #endif
                if(deleted_more_1 ){
                    assert(N_map[b].is_zero(indices_1));
                    if(!N_map[b].is_zero(indices_2)){
                        insert_into_sorted_vector(blocks_2, b);
                    }
                    erase_from_sorted_vector(blocks_1, b);
                    auto it = blocks_excl_1.erase(--itr.base());
                    itr = std::reverse_iterator<decltype(it)>(it);  
                    #if DETAILS
                        std::cout << "    In Step 2, deleted " << b << " from side 1." << std::endl;
                        if(std::find(blocks_2.begin(), blocks_2.end(), b) != blocks_2.end()){
                            std::cout << "    ..but added " << b << " to side 2." << std::endl;
                        }
                    #endif
                } else {
                    itr++;
                }
            }
            assert( blocks_conflict.size() > 0);
            // Now need to treat all of blocks_1 as one block and delete its rhs of N.
            #if DETAILS
                std::cout << "B_1: " << blocks_1 << " B_2: " << blocks_2 << " B_conflict: " << blocks_conflict << std::endl;
            #endif

            #if TIMERS
                alpha_decomp_timer.stop();
                misc_timer.resume();
            #endif
            conflict_resolution = use_either_block_reduce(A, blocks_1, N_map, batch_indices, true, 
                pierced_blocks, block_map, indices_2, base_change, row_map, hom_spaces, brute_force, compare_both, indices_1, true);
            #if TIMERS
                dispose_S_timer.stop();
                alpha_decomp_timer.resume();
            #endif
            if(conflict_resolution){
                for(index b : blocks_1){
                    erase_from_sorted_vector(blocks_2, b);
                    blocks_conflict.erase(b);
                    assert(N_map[b].is_zero(indices_2));
                }
            }
        }
        
        if (conflict_resolution){ 
            assert(blocks_conflict.size() == 0);
            // A valid decomposition has been found. Continue here.
            auto partition_1 = exhaustive_alpha_decomp(A, B_list, block_map, blocks_1, batch_indices, indices_1, 
                e_vec, N_map, vector_space_decompositions, config, base_change,
                hom_spaces, row_map, brute_force, compare_both);
            auto partition_2 = exhaustive_alpha_decomp(A, B_list, block_map, blocks_2, batch_indices, indices_2, 
                e_vec, N_map, vector_space_decompositions, config, base_change,
                hom_spaces, row_map, brute_force, compare_both);
            partition_1.insert(partition_1.end(), partition_2.begin(), partition_2.end());
            return partition_1;
        }
    }
    vec< Merge_data > result;
    result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
    return result;
} // exhaustive_alpha_decomp


/**
 * @brief Tries to delete a cocycle defining an extension from pro_block to b
 * 
 * @param A 
 * @param b 
 * @param b_non_zero_columns 
 * @param non_processed_blocks 
 * @param pro_block 
 * @param incoming_vertices 
 * @param pro_blocks 
 * @param deleted_cocycles_b 
 * @param hom_graph 
 * @param hom_spaces 
 * @param batch_transforms 
 * @param base_change 
 * @param block_map 
 * @param row_map 
 * @param N_map 
 * @return true 
 * @return false 
 */
inline bool alpha_extension_decomposition(GradedMatrix& A, index& b, bitset& b_non_zero_columns, vec<index>& non_processed_blocks,
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, std::shared_ptr<Base_change_virtual>& base_change, vec<index>& external_incoming_vertices, Row_transform_map& component_transforms, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){
    
    Block& B = *block_map[b];
    bitset& target_batch_indices = pro_block.second;

    // If N_b is already zero, then we do not need to do anything.
    if(N_map[b].is_zero(target_batch_indices)){
        return true;        
    }

    #if DETAILS
        std::cout << "  alpha_extension_decomposition called on blocks ";
        std::cout << b << " x " << pro_block.first << " / " << target_batch_indices << std::endl;
    #endif

    
    #if TIMERS
        misc_timer.stop();    
        constructing_linear_system_timer.resume();
    #endif
    vec<index> y;
    linearise_sub_batch_entries(y, N_map[b], target_batch_indices, row_map);
    vec<index> solution;    
    Sparse_Matrix S(0,0); vec<hom_info> hom_storage; index E_threshold;  index N_threshold; index M_threshold;
    construct_linear_system_extension(S, hom_storage, E_threshold, N_threshold, M_threshold,
    b, b_non_zero_columns, 
    pro_block, incoming_vertices, pro_blocks, deleted_cocycles_b,
    hom_graph, hom_spaces, batch_transforms, 
    block_map, row_map, N_map);


    
    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
    #endif
    S.compute_num_cols();
    bool reduced_to_zero = S.solve_col_reduction(y, solution);
    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b << " and " << target_batch_indices << std::endl;
        #endif
        update_matrix_extension(A, N_map, block_map, hom_spaces, base_change, non_processed_blocks, row_map, 
        solution, E_threshold, N_threshold, M_threshold, hom_storage, batch_transforms, pro_blocks, pro_block);
        for(index i = target_batch_indices.find_first(); i != bitset::npos; i = target_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
        }
        N_map[b].compute_rows_forward_map(row_map);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif
    #if DETAILS
        if(reduced_to_zero){
            std::cout << "Deleted Cocycle" << std::endl;
        } else {
            std::cout << "Did not delete Cocycle" << std::endl;
        }
    #endif
    return reduced_to_zero;

}

/**
 * @brief tries to solve the linear system needed to delete a cocycle and returns a solution if successful.
 * 

 */
inline bool virtual_alpha_extension_decomposition(GradedMatrix& A, index& b, bitset& b_non_zero_columns, vec<index>& non_processed_blocks,
    Merge_data& pro_block, vec<index>& incoming_vertices, vec<Merge_data>& pro_blocks, bitset& deleted_cocycles_b,
    Graph& hom_graph, Hom_map& hom_spaces, Transform_Map& batch_transforms, std::shared_ptr<Base_change_virtual>& base_change, 
    vec<Block_iterator>& block_map, vec<index>& row_map, Sub_batch& N_map){
    
    Block& B = *block_map[b];
    bitset& target_batch_indices = pro_block.second;
    //In general version the following should be the effect of a hom on a block.
    Sparse_Matrix& Cocycle_container = N_map[b]; 

    // If N_b is already zero, then we do not need to do anything. 
    // Should be checked before calling, so not necessary to check again.
    // if(N_map[b].is_zero(target_batch_indices)){
    //     return true;        
    // }

    //TO-DO: Continue here.
    #if DETAILS
        std::cout << "  alpha_extension_decomposition called on blocks ";
        std::cout << b << " x " << pro_block.first << " / " << target_batch_indices << std::endl;
    #endif

    #if TIMERS
        misc_timer.stop();    
        constructing_linear_system_timer.resume();
    #endif
    vec<index> y;
    linearise_sub_batch_entries(y, Cocycle_container, target_batch_indices, row_map);
    vec<index> solution;    
    Sparse_Matrix S(0,0); vec<hom_info> hom_storage; index E_threshold;  index N_threshold; index M_threshold;
    construct_linear_system_extension(S, hom_storage, E_threshold, N_threshold, M_threshold,
    b, b_non_zero_columns, 
    pro_block, incoming_vertices, pro_blocks, deleted_cocycles_b,
    hom_graph, hom_spaces, batch_transforms, 
    block_map, row_map, N_map);
    
    #if TIMERS
        constructing_linear_system_timer.stop();
        solve_linear_system_timer.resume();
    #endif
    #if SYSTEM_SIZE
        std::cout << "  Solving linear system of size " << S.get_num_cols() << "." << std::endl;
    #endif
    S.compute_num_cols();
    bool reduced_to_zero = S.solve_col_reduction(y, solution);
    #if TIMERS
        solve_linear_system_timer.stop();  
        update_matrix_timer.resume();
    #endif

    if(reduced_to_zero){
        #if DETAILS
            std::cout << "      Deleted N at: " << b << " and " << target_batch_indices << std::endl;
        #endif
        update_matrix_extension(A, N_map, block_map, hom_spaces, base_change, non_processed_blocks, row_map, 
        solution, E_threshold, N_threshold, M_threshold, hom_storage, batch_transforms, pro_blocks, pro_block);
        for(index i = target_batch_indices.find_first(); i != bitset::npos; i = target_batch_indices.find_next(i)){
                N_map[b].data[i].clear();
        }
        N_map[b].compute_rows_forward_map(row_map);
    }

    #if TIMERS
        update_matrix_timer.stop();
        dispose_S_timer.resume();
    #endif

    return reduced_to_zero;
}

inline Row_transform compute_internal_row_ops(bitset& source_deleted_cocycles, bitset& target_deleted_cocycles){
    // TO-DO: Implement this.
    // Question is: can the collection of cocycles in source_deleted_cocycles be deleted when in target_deleted_cocycles?
    // Right now I will only program this for the case where we deal with components which are cyclic modules.
    
    // We should be able to iterate over all non-zero entries in source_deleted_cocycles, 
    // pretend that their entries lie in the target row and 
    // check if we can delete each seperately via alpha-extension decomposition.

    for(index i = source_deleted_cocycles.find_first(); i != bitset::npos; i = source_deleted_cocycles.find_next(i)){
        // bool deleteion = virtual_alpha_extension_decomposition(
                            
    }

    return Row_transform();
}

/**
 * @brief 
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 * @param pierced_blocks 
 * @param batch_indices 
 * @param N_column_indices 
 * @param e_vec 
 * @param N_map 
 * @param config
 * @param base_change 
 * @param hom_spaces 
 * @param row_map 
 * @param computation_order 
 * @param scc 
 * @param condensation 
 * @param is_resolvable_cycle
 * @return vec< Merge_data > 
 */
inline vec< Merge_data > automorphism_sensitive_alpha_decomp( GradedMatrix& A, Block_list& B_list, vec<Block_iterator>& block_map, 
    vec<index>& local_pierced_blocks, vec<index>& batch_indices, const bitset& N_column_indices, 
    vec<bitset>& e_vec, Sub_batch& N_map, vec<vec<transition>>& vector_space_decompositions,
    AIDA_config& config, std::shared_ptr<Base_change_virtual>& base_change,
    Hom_map& hom_spaces, vec<index>& row_map, Graph& hom_graph,
    vec<index>& computation_order, vec<vec<index>>& scc, Graph& condensation, vec<bool>& is_resolvable_cycle){

    index k = batch_indices.size();

    Transform_Map batch_transforms; // storage for transforms; maps pairs of virtual (processed) blocks to the admissible batch-transforms.
    vec<Merge_data> pro_blocks; // Processed blocks may have already merged, though not formally so.
    Graph pro_graph; // graph on processed blocks given by existence of associated column-operations.
    // There should be an optimisation which mostly avoids computing this ->
    array<index> pro_scc;
    Graph pro_condensation; 
    vec<index> pro_computation_order;
    bitset non_pro = N_column_indices;
    vec<index> non_processed_blocks = local_pierced_blocks;
    

    #if DETAILS
        std::cout << "      Automorphism Sensitive Alpha Decomposition" << std::endl;
        std::cout << "Blocks: " << local_pierced_blocks << ", col-support: " << N_column_indices << " with hom-graph: " << std::endl;
        print_graph_with_labels(hom_graph, local_pierced_blocks);
        vec<indtree> scc_labels;
        // Create the labels for the SCCs for better readability.
        for( vec<index>& s : scc){
            scc_labels.push_back(indtree());
            for(index number : s){
                scc_labels.back().insert(local_pierced_blocks[number]);
            }
        }
        std::cout << " and condensation: " << std::endl;
        print_graph_with_labels<indtree>(condensation, scc_labels);
        vec<vec<bool>> N_map_indicator = vec<vec<bool>>(local_pierced_blocks.size(), vec<bool>(k, 0));
        for(index i = 0; i < local_pierced_blocks.size(); i++){
            for(index j = 0; j < k; j++){
                N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
            }
        }
    #endif

    for(auto it = computation_order.rbegin(); it != computation_order.rend(); it++){
        
        vec<bitset> component_non_zero_columns; // Stores the column supports for N^{zero} for each block in the component.
        index component_index = *it;
        vec<index>& component_blocks_numbers = scc[component_index];
        vec<index> component_blocks = vec_restriction(local_pierced_blocks, component_blocks_numbers);
        // This actually removes component blocks from non_processed_blocks of course.
        CT::add_to(component_blocks, non_processed_blocks);
        #if DETAILS
            std::cout << "Processing component " << component_index << " with vertices { " << 
            component_blocks_numbers << "} and blocks { " << component_blocks << "}"  << std::endl;
            std::cout << "Processed virtual blocks: ";
            for(auto& p : pro_blocks){
                std::cout << " ( " << p.first << ") ";
            }
            std::cout << std::endl;
            std::cout << "Internal col-op graph:" << std::endl;
                vec<vec<index>> pro_graph_labels = vec<vec<index>>();
                for(Merge_data virtual_block : pro_blocks){
                    pro_graph_labels.push_back(virtual_block.first);
                }
                print_graph_with_labels(pro_graph, pro_graph_labels);
        #endif
        //Should not need to do this, if there arent many entries-> TO-DO: Check if this is necessary.
        bitset non_zero_columns = non_pro;
        if(non_pro.any()){
            non_zero_columns = simultaneous_column_reduction(N_map, component_blocks, local_pierced_blocks, non_pro);
            non_pro ^= non_zero_columns;
            // simultaneous_align(N_map, local_pierced_blocks, processed_support, non_zero_columns); -> Do we need this?  Update non-zero cols if align is used. 
        }
        for( index b : component_blocks){
            N_map[b].compute_rows_forward_map(row_map);
        }
        #if DETAILS
            std::cout << " N^{0}-support: " << non_zero_columns << std::endl;
            std::cout << " Non-processed columns: " << non_pro << std::endl;
            for(index i = 0; i < local_pierced_blocks.size(); i++){
                for(index j = 0; j < k; j++){
                    N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                }
            }
        #endif

        Graph component_graph;
        // TO-DO: Need to store which colums-ops correspond to an operation that is legal via this graph.
        Row_transform_map component_transforms;

        if(component_blocks.size() == 1){

            component_non_zero_columns = vec<bitset>(1, non_zero_columns);
            // No further reduction needed.
        } else {
            // If the component has more than one block, we need to consider the automorphism group of the blocks when extended with their new columns.
            // This means forming a new sub graph of \B
            component_graph = induced_subgraph(hom_graph, component_blocks_numbers);
            #if DETAILS
                std::cout << "  Induced subgraph of hom-graph: " << std::endl;
                print_graph_with_labels(component_graph, component_blocks);
            #endif

            if(is_resolvable_cycle[component_index]){
                component_non_zero_columns = vec<bitset>(component_blocks.size(), bitset(k, 0));
                // With this we can track which col ops need to be done if we perform a row op internal to the component.
                // -> We can also use all row-operations and thus reduce the sub-matrix N^zero even more if needed without calling Naive Decomposition
                if(non_zero_columns.count() < component_blocks.size() && non_zero_columns.any()){
                    // If the reduced Matrix has less columns than rows (each block is one row), only then are additional row-operations necessary:
                    simultaneous_row_reduction_on_submatrix(N_map, component_blocks, non_zero_columns, A);
                    for(index b : component_blocks){
                        N_map[b].compute_columns_from_rows((*block_map[b]).rows);
                    }
                }
                if(non_zero_columns.any()){
                    for(index j = 0; j < component_blocks.size(); j++){
                        bool all_zero = true;
                        for(index i = non_zero_columns.find_first(); i != bitset::npos; i = non_zero_columns.find_next(i)){
                            if( N_map[component_blocks[j]].is_nonzero(i)){
                                all_zero = false;
                                component_non_zero_columns[j].set(i);
                                assert(component_non_zero_columns[j].count() == 1);
                            }
                        }
                        if(all_zero){
                            delete_incoming_edges(component_graph, j);
                        }
                    }
                }

            } else {
                // At the moment, this should never be called, because I do not have the time to 
                // write a program which generates the sub-group of GL_k(F_2) acting on the columns via 
                // homomorphisms and from there generate all decompositions accesible from a group action.
                assert(false);
            }
        }

        // After N^{0} has been reduced we use these new columns to reduce N_b for the processed blocks.
        // This bitset has a true/1 entry whenever a cocycle has not(!) been deleted.
        vec<bitset> deleted_cocycles = vec<bitset>(component_blocks.size(), bitset(pro_blocks.size(), 0));
        for(index i = 0; i < component_blocks.size(); i++){
            for(index j = 0; j < pro_blocks.size(); j++){
                if(N_map[component_blocks[i]].is_nonzero(pro_blocks[j].second)){
                    deleted_cocycles[i].set(j);
                }
            }
        }

        for(auto it = pro_computation_order.rbegin(); it != pro_computation_order.rend(); it++){
            index& pro_component_index = *it;
            vec<index>& pro_component_blocks_numbers = pro_scc[pro_component_index];
            Vertex current_vertex = boost::vertex(pro_component_index, pro_condensation);
            vec<Merge_data> current_pro_blocks = vec_restriction(pro_blocks, pro_component_blocks_numbers); 

            #if DETAILS
                std::cout << "  Deleting above " << pro_component_index << " with ";
                for(auto& p : current_pro_blocks){
                    std::cout << " ( " << p.first << ") -> ";
                    std::cout << p.second;
                }
                std::cout << std::endl;
            #endif
            if (component_blocks.size() == 1) {
                index& b = component_blocks[0];
                bitset& b_non_zero_columns = component_non_zero_columns[0];
                for(index pro_b : pro_component_blocks_numbers){
                    Merge_data& pro_block = pro_blocks[pro_b];
                    // vec<index>& pro_block_blocks = pro_block.first;
                    // bitset& pro_support = pro_block.second;
                    Vertex internal_current_vertex = boost::vertex(pro_b, pro_graph);
                    vec<index> incoming_vertices = incoming_edges<index>(pro_graph, internal_current_vertex); // Those virtual blocks from which there are internal column operations to the current block.
                    vec<index> external_incoming_vertices = vec<index>(); // No other blocks in the component.
                    if(deleted_cocycles[0][pro_b]){
                        deleted_cocycles[0][pro_b] = ! alpha_extension_decomposition(
                            A, b, b_non_zero_columns, non_processed_blocks, pro_block, incoming_vertices, pro_blocks, deleted_cocycles[0],
                            hom_graph, hom_spaces, batch_transforms, base_change, external_incoming_vertices, component_transforms,
                            block_map, row_map, N_map);
                        if(deleted_cocycles[0][pro_b]){
                            #if DETAILS
                                std::cout << "      No deletion of cocycle at " << b << " and " << pro_block << std::endl;
                            #endif
                        } else {
                            #if DETAILS
                                std::cout << "      Deleted cocycle at " << b << " and " << pro_block << std::endl;
                            #endif
                        }
                    } else {
                        #if DETAILS
                        std::cout << "      0 cocyle at " << b << " and " << pro_block << std::endl;
                        #endif
                    }
                    #if DETAILS
                        for(index i = 0; i < local_pierced_blocks.size(); i++){
                            for(index j = 0; j < k; j++){
                                N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                            }
                        }
                    #endif
                }
            } else if (pro_component_blocks_numbers.size() == 1){
                index& pro_b = pro_component_blocks_numbers[0];
                Merge_data& pro_block = pro_blocks[pro_b];
                // vec<index>& pro_block_blocks = pro_block.first;
                // bitset& pro_support = pro_block.second;
                Vertex internal_current_vertex = boost::vertex(pro_b, pro_graph);
                vec<index> internal_incoming_vertices = incoming_edges<index>(pro_graph, internal_current_vertex); // Those virtual blocks from which there are internal column operations to the current pro block.

                for(index i = 0; i < component_blocks.size(); i++){
                    index& b = component_blocks[i];
                    Vertex external_current_vertex = boost::vertex(i, component_graph);
                    assert(external_current_vertex == i);
                    vec<index> external_incoming_vertices = incoming_edges<index>(component_graph, external_current_vertex); // Those blocks in the component from which there are row operations to the current block.
                    bitset& b_non_zero_columns = component_non_zero_columns[i];
                    if(deleted_cocycles[i][pro_b]){
                        deleted_cocycles[i][pro_b] = ! alpha_extension_decomposition(
                            A, b, b_non_zero_columns, non_processed_blocks, pro_block, internal_incoming_vertices, pro_blocks, deleted_cocycles[i],
                            hom_graph, hom_spaces, batch_transforms, base_change, external_incoming_vertices, component_transforms,
                            block_map, row_map, N_map); 
                        if(deleted_cocycles[i][pro_b]){
                            #if DETAILS
                                std::cout << "      No deletion of cocycle at " << b << " and " << pro_block << ", " << i << std::endl;
                            #endif
                        } else {
                            #if DETAILS
                                std::cout << "      Deleted cocycle at " << b << " and " << pro_block << ", " << i << std::endl;
                            #endif
                        }
                        #if DETAILS
                            for(index i = 0; i < local_pierced_blocks.size(); i++){
                                for(index j = 0; j < k; j++){
                                    N_map_indicator[i][j] = N_map[local_pierced_blocks[i]].is_nonzero(j);
                                }
                            }
                        #endif
                    } else {
                        #if DETAILS
                        std::cout << "      0 cocyle at " << b << " and " << pro_block << std::endl;
                        #endif
                    }
                }
                // Now recompute allowed row-operations for the component.  
                
                for(index i = 0; i < component_blocks.size(); i++){
                    index& b = component_blocks[i];
                    bitset& b_non_zero_columns = component_non_zero_columns[i];
                    bitset& b_cocycles = deleted_cocycles[i];
                    vec<index> sources = incoming_edges<index>(component_graph, i);
                    for(index j : sources){
                        index& c = component_blocks[j];
                        bitset& c_non_zero_columns = component_non_zero_columns[j];
                        bitset& c_cocycles = deleted_cocycles[j];
                        // This should compute if the effect of a row operation on the cocycles can be reverted by column operations.
                        // component_transforms[{i,j}] = compute_internal_row_ops(b_cocycles, c_cocycles);
                        //TO-DO: Implement this.
                    }
                }
            } else {
                assert(false);
                // Right now, I dont think I want this to happen.
                // If the graph on the processed blocks has no cycles this should be done block-by-block, otherwise:
                // TO-DO: Determine the the automorphism group of the blocks when extended with their new columns in
                // component_non_zero_columns as a subgroup of GL_comp_blocks.size ( F_2 )
                // and iterate over these ? 
                // -> Maybe call exhaustive decomposition instead.
            }
        }
        // Virtually merge the current blocks with the processed virtual blocks based on deleted cocycles.
        if (component_blocks.size() == 1){
            assert(deleted_cocycles.size() == 1);
            assert(component_non_zero_columns.size() == 1);
            Merge_data new_pro_block = {component_blocks, component_non_zero_columns[0]};
            for(index j = deleted_cocycles[0].size()-1; j > -1 ; j--){
                if(deleted_cocycles[0][j]){
                    merge_virtual_blocks(new_pro_block, pro_blocks[j]);
                    pro_blocks.erase(pro_blocks.begin() + j);
                }
            }
            pro_blocks.push_back(new_pro_block);
        } else {
            vec<Merge_data> new_pro_blocks = vec<Merge_data>();
            for(index i = 0; i < component_blocks.size(); i++){
                new_pro_blocks.push_back({{component_blocks[i]}, component_non_zero_columns[i]});
            }
            assert(new_pro_blocks.size() == component_blocks.size());
            for(index j = pro_blocks.size() -1 ; j > -1; j--){
                assert(deleted_cocycles.size() == new_pro_blocks.size());
                index first_false = -1;
                vec<index> component_block_merges = vec<index>();
                for(index b = 0; b < new_pro_blocks.size(); b++){
                    if(deleted_cocycles[b][j]){
                        if(first_false == -1){
                            first_false = b;
                            merge_virtual_blocks(new_pro_blocks[b], pro_blocks[j]);
                            pro_blocks.erase(pro_blocks.begin() + j);
                        } else {
                            component_block_merges.push_back(b);
                        }
                    }
                }
                if (!component_block_merges.empty()){
                    // We can merge the blocks in the component and the bitsets indicating which of the cocycles have been deleted.
                    for(auto it = component_block_merges.rbegin(); it != component_block_merges.rend(); it++){
                        merge_virtual_blocks(new_pro_blocks[first_false], new_pro_blocks[*it]);
                        new_pro_blocks.erase(new_pro_blocks.begin() + *it); // Is this a problem? will the iterator become invalid?
                        deleted_cocycles[first_false] |= deleted_cocycles[*it];
                        deleted_cocycles.erase(deleted_cocycles.begin() + *it);
                    }
                }
            }
            pro_blocks.insert(pro_blocks.end(), new_pro_blocks.begin(), new_pro_blocks.end());  
        }
        // Update the processed graph. Right now this is very non-optimised.   
        for(auto it = pro_blocks.rbegin(); it != pro_blocks.rend(); it++){
            for( auto it2 = pro_blocks.rbegin(); it2 != pro_blocks.rend(); it2++){
                if( it != it2){
                if( batch_transforms.find({*it, *it2}) == batch_transforms.end() ){
                    batch_transforms[{*it, *it2}] = compute_internal_col_ops(*it, *it2, N_map, hom_spaces, row_map, k, block_map);
                }
                }
            }
        }
        pro_graph = construct_batch_transform_graph(batch_transforms, pro_blocks);
        vec<index> component = vec<index>(boost::num_vertices(pro_graph));
        pro_condensation = compute_scc_and_condensation(pro_graph, component, pro_scc);
        pro_computation_order = compute_topological_order<index>(pro_condensation); 

    }
    return pro_blocks;
}


/**
 * @brief Check for directed cycles
 * 
 * @param pierced_blocks 
 * @param scc 
 * @param has_cycle 
 * @param has_unresolvable_cycle 
 * @param has_multiple_cycles 
 */
inline void get_cycle_information(vec<index>& pierced_blocks, vec<vec<index>>& scc, vec<Block_iterator>& block_map,
    bool& has_cycle,
    bool& has_unresolvable_cycle,
    bool& has_multiple_cycles,
    vec<bool>& is_resolvable_cycle){

    for(index i = 0; i < scc.size(); i++){
        if(scc[i].size() > 1){
            if(has_cycle){
                has_multiple_cycles = true;
                #if DETAILS
                    std::cout << "Multiple cycles detected." << std::endl;
                #endif
                break;
            }
            has_cycle = true;
            #if DETAILS
            std::cout << "Blocks in cycle: ";
            #endif
            for(index vertex_number : scc[i]){
                index block_index = pierced_blocks[vertex_number];
                #if DETAILS
                std::cout << block_index << " ";
                #endif
                if(block_map[block_index]->type == BlockType::NON_INT){
                    has_unresolvable_cycle = true;
                    is_resolvable_cycle[i] = false;
                    break;
                }
            }
            #if DETAILS
            std::cout << std::endl;
            #endif
        }
    }
}

inline void reduce_hom_alpha_graph(Hom_map& hom_spaces, vec<index>& local_pierced_blocks, 
    Graph& hom_digraph, r2degree& alpha, vec<Block_iterator>& block_map) {

    auto edges = boost::edges(hom_digraph);
    std::vector<std::pair<Graph::vertex_descriptor, Graph::vertex_descriptor>> edges_to_remove;
    for (auto edge_it = edges.first; edge_it != edges.second; ++edge_it) {
        // Get the source and target vertices of the edge
        auto source_vertex = boost::source(*edge_it, hom_digraph);
        auto target_vertex = boost::target(*edge_it, hom_digraph);

        index c = local_pierced_blocks[source_vertex];
        index b = local_pierced_blocks[target_vertex];

        // Access the blocks C and B
        Block& C = *block_map[c];
        Block& B = *block_map[b];

        // Perform the steps already present
        if (B.local_basislift_indices.empty()) {
            B.compute_local_basislift(alpha);
        }
        if (C.local_basislift_indices.empty()) {
            C.compute_local_basislift(alpha);
        }
        if (B.local_cokernel == nullptr) {
            B.compute_local_cokernel();
        }

        bool is_zero = hom_quotient_zero(hom_spaces[{c,b}] , *B.local_cokernel, C.local_basislift_indices, C.local_admissible_rows, B.local_admissible_rows, C.rows);
        
        if (is_zero) {
            #if DETAILS
                std::cout << "  alpha-reduction: Deleted " << c << " to " << b << std::endl;
            #endif
            hom_spaces[{c,b}].first.data.clear();
            hom_spaces[{c,b}].first.set_num_cols(0);
            edges_to_remove.emplace_back(source_vertex, target_vertex);
        }
    }

    for (const auto& edge : edges_to_remove) {
        boost::remove_edge(edge.first, edge.second, hom_digraph);
    }
}


/**
 * @brief Given a list of blocks and all hom_spaces, constructs the digraph on the blocks 
 * where there is a directed edge from block i to block j if Hom(B_i, B_j) != 0, as well as
 * a condensation of this graph and a topological order on the condensation.
 * 
 * @param hom_digraph 
 * @param component 
 * @param scc 
 * @param condensation 
 * @param computation_order 
 * @param pierced_blocks 
 * @param hom_spaces 
 * @param cyclic_counter 
 * @param resolvable_cyclic_counter 
 * @param acyclic_counter 
 * @return false if there are unresolvable cycles 
 */
inline bool construct_graphs_from_hom(Graph& hom_digraph, std::vector<index>& component, vec<vec<index>>& scc, vec<bool>& is_resolvable_cycle,
        Graph& condensation, vec<index>& computation_order, vec<index>& pierced_blocks, Hom_map& hom_spaces, 
        AIDA_runtime_statistics& statistics, AIDA_config& config, index& t, vec<Block_iterator>& block_map, r2degree& alpha){
    
    // Graph on pierced blocks representing Hom(C,B) != 0 
    hom_digraph = construct_hom_digraph(hom_spaces, pierced_blocks);
    
    bool test_alpha_cycles = true;

    bool test_has_cycle = false;
    bool test_has_unresolvable_cycle = false;
    bool test_has_multiple_cycles = false;
    vec<bool> test_is_resolvable_cycle;

    if(test_alpha_cycles){
        std::vector<index> test_component = vec<index>(boost::num_vertices(hom_digraph));
        vec<vec<index>> test_scc;
        Graph test_condensation = compute_scc_and_condensation(hom_digraph, test_component, test_scc);
        test_is_resolvable_cycle = vec<bool>(test_scc.size(), true);
        get_cycle_information(pierced_blocks, test_scc, block_map, test_has_cycle, test_has_unresolvable_cycle, test_has_multiple_cycles, test_is_resolvable_cycle);
    }

    if(config.alpha_hom){
      //  reduce_hom_alpha_graph(hom_spaces, pierced_blocks, hom_digraph, alpha, block_map);
    }

    // Components assigns to each block the index of the strongly connected component it is in.
    component = vec<index>(boost::num_vertices(hom_digraph));
    // SCC is a vec< set<index> >, where each set contains the indices of the blocks in the SCC., condensation is a graph on the SCCs.
    condensation = compute_scc_and_condensation(hom_digraph, component, scc);
    // Contains the order in which to process the SCCs in reverse
    computation_order = compute_topological_order<index>(condensation); 
    
    is_resolvable_cycle = vec<bool>(scc.size(), true);

    #if DETAILS
        print_graph(hom_digraph);
        std::cout << "Component " <<  component << std::endl;
        std::cout << "SCCs " << scc << std::endl;
        print_graph(condensation);
        std::cout << "Computation order " << computation_order << std::endl;
    #endif


    // Check if there are any cycles in the hom-digraph by checking if there is a strongly connected component of size > 1
    bool has_cycle = false;
    bool has_unresolvable_cycle = false;
    bool has_multiple_cycles = false;

    get_cycle_information(pierced_blocks, scc, block_map, has_cycle, has_unresolvable_cycle, has_multiple_cycles, is_resolvable_cycle);

    if(test_alpha_cycles){
        if(test_has_unresolvable_cycle && !has_unresolvable_cycle){
            statistics.alpha_cycle_avoidance ++;
        } else if (test_has_multiple_cycles && !has_multiple_cycles){
            // record this?
        } else if (test_has_cycle && !has_cycle){
            // record this?
        }
    }

    #if DETAILS
        std::cout << "Batch " << t << " is ";
    #endif
    if(has_unresolvable_cycle){
        statistics.cyclic_counter++;
        #if DETAILS
            std::cout << "un-resolvable cyclic." << std::endl;
        #endif
    } else if(has_multiple_cycles){
        #if DETAILS
            std::cout << "polycyclic." << std::endl;
        #endif
    } else if(has_cycle) {
        statistics.resolvable_cyclic_counter++;
        #if DETAILS
            std::cout << "resolvable cyclic." << std::endl;
        #endif
    } else {
        statistics.acyclic_counter++;
        #if DETAILS
            std::cout << "acyclic." << std::endl;
        #endif
    }
    return !has_unresolvable_cycle && !has_multiple_cycles;
}

/**
 * @brief Groups the blocks so that every group occupies a seperate subset of the columns.
 * 
 * @param N_map 
 * @param blocks 
 * @param support 
 * @return vec<Merge_data>
 */
inline vec<Merge_data> find_prelim_decomposition(Sub_batch& N_map, const vec<index>& blocks, const bitset& support){
    index k = support.size();
    assert(k == N_map[blocks[0]].get_num_cols());
    vec<Merge_data> block_to_columns; 
    for(index i = 0; i< blocks.size(); i++){
        index b = blocks[i];
        block_to_columns.push_back({{b}, bitset(k, 0)});
        for(index col = support.find_first(); col != bitset::npos ; col = support.find_next(col)){
            if(N_map[b].col_last(col) != -1){
                block_to_columns[i].second.set(col);
            }
        }
    }
    for(index col = support.find_first(); col!= bitset::npos; col = support.find_next(col)){
        auto first_occurence = block_to_columns.end();
        for(auto it = block_to_columns.begin(); it != block_to_columns.end();){
            if((*it).second.test(col)){
                if(first_occurence == block_to_columns.end()){
                    first_occurence = it;
                    it++;
                } else {
                    // Merge the two blocks
                    (*first_occurence).first.insert((*first_occurence).first.end(), (*it).first.begin(), (*it).first.end() );
                    (*first_occurence).second |= (*it).second;
                    it = block_to_columns.erase(it);
                }
            } else {
                it++;
            }
        }       
    }
    return block_to_columns;
}



/**
 * @brief Compares the sets of merges at each batch. The first input is assumed to be the stable one.
 * 
 * @param merge_info_1 
 * @param merge_info_2 
 */
inline void compare_merge_info(Full_merge_info& merge_info_1, Full_merge_info& merge_info_2){
    assert(merge_info_1.size() == merge_info_2.size());
    bool success = true;
    for(index i = 0; i < merge_info_1.size(); i++){
        auto& merge_vec_1 = merge_info_1[i];
        auto& merge_vec_2 = merge_info_2[i];

        for(Merge_data& merge : merge_vec_1){
            std::sort(merge.first.begin(), merge.first.end());
            assert(is_sorted(merge.first));
        }
        for(Merge_data& merge : merge_vec_2){
            std::sort(merge.first.begin(), merge.first.end());
            assert(is_sorted(merge.first));
        }
        std::sort(merge_vec_1.begin(), merge_vec_1.end(), merge_comparator);
        std::sort(merge_vec_2.begin(), merge_vec_2.end(), merge_comparator);

        if(merge_vec_1.size() != merge_vec_2.size()){
            success = false;
            std::cout << "Different number of merges at batch " << i << std::endl;
            for(auto& merge : merge_vec_1){
                std::cout << "(" << merge.first << ") ";
            }
            std::cout << std::endl;
            std::cout << " vs " << std::endl;
            for(auto& merge : merge_vec_2){
                std::cout << "(" << merge.first << ") ";
            }
            std::cout << std::endl;
        } else {
            
            for(index j = 0; j < merge_vec_1.size(); j++){
                Merge_data& merge_1 = merge_vec_1[j];
                Merge_data& merge_2 = merge_vec_2[j];
                if(merge_1.first.size() != merge_2.first.size()){
                    success = false;
                    std::cout << "Different number of blocks in merge at batch " << i << std::endl;
                    for(auto& block : merge_1.first){
                        std::cout << block << " ";
                    }
                    std::cout << std::endl;
                    std::cout << " vs " << std::endl;
                    for(auto& block : merge_2.first){
                        std::cout << block << " ";
                    }
                    std::cout << std::endl;
                } else {
                    /** 
                    auto it1 = merge_1.first.begin();
                    auto it2 = merge_2.first.begin();
                    for(; it1 != merge_1.first.end(); it1++, it2++){
                        if(*it1 != *it2){
                            std::cout << "Different blocks in merge at batch " << i << std::endl;
                            for(auto& block : merge_1.first){
                                std::cout << block << " ";
                            }
                            std::cout << std::endl;
                            std::cout << " vs " << std::endl;
                            for(auto& block : merge_2.first){
                                std::cout << block << " ";
                            }
                            std::cout << std::endl;
                        }
                    }
                    */
                }
            }
        }
    }
    if(success){
        std::cout << "The number of blocks at each merge point is the same." << std::endl;
    }
} // Compare_merge_info


} // namespace aida


#endif // AIDA_HPP
