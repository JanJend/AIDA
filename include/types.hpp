
/**
 * @file option_parser.hpp
 * @author Jan Jendrysiak
 * @version 0.9.0
 * @date 2025-10-21
 * @brief  defines types used throughout the AIDA library
 * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */


#pragma once
#ifndef AIDA_TYPES_HPP
#define AIDA_TYPES_HPP

#include <vector>
#include <cstdint>
#include <utility>
#include <set>           
#include <unordered_map> 
#include <tuple>         
#include <grlina/graded_linalg.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/storage/csc.hpp>
#include <boost/timer/timer.hpp>  

namespace aida{

using namespace graded_linalg;
using namespace boost::timer;

using index = int;

template<typename T>
using vec = std::vector<T>;
template<typename T>
using array = vec<vec<T>>;

using Sparse_Matrix = SparseMatrix<index>;
using CT = Column_traits<vec<index>, index>;
using GradedMatrix = GridGradedSparseMatrix<double, 2, index>;
using CSCGradedMatrix = R2GradedSparseMatrix<index, CSCStorage<index>>;
using Degree = base_degree<GradedMatrix>;
using GridD = typename GradedMatrix::GridD;
using Module = graded_linalg::Module<GradedMatrix>;
using PersistenceModule = Module; // compatibility alias
template<typename index>
using indtree_t = std::set<index>;

// a list of blocks with a corresponding subset of the columns of the current batch
// Can be treated like an indecomposable/block itself.
template<typename index>
using Merge_data_t = std::pair<vec<index>, bitset>;


template<typename index>
struct vec_index_hash_t {
    std::size_t operator()(const std::vector<index>& v) const {
        std::size_t seed = 0;
        for (const auto& elem : v) {
            seed ^= std::hash<index>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


/**
 * @brief Careful, this only considers the indices of the blocks not the columns!
 * 
 */
template<typename index>
struct virtual_block_pair_hash_t {
    std::size_t operator()(const std::pair<Merge_data_t<index>, Merge_data_t<index>>& p) const {
        vec_index_hash_t<index> vector_hasher;
        auto hash1 = vector_hasher(p.first.first);
        auto hash2 = vector_hasher(p.second.first);
        return hash1 ^ (hash2 << 1); 
    }
};

// A list of blocks with a corresponding subset of the columsn of the current batch.
// Can be treated like a virtual block, as in that it is an indecomposable even before it is formally merged.
template<typename index>
using Full_merge_info_t = vec<vec<Merge_data_t<index>>>;
template<typename index>
using pair_t = std::pair<index, index>;
template<typename index>
using op_info_t = std::pair<pair_t<index>, pair_t<index>>;
template<typename index>
using hom_info_t = std::pair<index, pair_t<index>>;
template<typename index>
using edge_list_t = array<index>;

// Every column of the sparse matrix is a homomorphism, where the pair corresponding to each entry 
    // is the source and target index (internal to the blocks) of a row operation.
template<typename index>
using Hom_space_t = std::pair< SparseMatrix<index>, vec<pair_t<index>>>;

    // The key is the block index, the value is the sub-matrix given by restricting the batch to this block.
template<typename index>
using Sub_batch_t = std::unordered_map<index, SparseMatrix<index>>;

    // The key is a pair of block indices, the value is a basis for the space of homomorphism from the first to the second.
template<typename index>
using Hom_map_t = std::unordered_map<pair_t<index>, Hom_space_t<index>, pair_hash<index>> ;
    
    // The Matrix encodes an admissible linear transformation of the (columns of the) batch
    // The vector of tuples encodes the associated homomorphisms: (c, b, i) means 
    // the i-th homomorphism from c to b in the linear representation of Hom(c,b) in the hom_map.
template<typename index>
using Batch_transform_t = std::pair< DenseMatrix, vec<std::tuple<index,index,index>> > ;


    //  For some pair of blocks, this should contain a list of associated column operations. 
    //  Each entry is a pair of virtual blocks, 
    //  together with a linear combination of basis elements of the space of allowable column transformations
template<typename index>
using Row_transform_t = vec< std::pair<pair_t<index>, vec<index>> >;

    // The key is a pair of block indices, the value is a Row_transform.
template<typename index>
using Row_transform_map_t = std::unordered_map<pair_t<index>, Row_transform_t<index>, pair_hash<index>>;


template<typename index>
using Transform_Map_t = std::unordered_map< std::pair<Merge_data_t<index>, Merge_data_t<index>>, vec<Batch_transform_t<index>>, virtual_block_pair_hash_t<index>>;


using indtree = indtree_t<int>;
using Merge_data = Merge_data_t<int>;
using Full_merge_info = Full_merge_info_t<int>;
using pair = pair_t<int>;
using op_info = op_info_t<int>;
using hom_info = hom_info_t<int>;
using edge_list = edge_list_t<int>;
using Hom_space = Hom_space_t<int>;
using Sub_batch = Sub_batch_t<int>;
using Hom_map = Hom_map_t<int>;
using Batch_transform = Batch_transform_t<int>;
using Row_transform = Row_transform_t<int>;
using Row_transform_map = Row_transform_map_t<int>;
using Transform_Map = Transform_Map_t<int>;
using vec_index_hash = vec_index_hash_t<int>;
using virtual_block_pair_hash = virtual_block_pair_hash_t<int>;

enum class BlockType {
    FREE, // 1 Generator, 0 Relations
    CYC,  // 1 Generator, >0 Relation (cyclic, non free)
    INT,  // Interval, non cyclic.
    NON_INT // Non-Interval
};



}

#endif // AIDA_TYPES_HPP
