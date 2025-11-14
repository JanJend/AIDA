
/**
 * @file option_parser.hpp
 * @author Jan Jendrysiak
 * @version 0.2
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
#include <utility>
#include <set>           
#include <unordered_map> 
#include <tuple>         
#include <grlina/graded_linalg.hpp>  
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
using GradedMatrix = R2GradedSparseMatrix<index>;
using indtree = std::set<index>;
using CT = Column_traits<vec<index>, index>;
// a list of blocks with a corresponding subset of the columns of the current batch
// Can be treated like an indecomposable/block itself.
using Merge_data = std::pair<vec<index>, bitset>; 


struct vec_index_hash {
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
struct virtual_block_pair_hash {
    std::size_t operator()(const std::pair<Merge_data, Merge_data>& p) const {
        vec_index_hash vector_hasher;
        auto hash1 = vector_hasher(p.first.first);
        auto hash2 = vector_hasher(p.second.first);
        return hash1 ^ (hash2 << 1); 
    }
};

// A list of blocks with a corresponding subset of the columsn of the current batch.
// Can be treated like a virtual block, as in that it is an indecomposable even before it is formally merged.
using Full_merge_info = vec<vec<Merge_data>>;
using pair = std::pair<index, index>;
using op_info = std::pair<pair, pair>;
using hom_info = std::pair<index, pair>;
using edge_list = array<index>;

// Every column of the sparse matrix is a homomorphism, where the pair corresponding to each entry 
    // is the source and target index (internal to the blocks) of a row operation.
using Hom_space = std::pair< Sparse_Matrix, vec<pair>>; 

    // The key is the block index, the value is the sub-matrix given by restricting the batch to this block.
using Sub_batch = std::unordered_map<index, Sparse_Matrix>; 

    // The key is a pair of block indices, the value is a basis for the space of homomorphism from the first to the second.
using Hom_map = std::unordered_map<pair, Hom_space, pair_hash<index>> ;
    
    // The Matrix encodes an admissible linear transformation of the (columns of the) batch
    // The vector of tuples encodes the associated homomorphisms: (c, b, i) means 
    // the i-th homomorphism from c to b in the linear representation of Hom(c,b) in the hom_map.
using Batch_transform = std::pair< DenseMatrix, vec<std::tuple<index,index,index>> > ;


    //  For some pair of blocks, this should contain a list of associated column operations. 
    //  Each entry is a pair of virtual blocks, 
    //  together with a linear combination of basis elements of the space of allowable column transformations
using Row_transform = vec< std::pair<pair, vec<index>> >; 

    // The key is a pair of block indices, the value is a Row_transform.
using Row_transform_map = std::unordered_map<pair, Row_transform, pair_hash<index>>;


using Transform_Map = std::unordered_map< std::pair<Merge_data, Merge_data>, vec<Batch_transform>, virtual_block_pair_hash>;


enum class BlockType {
    FREE, // 1 Generator, 0 Relations
    CYC,  // 1 Generator, >0 Relation (cyclic, non free)
    INT,  // Interval, non cyclic.
    NON_INT // Non-Interval
};



}

#endif // AIDA_TYPES_HPP