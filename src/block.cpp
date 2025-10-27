
#include "block.hpp"

namespace aida {

    /**
 * @brief Constructs the Blocks of an empty Matrix whose rows are given by A.
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 */
void initialise_block_list(const GradedMatrix& A, Block_list& B_list, vec<Block_list::iterator>& block_map) {
    B_list.clear();
    B_list = Block_list();
    for(int i=0; i < A.get_num_rows(); i++) {
        // Block B({},{i}, BlockType::FREE);
        // B.set_num_rows(1);
        auto it = B_list.emplace(B_list.end(), std::vector<index>{}, std::vector<index>{i}, BlockType::FREE);
        // B_list.back().set_num_cols(0);
        B_list.back().set_num_rows(1);
        block_map.push_back(it);
        (*it).row_degrees[0] = A.row_degrees[i];
        (*it)._rows = vec<vec<index>>(1);
        (*it).rows_computed = true;
    }
}

/**
 * @brief Displays the degrees of each block in the block list.
 * 
 * @param B_list 
 */
void print_block_list_status(Block_list& B_list) {
    std::cout << "Status: " << B_list.size() << " blocks:\n";
    index count=0;
    for(Block& b : B_list) {
      std::cout << "Block " << count++ << ":" << std::endl;
      b.print_degrees();
      std::cout << std::endl;
    }
}

/**
 * @brief Extends the block B by the columns of N given by the batch_indices and the batch_positions.
 * 
 * @param B 
 * @param N 
 * @param batch_positions 
 * @param batch_indices 
 */
void extend_block(Block& B, Sparse_Matrix& N, vec<index> batch_indices, bitset& batch_positions, r2degree& alpha) {
    if(batch_positions.empty()){
        batch_positions = bitset(N.get_num_cols(), true);
    }
    
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        B.columns.push_back(batch_indices[i]);
        B.data.push_back(N.data[i]);
        B.col_degrees.push_back(alpha);
        // Directly compute the rows for efficiency:
        auto it = N.data[i].begin();
        for(index j = 0; j < B.rows.size() && it != N.data[i].end() ; j++){
            if(*it == B.rows[j]){
                B._rows[j].push_back(i);
                it++;
            }
        }   
    }
    B.increase_num_cols(batch_positions.count());
    assert(B.get_num_cols() == B.columns.size());
    assert(B.get_num_cols() == B.data.size());

    if(B.type == BlockType::FREE){
        B.type = BlockType::CYC;
    } 
    
}


}