
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


/**
 * @brief Merges the content of all blocks and a restriction of N into a new block.
 * While the rows stay sorted, the columns are not.
 *          
 * @param block_indices 
 * @param block_map 
 * @param new_block 
 * @param N_map 
 * @param batch_positions 
 * @param batch_indices 
 */
void merge_blocks_into_block(vec<index>& block_indices, vec<Block_list::iterator>& block_map, Block& new_block, 
                            Sub_batch& N_map, bitset& batch_positions, vec<index>& batch_indices, 
                            vec<index>& row_map, r2degree& alpha){
    
    std::priority_queue<block_position, vec<block_position>, compare_block_position_row> row_heap;
    std::priority_queue<block_position, vec<block_position>, compare_block_position_column> column_heap;

    // maps the initial index of the block to a vector containing the pairs: 
    // (batch index, iterator to the associated column of N)
    /*
    std::map<index, vec<std::pair<index, vec<index>::iterator>> > N_iterators;
    */


    bool input_is_interval = true;

    for(index i : block_indices){
        auto B = block_map[i];
        if(B->type == BlockType::NON_INT){
            input_is_interval = false;
        }
        row_heap.push({0, B});
        if(!(B->columns.empty())){
            column_heap.push({0, B});
        }
        /*
        new_block.columns.insert(new_block.columns.end(), B->columns.begin(), B->columns.end());
        new_block.data.insert(new_block.data.end(), B->data.begin(), B->data.end());
        new_block.col_degrees.insert(new_block.col_degrees.end(), B->col_degrees.begin(), B->col_degrees.end());
        */
        new_block.increase_num_rows( B->get_num_rows());
        new_block.increase_num_cols( B->get_num_cols());
    }
    
    new_block.type = BlockType::NON_INT;


    // Check if we stay being an interval
    Degree_traits<r2degree> traits;
    if(batch_positions.count() == 1 && block_indices.size() == 2 && input_is_interval){
        auto i = batch_positions.find_first();
        bool N_is_length_two = true;
        vec<r2degree> generators;
        for(index b : block_indices){
            if( N_map[b].data[i].size() != 1 ){
                N_is_length_two = false;
                break;
            } else {
                generators.push_back(block_map[b]->row_degrees[row_map[N_map[b].data[i].front()]]);
            }
        }
        if(N_is_length_two){
            if( traits.equals(alpha,  traits.join(generators[0], generators[1])) ){
                new_block.type = BlockType::INT;
            }
        }
    }



    index batch_threshold = new_block.get_num_cols();

    new_block.rows.reserve(new_block.get_num_rows());
    new_block._rows.reserve(new_block.get_num_rows());
    new_block.row_degrees.reserve(new_block.get_num_rows());
    new_block.columns.reserve(new_block.get_num_cols());
    new_block.data.reserve(new_block.get_num_cols());
    new_block.col_degrees.reserve(new_block.get_num_cols());

    // Add columns of the blocks according to the column_heap.

    while (!column_heap.empty()) {
        auto current_col = column_heap.top();
        column_heap.pop();
        new_block.columns.push_back(current_col.second->columns[current_col.first]);
        new_block.col_degrees.push_back(current_col.second->col_degrees[current_col.first]);
        new_block.data.push_back(current_col.second->data[current_col.first]);
        if (current_col.first + 1 < current_col.second->columns.size()) {
            column_heap.push({current_col.first + 1, current_col.second});
        }
    }

    // Add columns of N and initialise all iterators to the columns of N.
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        new_block.columns.push_back(batch_indices[i]);
        new_block.col_degrees.push_back(alpha);
        new_block.data.push_back(vec<index>());
        for(index j : block_indices){
            new_block.data.back().insert(new_block.data.back().end(), N_map[j].data[i].begin(), N_map[j].data[i].end());
            /*
            if( N_map[j].data[i].empty() ){
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            } else {
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            }
            */
        }
        std::sort(new_block.data.back().begin(), new_block.data.back().end());
    }
    new_block.increase_num_cols( batch_positions.count());
    // The minheap sorts the row-indices of the blocks in ascending order. 
    // Iteratively, we add this row index, the associated row from the block, and append entries, if the columns of N permit us.
    // TO-DO: Maybe this is much slower than simply sorting everything, I do not know.

    new_block._rows = vec<vec<index>>(new_block.get_num_rows());
    index row_counter = 0;
    while (!row_heap.empty()) {
        block_position current = row_heap.top();
        Block& B = *current.second;
        row_heap.pop();
        new_block.rows.push_back(B.rows[current.first]);
        new_block.row_degrees.push_back(B.row_degrees[current.first]);
        // Reevaluating the map from A.rows to new_block.rows
        row_map[B.rows[current.first]] = row_counter;
        /* Since we are now using internally indexed rows it doe not make sense to merge them like this without a map from col_indices to internal indexes.
        Instead we should recompute them, although this might be costly. Maybe optimise in a later version.
        new_block._rows.push_back(B._rows[current.first]);
        
        auto& itvec = N_iterators[B.rows.front()];
        index internal_col_index = batch_positions.find_first();
        for(index i = 0; i < itvec.size(); i++){
            if( itvec[i].second == N_map[B.rows.front()].data[internal_col_index].end() ){
                internal_col_index = batch_positions.find_next(internal_col_index);
                continue;    
            }
            internal_col_index = batch_positions.find_next(internal_col_index);
            if( *itvec[i].second == B.rows[current.first]){
                new_block.data[batch_threshold + i].push_back(*itvec[i].second);
                new_block._rows[row_counter].push_back(itvec[i].first);
                itvec[i].second++;
            }
        }
        */
        if (current.first + 1 < B.rows.size()) {
            row_heap.push({current.first + 1, current.second});
        }
        row_counter++;
    }   

    assert(new_block._rows.size() == new_block.row_degrees.size());
    assert(new_block.rows.size() == new_block.row_degrees.size());
    assert(new_block.columns.size() == new_block.data.size());
    assert(new_block.data.size() == new_block.get_num_cols());

}


/**
 * @brief Merges the blocks and updates the B_list.
 * 
 * @param A 
 * @param B_list 
 * @param N_map 
 * @param block_map 
 * @param block_partition 
 */
void merge_blocks(Block_list& B_list, Sub_batch& N_map, 
                    vec<Block_list::iterator>& block_map, vec<Merge_data>& block_partition, vec<index>& batch_indices, 
                    vec<index>& row_map, r2degree& alpha){ 
    for(auto& partition : block_partition){
        vec<index>& block_indices = partition.first;
        bitset& batch_positions = partition.second; 
        index first = *block_indices.begin();
        if(block_indices.size() == 1){
            extend_block(*block_map[first], N_map[first], batch_indices, batch_positions, alpha);
        } else {
            if(block_indices.size() < 1){
                std::cout << "  Warning: No Merge info at batch indices " <<  batch_indices << std::endl;
                assert(false);
            }
            Block new_block({}, {});
            // TO-DO: In many cases it might be better to find the largest block and merge the others into it.
            auto new_it = B_list.insert(B_list.end(), new_block);
            merge_blocks_into_block(block_indices, block_map, *new_it, N_map, batch_positions, batch_indices, row_map, alpha); 

            for(index i : block_indices){
                auto del_it = block_map[i];
                for(index j : block_map[i]->rows){
                    block_map[j] = new_it;
                }
                B_list.erase(del_it);

            }
        }          
    }
}


}