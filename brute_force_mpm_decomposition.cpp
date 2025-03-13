#include <iostream>
#include <grlina/graded_linalg.hpp>
#include <unordered_set>
#include <list>
#include <boost/timer/timer.hpp>
#include <memory_resource>
#include <cstdlib>
#include <filesystem>
#include <queue>
#include <functional>
namespace aida{

using namespace graded_linalg;
namespace fs = std::filesystem;
using namespace boost::timer;


#define TIMERS 1
#define DETAILS 0
#define ANALYSIS 0

template <typename T>
using vec = vec<T>;
template <typename T>
using array = vec<vec<T>>;

using index = int; //Change to large enough int type.
using indvec = vec<index>;
using Sparse_Matrix = SparseMatrix<index>;
using GradedMatrix = R2GradedSparseMatrix<index>;
using indtree = std::set<index>;
using Merge_data = std::pair<indtree, bitset>;
using op_info = std::pair< std::pair<index, index>, std::pair<index, index> >;
using edgeList = array<index>;
using pair = std::pair<index, index>;

struct Block {

    indvec rows;
    indvec columns;

    bool rows_computed;

    Sparse_Matrix indecomp; // This will store the content of the block in column and row form. For now as a sparse matrix.

    indvec admissible_cols; // Stores the indices of the columns which can be used for column operations.
    // This will store the columns which can be added to the current batch, i.e. local_data = data | admissable_cols.
    std::shared_ptr<Sparse_Matrix> local_data; 


    /**
     * @brief If admissable_cols has been set, this fetches the columns for further processing.
     * 
     */
    void compute_local_data(GradedMatrix& A, index target){
        // local_data = std::make_shared<Sparse_Matrix>(Sparse_Matrix(0,0));
        for(index i = 0; i < columns.size(); i++){
            if(A.is_admissible_column_operation(columns[i], target)){
                // std::cout << "  found an addmisible col op from column " << i << ": ";
                // std::cout << A.col_degrees[columns[i]].first << " " << A.col_degrees[columns[i]].second << " to " <<
                //     A.col_degrees[target].first << " " << A.col_degrees[target].second << std::endl;
                admissible_cols.push_back(i);
            }
        }
        // std::cout << "  Calculating local data at indices " << admissible_cols << std::endl;
        if(admissible_cols.empty()){
            local_data = nullptr;
        } else {
            local_data = std::make_shared<Sparse_Matrix>(indecomp.restricted_domain_copy(admissible_cols));
        }
        
    }

    /**
     * @brief Resets the local data after processing a batch.
     * 
     */
    void delete_local_data(){
        admissible_cols.clear();
        local_data.reset();
    }

    /**
     * @brief Tries to delete N with the columns in local_data.
     * 
     * @param N 
     * @return true 
     * @return false 
     */
    bool reduce_N( Sparse_Matrix& N){
        // Tries to delete N with the columns in local_data.
        if(local_data == nullptr){
            return false;
        }

        // std::cout << "The local data is ";
        // local_data->print();
        // std::cout << "N is: ";
        //N.print();
        return local_data->solve_col_reduction(N);
    }

    /**
     * @brief Compute the list of rows format for the indecomposable matrix belonging to the block.
     * 
     */
    void compute_rows(){
        if(!rows_computed){
            indecomp.compute_rows_forward(columns, rows);
            rows_computed = true;
        } else {
            // std::cerr << "Rows have already been computed for the block at: " << columns << " x " << rows << std::endl;
        }
    }


    //To-Do: Update the constructors, so that memory for the data  is reserved??
    Block(const indvec& c, const indvec& r) : indecomp(c.size(),r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
    }

    
    Block() {}
    
    void clear() {
      this->rows.clear();
      this->columns.clear();
    }
    
    void print(bool with_content=true) {
        std::cout << "  Columns: ";
        for(int i : this->columns) {
            std::cout << i << " ";
        }
        std::cout << "\n  Rows: ";
        for(int i : this->rows) {
            std::cout << i << " ";
        }

        if(with_content){
            std::cout << "  Data: " << std::endl;
            this->indecomp.print(false, true);
        }
    }

    void compute_degree_graph(GradedMatrix& A){

    }
};



struct Operation_info {
    bool row_op;
    index source;
    index target;
    Operation_info(bool row_op,index source,index target) 
      : row_op(row_op), source(source), target(target) 
    {}
  };

// It might be computationally efficient to store the content of a block as a dense matrix 
// to solve the linear systems in each step faster.
struct DenseBlock : public Block {
    // GradedDenseMatrix matrix;
    DenseBlock(const indvec& r, const indvec& c) : Block(r,c) {}
    // DecBlock(const DecBlock& b) : Block(b.rows,b.columns), matrix(b.matrix) {}

};


// Question(Jan): Is an std::list the best thing to do this? 
typedef std::list<Block> Block_list;
typedef Block_list::iterator Block_iterator;

void initialise_block_list(const GradedMatrix& A, Block_list& B_list, vec<Block_list::iterator>& block_map) {
    B_list.clear();
    for(int i=0; i<A.get_num_rows(); i++) {
        Block B({},{i});
        auto it = B_list.insert(B_list.end(), B);
        block_map.push_back(it);
        (*it).indecomp = Sparse_Matrix(0,1);
        (*it).indecomp._rows = vec<indvec>(1);
    }
}

void print_block_list_status(Block_list& B_list, GradedMatrix& A) {
    std::cout << "Status: " << B_list.size() << " blocks:\n";
    index count=0;
    for(Block& b : B_list) {
      std::cout << "Block " << count++ << ":" << std::endl;
      std::cout << "Generators at: ";
      for(int i : b.rows) {
	    std::cout << "(" << A.row_degrees[i].first << ", " << A.row_degrees[i].second << ") ";
      }
      std::cout << "\nRelations at: ";
      for(int i : b.columns) {
	    std::cout << "(" << A.col_degrees[i].first << ", " << A.col_degrees[i].second << ") ";
      }
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
void extend_block(Block& B, Sparse_Matrix& N, indvec batch_indices, bitset& batch_positions) {
    // std::cout << "Extending block " << B.rows.front() << " by " << batch_positions.size() << " with indices ";
    if(batch_positions.empty()){
        batch_positions = bitset(N.num_cols, true);
    }
    
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        // std::cout << batch_indices[i] << " ";
        B.columns.push_back(batch_indices[i]);
        B.indecomp.data.push_back(N.data[i]);
        auto it = N.data[i].begin();
        for(index j = 0; j < B.rows.size() && it != N.data[i].end() ; j++){
            if(*it == B.rows[j]){
                B.indecomp._rows[j].push_back(batch_indices[i]);
                it++;
            }
        }   
        B.indecomp.num_cols++;
    }
    // std::cout << std::endl;
    B.rows_computed = true;
}

/**
 * @brief Extends the block B by the columns of N given by the batch_indices.
 * 
 * @param B 
 * @param N 
 * @param batch_indices 
 */
void extend_block(Block& B, Sparse_Matrix& N, indvec batch_indices){
    // std::cout << "Extending block " << B.rows.front() << " with col indices " << batch_indices << std::endl;
    for(index i = 0; i < batch_indices.size(); i++){
        B.columns.push_back(batch_indices[i]);
        B.indecomp.data.push_back(N.data[i]);
        B.indecomp.num_cols++;
    }
    B.print(false);
    B.rows_computed = false;
}

using row_pair = std::pair<index, Block_list::iterator>; 
/**
 * @brief Returns *lhs < *rhs
 * 
 */
struct compare_row_pair {
    bool operator()(const row_pair& lhs, const row_pair& rhs) const {
        return lhs.second->rows[lhs.first] > rhs.second->rows[rhs.first];
    }
};

/**
 * @brief Merges the content of all blocks and N_, restricted to the batch indices given by batch_positions given by block_indices, into a new block.
 * 
 * @param block_indices 
 * @param block_map 
 * @param new_block 
 * @param N_map 
 * @param batch_positions 
 * @param batch_indices 
 */
void merge_blocks_into_block(indtree& block_indices, vec<Block_list::iterator>& block_map, Block& new_block, 
                            std::unordered_map<index, Sparse_Matrix>& N_map, bitset& batch_positions, indvec batch_indices) {
    
                               

    std::priority_queue<row_pair, vec<row_pair>, compare_row_pair> minHeap;

    // maps the initial index of the block to a vector containing the pairs: <batch index, iterator to the associated column>
    std::map<index, vec<std::pair<index, indvec::iterator>> > N_iterators;
    #if DETAILS
        std::cout << "Merging the following blocks:";
        for(index i_0 : block_indices){
             std::cout << " " << i_0;
        }
        std::cout << std::endl;
    #endif
    for(index i : block_indices){
        auto B = block_map[i];
        minHeap.push({0, B});
        new_block.columns.insert(new_block.columns.end(), B->columns.begin(), B->columns.end());
        new_block.indecomp.data.insert(new_block.indecomp.data.end(), B->indecomp.data.begin(), B->indecomp.data.end());
    }
        
    

    index batch_threshold = new_block.columns.size();

    new_block.rows.reserve(2*new_block.columns.size());
    new_block.indecomp._rows.reserve(2*new_block.columns.size());

    // Add columns (unsorted for now, if not needed otherwise) and initialise all iterators to the columns of N.
    for(auto i = batch_positions.find_first(); i != bitset::npos; i = batch_positions.find_next(i)){
        new_block.columns.push_back(batch_indices[i]);
        new_block.indecomp.data.push_back(indvec());
        for(index j : block_indices){
            if( N_map[j].data[i].empty() ){
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            } else {
                N_iterators[j].push_back({batch_indices[i], N_map[j].data[i].begin()});
            }
        }
    }
    
    // The minheap sorts the row-indices of the blocks in ascending order. 
    // Iteratively, we add the this row index, the associated row from the block, and append entries, if the columns of N permit us.
    // Maybe this is much slower than simply sorting everything, I do not know.
    index row_counter = 0;
    while (!minHeap.empty()) {
        auto current = minHeap.top();
        Block& B = *current.second;
        minHeap.pop();
        new_block.rows.push_back(B.rows[current.first]);
        new_block.indecomp._rows.push_back(B.indecomp._rows[current.first]);
        auto& itvec = N_iterators[B.rows.front()];
        for(index i = 0; i < itvec.size(); i++){
            if( itvec[i].second == N_map[B.rows.front()].data[i].end() ){
                continue;
            }
            if( *itvec[i].second == B.rows[current.first]){
                new_block.indecomp.data[batch_threshold + i].push_back(*itvec[i].second);
                new_block.indecomp._rows[row_counter].push_back(itvec[i].first);
                itvec[i].second++;
            }
        }
        if (current.first + 1 < B.rows.size()) {
            minHeap.push({current.first + 1, current.second});
        }
        row_counter++;
    }

    assert(new_block.columns.size() == new_block.indecomp.data.size());
    new_block.indecomp.num_cols = new_block.indecomp.data.size();
    new_block.rows_computed = true;
    // std::cout << "Merged block has rows " << new_block.rows << std::endl;
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
void merge_blocks(GradedMatrix& A, Block_list& B_list, std::unordered_map<index, Sparse_Matrix>& N_map, 
                    vec<Block_list::iterator>& block_map, std::list<Merge_data>& block_partition, indvec& batch_indices) {
    for(auto& partition : block_partition){
        indtree& block_indices = partition.first;
        bitset& batch_positions = partition.second; 
        auto fit = block_indices.begin();
        index first = *fit;
        if(block_indices.size() == 1){
            extend_block(*block_map[first], N_map[first], batch_indices, batch_positions);
        } else {
            Block new_block;
            auto new_it = B_list.insert(B_list.end(), new_block);
            merge_blocks_into_block(block_indices, block_map, *new_it, N_map, batch_positions, batch_indices); 

            for(index i : block_indices){
                auto del_it = block_map[i];
                for(index j : block_map[i]->rows){
                    block_map[j] = new_it;
                }
                B_list.erase(del_it);
                // std::cout << std::endl;
            }
        }          
    }
}

/**
 * @brief Fills c with the linearised entries of N_B restricted by a bitset.
 * 
 */
void linearise_prior( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>> Ns, indvec& batch_indices, indvec& result, bitset& sub_batch_indices) {
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
}

/**
 * @brief Fills c with the linearised entries of N_B using the full batch.
 * 
 */
void linearise_prior( GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>> Ns, indvec& batch_indices, indvec& result){
    for(auto& ref : Ns){
        Sparse_Matrix& N = ref.get();
        for(index i = 0; i < batch_indices.size(); i++){
            for(index j : N.data[i]){
                result.push_back(A.linearise_position_reverse(batch_indices[i], j));
            }
        }
    }
}

void test_rows_of_A(GradedMatrix& A, index batch){
    for(index i = 0; i < A.num_rows; i++){
        if(!A._rows[i].empty()){
            for(index j : A._rows[i]){
                if(j < batch){
                    std::cout << "Warning: The row " << i << " has an entry smaller than the current batch: " << j << std::endl;
                }
                if(j > A.num_cols){
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
void construct_linear_system(GradedMatrix& A, std::vector<std::reference_wrapper<Sparse_Matrix>> Ns, indvec& sub_batch_indices, bool restricted_batch,
                            indtree& relevant_blocks, vec<Block_list::iterator>& block_map, 
                            Sparse_Matrix& S, vec< op_info >& ops, 
                            indtree& b_vec, std::unordered_map<index, Sparse_Matrix>& N_map,
                            const bitset& extra_columns = bitset(0)){
    
    //TO-DO: Check if all admissible row-operations from C to B do not change N_B and in that case dont consider them.

    bool has_effect_on_N; // Not needed yet.
    
    
    //TO-DO: this can be further optimised as follows: If a block C turns out to contribute 
    // and the associated row-operation would change not only N, but also the area B.rows*C.cols,
    // then there need to be admissible non-zero operations from B.col to C.col to balance this operations
    // and we can skip the row-operation otherwise. (This is essentially a test for Hom^{omega}(C, B) = 0)

    //TO-DO: Parallelise this whole subroutine.

    Block& B_first = *block_map[*b_vec.begin()];  
    Block& B_probe = *block_map[*relevant_blocks.begin()];
    // TO-DO: Got a bad alloc in the following line, so in fact this is too large.
    // When the bad alloc happend  we were trying to allocate > 7* 10^6 entries. 
    size_t buffer = 0;
    // Is this a good number?
    size_t max_buffer_size = 200000;
    if(B_first.local_data != nullptr){
        buffer = B_first.local_data->data.size();
    }
    buffer += b_vec.size()*relevant_blocks.size()*(B_first.rows.size()*B_probe.rows.size() 
    + B_probe.columns.size()*B_first.columns.size());
    buffer = std::min(buffer, max_buffer_size);
    S.data.reserve(buffer);
    index S_index = 0;

    // First find all blocks which can actually contribute by having a non-zero admissible row operation to any row of B:
    // While doing that, construct the associated columns of S belonging to these row operations.
    indtree admissible_relevant_blocks;

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

            #if DETAILS
                std::cout << "Checking row ops from block " << c << " to block " << b << std::endl;
            #endif

            Block& C = *block_map[c];
            Sparse_Matrix& N_C = N_map[c];
            for(index i = 0; i < C.rows.size(); i++){
                for(index j = 0; j < B.rows.size(); j++){
                    auto source_index = C.rows[i];
                    auto target_index = B.rows[j];
                    if(A.is_admissible_row_operation(source_index, target_index)){
                        #if DETAILS
                            std::cout << "Found admissible row operation from " << source_index << " to " << target_index << std::endl;
                            std::cout << "Degrees: " << A.row_degrees[source_index].first << " " << A.row_degrees[source_index].second << " to " 
                            << A.row_degrees[target_index].first << " " << A.row_degrees[target_index].second << std::endl; 
                        #endif
                        S.data.push_back(indvec());
                        ops.emplace_back( std::make_pair(std::make_pair(i , j), std::make_pair(c, b)) );

                        // Fill the column of S belonging to the operation first with the row in N_C, 
                        // then with the row in C.indecomp, so that no sorting is needed.

                        auto row_it = N_C._rows[i].rbegin();
                        auto batch_it = sub_batch_indices.rbegin();
                        while(row_it != N_C._rows[i].rend()){
                            // If there is no restriction, then we can use all the entries of the row.
                            if(!restricted_batch || *row_it == *batch_it){ 
                                S.data[S_index].emplace_back(A.linearise_position_reverse(*row_it, target_index));
                                row_it++;
                                batch_it++;
                            } else if (*row_it > *batch_it){
                                row_it++;
                            } else {
                                batch_it++;
                            }
                        }    

                        // Now the content of C.indecomp

                        

                        if(!C.indecomp._rows[i].empty()){


                            for(auto row_it2 = C.indecomp._rows[i].rbegin(); row_it2 != C.indecomp._rows[i].rend(); row_it2++){
                                // only insert if the row operation has an effect on B.rows*C.columns.

                                if(!no_new_inserts){
                                    // std::cout << "Inserting " << c << " into admissible_relevant_blocks" << std::endl;
                                    auto result = admissible_relevant_blocks.insert(c);
                                    no_new_inserts = result.second;
                                }
                                auto effect_position = A.linearise_position_reverse(*row_it2, target_index);
                                S.data[S_index].emplace_back(effect_position);
                            }
                        }
                        S_index++;
                    } 
                }
            }
            no_new_inserts = false;
        }
    
    

        // Next add all col ops from all blocks in b_vec to the columns of the blocks which could contribute.
        // std::cout << "Adding non-basic column operations " << std::endl;
        for(index c : admissible_relevant_blocks){
            auto it = block_map[c];
            Block& C = *it;
            for(index i = 0; i < C.columns.size(); i++){
                for(index j = 0; j < B.columns.size(); j++){
                    if(A.is_admissible_column_operation(B.columns[j], C.columns[i])){
                        S.data.push_back(indvec());
                        for(index row_index : B.indecomp.data[j]){
                            S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[i], row_index));
                        }
                        S_index++;
                    }
                }
            }
        }


        // At last, add the basic column-operations from B to N which have already been computed
        // std::cout << "Adding basic column operations." << std::endl;
        // TO-DO: This doesnt work yet, somehow local data is an empty matrix instead of having a nullptr.
        if(B.local_data != nullptr){
            for(index i : sub_batch_indices){
                for(indvec& column : (*B.local_data).data){
                    for(index j : column){
                        S.data.push_back(indvec());
                        S.data[S_index].emplace_back(A.linearise_position_reverse(i, j));
                        S_index++;
                    }
                }
            }
        }
    }

} //construct_linear_system


/**
 * @brief Use column-operations to try and delete the block (A_t)_B
 * 
 * @param A 
 * @param B 
 * @param N
 * @param col_indices 
 */
bool delete_with_col_ops(GradedMatrix& A, Block& B, indvec batch_indices, Sparse_Matrix& N) {

    B.compute_local_data(A, batch_indices[0]);

    // Here I want the column-reduction to be performed in-place:
    return B.reduce_N(N);
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
 * @param extra_columns if naive_first is true, this is the set of columns-indices of the batch which belong to the second subspace tested in naive decomposition.
 *                      If naive_first is false, this is the set of columns-indices of the batch which belong to the first subspace tested in naive decomposition.
 *                      It should be empty if block reduce is not called from naive decomposition.
 */                     
bool block_reduce(GradedMatrix& A, indtree& b_vec, std::unordered_map<index, Sparse_Matrix>& N_map, indvec& batch_indices,
                bool restricted_batch, indtree& relevant_blocks, vec<Block_iterator>& block_map, bitset& sub_batch_indices, vec<pair>& performed_row_ops,
                cpu_timer& timer_1_1_block, cpu_timer& linearise_timer, cpu_timer& constructing_linear_system_timer, cpu_timer& solve_linear_system_timer, cpu_timer& dispose_S_timer, cpu_timer& update_matrix_timer,
                const bitset& extra_columns = bitset(0), bool naive_first = false, bool suppress_output = true) {

    std::vector<std::reference_wrapper<Sparse_Matrix>> Ns;
    for(index i : b_vec){
        Ns.push_back(N_map[i]);
    }

    #if DETAILS
        std::cout << "  Block_reduce called on (b_vec) ";
        for(index b : b_vec){ std::cout << b << " ";}
        std::cout << std::endl;
        std::cout << "      Ns: \n";
        for(auto& ref : Ns){
            ref.get().print();
        }
        std::cout << "      batch_indices: " << batch_indices << std::endl;
        if(restricted_batch){
        std::cout << "      sub_batch_indices: " << sub_batch_indices << std::endl;
        }
    #endif

    auto& B_first = *block_map[*b_vec.begin()];

    bool reduced_to_zero = false;

        bool is_1_1_block = (B_first.rows.size()==1  && B_first.columns.size()==1);
        if(is_1_1_block) {
            #if TIMERS
                timer_1_1_block.resume();
            #endif
            index row_id = B_first.rows.back();
            index col_id = B_first.columns.back();
            if(!suppress_output){
                std::cout << "1-1-Block: " << row_id << " " << col_id << std::endl;
            }
            //TODO: Implement this (if there is a way to do this efficiently)

            #if TIMERS
                timer_1_1_block.stop();
            #endif
        }


    #if TIMERS
        linearise_timer.resume();
    #endif

    indvec c;        
    // Uses the Columns of Ns
    // Probably this doesnt really make a difference:
    if(restricted_batch){
        linearise_prior(A, Ns, batch_indices, c, sub_batch_indices);
    } else {
        linearise_prior(A, Ns, batch_indices, c);
    }

    #if TIMERS
        linearise_timer.stop();
    #endif
        
    #if TIMERS    
        constructing_linear_system_timer.resume();
    #endif

    indvec sub_batch_indices_alt;
    if(restricted_batch){
        for(index i = sub_batch_indices.find_first(); i != bitset::npos; i = sub_batch_indices.find_next(i)){
            sub_batch_indices_alt.push_back(batch_indices[i]);
        }
    } 

    Sparse_Matrix S(0,0);
    // Only need to store row-operations. This vector has as many entries as columns of S are associated to row-operations.
    // only uses the _rows of N.
    vec< op_info> ops;
    if(restricted_batch){
        construct_linear_system(A, Ns, sub_batch_indices_alt, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map, extra_columns);
    } else {
        construct_linear_system(A, Ns, batch_indices, restricted_batch, relevant_blocks, block_map, S, ops, b_vec, N_map, extra_columns);
    }

    index row_op_limit = ops.size();

    #if TIMERS
        constructing_linear_system_timer.stop();
    #endif    

    #if TIMERS
        solve_linear_system_timer.resume();
    #endif

    index threshold = 0; // Right now, we do not use this.
    indvec solution;
    S.num_cols = S.data.size();
    #if DETAILS
        std::cout << "Solving the following linear system." << std::endl;
        S.print();
        std::cout << "c: " <<  c << std::endl;
    #endif
    reduced_to_zero = S.solve_with_col_reduction(c, threshold, solution);

    #if TIMERS
        solve_linear_system_timer.stop();
    #endif


    // Perform all row-operations and clear N_B if the system could be solved
    if(reduced_to_zero){
        #if TIMERS
            update_matrix_timer.resume();
        #endif


        for(index operation_index : solution){
            // std::cout << "Solution index: " << operation_index;
            if(operation_index >= row_op_limit){
                // std::cout << " does not belong to a row-operation." << std::endl;
            } else {
                auto op = ops[operation_index];
                auto& B_source = *block_map[op.second.first];
                auto& B_target = *block_map[op.second.second];
                // std::cout << " belongs to a row-operation: " << B_source.rows[op.first.first] << " " << B_target.rows[op.first.second] << std::endl;
                performed_row_ops.push_back( std::make_pair(B_source.rows[op.first.first], B_target.rows[op.first.second]));
                A.fast_rev_row_op(B_source.rows[op.first.first], B_target.rows[op.first.second]);
                if(restricted_batch && naive_first == true){
                    auto& N_source = N_map[op.second.first];
                    auto& N_target = N_map[op.second.second];
                    // TO-DO: Here we only change _rows of N. We should also change the columns/data. 
                    // We also dont need to change the part of N which we are looking at, because it can be reduced to zero with the column operations.
                    add_to(N_source._rows[op.first.first], N_target._rows[op.first.second]);
                    auto it = N_source._rows[op.first.first].begin();
                    for(index col_index = extra_columns.find_first(); col_index != bitset::npos; col_index = extra_columns.find_next(col_index)){
                        if(*it == batch_indices[col_index]){
                            // This meanse that the row-operation will change the part of N which we are not looking at, 
                            // but will have to next, so that a change to data is necessary.
                            // Need to test if this happens often, because right now this here -> is too costly.
                            auto p = std::find( N_source.data[col_index].begin(), N_source.data[col_index].end(), B_target.rows[op.first.second] );
                            if( p != N_source.data[col_index].end() ){
                                N_source.data[col_index].erase(p);
                            } else {
                                N_source.data[col_index].push_back(B_source.rows[op.first.first]);
                                std::sort(N_source.data[col_index].begin(), N_source.data[col_index].end());    
                            }
                        }
                    }
                    // This does not delete the parts of the rows which this subroutine is supposed to delete. 
                    // We can pretend that they are gone or we can actually delete them?
                }
            }
        }

        test_rows_of_A(A, batch_indices.back());

        #if TIMERS
            update_matrix_timer.stop();
        #endif  
    }

    #if TIMERS    
        dispose_S_timer.resume();
    #endif

    return reduced_to_zero;
} //Block_reduce



/**
 * @brief This computes a column-form of the entries in the batch, splits it up into the blocks which are touched by the batch, and stores the information.
 * 
 * @param active_blocks Stores the touched blocks.
 * @param block_map 
 * @param A the matrix
 * @param batch 
 */
void compute_touched_blocks(indtree& active_blocks, vec<Block_iterator>& block_map, 
                            GradedMatrix& A, indvec& batch, std::unordered_map<index, Sparse_Matrix>& N_map) {

    for(index j = 0; j < batch.size(); j++){
        index bj = batch[j];
        // Q: Is this the fastest way to do it? It is possible we want a specific sorting function.
        std::sort(A.data[bj].begin(), A.data[bj].end());
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
                N_map[initial].data = vec<indvec>(batch.size(), indvec());
            }
            N_map[initial].data[j].push_back(i);
            // Maybe we should also store the rows, but not sure right now.
            if( A._rows[i].back() != bj){
                std::cout << "Warning at row: " << i << ". " << A._rows[i].back() << " != " << bj << std::endl;
            }
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
std::list<std::pair< indtree , bitset>> naive_decomposition(
                        GradedMatrix& A, Block_list& B_list, vec<Block_iterator>& block_map, 
                        indtree& pierced_blocks, indvec& batch_indices, const bitset& N_column_indices, 
                        vec<bitset>& e_vec, std::unordered_map<index, Sparse_Matrix>& N_map,
                        vec<vec<transition>>& vector_space_decompositions, vec<pair>& performed_row_ops,
                        cpu_timer& timer_1_1_block, cpu_timer& linearise_timer, cpu_timer& constructing_linear_system_timer, cpu_timer& solve_linear_system_timer, cpu_timer& dispose_S_timer, cpu_timer& update_matrix_timer) {
                           
    //TO-DO: Implement a branch and bound strategy such that we do not need to iterate over those decompositions
    //TO-DO: Need to first decompoose/reduce the left-hand columns, create an updated temporary block-merge and then decompose the right-hand columns.
    int k = N_column_indices.count();    
    int num_b = pierced_blocks.size();

    // std::cout << "!!!   calling naive_decomposition with " << k << " columns and " << 
    //    num_b << " blocks at N_column_indices: " << N_column_indices << std::endl;
    if(k != 1){
        // Iterate over all decompositions of GF(2)^k into two subspaces.
        // TO-DO: Implement a branch and bound strategy such that we do not need to iterate over those decompositions 
        // again which have already been tried.
        for(auto transition : vector_space_decompositions[k-2]){
            auto& basechange = transition.first;
            auto& partition_indices = transition.second;
            bitset indices_1 = partition_indices;
            bitset indices_2 = partition_indices;
            indices_2.flip();
            // std::cout << "  Indices 1: " << indices_1 << " Indices 2: " << indices_2 << std::endl;
            indtree blocks_1;
            indtree blocks_2;
            indtree blocks_conflict;
            for(index b : pierced_blocks){
                // TO-DO: This is inefficient. We might want to store N as a dense matrix.
                N_map[b].multiply_dense_with_e_check(basechange, e_vec, N_column_indices);
                Block& B = *block_map[b];
                N_map[b].compute_rows_forward(batch_indices, B.rows);
            }

            for(index b : pierced_blocks){
                if(N_map[b].is_zero(indices_1)){
                    blocks_2.insert(b);
                } else if (N_map[b].is_zero(indices_2)){
                    blocks_1.insert(b);
                } else {
                    blocks_conflict.insert(b);
                    blocks_1.insert(b);
                    blocks_2.insert(b);
                }
            }

            bool conflict_resolution = true;

            // Optimisation step: If we could not delete the lhs of a block in conflict, then try to delete the rhs without all operations. 
            // Only stop doing this if this does not work for one block because in this case we will have to try to delete all of these blcoks at once anyways.

            if (blocks_conflict.size() > 0){
                // std::cout << "Conflict in naive decomposition. Trying to resolve directly." << std::endl;
                for(auto itr = blocks_conflict.rbegin(); itr != blocks_conflict.rend();){
                    index b = *itr;
                    indtree b_vec = {b};
                    bool deleted_N1 = block_reduce(A, b_vec, N_map, batch_indices, true, blocks_1, block_map, indices_1, performed_row_ops,
                                                    timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer
                                                    );
                    #if TIMERS
                        dispose_S_timer.stop();
                    #endif
                    if(deleted_N1){
                        blocks_1.erase(b);
                        auto it = blocks_conflict.erase(--itr.base());
                        itr = std::reverse_iterator(it); 
                    } else if ( conflict_resolution ) {
                        bool deleted_N2 = block_reduce(A, b_vec, N_map, batch_indices, true, blocks_2, block_map, indices_2, performed_row_ops,
                                                        timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer
                                                        );
                        #if TIMERS
                            dispose_S_timer.stop();
                        #endif
                        if(deleted_N2){
                            blocks_2.erase(b);
                            auto it = blocks_conflict.erase(--itr.base());
                            itr = std::reverse_iterator(it);
                            
                        } else {
                            conflict_resolution = false;
                            itr++;
                        }
                    } else {
                        itr++;
                    
                    }
                }
            }

            if(!conflict_resolution){
                // We should not need to look again at those blocks which where in blocks_conflict once, since everything was being tried for them.
                // std::cout << "Conflict could not be resolved. First reducing N_1 as much as possible. (Should count this?)" << std::endl;
                for(auto itr = blocks_1.rbegin(); itr != blocks_1.rend();){
                    index b = *itr;
                    indtree b_vec = {b};
                    bool deleted_more_1 = block_reduce(A, b_vec, N_map, batch_indices, true, blocks_1, block_map, indices_1, performed_row_ops,
                                timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer
                                );
                    #if TIMERS
                        dispose_S_timer.stop();
                    #endif
                    if(deleted_more_1){
                        auto it = blocks_1.erase(--itr.base());
                        itr = std::reverse_iterator(it);    
                        // std::cout << "Could delete more blocks with the help of the blocks in conflict. (Should count this.)" << std::endl;
                    } else {
                        itr++;
                    }
                }
                assert( blocks_conflict.size() > 0);
                // Now need to treat all of blocks_1 as one block and delete its rhs of N.
                conflict_resolution = block_reduce(A, blocks_1 , N_map, batch_indices, false, pierced_blocks, block_map, indices_2, performed_row_ops,
                        timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer,
                        N_column_indices);
            }
            
            if (conflict_resolution) {
                // A valid decomposition has been found. Continue here.
                auto partition_1 = naive_decomposition(A, B_list, block_map, blocks_1, batch_indices, indices_1, e_vec, N_map, vector_space_decompositions, performed_row_ops,
                                                        timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer);
                auto partition_2 = naive_decomposition(A, B_list, block_map, blocks_2, batch_indices, indices_2, e_vec, N_map, vector_space_decompositions, performed_row_ops,
                                                        timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer);
                partition_1.splice(partition_1.end(), partition_2);
                return partition_1;
            }
        }
    }
    // If there was no decomposition of this batch:
    std::list< Merge_data > result;
    result.emplace_back( make_pair(pierced_blocks, N_column_indices) );  
    return result;
} // naive_decomposition

/**
 * @brief 
 * 
 * Number of iterations for k = 1, 2, 3, 4, 5, 6, 7, 8, ...: per batch is
 * 1, 2, 7, 43, 186, 1965, 14605, 297181, ... 
 * 
 * @param A 
 * @param B_list 
 * @param vector_space_decompositions 
 */
void decompose(GradedMatrix& A, Block_list& B_list, vec<vec<transition>> vector_space_decompositions) {
    
    index batches = A.col_batches.size();
    std::cout << "Decomposing matrix with " << A.get_num_rows() << " rows and " << A.get_num_cols() << " columns." << std::endl;
    std::cout << "k_max is " << A.k_max << " and there are " << batches << " batches." << std::endl;
    // A.print_graded();

    #if ANALYSIS
        edgeList column_edges = minimal_directed_graph<degree, index>(A.col_degrees);
        edgeList row_edges = minimal_directed_graph<degree, index>(A.row_degrees);
        std::cout << "Column graph: " << std::endl;
        print_edge_list(column_edges);
        std::cout << "Row graph: " << std::endl;
        print_edge_list(row_edges);
    #endif

    #if TIMERS
        cpu_timer full_timer;
        full_timer.start();
        cpu_timer timer_1_1_block;
        timer_1_1_block.stop();
        cpu_timer linearise_timer;
        linearise_timer.stop();
        cpu_timer constructing_linear_system_timer;
        constructing_linear_system_timer.stop();
        cpu_timer solve_linear_system_timer;
        solve_linear_system_timer.stop();
        cpu_timer dispose_S_timer;
        dispose_S_timer.stop();
        cpu_timer update_matrix_timer;
        update_matrix_timer.stop();
    #endif

    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();
    auto& _rows = A._rows;

    indvec num_subspace_iterations = {1, 2, 7, 43, 186, 1965, 14605, 297181};

    // Only for optimisation purposes
    index counter_no_comp = 0;
    index counter_only_col = 0;
    index counter_only_row = 0;
    indvec only_row_indices;
    index counter_naive_deletion = 0;
    index counter_naive_full_iteration = 0;
    index counter_extra_iterations = 0;
    vec<pair> performed_row_ops;
    index counter_col_deletion = 0;
    index counter_row_deletion = 0;

    vec<Block_iterator> block_map;
    initialise_block_list(A, B_list, block_map);

    std::unordered_map<index, Sparse_Matrix> N_map;
    
    vec<bitset> e_vec = compute_standard_vectors(A.k_max);
    vec<bitset> count_vector = compute_sum_of_standard_vectors(A.k_max);


    A.compute_revrows();
    // A.print_rows();

    bool no_further_comp = false;
    bool no_row_dels = true;
    indvec num_of_merges_at_t;

    
    // Iterate over all batches
    for(index t = 0; t < batches; t++){
        
        #if !DETAILS
            static index last_percent = -1;
            index percent = static_cast<index>((static_cast<double>(t) / batches) * 100);
            if (percent % 2 == 0 && percent != last_percent) {
                // Calculate the number of symbols to display in the progress bar
                int num_symbols = percent / 2;
                std::cout << "\rProgress: [";

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
        #endif
        int k_ = A.k_vec[t]; // Number of columns in the batch
        indvec batch_indices = A.col_batches[t]; // Indices of the columns in the batch
        #if DETAILS
            std::cout << std::endl;
            std::cout << "Processing batch " << t << " with " << k_ << " columns at the indices " << batch_indices << std::endl;
        #endif
        N_map.clear();
        no_further_comp = false;

        // Get the batch as a set of columns from the rows and identify the blocks which need to be processed.
        indtree active_blocks;

        // Question: Which datatype is best here? Maybe I want a vector? This stores N_B for each block B.
        // Another Question: How to tell the compiler to allocate enough memory here 
        // This could be optimised a lot: The algorithm could store somewhere which row operations have been done and how they change the columns without performing these column operations.
        compute_touched_blocks(active_blocks, block_map, A, batch_indices, N_map); 

        #if DETAILS
             std::cout << "  !! There are "  << active_blocks.size() << " touched blocks with the following initial indices: ";
            for(index i : active_blocks){
                std::cout << i << " ";
            }
            std::cout << std::endl;
        #endif
        // First try to delete every whole sub-batch only with column operations. That is, compute the *affected* blocks.
        std::list<Merge_data> block_partition = {};


        if(active_blocks.size() == 1){
            counter_no_comp++;
            block_partition = {{{*active_blocks.begin()}, count_vector[k_-1]}};
            no_further_comp = true;
        } else {
            for(auto it = active_blocks.begin(); it != active_blocks.end();){
                index j = *it;
                Block& B = *block_map[j];
                // No need to do anyhting here if the block is empty. Careful though, there will be no local data in later steps.
                    if(B.columns.size() == 0){
                        it++;
                        B.local_data = std::make_shared<Sparse_Matrix>(0,0);
                        N_map[j].compute_rows_forward(batch_indices, B.rows);
                        continue;}
                auto& N = N_map[j];
                bool only_col_ops = delete_with_col_ops(A, B, batch_indices, N);
                if(only_col_ops){
                    #if DETAILS
                        std::cout << "      Successfully deleted N at index " << j << " with column ops." << std::endl;
                    #endif
                    counter_col_deletion++;
                    B.delete_local_data();
                    N_map.erase(j);
                    it = active_blocks.erase(it);
                } else {
                    it++;
                    // If the block could not be deleted with column operations, then it is still active.
                    // We will need its row-information later.
                    B.compute_rows(); 
                    N_map[j].compute_rows_forward(batch_indices, B.rows);
                }
            }
        }

        // TO-DO: We should go through the blocks in reverse order here to reduce the number of row-operations and decrease the size of S:
        indtree pierced_blocks = indtree();
        if(active_blocks.size() == 1 && !no_further_comp){
            counter_only_col++;
            no_further_comp = true;
            block_partition = {{{*active_blocks.begin()}, count_vector[k_-1]}};
        } else if ( !no_further_comp ) {
            #if DETAILS
                std::cout << "   !! There are "  << active_blocks.size() << " affected blocks with the following indices: ";
                for(index i : active_blocks){
                   std::cout << i << " ";
                }
                std::cout << std::endl;
            #endif
            // Then with the rest, if there is more than one left, try also with row-operations.
            for(auto itr = active_blocks.rbegin(); itr != active_blocks.rend();) {
                index b = *itr;
                indtree b_vec = {b};
                Block_iterator bit = block_map[b];
                Block& B = *bit;
                // TODO: This Function should store the matrices it generates for later. 
                #if DETAILS
                    std::cout << "      Trying to delete N at index " << b << " with row ops" << std::endl;
                #endif
                bool block_reduce_result = block_reduce(A, b_vec, N_map, batch_indices, false, active_blocks, block_map, count_vector[k_-1], performed_row_ops,
                                                        timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer
                                                        );
                #if TIMERS
                    dispose_S_timer.stop();
                #endif
                if(block_reduce_result){
                    #if DETAILS 
                        std::cout << " - successfully." << std::endl;
                    #endif
                    counter_row_deletion++;
                    no_row_dels = false;
                    B.delete_local_data();
                    N_map.erase(b);
                    auto it = active_blocks.erase(--itr.base());
                    itr = std::reverse_iterator(it);
                } else {
                    itr++;
                    no_row_dels = true;
                    #if DETAILS
                        std::cout << " - unsuccessfully." << std::endl;
                    #endif
                    pierced_blocks.insert(b);
                }
            }
        } 

        if( k_ == 1 && !no_further_comp ){
            if(no_row_dels){
                counter_only_col++;
                no_further_comp = true;
                block_partition = {{pierced_blocks, count_vector[k_-1]}};
            } else {
                counter_only_row++;
                only_row_indices.push_back(t);
                no_further_comp = true;
                block_partition = {{pierced_blocks, count_vector[k_-1]}};
            }
        } else if((pierced_blocks.size() == 1 || k_ == 1) && !no_further_comp){
            counter_only_row++;
            only_row_indices.push_back(t);
            no_further_comp = true;
            block_partition = {{pierced_blocks, count_vector[k_-1]}};
        } else if ( !no_further_comp) {
            // std::cout << "There are "  << pierced_blocks.size() << " pierced blocks with the following indices: ";
            // for(index i : active_blocks){
            //     std::cout << i << " ";
            // }
            
            // Could check if there is already an "almost" decomposition before calling the loop.

            // std::cout << std::endl;
            block_partition = naive_decomposition(A, B_list, block_map, active_blocks, batch_indices, count_vector[k_-1], 
                                                    e_vec, N_map, vector_space_decompositions, performed_row_ops,
                                                    timer_1_1_block, linearise_timer, constructing_linear_system_timer, solve_linear_system_timer, dispose_S_timer, update_matrix_timer);
            if(block_partition.size() == 1){
                counter_naive_full_iteration++;
                counter_extra_iterations += num_subspace_iterations[k_-1];
            } else {
                counter_naive_deletion++;
            }
        }
        // Deleting remaining local data.
        for(index i : active_blocks){
            block_map[i]->delete_local_data();
        }
        merge_blocks(A, B_list, N_map, block_map, block_partition, batch_indices);
        num_of_merges_at_t.push_back(block_partition.front().first.size()-1);
    }
    full_timer.stop();
    std::cout << "Finished decomposition. Number of indecomposables: " << B_list.size() << std::endl;
    std::cout << "Statistics: " << std::endl;
    std::cout << "  Total number of merges: " << std::accumulate(num_of_merges_at_t.begin(), num_of_merges_at_t.end(), 0) << std::endl;
    std::cout << "  No computation: " << counter_no_comp << std::endl;
    std::cout << "  Only column operations: " << counter_only_col << std::endl;
    std::cout << "  Only row operations: " << counter_only_row << std::endl;
    std::cout << "  Naive deletion: " << counter_naive_deletion << std::endl;
    std::cout << "  Naive full iteration: " << counter_naive_full_iteration << std::endl;
    std::cout << "  Column deletion: " << counter_col_deletion << std::endl;
    std::cout << "  Row deletion: " << counter_row_deletion << std::endl;
    std::cout << "  Extra iterations: " << counter_extra_iterations << std::endl;
    std::cout << "  Linearise timer: " << linearise_timer.elapsed().wall/1e9 << "s" << std::endl;
    std::cout << "  Constructing linear system timer: " << constructing_linear_system_timer.elapsed().wall/1e9 << "s" << std::endl;
    std::cout << "  Solve linear system timer: " << solve_linear_system_timer.elapsed().wall/1e9 << "s" << std::endl;
    std::cout << "  Dispose S timer: " << dispose_S_timer.elapsed().wall/1e9 << "s" << std::endl;
    std::cout << "  Update matrix timer: " << update_matrix_timer.elapsed().wall/1e9 << "s" << std::endl;
    std::cout << "  Total time: " << full_timer.elapsed().wall/1e9 << " vs accumulated " << 
    linearise_timer.elapsed().wall/1e9 + constructing_linear_system_timer.elapsed().wall/1e9 + solve_linear_system_timer.elapsed().wall/1e9 + dispose_S_timer.elapsed().wall/1e9 + update_matrix_timer.elapsed().wall/1e9 << "s" << std::endl;
    #if DETAILS
        std::cout << "Full merge details: " << std::endl;
        for(index i = 0; i < num_of_merges_at_t.size(); i++){
            std::cout << "  #Merges at batch " << i << ": " << num_of_merges_at_t[i] << std::endl;
        }
        std::cout << "  checked for row ops at batches: " << only_row_indices << std::endl; 
        std::cout << "  Performed row ops: " << std::endl;
        for(auto& p : performed_row_ops){
            std::cout << "  " << p.first << " -> " << p.second << std::endl;
        }
    #endif
} //decompose

/**
 * @brief Prints all blocks with their rows, columns and content.
 * 
 * @param B_list 
 */
void print_block_list(GradedMatrix& A, Block_list& B_list) {
    std::cout << "Indecomposables: \n";
    for(auto it = B_list.begin(); it != B_list.end(); it++){
        Block& B = *it;
        B.indecomp.reorder_via_comparison(B.columns);
        std::sort(B.columns.begin(), B.columns.end());
        vec<degree> row_degrees;
        for(index i : B.rows){
            row_degrees.push_back(A.row_degrees[i]);
        }
        vec<degree> col_degrees;
        for(index i : B.columns){
            col_degrees.push_back(A.col_degrees[i]);
        }
        B.indecomp.num_rows = B.rows.size();
        B.print();
        B.indecomp.compute_normalisation(B.rows);
        B.indecomp.print();
        
        for(index i = 0; i< col_degrees.size(); i++){
            std::cout << "Degree " << i << " : " << col_degrees[i] << std::endl;
        }

        edgeList E = minimal_directed_graph<degree, index>(row_degrees);
        
        edgeList F = minimal_directed_graph<degree, index>(col_degrees);
        std::cout << "Row graph on " << row_degrees.size() << " vertices: \n";
        print_edge_list(E);
        std::cout << "Column graph on " << col_degrees.size() << " vertices: \n";
        print_edge_list(F);
    }
}

} // namespace aida


int main(int argc, char** argv){

    bool print_output=false;

    std::string test_matrix;
    
    if(argc<2) {
	std::string foldername = "test_presentations/";
	std::string filename1 = "small_pres_with_nonzero_k_min_pres.firep";
	std::string filename2 = "noisy_circle_firep_8_0_min_pres.firep";
	std::string filename3 = "test_pres_with_nonzero_k_min_pres.firep";
	std::string filename4 = "k_fold_46_10_2_min_pres.firep";
	std::string filename5 = "k_fold_175_10_1_min_pres.firep";
	std::string filename6 = "multi_cover_050_10_1_min_pres.scc";
    std::string min_example = "function_delaunay_7_2.scc";
    std::string k1_test = "full_rips_size_1_instance_5_min_pres.scc";
	std::string k1_large_test = "off_dragon_min_pres.scc";
	test_matrix = foldername + k1_large_test;
    } else {
        test_matrix=std::string(argv[1]);
    }
    
    // This fills the matrix with the data from the file and also computes the batches and k_max.
    aida::GradedMatrix A(test_matrix, true);


    // This stores all vector space decompositions for all batch-sizes. 
    // DecompTree: bitset-> vector(vector( Pair of Matrices)), 
    // The first vector indicates the pivots, the second the Plücker Coordinates.

    std::vector<std::vector<graded_linalg::transition>> vector_space_decompositions;

    const std::string tran_path = "lists_of_decompositions/transitions_reduced_";
    const std::string command = "./generate_decompositions -at -cover -transitions ";

    for(int k = 2; k <= A.k_max; k++) {
        if(!aida::fs::exists(tran_path + std::to_string(k) + ".bin")){
            std::cout << "Could not find " << tran_path + std::to_string(k) + ".bin" << std::endl;
            int result = system( (command + std::to_string(k)).c_str() );
        } else {
            try { 
                vector_space_decompositions.emplace_back(aida::load_transition_list( tran_path + std::to_string(k) + ".bin"));
            } catch (std::exception& e) {
                std::cout << "Could not load transitions_reduced_" << k << ".bin " << std::endl;
                abort();
            }
        }
    }

    aida::Block_list B_list;
    aida::decompose(A, B_list, vector_space_decompositions);
    // aida::print_block_list(A, B_list);
    std::cout << "Indecomposable summands: " << B_list.size() << std::endl;
    return B_list.size();
} //main
