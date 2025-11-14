#pragma once

#ifndef AIDA_BLOCK_HPP
#define AIDA_BLOCK_HPP

#include "types.hpp"

namespace aida{


/**
 * @brief Holds the indecomposable summands in the matrix
 * 
 */
struct Block : GradedMatrix {

    vec<index> rows;
    vec<index> columns;

    Block& operator= (const Block& other) {
        if (this != &other) {
            GradedMatrix::operator=(other);
            rows = other.rows;
            columns = other.columns;
        }
        return *this;
    }

    Block& operator= (Block&& other) {
        if (this != &other) {
            GradedMatrix::operator=(std::move(other));
            rows = std::move(other.rows);
            columns = std::move(other.columns);
        }
        return *this;
    }

    Block() : GradedMatrix() {
        rows = vec<index>();
        columns = vec<index>();
    }

    Block(const Block& other) : GradedMatrix(other) {
        rows = other.rows;
        columns = other.columns;
    }
    Block(Block&& other) : GradedMatrix(std::move(other)) {
        rows = std::move(other.rows);
        columns = std::move(other.columns);
    }

    BlockType type; // 0 for free module, 1 for cyclic, 2 for interval, 3 for non-interval

    vec<index> local_admissible_cols; // Stores the indices of the columns which can be used for column operations.
    vec<index> local_admissible_rows; // Stores the actual row_indices which can be used for row operations.
    vec<index> local_basislift_indices; // Stores a set of indices defining a subset of local_admissible_rows which forms a basis at the local r2degree.
    // Careful, these are indices only locally for this block!

    // This will store the columns which can be added to the current batch, i.e. local_data = data | admissible_cols.
    std::shared_ptr<Sparse_Matrix> local_data; 
    std::shared_ptr<Sparse_Matrix> local_data_normalised; // Stores the normalised local_data.
    std::shared_ptr<Sparse_Matrix> local_cokernel; // Stores the cokernel of the local_data.

    /**
     * @brief Get the type of indecomposable (free, interval, ...)
     * 
     * @return std::string 
     */
    std::string get_type(){
        if(type == BlockType::FREE){
            return "free";
        } else if(type == BlockType::CYC){
            return "cyclic";
        } else if(type == BlockType::INT){
            return "interval";
        } else {
            return "non-interval";
        }
    }

    /**
     * @brief Returns the indices of the relations with only one entry. 
     * If the presented module is an interval, these limit the end of the interval.
     * 
     * @return vec<index> 
     */
    vec<r2degree> endpoints(){
        vec<r2degree> result;
        for(index i = 0; i < get_num_cols(); i++){
            if(data[i].size() == 1){
                result.push_back(col_degrees[i]);
            }
        }
        return result;
    }



    /**
     * @brief If admissible_cols has been set, this fetches the columns for further processing.
     * 
     */
    void compute_local_data(r2degree d){
        // TO-DO: Can we make it so that this is already reduced?
        local_data = std::make_shared<Sparse_Matrix>(this->map_at_degree(d, local_admissible_cols));
    }

    /**
     * @brief stores the rows of r2degree <= d for repeated usage.
     * 
     * @param d 
     */
    void compute_local_generators(r2degree d){
        local_admissible_rows = vec<index>();
        for( index i = 0; i < get_num_rows(); i++){
            if( is_admissible_row_operation(d, i)){
                local_admissible_rows.push_back(rows[i]);
            }
        }
        local_data_normalised = std::make_shared<Sparse_Matrix>(*local_data);
        local_data_normalised->set_num_rows(local_admissible_rows.size());
        local_data_normalised->compute_normalisation_with_pivots(local_admissible_rows);
    }


    /**
     * @brief Computes the local basislift for the block.
     *  Assumes, that local data has been computed and reduced.
     * @param d 
     */
    void compute_local_basislift(r2degree d){
        compute_local_generators(d);
        // The following returns a subset of the indeices in local_admissible_row_indices.
        local_basislift_indices = local_data_normalised->coKernel_basis(local_admissible_rows, true);
    }

    void compute_local_cokernel(){
        local_cokernel = std::make_shared<Sparse_Matrix>(local_data_normalised->coKernel_transposed_without_prelim(local_basislift_indices));
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
        if(local_data->get_num_cols() == 0){
            return false;
        }
        return local_data->solve_col_reduction(N);
    }

    /**
     * @brief Tries to delete N with the columns in local_data.
     * 
     * @param N 
     * @return true 
     * @return false 
     */
    bool reduce_N_fully( Sparse_Matrix& N, bool is_diagonal){
        // Tries to delete N with the columns in local_data.
        if(local_data->get_num_cols() == 0){
            return false;
        }
        return local_data->reduce_fully(N, is_diagonal);
    }


    /**
     * @brief Use column-operations to try and delete the block (A_t)_B
     * 
     * @param N 
     */
    bool delete_with_col_ops(r2degree d, Sparse_Matrix& N, bool no_deletion = false) {

        this->compute_local_data(d);

        // Here I would like a version that also first tries column-reduction to be performed in-place
        // TO-DO: implement
        if(no_deletion){
            return false;
        }
        return local_data->solve_col_reduction(N);
    }

    /**
     * @brief Resets the local data after processing a batch.
     * 
     */
    void delete_local_data(){
        local_admissible_cols.clear();
        local_data.reset();
        local_data_normalised.reset();
        local_admissible_rows.clear();
        local_cokernel.reset();
        local_basislift_indices.clear();
    }

    

    /**
     * @brief Compute the list of rows format for the indecomposable matrix belonging to the block.
     * Careful, the entries of the rows are given with internal numbering of the columns!
     * If you want to change this, give the column indices as an argument to compute_rows_forward.
     * 
     */
    void compute_rows(vec<index>& row_map){
        compute_rows_forward_map(row_map);
    }


    //To-Do: Update the constructors, so that memory for the data  is reserved??
    Block(const vec<index>& c, const vec<index>& r) : GradedMatrix(c.size(), r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
    }

    Block(const vec<index>& c, const vec<index>& r, BlockType t) : GradedMatrix(c.size(), r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
        this->type = t;
    }

    
    
    void clear() {
      this->rows.clear();
      this->columns.clear();
    }
    
    /**
     * @brief Outputs either only the position or also the indecomposable of the block.
     * 
     * @param with_content 
     */
    void print_block(bool with_content=true) {
        std::cout << "  Columns: ";
        std::cout << columns << " ";

        std::cout << "\n  Rows: ";
        std::cout << rows << " ";

        if(with_content){
            std::cout << "  Data: " << std::endl;
            this->print(false, true);
        }
    }
}; //Block	

typedef std::list<Block> Block_list;
typedef Block_list::iterator Block_iterator;
using block_position = std::pair<index, Block_list::iterator>; 

/**
 * @brief Returns *lhs < *rhs
 * 
 */
struct compare_block_position_row {
    bool operator()(const block_position& lhs, const block_position& rhs) const {
        return lhs.second->rows[lhs.first] > rhs.second->rows[rhs.first];
    }
};

struct compare_block_position_column {
    bool operator()(const block_position& lhs, const block_position& rhs) const {
        return lhs.second->columns[lhs.first] > rhs.second->columns[rhs.first];
    }
};

/**
 * @brief Constructs the Blocks of an empty Matrix whose rows are given by A.
 * 
 * @param A 
 * @param B_list 
 * @param block_map 
 */
void initialise_block_list(const GradedMatrix& A, Block_list& B_list, 
    vec<Block_list::iterator>& block_map);


/**
 * @brief Displays the degrees of each block in the block list.
 * 
 * @param B_list 
 */
void print_block_list_status(Block_list& B_list);



/**
 * @brief Extends the block B by the columns of N given by the batch_indices and the batch_positions.
 * 
 * @param B 
 * @param N 
 * @param batch_positions 
 * @param batch_indices 
 */
void extend_block(Block& B, Sparse_Matrix& N, 
    vec<index> batch_indices, bitset& batch_positions, 
    r2degree& alpha);




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
                            vec<index>& row_map, r2degree& alpha);

                            
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
                    vec<index>& row_map, r2degree& alpha);

} //namespace aida

#endif // AIDA_BLOCK_HPP