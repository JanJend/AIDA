#pragma once

#ifndef AIDA_BLOCK_HPP
#define AIDA_BLOCK_HPP

#include "types.hpp"

namespace aida{


/**
 * @brief Holds the indecomposable summands in the matrix
 * 
 */
template<class Matrix>
struct Block_t : Matrix {
    using index = matrix_index_t<Matrix>;
    using Degree = base_degree<Matrix>;
    using Sparse_Matrix = SparseMatrix<index>;
    using CoordinateBase = typename Matrix::Base;

    vec<index> rows;
    vec<index> columns;

    Block_t& operator= (const Block_t& other) {
        if (this != &other) {
            Matrix::operator=(other);
            rows = other.rows;
            columns = other.columns;
        }
        return *this;
    }

    Block_t& operator= (Block_t&& other) {
        if (this != &other) {
            Matrix::operator=(std::move(other));
            rows = std::move(other.rows);
            columns = std::move(other.columns);
        }
        return *this;
    }

    explicit Block_t(const Matrix& matrix) : Matrix(matrix) {}
    explicit Block_t(Matrix&& matrix) : Matrix(std::move(matrix)) {}

    Block_t() : Matrix() {
        rows = vec<index>();
        columns = vec<index>();
    }

    Block_t(const Block_t& other) : Matrix(other) {
        rows = other.rows;
        columns = other.columns;
    }
    Block_t(Block_t&& other) : Matrix(std::move(other)) {
        rows = std::move(other.rows);
        columns = std::move(other.columns);
    }

    BlockType type; // 0 for free module, 1 for cyclic, 2 for interval, 3 for non-interval

    vec<index> local_admissible_cols; // Stores the indices of the columns which can be used for column operations.
    vec<index> local_admissible_rows; // Stores the actual row_indices which can be used for row operations.
    vec<index> local_basislift_indices; // Stores a set of indices defining a subset of local_admissible_rows which forms a basis at the local Degree.
    // Careful, these are indices only locally for this block!

    // This will store the columns which can be added to the current batch, i.e. local_data = this->data | admissible_cols.
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
    vec<Degree> endpoints(){
        vec<Degree> result;
        for(index i = 0; i < this->get_num_cols(); i++){
            if(this->data[i].size() == 1){
                result.push_back(this->col_degrees[i]);
            }
        }
        return result;
    }



    /**
     * @brief If admissible_cols has been set, this fetches the columns for further processing.
     * 
     */
    // NEW: blocks retain source row indices until the final normalisation.
    // Restrict degrees without validating those temporary indices as local rows.
    std::pair<Sparse_Matrix, vec<index>> map_at_stored_degree_pair(const Degree& degree,
                                                               bool shifted = true) const {
        vec<index> selected_rows;
        for (index i = 0; i < this->get_num_rows(); ++i)
            if (CoordinateBase::is_admissible_row_operation(degree, i)) selected_rows.push_back(i);
        vec<index> selected_columns;
        auto matrix = CoordinateBase::map_at_degree(degree, selected_columns);
        matrix.set_num_rows(selected_rows.size());
        if (selected_rows.empty()) {
            matrix.data.clear();
            matrix.set_num_cols(0);
        } else if (shifted) {
            matrix.transform_data(shiftIndicesMap(selected_rows));
        }
        return {std::move(matrix), std::move(selected_rows)};
    }

    void compute_local_data(Degree d){
        // TO-DO: Can we make it so that this is already reduced?
        local_data = std::make_shared<Sparse_Matrix>(CoordinateBase::map_at_degree(d, local_admissible_cols));
    }

    /**
     * @brief stores the rows of Degree <= d for repeated usage.
     * 
     * @param d 
     */
    void compute_local_generators(Degree d){
        local_admissible_rows = vec<index>();
        for( index i = 0; i < this->get_num_rows(); i++){
            if( CoordinateBase::is_admissible_row_operation(d, i)){
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
    void compute_local_basislift(Degree d){
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
    bool delete_with_col_ops(Degree d, Sparse_Matrix& N, bool no_deletion = false) {

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
        this->compute_rows_forward_map(row_map);
    }


    //To-Do: Update the constructors, so that memory for the data is reserved??
    Block_t(const vec<index>& c, const vec<index>& r) : Matrix(c.size(), r.size()){
        std::copy(c.begin(),c.end(),std::back_inserter(this->columns));
        std::copy(r.begin(),r.end(),std::back_inserter(this->rows));
    }

    Block_t(const vec<index>& c, const vec<index>& r, BlockType t) : Matrix(c.size(), r.size()){
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
}; //Block_t

// NEW: compare exported summands by real coordinates, including separate grids.
struct CompareBlocksByDegrees {
    template<class Matrix>
    bool operator()(const Block_t<Matrix>& lhs, const Block_t<Matrix>& rhs) const {
        using index = matrix_index_t<Matrix>;
        using DT = Degree_traits<typename Matrix::degree_type>;
        if (lhs.parameter_count() != rhs.parameter_count())
            return lhs.parameter_count() < rhs.parameter_count();
        for (index i = 0; i < std::min(lhs.get_num_rows(), rhs.get_num_rows()); ++i) {
            auto a = geometric_degree(static_cast<const Matrix&>(lhs), lhs.row_degrees[i]), b = geometric_degree(static_cast<const Matrix&>(rhs), rhs.row_degrees[i]);
            if (DT::lex_order(a, b)) return true;
            if (DT::lex_order(b, a)) return false;
        }
        if (lhs.get_num_rows() != rhs.get_num_rows())
            return lhs.get_num_rows() < rhs.get_num_rows();
        for (index i = 0; i < std::min(lhs.get_num_cols(), rhs.get_num_cols()); ++i) {
            auto a = geometric_degree(static_cast<const Matrix&>(lhs), lhs.col_degrees[i]), b = geometric_degree(static_cast<const Matrix&>(rhs), rhs.col_degrees[i]);
            if (DT::lex_order(a, b)) return true;
            if (DT::lex_order(b, a)) return false;
        }
        return lhs.get_num_cols() < rhs.get_num_cols();
    }
};

template<class Matrix>
using Block_list_t = std::list<Block_t<Matrix>>;
template<class Matrix>
using Block_iterator_t = typename Block_list_t<Matrix>::iterator;
template<class Matrix>
using block_position_t = std::pair<matrix_index_t<Matrix>, Block_iterator_t<Matrix>>;
using Block = Block_t<GradedMatrix>;
using Block_list = Block_list_t<GradedMatrix>;
using Block_iterator = Block_iterator_t<GradedMatrix>;
using block_position = block_position_t<GradedMatrix>;

/**
 * @brief Returns *lhs < *rhs
 * 
 */
struct compare_block_position_row {
    template<class Position>
    bool operator()(const Position& lhs, const Position& rhs) const {
        return lhs.second->rows[lhs.first] > rhs.second->rows[rhs.first];
    }
};

struct compare_block_position_column {
    template<class Position>
    bool operator()(const Position& lhs, const Position& rhs) const {
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
template<class Matrix, typename index = matrix_index_t<Matrix>>
void initialise_block_list(const Matrix& A, Block_list_t<Matrix>& B_list,
    vec<Block_iterator_t<Matrix>>& block_map);


/**
 * @brief Displays the degrees of each block in the block list.
 * 
 * @param B_list 
 */
template<class Matrix, typename index = matrix_index_t<Matrix>>
void print_block_list_status(Block_list_t<Matrix>& B_list);



/**
 * @brief Extends the block B by the columns of N given by the batch_indices and the batch_positions.
 * 
 * @param B 
 * @param N 
 * @param batch_positions 
 * @param batch_indices 
 */
template<class Matrix, typename index = matrix_index_t<Matrix>>
void extend_block(Block_t<Matrix>& B, SparseMatrix<index>& N,
    vec<index> batch_indices, bitset& batch_positions, 
    base_degree<Matrix>& alpha);




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
template<class Matrix, typename index = matrix_index_t<Matrix>>
void merge_blocks_into_block(vec<index>& block_indices, vec<Block_iterator_t<Matrix>>& block_map, Block_t<Matrix>& new_block,
                            Sub_batch_t<index>& N_map, bitset& batch_positions, vec<index>& batch_indices,
                            vec<index>& row_map, base_degree<Matrix>& alpha);

                            
/**
 * @brief Merges the blocks and updates the B_list.
 * 
 * @param A 
 * @param B_list 
 * @param N_map 
 * @param block_map 
 * @param block_partition 
 */
template<class Matrix, typename index = matrix_index_t<Matrix>>
void merge_blocks(Block_list_t<Matrix>& B_list, Sub_batch_t<index>& N_map,
                    vec<Block_iterator_t<Matrix>>& block_map, vec<Merge_data_t<index>>& block_partition, vec<index>& batch_indices,
                    vec<index>& row_map, base_degree<Matrix>& alpha);

} //namespace aida

#include "block_impl.hpp"

#endif // AIDA_BLOCK_HPP
