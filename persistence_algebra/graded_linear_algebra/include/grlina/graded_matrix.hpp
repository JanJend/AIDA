// graded_matrix.hpp 

#pragma once

#ifndef GRADED_MATRIX_HPP
#define GRADED_MATRIX_HPP

#include <iostream>
#include <vector>
#include <grlina/sparse_matrix.hpp>
#include <grlina/orders_and_graphs.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>



namespace graded_linalg {




/**
 * @brief A graded matrix with generic degree-type. 
 * 
 * @tparam D 
 * @tparam index 
 */
template <typename D, typename index>
struct GradedSparseMatrix : public SparseMatrix<index> {
    
    vec<D> col_degrees;
    vec<D> row_degrees;

    vec<vec<index>> col_batches;


    // admissible_col[i] stores to what column i can be added
    array<index> admissible_col;
    // I actually want to store the indices of the columns that can be added to i strictly, i.e. without the batch
    array<index> admissible_col_strict_dual;
    // admissible_row[i] stores what can be added to i
    array<index> admissible_row;
    // Same here, but not strict.
    array<index> admissible_row_dual;

    vec<index> rel_k = vec<index>(100, 0);
	vec<index> gen_k= vec<index>(100, 0);
    index k_max = 1;

    GradedSparseMatrix() : SparseMatrix<index>() {};

    GradedSparseMatrix(index m, index n) : SparseMatrix<index>(m, n), col_degrees(vec<D>(m)), row_degrees(vec<D>(n)) {}

    GradedSparseMatrix(index m, index n, vec<D> c_degrees, vec<D> r_degrees) : SparseMatrix<index>(m, n), col_degrees(c_degrees), row_degrees(r_degrees) {}
    
    /**
     * @brief Infer the number of rows from the list of row degrees.
     * 
     */
    void compute_num_rows(){
        this->num_rows = this->row_degrees.size();
    }

    /**
     * @brief Prints the content in scc format to a stream. Partially from MPP_UTILS print_in_rivet_format in Graded_matrix.h
     * 
     */
    template <typename OutStream>
    void to_stream(OutStream& out, bool header=true){

        // Set the precision for floating-point output
        out << std::fixed << std::setprecision(17);

        if(row_degrees.size() == 0){
            std::cerr << "No rows in this matrix." << std::endl;
            return;
        }

        int dimension = Degree_traits<D>::position(this->row_degrees[0]).size();
        if(header) {
            out << "scc2020\n" << dimension << std::endl;
            out << this->get_num_cols() << " " << this->get_num_rows() << " 0" << std::endl;
        }
        for(index i = 0; i < this->get_num_cols(); i++){
            out << Degree_traits<D>::position(this->col_degrees[i]) << " ;";
            for(index j : this->data[i]){
                out << " " << j;
            }
            out << std::endl;
        }
        for(index j = 0; j < this-> row_degrees.size(); j++){
            out << Degree_traits<D>::position(this->row_degrees[j]) << " ;" << std::endl;
        }
    }


    /**
     * @brief computes the linear map induced at a single degree by cutting all columns and rows of a higher degree.
     * 
     * @param d 
     * @param shifted if true then this reshifts to normalise the entries
     * @return std::pair<SparseMatrix, vec<index>> 
     */
    std::pair<SparseMatrix<index>, vec<index>> map_at_degree_pair(D d, bool shifted = false) const {
        vec<index> selectedRowDegrees;

        // assert(row_degrees.size() == num_rows);
        // assert(col_degrees.size() == num_cols);
        for(index i = 0; i < this->num_rows; i++) {
            if( Degree_traits<D>::smaller_equal(row_degrees[i], d) ) {
                selectedRowDegrees.push_back(i);
            }
        }
        index new_row = selectedRowDegrees.size();
        SparseMatrix<index> result;
        result.num_rows = new_row;
        if(new_row == 0){
            result.num_cols = 0;
            return std::move(std::make_pair(result, selectedRowDegrees));
        }
        for(index i = 0; i < this->num_cols; i++) {
            if( Degree_traits<D>::smaller_equal(col_degrees[i], d) ) {
                result.data.emplace_back(this->data[i]);
            }
        }

        result.compute_num_cols();

        if(shifted){
            transform_matrix(result.data, shiftIndicesMap(selectedRowDegrees), true);
        }

        return std::move(std::make_pair(result, selectedRowDegrees));
    }

    /**
     * @brief computes the linear map induced at a single degree by cutting all columns of higher degrees.
     * 
     * @param d 
     * @return std::pair<SparseMatrix, vec<index>> 
     */
    SparseMatrix<index> map_at_degree(D d, vec<index>& local_admissible_columns) const  {
        // local_data = std::make_shared<Sparse_Matrix>(Sparse_Matrix(0,0));
        for(index i = 0; i < this->num_cols; i++){
            if(is_admissible_column_operation(i, d)){
                // std::cout << "  found an addmisible col op from column " << i << ": ";
                // std::cout << A.col_degrees[columns[i]].first << " " << A.col_degrees[columns[i]].second << " to " <<
                //     A.col_degrees[target].first << " " << A.col_degrees[target].second << std::endl;
                local_admissible_columns.push_back(i);
            }
        }
        return this->restricted_domain_copy(local_admissible_columns);
    }

    /**
     * @brief Returns all row indices whose degree is smaller or equal to d.
     * 
     * @param d 
     * @return vec<index> 
     */
    vec<index> admissible_row_indices(D d) {
        vec<index> result;
        for(index i = 0; i < this->num_rows; i++){
            if(is_admissible_row_operation(i, d)){
                result.push_back(i);
            }
        }
        return result;
    }

    bool is_admissible_column_operation(index i, index j) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], col_degrees[j]) && i != j;
    }

    bool is_admissible_column_operation(index i, const D d) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], d);
    }

    bool is_admissible_row_operation(index i, index j) const {
        assert(i != j);
        return Degree_traits<D>::greater_equal( row_degrees[i], row_degrees[j] );
    }

    bool is_strictly_admissible_column_operation(index i, index j) const {
        return Degree_traits<D>::smaller( col_degrees[i], col_degrees[j]);
    }

    /**
     * @brief Stores the admissible column and row operations when we expect to use these multiple times.
     * 
     */
    void precompute_admissible() {
        admissible_col.resize(this->get_num_cols());
        for(index j=0; j<this->get_num_cols(); j++) {
            for(index i=0; i< j; i++) {
                if(this->is_strictly_admissible_column_operation(i,j)) {
                    this->admissible_col_strict_dual[j].push_back(i);
                }
            }
        }
        admissible_row.resize(this->get_num_rows());
        for(index j=0; j<this->get_num_rows(); j++) {
            for(index i=0; i<this->get_num_rows(); i++) {
                if(this->is_admissible_row_operation(i,j)) {
                    this->admissible_row_dual[j].push_back(i);
                }
            }
        }
    }

    /**
     * @brief Returns a vector containing the degrees of the columns and rows.
     * 
     * @return degree_list 
     */
    vec<D> discrete_support() {
        assert(this->col_degrees.size() == this->num_cols);
        assert(this->row_degrees.size() == this->num_rows);
        vec<D> result = col_degrees;
        result.insert(result.end(), row_degrees.begin(), row_degrees.end());
        return result;
    }  

    /**
     * @brief Prints the matrix as well as the column and row degrees.
     * 
     * @param suppress_description 
     */
    void print_graded(bool suppress_description = false) {
        this->print(suppress_description);
        std::cout << "Column Degrees: " ;
        for(D d : col_degrees) {
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << "\n Row Degrees: ";
        for(D d : row_degrees) {
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief groups the columns by degree.
     * 
     * @param get_statistics 
     */
    void compute_col_batches(bool get_statistics = false){
        this->col_batches.clear();
        this->col_batches.reserve(this->get_num_cols());
        D last_degree = col_degrees[0];
        index j = 0;
        this->col_batches.push_back(vec<index>(1, 0));
        index counter = 1;
        for(index i = 1; i < this->get_num_cols(); i++) {
            if( Degree_traits<D>::equals(col_degrees[i], last_degree) ) {
                counter++;
                if(counter > k_max) {
                    k_max = counter;
                }
            } else {
                col_batches.push_back(vec<index>());
                last_degree = col_degrees[i];
                j++;
                counter = 1;
                if(get_statistics) {
                    rel_k[counter]++;
                }
            }
            this->col_batches[j].push_back(i);
        }
        if(get_statistics) {
            rel_k[counter]++;
        }

        if(get_statistics) {
            counter = 1;
            last_degree  = row_degrees[0];
            for(index i = 1; i < this->get_num_rows(); i++){
                if( Degree_traits<D>::equals(row_degrees[i], last_degree) ){
                    counter++;
                } else {
                    gen_k[counter]++;
                    counter = 1;
                    last_degree = row_degrees[i];
                }
            }
            gen_k[counter]++;
        }
    }

    /**
     * @brief Count the number of repeating degrees. Assumes the degree lists to be sorted.
     * 
     */
    void get_k_statistics(){
		D tmp = col_degrees[0];
		index counter = 1;
		for(index i = 1; i < this->num_cols; i++){
			if( Degree_traits<D>::equals(col_degrees[i], tmp) ){
				counter++;
			} else {
				rel_k[counter]++;
				counter = 1;
				tmp = col_degrees[i];
			}
		}
        rel_k[counter]++;

        counter = 1;
        tmp = row_degrees[0];
        for(index i = 1; i < this->num_rows; i++){
            if( Degree_traits<D>::equals(row_degrees[i], tmp) ){
                counter++;
            } else {
                gen_k[counter]++;
                counter = 1;
                tmp = row_degrees[i];
            }
        }

        gen_k[counter]++;
	}

    /**
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the columns.
     * 
     * @return array<index> 
     */
    array<index> get_column_graph() {
        return minimal_directed_graph<D, index>(col_degrees);
    }

    /**
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the rows.
     * 
     * @return array<index> 
     */
    array<index> get_row_graph() {
        return minimal_directed_graph<D, index>(row_degrees);
    }

    /**
     * @brief Sorts the columns lexicographically by degree, 
     *  using a pointer which points two both the column degrees and the data.
     * 
     */
    void sort_columns_lexicographically_with_pointers() {
        sort_simultaneously<D, vec<index>>(col_degrees, this->data);
    }

    /**
     * @brief Sorts the columns lexicographically by degree, 
     * saves the permutation used to do so and applies it to the data.
     * 
     */
    void sort_columns_lexicographically() {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->col_degrees, Degree_traits<D>::lex_lambda);
        array<index> new_data = array<index>(this->data.size());
        for(index i = 0; i < this->data.size(); i++) {
            new_data[i] = this->data[permutation[i]];
        }
        this->data = new_data;
    }

    /**
     * @brief Sorts the rows lexicographically by degree, then transforming the data accordingly.
     * 
     */
    void sort_rows_lexicographically(){

        vec<index> permutation = sort_and_get_permutation<D, index>(this->row_degrees, Degree_traits<D>::lex_lambda);
        // Need inverse of permutation
        vec<index> reverse = vec<index>(permutation.size());
        for (int i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
    }

    /**
     * @brief Outputs the lists of generators and relations
     * 
     */
    void print_degrees() {
        std::cout << "Generators at: ";
            for(D d : row_degrees) {
                Degree_traits<D>::print_degree(d);
                std::cout << " ";
            }
            std::cout << "\nRelations at: ";
            for(D d : col_degrees) {
                Degree_traits<D>::print_degree(d);
                std::cout << " ";
            }
    }
}; // GradedSparseMatrix


/**
 * @brief Compares two graded matrices by their degrees.
 * 
 * @tparam D 
 * @tparam index 
 */
template <typename D, typename index>
struct Compare_by_degrees {

    /**
     * @brief -1 if a<b, 0 if a=b, 1 if a>b
     * 
     * @param a 
     * @param b 
     * @return int 
     */
    static int compare_three_way(const GradedSparseMatrix<D, index>& a, const GradedSparseMatrix<D, index>& b) {
        // Compare row degrees
        for (size_t i = 0; i < std::min(a.row_degrees.size(), b.row_degrees.size()); ++i) {
            if (Degree_traits<D>::smaller(a.row_degrees[i], b.row_degrees[i])) {
                return -1;
            }
            if (Degree_traits<D>::smaller(b.row_degrees[i], a.row_degrees[i])) {
                return 1;
            }
        }
        if (a.row_degrees.size() != b.row_degrees.size()) {
            return a.row_degrees.size() < b.row_degrees.size() ? -1 : 1;
        }

        // Compare column degrees
        for (size_t i = 0; i < std::min(a.col_degrees.size(), b.col_degrees.size()); ++i) {
            if (Degree_traits<D>::smaller(a.col_degrees[i], b.col_degrees[i])) {
                return -1;
            }
            if (Degree_traits<D>::smaller(b.col_degrees[i], a.col_degrees[i])) {
                return 1;
            }
        }
        if (a.col_degrees.size() != b.col_degrees.size()) {
            return a.col_degrees.size() < b.col_degrees.size() ? -1 : 1;
        }

        return 0;
    }

    bool operator()(const GradedSparseMatrix<D, index>& a, const GradedSparseMatrix<D, index>& b) const {
        return compare_three_way(a, b) == -1;
    }
};

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 *  make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space(const GradedSparseMatrix<D, index>& A, const GradedSparseMatrix<D, index>& B, 
    const vec<index>& row_indices_A = vec<index>(), const vec<index>& row_indices_B = vec<index>())  {
    
    assert(A.rows_computed);

    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.num_rows + B.num_rows + 1);
    index S_index = 0;

    // TO-DO: Right now we compute map_at_degree possibly multiple times! Fix this.
    for(index i = 0; i < A.num_rows; i++) {
        // Compute the target space B_alpha for each generator of A to minimise the number of variables.
        auto [B_alpha, rows_alpha] = B.map_at_degree_pair(A.row_degrees[i]);
        vec<index> basislift = B_alpha.coKernel_basis(rows_alpha, row_indices_B );

        // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
    	    variable_positions.push_back(std::make_pair(i, j));
            for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                auto& column_index = *rit;
                S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.num_cols, B.num_rows));
            }
            S_index++;
        }
    }
    
    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }

    // Then all column-operations from B to A
    for(index i = A.num_cols-1; i > -1; i--){
        for(index j = 0; j < B.num_cols; j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    if(row_indices_B.size() != 0){
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.num_cols, B.num_rows));
                    } else {
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.num_cols, B.num_rows));        
                    }
                }
                S_index++;
            }
        }
    }



    S.compute_num_cols();
    auto K = S.get_kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();
    K.column_reduction_triangular(true);

    return std::make_pair(K, variable_positions);
}

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 * 
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index>
std::pair< SparseMatrix<index>, vec<std::pair<index, index> > > block_hom_space_without_optimisation(const GradedSparseMatrix<D, index>& A, const GradedSparseMatrix<D, index>& C, const GradedSparseMatrix<D, index>& B,
        vec<index>& C_rows, vec<index>& B_rows, bool system_size = false)  { 
    vec<std::pair<index, index>> row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.
    SparseMatrix K(0,0);
    SparseMatrix S(0,0);
    S.data.reserve( C_rows.size() + B_rows.size() + 1);
    index S_index = 0;
    // First add all row-operations from C to B
    for(index i = 0; i < C_rows.size(); i++){
        for(index j = 0; j < B_rows.size(); j++){
            auto source_row_index = C_rows[i];
            auto target_row_index = B_rows[j];
            if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                row_ops.push_back({source_row_index, target_row_index});
                S.data.push_back(vec<index>());
                for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(A.linearise_position_reverse(column_index, target_row_index));
                }
                S_index++;
            }
        }
    }

    index row_op_threshold = S_index;
    assert( row_ops.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return {SparseMatrix<index>(), row_ops};
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

    S.compute_num_cols();

    if(system_size){
        std::cout << "System size: " << S.num_cols << std::endl;
    }

    // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
    K = S.get_kernel();
    // To see how much the following reduces K: index K_size = K.data.size();
    // Now we need to delete the entries of K which correspond to the row-operations.
    K.cull_columns(row_op_threshold, false);
    
    // Last we need to quotient out those Q where for every i the column Q_i - with its degree alpha_i -
    // lies in the image of N, that is, it lies in the image of N|alpha_i.
    // That is equivalent to locally reducing every column of Q.
    
    for(index i = 0; i < C_rows.size(); i++){
        D alpha = C.row_degrees[i];
        vec<index> local_admissible_columns;
        auto B_alpha = B.map_at_degree(alpha, local_admissible_columns);
        std::unordered_map<index, index> shiftIndicesMap;
        for(index j : B_rows){
            shiftIndicesMap[j] = A.linearise_position_reverse(C_rows[i], j);
        }
        B_alpha.transform_data(shiftIndicesMap);
        B_alpha.reduce_fully(K);
    }
    
    //  delete possible linear dependencies.
    K.column_reduction_triangular(true);
    return std::make_pair(K, row_ops);
}

} // namespace graded_linalg

#endif // GRADED_MATRIX_HPP
