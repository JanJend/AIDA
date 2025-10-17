
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <concepts>
#include <omp.h>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <filesystem>

namespace presentation_to_quiver {

// Change this to whatever type you need
using index = int;

template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;
using indvec = vec<index>;
using bitset = boost::dynamic_bitset<>;
using edgeList = vec<std::pair<index,index>>;



// All functions and classes for partially ordered sets over a set of degrees


using degree_list = vec<degree>;

void print_degree_list(const degree_list& degrees){
    for(degree d : degrees){
        std::cout << d << " ";
    }
    std::cout << std::endl;
}

using point = vec<double>; // Change if you need another type
using point_list = vec<point>;

bool operator==(const point& lhs, const point& rhs) {
    bool equal = true;
    for (int i = 0; i < lhs.size(); ++i) {
        equal = equal && (lhs[i] == rhs[i]);
    }
    return equal;
}

bool operator<(const point& lhs, const point& rhs) {
    bool lessequal = true;
    bool equal = true;
    for (int i = 0; i < lhs.size(); ++i) {
        lessequal = lessequal && (lhs[i] <= rhs[i]);
        equal = equal && (lhs[i] == rhs[i]);
    }
    return lessequal && !equal;
}

bool operator<=(const point& lhs, const point& rhs) {
    bool lessequal = true;
    for (int i = 0; i < lhs.size(); ++i) {
        lessequal = lessequal && (lhs[i] <= rhs[i]);
    }
    return lessequal;
}

bool lex_order(const point& a, const point& b) {
    bool lexsmaller = true;
    for(int i = 0; i < a.size(); ++i) {
        if (a[i] < b[i]){
            return true;
        } else if (a[i] > b[i]){
            return false;
        }
    }
    return false;
}

point position(const point& a) {
    return a;
}


template <Degree D>
void writeDegreeListToCSV(const std::string& filename, const vec<D>& degrees) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(15);

    int dim = position(degrees[0]).size();
    for (size_t i = 0; i < dim; ++i) {
        outfile << "Column_" << i << ",";
    }
    outfile << std::endl;

    for (const auto& element : degrees) {
        const vec<std::string>& pos = position(element);
        for (const std::string& coord : pos) {
            outfile << coord << ",";
        }
        outfile << std::endl;
    }

    std::cout << "Degree list written to: " << filename << std::endl;
}

void writeEdgeListToCSV(const std::string& filename, const edgeList& edges) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header if needed
     outfile << "Source,Target" << std::endl;

    // Write integer pairs to the file
    for (const auto& pair : edges) {
        outfile << pair.first << "," << pair.second << std::endl;
    }

    std::cout << "Integer pairs written to: " << filename << std::endl;
}

void print_edge_list(const edgeList& edges){
    for(auto e : edges){
        std::cout << e.first << " -> " << e.second << std::endl;
    }
}

/**
 * @brief Computes the Hasse Diagram of a sorted list of unique degrees
 * 
 * @tparam D 
 * @param degrees needs to be sorted in any linear extension of the order
 * @return edgeList 
 */
template <Degree D>
edgeList minimal_directed_graph(vec<D>& degrees) {

    edgeList edges;

    // writeDegreeListToCSV("degreeList.csv", vertices);
    
    // Add directed edges based on the relationships in the degreeList
    // This computes a transitive reduction of the full poset graph

    vec<D> directSuccessors;
    for (index i = 0; i < degrees.size(); ++i) {
      directSuccessors.clear();
      D d = degrees[i];
        for (index j = i + 1; j < degrees.size(); ++j) {
          bool comesAfterSuccessor = false;
          D nextd = degrees[j];
            if( (d < nextd) ){
              for(auto s : directSuccessors){
                  if( (s < nextd) ) {
                    comesAfterSuccessor = true;
                    break;
                  }
              }
              if(!comesAfterSuccessor){
                edges.push_back(std::make_pair(i,j));
                directSuccessors.push_back(nextd);
              }
            }
        }
    }
    // writeEdgeListToCSV("edgesBeforeReduction.csv", edgesForPython);
    return edges;
}





// Interface for a matrix
template<typename V, typename Derived>
struct MatrixUtil{

    index num_cols;
    index num_rows;

    vec<V> data; //stores the columns of the matrix
    
    index get_num_rows(){return num_rows;};
    index get_num_cols(){return num_cols;};
    
    void set_num_rows(index m){num_rows = m;};
    void set_num_cols(index n){num_cols = n;};
    
    void compute_num_cols(){
        num_cols = data.size();
    };

    // Define these for your own type which designates a column of a matrix
    
    virtual bool is_nonzero_at(V& a, index i) = 0;
    virtual void add_to(V& a, V& b) = 0;
    virtual index last_entry_index(V& a) = 0;
	
    
    //Shouldnt need to implement the following functions again.

   
    void col_op(index i, index j){
        add_to(data[i], data[j]);
    };

    void row_op(index i, index j){};

    bool is_nonzero_entry(index i, index j){
        return is_nonzero_at(data[i] , j);   
    };

    std::unordered_map<index,index> pivots;

    index col_last(index i){
        return last_entry_index(data[i]);
    };
    
    array<index> col_last_vec; // for each row index, lists the column indices that have the largest non-zero entry in that row

    void print(){
        assert(data.size() == num_cols);
        std::cout << "Cols: " << num_cols << " rows: " << num_rows << std::endl;
        for(index i=0;i<num_cols;i++) {
            std::cout << "Column " << i << ": " << data[i] << std::endl;
        }
    };
    
    /**
     * @brief Points from a row index to all the columns that have a largest non-zero entry in that row
     * 
     */
    void compute_col_last(){
        col_last_vec = vec<indvec>(num_rows);
        for (index i = 0; i < num_cols; i++){
            index l = col_last(i);
            if(l>=num_rows){
                throw std::out_of_range("There is an index in the column that is larger than the number of rows at: " + std::to_string(i) + " and the entry is " + std::to_string(l) + " and the number of rows is " + std::to_string(num_rows));
            } 
            if(l>=0){
                col_last_vec[l].push_back(i);
            }
        }
    }

   /**
    * @brief Brings Matrix in reduced Column Echelon form.
    * 
    */
    void column_reduction() {
        col_last_vec = vec<indvec>(num_rows);
        for (index col = 0; col < num_cols; ++col) {
            index pivotRow = col_last(col);
            if(pivotRow >= num_rows){
                throw std::out_of_range("There is an index in the column that is larger than the number of rows at column: " + std::to_string(col) + " and the entry is " + std::to_string(pivotRow) + " and the number of rows is " + std::to_string(num_rows));
            } 
            
            if (pivotRow != -1) {
                col_last_vec[pivotRow].push_back(col);
                for (index otherCol = 0; otherCol < num_cols; ++otherCol) {
                    if (col != otherCol && is_nonzero_entry(otherCol, pivotRow)) {
                        col_op(col, otherCol);
                    }
                }
            }
        }
    }
    
    /**
    * @brief Returns a copy with only the columns at the indices given in colIndices.
    * 
    * @param colIndices 
    * @return sparseMatrix 
    */
    Derived restricted_domain_copy(indvec& colIndices){
        for(index i : colIndices){
            assert(i < num_cols);
        }
        Derived result(colIndices.size(), num_rows);
        for(index i = 0; i < colIndices.size(); i++){
            result.data[i] = data[colIndices[i]];
        }
        return result;
    }

    bool equals(MatrixUtil& other){
        if(num_cols != other.num_cols){
        std::cout << "#columns dont match.";
        return false;
        }
        if(num_rows != other.num_rows){
            std::cout << "#rows dont match.";
            return false;
        }
        for(index i = 0; i< num_cols; i++){
            if( !is_equal(data[i], other.data[i]) ){
            std::cout << "columns at index " << i << " dont match.";
            return false;
            }
        }
        return true;
    }

    MatrixUtil() {};

    MatrixUtil(index m, index n) : num_cols(m), num_rows(n), data(vec<V>(m)) {}

    // Copy constructor
    MatrixUtil(const MatrixUtil& other) : data(other.data), num_cols(other.num_cols), num_rows(other.num_rows) {}

    // Copy assignment operator
    MatrixUtil& operator=(const MatrixUtil& other) {
        if (this != &other) {
            data = other.data; // Assuming Matrix has a proper copy assignment operator
            num_cols = other.num_cols;
            num_rows = other.num_rows;
        }
        return *this;
    }

    // Move constructor
    MatrixUtil(MatrixUtil&& other) noexcept : data(std::move(other.data)), num_cols(other.num_cols), num_rows(other.num_rows) {
        // Reset the source object
        other.num_cols = 0;
        other.num_rows = 0;
        other.data = nullptr;
    }

    // Destructor
    ~MatrixUtil() {
        // std::cout << "MatrixUtil Destructor Called on the instance of size" << get_num_cols() << " x "<< get_num_rows() << std::endl;
    }
};

// Helperfunctions for index vectors and sparse matrices

/**
    * @brief sparse column addition over F_2. Adds a to b.
    * 
    * @param a a vector containing integers representing the indices of the nonzero entries of the column
    * @param b a vector containing integers representing the indices of the nonzero entries of the column
    */
    void add_to(indvec& a, indvec& b) {
        //std::cout << "ADDITION" << std::endl;
	    indvec c;
        auto a_it = a.begin();
        auto b_it = b.begin();
        while(a_it!=a.end() || b_it!=b.end()) {
        if(a_it==a.end()) {
            c.push_back(*b_it);
            b_it++;
            continue;
        }
        if(b_it==b.end()) {
            c.push_back(*a_it);
            a_it++;
            continue;
        }
        if(*a_it<*b_it) {
            c.push_back(*a_it);
            a_it++;
        } else if(*a_it>*b_it) {
            c.push_back(*b_it);
            b_it++;
        } else { // *a_it==*b_it
            assert(*a_it==*b_it);
            a_it++;
            b_it++;
        }      
        }
        b = c;
    }

/**
 * @brief Returns the scalar product of two sparse vectors over F_2
 * 
 * @param v 
 * @param w 
 * @return true 
 * @return false 
 */
bool scalar_product(indvec& v, indvec& w){

  auto it_v = v.begin();
  auto it_w = w.begin();
  index count = 0;

  while (it_v != v.end() && it_w != w.end()) {
      if (*it_v < *it_w) {
          // Move iterator of v because the current index in v is smaller
          ++it_v;
      } else if (*it_w < *it_v) {
          // Move iterator of w because the current index in w is smaller
          ++it_w; 
      } else {
          // Indices are equal, move both iterators without counting
          ++it_v;
          ++it_w;
          count++;  
      }
  }
  return count % 2 != 0;
}

bool is_equal(indvec& v,indvec& w){ return v == w; }
index length(indvec& v) { return v.size();}
index last_entry_index(indvec& v){
    if(v.size() == 0){
        return -1;
    } else {
        return v.back();
    } 
}
bool is_nonzero_at(indvec& v, index i){ return std::binary_search(v.begin(), v.end(), i); }

std::ostream& operator<< (std::ostream& ostr, const indvec& c) {
    for(index i:c) {
      ostr << i << " ";
    }
    return ostr;
}


void print_vec(indvec& c){
  std::cout << c << std::endl;
}

/**
 * @brief This function is used to delete rows in a LOC sparse matrix. 
 * It creates a map which maps the old indices to the new indices.
 * 
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index> 
 */
std::unordered_map<index, index> shiftIndicesMap(const indvec& indices) {
    std::unordered_map<index, index> indexMap;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indexMap[indices[i]] = i;
    }
    return indexMap;
}

/**
 * @brief Parallelized function to apply a transformation to a vector of indices.
 *
 * @param target
 * @param indexMap
 * @param needsNoDeletion If the target vector only contains indices which are in the indexMap, this can be set to true.
 */
void apply_transformation(indvec& target, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion = false) {
    if (!needsNoDeletion) {
#pragma omp parallel for
        for (int i = static_cast<int>(target.size()) - 1; i >= 0; --i) {
            const index& element = target[i];
#pragma omp critical
            {
                if (indexMap.find(element) == indexMap.end()) {
                    target.erase(target.begin() + i);
                }
            }
        }
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = indexMap.at(target[i]);
    }
}

/**
 * @brief Parallelised function to change a sparse matrix by applying the indexMap to each entry.
 *
 * @param S
 * @param indexMap
 */
void transform_matrix(array<index>& S, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion) {
#pragma omp parallel for
    for (std::size_t i = 0; i < S.size(); ++i) {
        apply_transformation(S[i], indexMap, needsNoDeletion);
    }
}

indvec transformed_copy(indvec& target , const std::unordered_map<index, index>& indexMap) {
    indvec result;
    result.reserve(target.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < target.size(); ++i) {
        result[i] = indexMap.at(target[i]);
    }
    return result;
}

/**
 * @brief Get all indices for which an element in the first vector is contained in the second vector. Both inputs should be ordered.
 * 
 * @param target 
 * @param subset 
 * @return vec<index> 
 */
indvec getIndicatorVector(vec<index> target, indvec subset){
  vec<index> result;
  auto itS = subset.begin();
  for(index i = 0; i < target.size() ; i++){
    if(itS != subset.end()){
    if(*itS == target[i]){
      result.push_back(i);
      itS++;
    }
    }
  }
  assert(itS == subset.end() && "Not all elements of the subset were found in the target");
  return result;
}

/**
 * @brief Extracts the elements of a vector at the indices given in the second vector.
 * 
 * @param target 
 * @param indices 
 * @return vec<T> 
 */
template <typename T>
vec<T> extractElements(const vec<T>& target, const indvec& indices) {
    vec<T> result;
    result.reserve(indices.size());
    for (index i : indices) {
        assert(i >= 0 && i < target.size() && "Index out of bounds");
        result.push_back(target[i]);
    }
    return result;
}

/**
 * @brief The first vector in each argument is a set of row indices of the original matrix telling us which generators form a basis of the cokernel
 * The second vector is a set of column indices of the cokernel matrix defining a section of the cokernel.
 * 
 * @param source 
 * @param target 
 * @return vec<index> 
 */
indvec basisLifting(std::pair<indvec, indvec>& source, std::pair<indvec, indvec>& target){
    //sanity check, comment out in real run
    vec<index> subsetIndicator1 = getIndicatorVector(target.first, source.first);
    // Probably this doesnt need to be its own function
    vec<index> result = getIndicatorVector(target.first, extractElements(source.first, source.second));
    return result;
}

template <Degree D>
index delete_multiples(vec<D>& degrees){
    
    std::sort(degrees.begin(), degrees.end(), [](const D& a, const D& b) {
        return lex_order(a, b);
    });
    

    index kMax = 1;
    D compare;
    if(degrees.size() ==0 ){
        return kMax;
    } else if (degrees.size() == 1){
        return 1;
    } else {
        compare = degrees[0];
    }

    indvec positions = {0};
    positions.reserve(degrees.size());


    index kCounter = 1;
    for (index i = 1; i < degrees.size(); i++) {
        if ( degrees[i] != compare){
            compare = degrees[i] ;
            kCounter = 1;
            positions.push_back(i);
        } else {
            kCounter++;
            if(kCounter > kMax){
              kMax = kCounter;
            }
        }

    }

    degrees = extractElements(degrees, positions);
    return kMax;
}

/**
 * @brief Every column is stored as a list of non-zero entries.
 * 
 */
struct SparseMatrix : public MatrixUtil<indvec, SparseMatrix>{

    SparseMatrix() : MatrixUtil<indvec, SparseMatrix>() {};

    SparseMatrix(index m, index n) : MatrixUtil<indvec, SparseMatrix>(m, n) {
		data = vec<indvec>(m);
	}

    SparseMatrix(const SparseMatrix& other) : MatrixUtil<indvec, SparseMatrix>(other)  {
        data = other.data;
    }

    

    void add_to(indvec& v, indvec& w) override {
		presentation_to_quiver::add_to(v, w);
	};

	bool is_nonzero_at(indvec& v, index i) override {
		return presentation_to_quiver::is_nonzero_at(v, i);
	};

    index last_entry_index(indvec& v) override {
		if(v.size() == 0){
            return -1;
        } else {
            return v.back();
        } 
	};

	/**
	std::ostream& operator<< (std::ostream& ostr, const indvec& c) {
		presentation_to_quiver::operator<<(ostr, c);
	}
    */


    void delete_last_entries(){
        #pragma omp parallel for
        for(std::size_t i = 0; i < data.size(); ++i) {
            vec<index>& c = data[i];
            #pragma omp critical
            if (!c.empty()) {
             c.pop_back();  // Delete the last entry if the column is not empty
            }
        }
    }
    
    /**
    * @brief Returns a transpose
    * 
    * @return sparseMatrix 
    */
    SparseMatrix transposed_copy() const {
        SparseMatrix result(num_rows, num_cols);
        for(index i=0;i<num_cols;i++) {
            for(index j : data[i]) {
                result.data[j].push_back(i);
            }
        }
        return result;
    }

    /**
     * @brief Computes the cokernel of a sparse matrix over F_2 by column reducing the matrix first
     * Notice that the result must be a cokernel to the non-reduced matrix, too, so we can also use a copy instead.
    * 
    * @param S 
    * @param isReduced
    * @return sparseMatrix 
    */
    SparseMatrix coKernel(bool isReduced = false, vec<index>* basisLift = nullptr){
  
		if(!isReduced){
            try {
                column_reduction();
            } catch (std::out_of_range& e) {
                std::cerr << "Error in coKernel Computation: " << e.what() << std::endl;
                print();
                std::abort();
            }
		}
        
		indvec quotientBasis;
	
		for(index i = 0; i < num_rows; i++){
			if(col_last_vec[i].empty()){
				quotientBasis.push_back(i);
			} else {
				// Check if matrix is really reduced and the last entry is unique
                if(col_last_vec[i].size() != 1) {
                    std::cerr << "Error: The matrix is not reduced. The last entry in row " << i << " is not unique, but of size " << col_last_vec[i].size() << std::endl;
                    print();
                    std::abort();
                };
			}
		}
		// print_vec(quotientBasis);

		auto indexMap = shiftIndicesMap(quotientBasis);
		SparseMatrix trunc(*this);
	
		trunc.delete_last_entries();

		transform_matrix(trunc.data, indexMap, true);
	
		index newRows = quotientBasis.size();
		index newCols = num_rows;
		SparseMatrix result(newCols, newRows);
		// std::cout << newCols << result.num_cols << " " << result.num_rows << result.data.size() << std::endl;
		index j = 0;
		for(index i = 0; i < newCols; i++){
		// Construct Identity Matrix on the generators which descend to the quotient basis. 
			if(j < quotientBasis.size() && quotientBasis[j] == i){
				result.data[i].push_back(j);
				j++;
			} else {
				// Locate the unqiue column with the last entry at i.
				index colForPivot = *col_last_vec[i].begin();
				result.data[i] = trunc.data[colForPivot];
			}
		}
		assert(j == quotientBasis.size() && "Not all quotient basis elements were used");
		if(basisLift){
			*basisLift = quotientBasis;
		}
		return std::move(result);
    }

	
};
    
/**
 * @brief Computes the transpose of M, then multiplies the columns.
 * 
 * @param M 
 * @param N 
 * @return product M*N over F_2 
 */
SparseMatrix multiply(SparseMatrix& M, SparseMatrix& N){
  SparseMatrix result(N.get_num_cols(), M.get_num_rows());
  SparseMatrix transpose = M.transposed_copy();
  for(index i = 0; i < N.get_num_cols(); i++){
    for(index j = 0; j < transpose.get_num_cols(); j++){
      if(scalar_product(transpose.data[j], N.data[i])){ 
        result.data[i].push_back(j);
      }
    }
  }
  return result;
}




/**
 * @brief graded sparse matrix.
 * 
 */
template <Degree D>
struct GradedMatrix : public SparseMatrix {
    
    vec<D> col_degrees;
    vec<D> row_degrees;

    GradedMatrix() : SparseMatrix() {};

    GradedMatrix(index m, index n) : SparseMatrix(m, n), col_degrees(vec<D>(n)), row_degrees(vec<D>(m)) {}

        /**
         * @brief computes the linear map induced at a single degree by cutting all columns and rows of a higher degree.
         * 
         * @param d 
         * @return std::pair<SparseMatrix, vec<index>> 
         */
        std::pair<SparseMatrix, indvec> map_at_degree(D d) {
            indvec selectedRowDegrees;
            bool timer = false;
            auto start = std::chrono::high_resolution_clock::now();

            // assert(row_degrees.size() == num_rows);
            // assert(col_degrees.size() == num_cols);
            for(index i = 0; i < num_rows; i++) {
                if(row_degrees[i] <= d) {
                selectedRowDegrees.push_back(i);
                }
            }
            
            index new_row = selectedRowDegrees.size();
            SparseMatrix result;
            result.num_rows = new_row;
            for(index i = 0; i < num_cols; i++) {
                if(col_degrees[i] <= d) {
                    result.data.emplace_back(data[i]);
                    // result.data.emplace_back(  transformed_copy(data[i],shiftIndicesMap(selectedRowDegrees))   );
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            if (timer) {
                std::chrono::duration<double> duration = end - start;
                std::cout << "Time for comparing degrees: " << duration.count() << " seconds" << std::endl;
            }

            result.num_cols = result.data.size();
            start = std::chrono::high_resolution_clock::now();

            transform_matrix(result.data, shiftIndicesMap(selectedRowDegrees), true);

            end = std::chrono::high_resolution_clock::now();
            if (timer) {
                std::chrono::duration<double> duration = end - start;
                std::cout << "Time for transform_matrix: " << duration.count() << " seconds" << std::endl;
            }
            return std::move(std::make_pair(result,selectedRowDegrees));
    }

};

/**

struct DenseMatrix : public MatrixUtil<bitset, DenseMatrix>{
    
	//Not yet done
	bool is_nonzero_at(bitset& v, index& i) override {
		return v[i];
	};

	void add_to(bitset& v, bitset& w) override {
		v ^= w;
	};

	index last_entry_index(bitset& v) override {
		return v.find_last();
	};

};
*/

template<typename T>
std::pair<vec<T>, indvec> parse_line_vector(const std::string& line, bool getEntries = true) {
    std::istringstream iss(line);
    vec<T> coordinates;
    indvec relation;

    // Parse degree
    T num;
    while (iss >> num) {
        coordinates.push_back(num);
    }

    // Skip the semicolon
    iss.ignore();

    // Parse relation
    if (getEntries) {
        index entry;
        while (iss >> entry) {
            relation.push_back(entry);
        }
    }

    return std::make_pair(coordinates, relation);
}





std::pair<degree, indvec> parse_line(const std::string& line, bool hasEntries = true) {
    std::istringstream iss(line);
    degree deg;
    indvec rel;


    // Parse degree
    iss >> deg.first >> deg.second;


    // Consume the semicolon
    std::string tmp;
    iss >> tmp;
    if(tmp != ";"){
        std::cerr << "Error: Expecting a semicolon. Invalid format in the following line: " << line << std::endl;
        std::abort();
    }


    // Parse relation
    if(hasEntries){
 
        index num;
        while (iss >> num) {

            rel.push_back(num);
        }
    }

    return std::move(std::make_pair(deg, rel));
}


struct QuiverRepresentation {
	degree_list degrees;
    indvec dimensionVector;
	edgeList edges;
	vec<SparseMatrix> matrices;
    
    QuiverRepresentation(){
        degrees = degree_list();
        dimensionVector = indvec();
        edges = edgeList();
        matrices = vec<SparseMatrix>();
    }

    void print(){
        print_degree_list(degrees);
        print_edge_list(edges);
        index size = degrees.size();
        assert(size == dimensionVector.size());
        index length = matrices.size();
        assert(length == edges.size());
        for(index i = 0; i < length; i++){
            std::cout << "Matrix at edge " << i << std::endl;
            matrices[i].print();
        }
    }

    QuiverRepresentation(degree_list degrees, indvec dimensionVector, edgeList edges, vec<SparseMatrix> matrices) : 
    degrees(degrees), dimensionVector(dimensionVector), edges(edges), matrices(matrices) {}

    void writeQPAFile(const std::string& fullFilePath, const std::string& quiverName = "Q") {
        
        std::string scriptPath = fullFilePath + ".g";
        std::string logFilePath = fullFilePath + "_log.txt";

        std::ofstream outfile(scriptPath);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file at path: " << scriptPath << std::endl;

        }

        index dimension = 0;
        for(index i = 0; i < dimensionVector.size(); i++){
            dimension += dimensionVector[i];
        }

        index num_vert = degrees.size();
        index num_edges = edges.size();

        // Write header information -  May need to be changed
        // outfile << "LoadPackage(\"qpa\");\n\n";

        // Define the quiver
        outfile << "# Dimension of Module: " << dimension << "\n";
        outfile << "LoadPackage(\"qpa\");\n";
        outfile << "start_read := Runtime();  # Record the starting time\n";
        
        outfile << quiverName << " := Quiver(" << num_vert << ", [ ";
        

        // Write arrows (edges)
        for (index j = 0; j < num_edges; j++) {
            index source = edges[j].first;
            index target = edges[j].second;

            // Write arrow
            outfile << "["  << source + 1 << "," << target + 1 << ", \"E" << j << "\"" << "], ";

            
            }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << "]);" << std::endl;

        // Define path algebra
        outfile << "A" << quiverName << " := PathAlgebra(GF(2), " << quiverName<< ");" << std::endl;

        // Define module
        // syntax: N := RightModuleOverPathAlgebra( A, [1,2,2], [ .. ["b", [[1,0], [-1,0]] ], ..] )
        // Use Z(2)^0 and 0*Z(2) for 1 and 0; (Need Z(p^k) for general finite field)
        outfile << "M" << quiverName << " := RightModuleOverPathAlgebra(A" << quiverName << ", [";
        for (index i = 0; i < num_vert; i++) {
            outfile << dimensionVector[i] << ",";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "], [ ";

        for (index j = 0; j < num_edges; j++) {
            auto source = edges[j].first;
            auto target = edges[j].second;
            // No need for this, qpa wants the transposed matrices in row format, so we just write our matrices column wise instead of row wise
            // No need for: SparseMatrix edgeMatrix = matrices[j].transposed_copy();

            // Check dimensions again
            if (matrices[j].get_num_cols() != dimensionVector[source]) {
                throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
            }
            if (matrices[j].get_num_rows() != dimensionVector[target]) {
                throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
            }

            if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
                continue;
            }
            // Write matrix at arrow; where each vector is a row

            
            outfile << "[\"E" << j << "\", [";
            for (index i = 0; i < matrices[j].num_cols; i++) {
                outfile << "[";
                auto it = matrices[j].data[i].begin();

                for (index k = 0; k < matrices[j].num_rows; k++) {
                    if (it != matrices[j].data[i].end()) {
                        if(*it == k){
                            outfile << "Z(2)^0" << ",";
                            ++it;
                        } else {
                            outfile << "0*Z(2)" << ",";
                        }
                    } else {
                        outfile << "0*Z(2)" << ",";
                    }
                }
                outfile.seekp(-1, std::ios::end);  // Remove trailing comma
                outfile << "],";
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "] ], ";
        }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << " ]);" << std::endl;
        outfile << "List" << quiverName << " := BlockDecompositionOfModule(M" << quiverName << ");" << std::endl;
        outfile << "end_read := Runtime(); \n";
        outfile << "time_taken := end_read - start_read; \n";
        outfile << "PrintTo( \"" << logFilePath << " \" , \"CPU time taken to read file and compute decomposition: \", time_taken, \" milliseconds\", List" << quiverName << "); \n";
        outfile << "QUIT;";
        std::cout << "Quiver file written to: " << fullFilePath << std::endl;
    }
};



struct R2GradedMatrix : GradedMatrix<degree> {

    R2GradedMatrix(index m, index n) : GradedMatrix<degree>(m, n) {}

    index kMax;

    /**
     * @brief Returns a vector containing the degrees of the columns and rows.
     * 
     * @return degree_list 
     */
    degree_list discreteSupport() {
      degree_list result;
      result.reserve(num_cols + num_rows);
      assert(col_degrees.size() == num_cols);
      assert(row_degrees.size() == num_rows);
        for(degree g : col_degrees) {
            result.push_back(g);
        }
        for(degree g : row_degrees) {
            result.push_back(g);
        }
      return result;
    }  

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     */
    R2GradedMatrix(const std::string& filepath) : GradedMatrix<degree>() {

        size_t dotPosition = filepath.find_last_of('.');
    
        if (dotPosition == std::string::npos) {
           // No dot found, invalid file format
            std::cerr << "Error: Invalid file format. File must have an extension (.scc or .firep)." << std::endl;
            std::abort();
        }

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filepath << std::endl;
            std::abort();
        }

    

        std::string extension = filepath.substr(dotPosition);
        std::string line;


        // Check the file extension and perform actions accordingly
        if (extension == ".scc" || extension == ".firep") {
            std::cout << "Reading presentation file: " << filepath << std::endl;
            std::getline(file, line);
            if (line.find("firep") != std::string::npos) {
                std::cout << "Reading FIREP presentation file: " << filepath << std::endl;
                // Skip 2 lines for FIREP
                std::getline(file, line);
                std::getline(file, line);
            } else if (line.find("scc2020") != std::string::npos) {
                std::cout << "Reading SCC2020 presentation file: " << filepath << std::endl;
                // Skip 1 line for SCC2020
                std::getline(file, line);
            } else {
            // Invalid file type
                std::cerr << "Error: Unsupported file type. Supported types are FIREP and SCC2020." << std::endl;
                std::abort();
            }
        } else {
            // Invalid file extension
            std::cerr << "Error: Unsupported file extension. Supported extensions are .scc and .firep." << std::endl;
            std::abort();
        }

        // Parse the first line after skipping
        std::getline(file, line);
        std::istringstream iss(line);
        index num_rel, num_gen, thirdNumber;
    
        // Check that there are exactly 3 numbers
        if (!(iss >> num_rel >> num_gen >> thirdNumber) || thirdNumber != 0) {
            std::cerr << "Error: Invalid format in the first line. Expecting exactly 3 numbers with the last one being 0." << std::endl;
            std::abort();
        }

        num_cols = num_rel;
        num_rows = num_gen;

        col_degrees.reserve(num_rel);
        row_degrees.reserve(num_gen);
        data.reserve(num_gen);
        index counter = 0;

        while (std::getline(file, line) && counter < num_rel + num_gen) {
        
            if(counter < num_rel){
                auto line_data = parse_line(line);
                col_degrees.push_back(line_data.first);  
                counter ++;
                data.push_back(line_data.second);
            } else {
                auto line_data = parse_line(line, false);
                counter ++;
                row_degrees.push_back(line_data.first);
            }  
    	}   

    
    }

	std::pair<indvec, indvec> get_k_statistics(){
		
		indvec rel_k(100);
		indvec gen_k(100);
		degree tmp;
		index counter = 0;
		for(index i = 0; i < num_cols; i++){
			if(col_degrees[i] == tmp){
				counter++;
			} else {
				rel_k[counter]++;
				counter = 0;
				tmp = col_degrees[i];
			}
		}
        counter = 0;
        tmp = degree(-1,-1);
        for(index i = 0; i < num_rows; i++){
            if(row_degrees[i] == tmp){
                counter++;
            } else {
                gen_k[counter]++;
                counter = 0;
                tmp = row_degrees[i];
            }
        }
        return std::make_pair(rel_k, gen_k);
	}

	/**
	 * @brief This function computes a quiver representation on the poset of unique degrees 
     * appearing for the columns and rows of the matrix.
	 */
	void computeQuiverRepresentation(QuiverRepresentation& rep){
		rep.degrees = this->discreteSupport();
 
        kMax = delete_multiples(rep.degrees);
    
        std::cout << "Maximal k: " << kMax << std::endl;

		rep.edges = minimal_directed_graph(rep.degrees );
        // print_edge_list(edges);
		index num_vert = rep.degrees .size();
		index num_edges = rep.edges.size();
		std::cout << "Created Hasse-Diagram of the Partial Order on the support of size " << num_vert << " V "<< num_edges << "E" << std::endl;
		// For each degree we want to store the cokernel, 
		// the row-indices of the generators which form its domain 
		// and a section of the cokernel given by column indices which are mapped to a basis
		vec< SparseMatrix > pointwise_Presentations;
        vec< std::pair< indvec , indvec> > pointwise_base;
		pointwise_Presentations.reserve(num_vert);
        pointwise_base.reserve(num_vert);
		rep.dimensionVector.reserve(num_vert);
        index progress_report = num_vert/10;
        if(progress_report == 0){
            progress_report = 1;
        }
        std::chrono::duration<double> map_duration_max = std::chrono::duration<double>::zero();
        // #pragma omp parallel for
		for (index i = 0; i < num_vert; i++) {
            // auto start = std::chrono::high_resolution_clock::now();

            if (i % progress_report == 0) {
                // #pragma omp critical
                std::cout << "Progress: " << i << " / " << num_vert << std::endl;
                // std::cout << "Map at degree time: " << map_duration_max.count() << " seconds" << std::endl;
            }

            // Measure time for map_at_degree
            auto map_start = std::chrono::high_resolution_clock::now();
            SparseMatrix S;
            indvec gens;
            std::tie(S, gens) = this->map_at_degree(rep.degrees[i]);
            auto map_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> map_duration = map_end - map_start;
            if(map_duration > map_duration_max){
                map_duration_max = map_duration;
            }
            // Measure time for column_reduction
            // auto column_reduction_start = std::chrono::high_resolution_clock::now();
            S.column_reduction();
            // auto column_reduction_end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> column_reduction_duration = column_reduction_end - column_reduction_start;

            indvec basisLift;

            // Measure time for coKernel
            // auto coKernel_start = std::chrono::high_resolution_clock::now(); 
            pointwise_Presentations.emplace_back(S.coKernel(true, &basisLift));
            // auto coKernel_end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> coKernel_duration = coKernel_end - coKernel_start;

            pointwise_base.emplace_back(std::make_pair(gens, basisLift));
			rep.dimensionVector.emplace_back(basisLift.size());

            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> duration = end - start;

            // Print durations
            
            // std::cout << "Map at degree time: " << map_duration.count() << " seconds" << std::endl;
            // std::cout << "Column reduction time: " << column_reduction_duration.count() << " seconds" << std::endl;
            // std::cout << "CoKernel computation time: " << coKernel_duration.count() << " seconds" << std::endl;
            // std::cout << "Total iteration time: " << duration.count() << " seconds" << std::endl;
		}
        std::cout << "computed " << pointwise_Presentations.size() << "  quotient spaces." <<std::endl;

        std::cout << "Max map_at_degree time: " << map_duration_max.count() << " seconds" << std::endl;
        
        // Now we want to compute the path actions of the representation
        rep.matrices.reserve(num_edges);
		for (index i = 0; i < num_edges; i++) {
				index source = rep.edges[i].first;
				index target = rep.edges[i].second;

				auto sourceBasis = pointwise_base[source];
				auto targetBasis = pointwise_base[target];

				assert(pointwise_Presentations[source].num_cols == sourceBasis.first.size());
				assert(pointwise_Presentations[target].num_cols == targetBasis.first.size());

				indvec lift_of_basis = basisLifting(sourceBasis, targetBasis);
				rep.matrices.emplace_back( pointwise_Presentations[target].restricted_domain_copy(lift_of_basis) );
                // rep.matrices.back().print();
			}

		std::cout << "Computed all " << rep.matrices.size() << "matrices of the representation" << std::endl;
		// Sanity check to see if dimensions match up
    
		for (index j = 0; j < num_edges; j++) { 
            if(rep.matrices[j].num_cols != rep.dimensionVector[rep.edges[j].first] || rep.matrices[j].num_rows != rep.dimensionVector[rep.edges[j].second]){
                throw std::runtime_error("Dimension mismatch in path action");
            }
        }      
	}

    
    void print_graded() {

        print();
        std::cout << "Column Degrees: " ;
        print_degree_list(col_degrees);
        std::cout << "Row Degrees: ";
        print_degree_list(row_degrees);
    }

    
};



namespace fs = std::filesystem;

std::tuple<double, index, index, index, index, index> processFile(std::string& filePath) {
    std::cout << "Processing file: " << filePath << std::endl;
    R2GradedMatrix A(filePath);

    // A.print_graded();

    index c = A.num_cols;
    index r = A.num_rows;
    // auto k_statistics = A.get_k_statistics();

    // foldername for the quiver representations
    std::string folderName = "quiver_representations";
    // Extract the folder path and file name
    size_t lastSlashPos = filePath.find_last_of('/');
    std::string folderPath = (lastSlashPos != std::string::npos) ? filePath.substr(0, lastSlashPos) : "";
    std::string fileName = (lastSlashPos != std::string::npos) ? filePath.substr(lastSlashPos + 1) : filePath;

    // Check if the filename has a file extension
    std::string endingToCheck = "_min_pres";

    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos) {
        fileName = fileName.substr(0, dotPos);
    }

    if (fileName.length() >= endingToCheck.length() &&
        fileName.compare(fileName.length() - endingToCheck.length(), endingToCheck.length(), endingToCheck) == 0) {
        // File path ends with "_min_pres", cut it off
        fileName.resize(fileName.length() - endingToCheck.length());
    } else {
        // File path doesn't end with "_min_pres"
        std::cerr << "Error: File path does not end with '_min_pres'" << std::endl;
    }

    // Build the full folder path
    std::string fullFolderPath = (folderPath.empty()) ? std::filesystem::current_path().string() + "/" + folderName : folderPath + "/" + folderName;

    // Create the folder
    std::filesystem::create_directories(fullFolderPath);

    // Build the full file path including the folder
    std::string fullFilePath = fullFolderPath + "/" + fileName;

    
    std::cout << "Matrix Dimensions: " << c << " Cols x " << r << " Rows" << std::endl;   
    auto startTime = std::chrono::high_resolution_clock::now();
    QuiverRepresentation rep;
    A.computeQuiverRepresentation(rep);
    std::cout << "Computed Quiver Representation" << std::endl;
    index kMax = A.kMax;
    // rep.print();
    rep.writeQPAFile(fullFilePath, fileName);

    std::pair<index, index> graphSize = std::make_pair(rep.dimensionVector.size(),rep.edges.size());
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration<double>(endTime - startTime).count();
    return std::make_tuple(elapsedTime, c, r, graphSize.first, graphSize.second, kMax);
}

void iterateOverFiles(const std::string& folderPath) {
  // Open a text file for writing
    std::ofstream resultFile(folderPath + "/runtime_statistics_for_pres_to_quiver.txt");
    if (!resultFile.is_open()) {
        std::cerr << "Error opening result file." << std::endl;
        return;
    }

    for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            
            // Process files with specific extensions
            if (filePath.size() >= 12 &&
                (filePath.compare(filePath.size() - 14, 14, "min_pres.firep") == 0 ||
                 filePath.compare(filePath.size() - 12, 12, "min_pres.scc") == 0)) {
                auto result = processFile(filePath);
                double elapsedTime = std::get<0>(result);
            index c = std::get<1>(result);
            index r = std::get<2>(result);
            index vertices = std::get<3>(result);
            index edges = std::get<4>(result);
            index kMax = std::get<5>(result);
            // Write results to the text file
            resultFile << "File: " << filePath ;
            resultFile << " Elapsed Time: " << elapsedTime << " seconds /n ";
            resultFile << "Matrix Dimensions: " << r << " x " << c << " with kMax of" << kMax << "\n";
            resultFile << "Graph Dimensions: " << vertices << " V " << edges << " E" << "\n \n";
            }
        }
    }
    resultFile.close();
}



} // namespace presentation_to_quiver





int main() {
  
    using grmatrix = presentation_to_quiver::R2GradedMatrix;

    std::string toyExample = "/home/wsljan/OneDrive/persistence_algebra/toy_examples/small_pres_with_nonzero_k_min_pres.firep";
    std::string folderPath = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets";
    std::string smallNontrivial = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets/k_fold_46_10_1_min_pres.firep";
    std::string largeExample = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets/k_fold_96_10_1_min_pres.firep";
    std::string smallExample = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_awith_rivet_datasets/points_on_sphere_7500_1_min_pres.firep";
    std::string largestExampleHere = "/home/wsljan/OneDrive/persistence_algebra/toy_examples/k_fold_175_10_1_min_pres.firep";
    std::string smallExampleHere = "/home/wsljan/OneDrive/persistence_algebra/toy_examples/k_fold_46_10_2_min_pres.firep";
    std::string smallExample2 = "/home/wsljan/OneDrive/persistence_algebra/toy_examples/noisy_circle_firep_8_0_min_pres.firep";
    std::string testExample = "/home/wsljan/OneDrive/persistence_algebra/toy_examples/test_pres_with_nonzero_k_min_pres.firep";
    std::string blabla = "/home/wsljan/rhomboidtiling/blabla_min_pres.firep";
    std::string folder1 = "/home/wsljan/benchmark_k_zero";
    std::string offhand = "/home/wsljan/benchmark_k_nontrivial/off.hand_min_pres.firep";
    std::string uniform_kfold = "/home/wsljan/generating_samples/point_sets/small_uniform";
    presentation_to_quiver::iterateOverFiles(uniform_kfold);

    // auto runTime = presentation_to_quiver::processFile(offhand);

    // std::cout << "Elapsed Time: " << std::get<0>(runTime) << " seconds" << std::endl;
    // std::cout << "Matrix Dimensions: " << std::get<1>(runTime) << " x " << std::get<2>(runTime) << " and kMax of " << std::get<5>(runTime) << std::endl;
    // std::cout << "Graph Dimensions: " << std::get<3>(runTime) << " V " << std::get<4>(runTime) << " E" << std::endl;
    
    return 0;
}



