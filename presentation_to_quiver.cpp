#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>
#include <algorithm>
#include <sys/stat.h>
#include <filesystem>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/transitive_reduction.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/graphviz.hpp>
#include <mpp_utils/create_graded_matrices_from_scc2020.h>
#include <mpp_utils/Graded_matrix.h>
#include <phat/boundary_matrix.h>
#include <scc/Scc.h>
#include <chrono>

using real = long double;
using degree = std::pair<real, real>;
using degreeList = std::vector<degree>;
using relation = std::vector<int>;
using decdegreeList = std::vector<std::pair<degree, relation>>;
using presentation = std::pair<decdegreeList, degreeList>;
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, degree>;
using GrMat = mpp_utils::Graded_matrix<phat::vector_vector>;

// Specialize std::hash for degree in the global namespace
namespace std {
    template <>
    struct hash<degree> {
        size_t operator()(const degree& d) const {
            // Combine hash values of first and second components of degree
            return hash<long double>()(d.first) ^ (hash<long double>()(d.second) << 1);
        }
    };
}

namespace pres_to_quiver {




/** 
 * @brief Index type for sparse vectors
*/
typedef mpp_utils::index index;

/**
 * @brief Sparse vector over F_2
 * 
 */
using Column = std::vector<index>;
/**
 * @brief 
 * 
 */
using Matrix = std::vector<Column>;


template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;

std::ostream& operator<< (std::ostream& ostr, const Column& c) {
    for(index i:c) {
      ostr << i << " ";
    }
    return ostr;
}

template<typename matrix>
struct sMatrix{
    virtual index get_num_cols() = 0;
    virtual index get_num_rows() = 0;
    virtual void col_op(index i, index j) = 0;
    virtual void row_op(index i, index j) = 0;
    virtual bool is_nonzero_col_entry(index i, index j) = 0;
    std::unordered_map<index,index> pivots;
    virtual index col_last(index i) = 0;
    virtual index row_last(index i) = 0;
    vec<std::set<index>> col_last_vec;

    /**
     * @brief Points from a row index to all the columns that have a largest non-zero entry in that row
     * 
     */
    void compute_col_last(){
      col_last_vec.resize(get_num_rows());
      for(index j = 0; j < get_num_rows(); j++){
        std::set<index> pivots;
        col_last_vec[j]=pivots;
      }
      for (index i = 0; i < get_num_cols(); i++){
        index l = col_last(i);
        if(l>=get_num_rows()){
          throw std::out_of_range("There is an index in the column that is larger than the number of rows at: " + std::to_string(i));
        } 
        if(l>=0){
          col_last_vec[l].insert(i);
        }
      }
    }

   
   /**
    * @brief Brings Matrix in reduced Column Echelon form.
    * 
    */
  void col_reduce() {

    // Iterate through each column
    for (index col = 0; col < get_num_cols(); ++col) {
        // Find the pivot row with the leftmost non-zero entry in the current column
        index pivotRow = col_last(col);

        if (pivotRow != -1) {
            for (index otherCol = 0; otherCol < get_num_cols(); ++otherCol) {
                if (col != otherCol && is_nonzero_col_entry(otherCol, pivotRow)) {
                    col_op(col, otherCol);
                }
            }
        }
    }
  }

    /**
     * @brief Not working right now! Brings Matrix in reduced Column Echelon form, but without direct swaps
     * 
     */
    void col_reduce_complete_sparse() {
        index s;
        compute_col_last();
        for (index i = get_num_rows()-1; i >= 0; i--){
          if(!col_last_vec[i].empty()){
              s = *col_last_vec[i].begin();
            } else {
              continue;
            }
            for(index j : col_last_vec[i]){
                if(j != s){
                    col_op(s,j);
                    col_last_vec[i].erase(j);
                    index l = col_last(j);
                    col_last_vec[l].insert(j);
                }
            }
            for (index k = i + 1 ; k < get_num_rows(); k++){
                for (index j : col_last_vec[k]){
                    if(is_nonzero_col_entry(j,i)){
                        col_op(s,j);
                    }
                }    
            }    
        }
    }

    void row_reduce() {

    }

   
};


/**
 * @brief checks if j is in C[i].
 * 
 * @param C 
 * @param i 
 * @param j 
 * @return true 
 * @return false 
 */
bool entry_at(Matrix& C, index i, index j) {
    return std::binary_search(C[i].begin(),C[i].end(),j);
  }

/**
 * @brief sparse column addition over F_2. Adds a to b.
 * 
 * @param a a vector containing integers representing the indices of the nonzero entries of the column
 * @param b a vector containing integers representing the indices of the nonzero entries of the column
 */
void add_to(Column& a, Column& b) {
    //std::cout << "ADDITION" << std::endl;
    Column c;
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
   * @brief Column reduces the matrix S. Pass a reference to an empty pivot map to get the pivots after reducing.
   *       
   * 
   * @param S Sparse Matrix over F_2  
   * @param pivots Should be empty, when the function is called. 
   * Maps a row index to the first column index where it is the largest entry.
   */
void reduce(Matrix& S, std::unordered_map<index,index>& pivots) {
  for(index j=0;j<S.size();j++) {
    //std::cout << "index: " << j << std::endl;
    Column& curr = S[j];
    while(!curr.empty()) {
      index p = curr.back();
      /*
      if(p<threshold) {
        break;
      }
      */	
      if(pivots.count(p)) {
        index i = pivots[p];
        add_to(S[i],curr);
        assert(curr.empty() || curr.back()<p);
        p = curr.back(); //Isn't this line superfluous?
      } else {
        pivots[p]=j;
        break;
      }
    }
  } 
}

/**
  * @brief Set the pivots without reducing object
  * 
  * @param S 
  * @param pivots 
*/
void set_pivots_without_reducing(Matrix& S, std::unordered_map<index,index>& pivots) {
  for(index j=0;j<S.size();j++) {
    //std::cout << "index: " << j << std::endl;
    Column& curr = S[j];
    if(!curr.empty()) {
      index p = curr.back();
      if(! pivots.count(p)) {
        pivots[p]=j;
      }
    }
  }
}

/**
  * @brief Column-reduces the matrix, then counts the number of non-empty columns.
  * 
  * @param S 
  * @return index 
  */
index rank(Matrix& S) {
  std::unordered_map<index,index> pivots;
  reduce(S,pivots);
  index _rank=0;
  for(index i=0;i<S.size();i++) {
    if(!S[i].empty()) {
	    _rank++;
    }
  }
  return _rank;
}



struct Point {
    double x;
    double y;
    Point(double x,double y) : x(x), y(y) {}
    
    bool operator< (const Point& p) {
      return this->x < p.y && this->y < p.y;
    }

    bool operator<= (const Point& p) {
      return this->x <= p.x && this->y <= p.y;
    }

    bool operator==(const Point& p) {
      return this->x==p.x && this->y==p.y;
    }
};
    
Point join(Point& a, Point& b) {
  return Point(std::max(a.x,b.x), std::max(a.y,b.y));
};


struct sparseMatrix : sMatrix<Matrix>{
    Matrix A;
    index no_cols;
    index no_rows;

    sparseMatrix() {
      no_cols=0;
      no_rows=0;
    }

  /**
   * @brief Construct a new sparse Matrix object
   * 
   * @param no_cols 
   * @param no_rows 
   */
  sparseMatrix(index no_cols, index no_rows) : no_cols(no_cols), no_rows(no_rows) {
    A = Matrix(no_cols);   
  }

  // Copy constructor
    sparseMatrix(const sparseMatrix& other) : A(other.A), no_cols(other.no_cols), no_rows(other.no_rows) {}

    // Copy assignment operator
    sparseMatrix& operator=(const sparseMatrix& other) {
        if (this != &other) {
            A = other.A; // Assuming Matrix has a proper copy assignment operator
            no_cols = other.no_cols;
            no_rows = other.no_rows;
        }
        return *this;
    }

    // Move constructor
    sparseMatrix(sparseMatrix&& other) noexcept : A(std::move(other.A)), no_cols(other.no_cols), no_rows(other.no_rows) {
        // Reset the source object
        other.no_cols = 0;
        other.no_rows = 0;
    }

    // Destructor
    ~sparseMatrix() {
        // std::cout << "sparseMatrix Destructor Called on the instance of size" << get_num_cols() << " x "<< get_num_rows() << std::endl;
        // Destructor of Matrix (A) is automatically called when sparseMatrix is destroyed.
    }

    index get_num_cols() {
      return no_cols;
    }  
    index get_num_rows() {
      return no_rows;
    }

    void col_op(index i, index j) {
      add_to(A[i],A[j]);
    }

    // AI generated - check if correct
    void row_op(index i, index j) {
      for(index k : A[i]) {
        auto it = std::find(A[j].begin(),A[j].end(),k);
        if(it==A[j].end()) {
          A[j].push_back(k);
        }
      }
      std::sort(A[j].begin(),A[j].end());
    }

    bool is_nonzero_col_entry(index i, index j) {
      return entry_at(A,i,j);
    }

    index col_last(index i) {
      if(A[i].empty()) {
        return -1;
      }
      return A[i].back();
    }

    // AI generated - check if correct
    index row_last(index i) {
      index result = -1;
      for(index j : A[i]) {
        result = std::max(result,j);
      }
      return result;
    }

  /**
   * @brief 
   * 
   * @return sparseMatrix 
   */
  sparseMatrix transposedCopy() const {
    sparseMatrix result(no_rows,no_cols);
    for(index i=0;i<no_cols;i++) {
      for(index j : A[i]) {
        result.A[j].push_back(i);
      }
    }
    return result;
  }

    void print(){
      assert(A.size()==no_cols);
      std::cout << "Cols: " << no_cols << " rows: " << no_rows << std::endl;
      for(index i=0;i<no_cols;i++) {
        std::cout << "Column " << i << ": " << A[i] << std::endl;
      }
    
    }
  bool equals(sparseMatrix other){
    if(no_cols != other.no_cols){
      std::cout << "#columns dont match.";
      return false;
    }
    if(no_rows != other.no_rows){
      std::cout << "#rows dont match.";
      return false;
    }
    for(index i = 0; i< no_cols; i++){
      if(A[i] != other.A[i]){
        std::cout << "columns at index " << i << " dont match.";
        return false;
      }
    }
    return true;
  }
  
};



void testTransposition() {
  sparseMatrix A(3,4);
  A.A[0] = {0,1};
  A.A[1] = {1,2};
  A.A[2] = {0,3};
  A.print();
  sparseMatrix B = A.transposedCopy();
  std::cout << B.get_num_cols() << "x" << B.get_num_rows() << std::endl;
  B.print();
}

template<typename Grade>
struct Decomp_matrix {
    
    Matrix _matrix;

    std::vector<Grade> column_grades;
    std::vector<Grade> row_grades;

    std::vector<std::unordered_set<index>> _rows;

    index _num_rows;
    index _num_cols;

    // admissable_col[i] stores to what column i can be added
    std::vector<Column> admissable_col;
    // admissable_row[i] stores what can be added to i
    std::vector<Column> admissable_row;

    Decomp_matrix(std::string filename) {
      presentation pres = read_presentation(filename);
      decdegreeList relations = pres.first;
      degreeList generators = pres.second;

        _num_rows = generators.size(); 
        _num_cols = relations.size(); 

        // May need some better memory allocation.  
        column_grades.reserve(_num_cols);
    _   matrix.reserve(_num_cols);

        for (const auto& relation : relations) {
            column_grades.push_back(relation.first);
            _matrix.push_back(relation.second);
        }

        row_grades = generators; 

    }

    degreeList discreteSupport() {
      degreeList result;
      for(Grade g : column_grades) {
        result.push_back(std::make_pair((real)g.first_val,(real)g.second_val));
      }
      for(Grade g : row_grades) {
        result.push_back(std::make_pair((real)g.first_val,(real)g.second_val));
      }
      return result;
    }  

    std::pair<Point,Point> bounding_box() {
      double x_min,x_max,y_min,y_max;
      if(column_grades.size()>0) {
        x_max=x_min=column_grades[0].first_val;
        y_max=y_min=column_grades[0].second_val;
      } else if(row_grades.size()>0) {
        x_max=x_min=row_grades[0].first_val;
        y_max=y_min=row_grades[0].second_val;
      } else {
	      x_max=x_min=y_max=y_min=0;
      }
      for(Grade g : column_grades) {
        double xval = g.first_val;
        x_min=std::min(x_min,xval);
        x_max=std::max(x_max,xval);
        double yval = g.second_val;
        y_min=std::min(y_min,yval);
        y_max=std::max(y_max,yval);
      }
      for(Grade g : row_grades) {
        double xval = g.first_val;
        x_min=std::min(x_min,xval);
        x_max=std::max(x_max,xval);
        double yval = g.second_val;
        y_min=std::min(y_min,yval);
        y_max=std::max(y_max,yval);
      }
      return std::make_pair(Point(x_min,y_min),Point(x_max,y_max));
    }

    long number_of_entries() {
      long result=0;
      for(int i=0;i<this->get_num_cols();i++) {
        Column col;
        this->get_col(i,col);
        result+=col.size();
      }
      return result;
    }

    template<typename GradedMatrix>
    void precompute_admissable(GradedMatrix& A) {
      admissable_col.resize(this->get_num_cols());
      for(index j=0;j<this->get_num_cols();j++) {
        for(index i=0;i<this->get_num_cols();i++) {
          if(this->is_admissable_column_operation(i,j) && i!=j) {
            this->admissable_col[i].push_back(j);
          }
        }
      }
      admissable_row.resize(this->get_num_rows());
      for(index j=0;j<this->get_num_rows();j++) {
        for(index i=0;i<this->get_num_rows();i++) {
          if(this->is_admissable_row_operation(i,j) && i!=j) {
            this->admissable_row[j].push_back(i);
          }
        }
      }
    }

    
    template<typename GradedMatrix>
    Decomp_matrix(GradedMatrix& A) {
      this->_matrix.resize(A.get_num_cols());
      this->_rows.resize(A.num_rows);
      std::copy(A.row_grades.begin(),A.row_grades.end(),std::back_inserter(this->row_grades));
      std::copy(A.grades.begin(),A.grades.end(),std::back_inserter(this->column_grades));
      for(index i=0;i<this->get_num_cols();i++) {
        Column col;
        A.get_col(i,col);
        this->_matrix[i]=col;
        for(index j : col) {
          this->_rows[j].insert(i);
        }  
      }
      precompute_admissable(A);
    }
      
    index get_num_cols() {
      return this->_matrix.size();
    }
    
    index get_num_rows() {
      return this->_rows.size();
    }

    void get_col(index i, Column& col) {
      col=this->_matrix[i];
    }

    // Return whether at least one operation has been performed
    bool eliminate_row(index row_id, index col_id) {
      assert(this->_matrix[col_id].size()==1);
      assert(this->_matrix[col_id].back()==row_id);
      bool result=false;
      Column row;
      this->get_row(row_id,row);
      for(index k : row) {
        if(k<=col_id) {
          continue;
        }
        if(this->is_admissable_column_operation(col_id,k)) {
          //std::cout << "Adding column " << col_id << " to column " << k << std::endl;
          this->col_op(col_id,k);
          result=true;
        }
      }
      return result;
    }

    // Return whether at least one operation has been performed
    bool eliminate_column(index col_id, index row_id) {
      assert(this->_rows[row_id].size()==1);
      assert(this->_rows[row_id].count(col_id));
      bool result=false;
      Column col;
      this->get_col(col_id,col);
      for(index k : col) {
        if(k==row_id) {
          break;
        }
        if(this->is_admissable_row_operation(row_id,k)) {
          //std::cout << "Adding row " << row_id << " to row " << k << std::endl;
          this->row_op(row_id,k);
          result=true;
        }
      }
      return result;
    }

    void sparsify_column(index j) {
      Column col;
      this->get_col(j,col);

      for(index i : col) {
        Column i_row;
        this->get_row(i,i_row);
        if(i_row.front()==j) {
          for(index k : col) {
            if(k==i) {
              break;
            }
            if(this->is_admissable_row_operation(i,k)) {
              //std::cout << "Adding row " << i << " to row " << k << std::endl;
              this->row_op(i,k);
              //std::cout << "Entries " << this->number_of_entries() << std::endl;
            }
          }
        }
      }
    }
      

    void set_col(index i, Column& col,bool prune=false) {
      //Remove the entries from the rows
      std::vector<index> row_indices_to_check;
      for(index j : this->_matrix[i]) {
        if(prune) {
          row_indices_to_check.push_back(j);
        }
        this->_rows[j].erase(i);
            }
            // Add the new entries to the rows
            for(index j : col) {
        this->_rows[j].insert(i);
        if(prune) {
          row_indices_to_check.push_back(j);
        }
            }
            this->_matrix[i]=col;
            if(prune) {
        if(col.size()==1) {
          index p = col.back();
          this->eliminate_row(p,i);
        }
        for(index k : row_indices_to_check) {
          if(this->_rows[k].size()==1) {
            index p = *(this->_rows[k].begin());
            this->eliminate_column(p,k);
          }
        }
      }
    }

    void get_row(index i, Column& row) {
      row.clear();
      std::copy(this->_rows[i].begin(),this->_rows[i].end(),std::back_inserter(row));
      std::sort(row.begin(),row.end());
    }    

    bool is_admissable_column_operation(index i, index j) {
      auto gr1 = this->column_grades[i];
      auto gr2 = this->column_grades[j];
      return gr1.first_val <= gr2.first_val && gr1.second_val <= gr2.second_val;
    }

    bool is_admissable_row_operation(index i, index j) {
      auto gr1 = this->row_grades[i];
      auto gr2 = this->row_grades[j];
      //std::cout << "Grade info: " << gr1.first_val << " " << gr1.second_val << " - " << gr1.second_val << " " << gr2.second_val << std::endl;
      return gr1.first_val >= gr2.first_val && gr1.second_val >= gr2.second_val;
    }

    
    void get_admissable_col(index i, Column& col) {
      col =  this->admissable_col[i];
    }

    void get_admissable_row(index i, Column& col) {
      col =  this->admissable_row[i];
    }
    
    void print(bool with_admissable=true) {
        std::cout << "Number of columns: " << this->get_num_cols() << std::endl;
        for(index i=0;i<this->get_num_cols();i++) {
            std::cout << "Column " << i << ": " << this->_matrix[i] << std::endl;
        }
        std::cout << "Number of rows: " << this->get_num_rows() << std::endl;
        for(index i=0;i<this->get_num_rows();i++) {
            Column row;
            this->get_row(i,row);
            std::cout << "Row " << i << ": " << row << std::endl;
        }
        if(with_admissable) {
            std::cout << "Admissable per column: " << std::endl;
        for(index i=0;i<this->get_num_cols();i++) {
            Column col;
            get_admissable_col(i,col);
            std::cout << "Column " << i << ": " << col << std::endl;
        }
        std::cout << "Admissable per row: " << std::endl;
        for(index i=0;i<this->get_num_rows();i++) {
            Column col;
            get_admissable_row(i,col);
            std::cout << "Row " << i << ": " << col << std::endl;
        }
      }
    }

    // Just for testing
    void col_op(index i,index j) {
      assert(i<j);
      Column col=this->_matrix[j];
      add_to(this->_matrix[i],col);
      // Needed to update the rows
      this->set_col(j,col);
    }

    void row_op(index i,index j) {
      assert(i>j);
      Column row;
      this->get_row(i,row);
      for(index k : row) {
        if(this->_rows[j].count(k)) {
          this->_rows[j].erase(k);
          auto col_it = std::find(this->_matrix[k].begin(),this->_matrix[k].end(),j);
          assert(col_it!=this->_matrix[k].end());
          this->_matrix[k].erase(col_it);
        } else {
          this->_rows[j].insert(k);
          auto col_it = this->_matrix[k].begin();
          while(col_it!=this->_matrix[k].end() && *col_it <j) {
            col_it++;
          }
          assert(col_it==this->_matrix[k].end() || *col_it > j);
          this->_matrix[k].insert(col_it,j);
        }
      }
    }

    /**
     * @brief computes the linear map induced by considering a single degree by cutting all columns and rows of a higher degree.
     * 
     * @param d 
     * @return std::pair<sparseMatrix, vec<index>> 
     */
    std::pair<sparseMatrix, vec<index>> map_at_degree(degree d) {
      std::vector<index> selectedRowDegrees;
      for(index i=0;i<this->get_num_rows();i++) {
        if(this->row_grades[i].first_val<=d.first && this->row_grades[i].second_val<=d.second) {
          selectedRowDegrees.push_back(i);
        }
      }
      int new_row = selectedRowDegrees.size();
      sparseMatrix result;
      result.no_rows = new_row;
      for(index i=0;i<this->get_num_cols();i++) {
        if(this->column_grades[i].first_val<=d.first && this->column_grades[i].second_val<=d.second) {
          result.A.push_back(this->_matrix[i]);
        }
      }
      result.no_cols = result.A.size();
      return std::move(std::make_pair(result,selectedRowDegrees));
    }

    void random_shuffle(index no_of_tries=100) {
      
      srand(time(NULL));
      int n = this->get_num_cols();
      int m = this->get_num_rows();
      
      for(int i=0;i<no_of_tries;i++) {
	
        bool col_op = (rand()%2==1);
        
        if(col_op) {
        int source = rand()%n;
        Column admissable_col_op;
        this->get_admissable_col(source,admissable_col_op);
        index no_cand = admissable_col_op.size();
        if(no_cand>0) {
            int target = rand()%no_cand;
            //std::cout << "Adding column " << source << " to " << admissable_col_op[target] << std::endl;
            this->col_op(source,admissable_col_op[target]);
        } else {
            //std::cout << "No admissable column operations found for column " << source << std::endl;
        }
        } else {
        int target = rand()%m;
        Column admissable_row_op;
        this->get_admissable_row(target,admissable_row_op);
        index no_cand = admissable_row_op.size();
        if(no_cand>0) {
            int source = rand()%no_cand;
            //std::cout << "Adding row " << admissable_row_op[source] << " to " << target << std::endl;
            this->row_op(admissable_row_op[source],target);
        } else {
            //std::cout << "No admissable row operations found for row " << target << std::endl;
        }
        
        }
      }
    }
};

bool operator==(const degree& lhs, const degree& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

bool operator<(const degree& lhs, const degree& rhs) {
    return ( (lhs.first < rhs.first) && (lhs.second <= rhs.second) ) 
    || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
}


bool compareLexicographically(const degree& a, const degree& b) {
    if (a.first != b.first) {
        return a.first < b.first;
    }
    return a.second < b.second;
}


/**
 * @brief Performs depth-first search (DFS) for transitive reduction.
 *
 * This function is a recursive helper function for DFS traversal used in transitive
 * reduction. It marks vertices as visited and removes transitive edges.
 *
 * @param start The starting vertex of the DFS.
 * @param current The current vertex in the DFS.
 * @param graph The graph to perform transitive reduction on.
 * @param visited A boolean vector to track visited vertices.
 */
void dfsTransitiveReduction(Graph::vertex_descriptor start,
                             Graph::vertex_descriptor current,
                             Graph& graph,
                             std::vector<bool>& visited) {
    visited[current] = true;

    // Iterate over out-edges of the current vertex
    for (auto outEdge : boost::make_iterator_range(out_edges(current, graph))) {

        
        auto targetVertex = target(outEdge, graph);

        // Check if the target vertex is reachable from the start vertex
        if (!visited[targetVertex]) {
            // Mark the target vertex as visited
            dfsTransitiveReduction(start, targetVertex, graph, visited);
        } else {
            // If the target vertex is already visited, remove the edge
            if (targetVertex != start) {
                remove_edge(current, targetVertex, graph);
            }
        }
    }
}

/**
 * @brief Applies transitive reduction to a directed graph.
 *
 * This function iterates over all vertices in the graph and performs depth-first
 * search (DFS) for transitive reduction.
 *
 * @param graph The directed graph to apply transitive reduction on.
 */
void transitiveReduction(Graph& graph) {
    // Iterate over all vertices in the graph
    for (auto v : boost::make_iterator_range(vertices(graph))) {
        // Initialize visited array for each DFS call
        std::vector<bool> visited(num_vertices(graph), false);

        // Perform DFS starting from vertex v
        dfsTransitiveReduction(v, v, graph, visited);
    }
}

using edgeList = vec<std::pair<index,index>>;

void writeDegreeListToCSV(const std::string& filename, const degreeList& degrees) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    outfile << std::fixed << std::setprecision(15);

    // Write header if needed
    outfile << "X,Y" << std::endl;


    for (const auto& pair : degrees) {
        outfile << pair.first << "," << pair.second << std::endl;
    }

    std::cout << "Real pairs written to: " << filename << std::endl;
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


edgeList getEdgesFromGraph(Graph& graph){
  edgeList edges;
  for (auto e : boost::make_iterator_range(boost::edges(graph))) {
    edges.push_back(std::make_pair(boost::source(e, graph), boost::target(e, graph)));
  }
  return edges;
}




std::pair<degree, relation> parse_line(const std::string& line, bool hasEntries = true) {
    std::istringstream iss(line);
    degree degree;
    relation rel;

    // Parse degree
    iss >> degree.first >> degree.second;

    // Skip the semicolon
    iss.ignore();

    // Parse relation
    if(hasEntries){
    int num;
    while (iss >> num) {
        rel.push_back(num);
    }
    }

    return std::make_pair(degree, rel);
}

presentation read_presentation(const std::string& filepath) {
    // Find the position of the dot in the file extension
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

    decdegreeList relations;
    degreeList generators;
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
    int noRel, noGen, thirdNumber;
    
    // Check that there are exactly 3 numbers
    if (!(iss >> noRel >> noGen >> thirdNumber) || thirdNumber != 0) {
        std::cerr << "Error: Invalid format in the first line. Expecting exactly 3 numbers with the last one being 0." << std::endl;
        std::abort();
    }

    int counter = 0;

    while (std::getline(file, line) && counter < noRel) {
        if(counter < noRel){
            relations.push_back(parse_line(line));  
            counter ++;
        } else {
            generators.push_back(parse_line(line, false).first);
        }  
    }   

    return std::make_pair(relations, generators);
}



// Need to delete duplicates. Might need annotation if gen or rel or both
degreeList presentation_to_degreeList(const presentation& inputVector) {
    std::vector<degree> resultVector;
    resultVector.reserve(inputVector.first.size() + inputVector.second.size()); // Reserve space for efficiency

    // Add degrees from decdegreeList part
    for (const auto& pair : inputVector.first) {
        const degree& degree = pair.first;
        resultVector.push_back(degree);
    }

    // Add degrees from degreeList part
    for (const auto& degree : inputVector.second) {
        resultVector.push_back(degree);
    }

    return resultVector;
}



std::vector<GrMat> load_matrix(std::string& infile) {
    typedef GrMat::Grade Grade;
    std::vector<GrMat> matrices;
    std::ifstream ifstr(infile);
    scc::Scc<> parser(ifstr);
    mpp_utils::create_graded_matrices_from_scc2020(parser,1,2,matrices);
    return matrices;
}

using Grade = GrMat::Grade;
using sgMatrix = Decomp_matrix<Grade>;


/**
 * @brief This function is used to delete rows in a LOC sparse matrix. 
 * It creates a map which maps the old indices to the new indices.
 * 
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index> 
 */
std::unordered_map<index, index> shiftIndicesMap(const vec<index>& indices) {
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
void applyTransformation(vec<index>& target, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion = false) {
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
 * @brief Deletes the last entry of every column. Parallelised.
 *
 * @param S
 */
void deleteLastEntry(Matrix& S) {
#pragma omp parallel for
    for (std::size_t i = 0; i < S.size(); ++i) {
        vec<index>& c = S[i];
#pragma omp critical
        if (!c.empty()) {
            c.pop_back();  // Delete the last entry if the column is not empty
        }
    }
}

/**
 * @brief Parallelised function to change a sparse matrix by applying the indexMap to each entry.
 *
 * @param S
 * @param indexMap
 */
void transformMatrix(Matrix& S, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion) {
#pragma omp parallel for
    for (std::size_t i = 0; i < S.size(); ++i) {
        applyTransformation(S[i], indexMap, needsNoDeletion);
    }
}

/**
 * @brief Reduces all entries by 1, so that the indices start at 0.
 * 
 * @param S 
 */
void normaliseEntries(sparseMatrix& S) {
  std::unordered_map<index, index> indexMap;
  for (std::size_t i = 1; i < S.get_num_rows()+1; ++i) {
    indexMap[i] = i - 1;
  }
  transformMatrix(S.A, indexMap, false);
}

void printCol(Column& c){
  for(auto i : c){
    std::cout << i << " ";
  }
  std::cout << std::endl;

}

/**
 * @brief Computes the cokernel of a sparse matrix over F_2 by column reducing the matrix first
 * Notice that the result must be a cokernel to the non-reduced matrix, too.
 * 
 * @param S 
 * @param isReduced
 * @return sparseMatrix 
 */
sparseMatrix coKernel(sparseMatrix &S, bool isReduced = false, vec<index>* basisLift = nullptr){
  
  if(!isReduced){
    S.col_reduce();
  }
  index no_cols = S.get_num_cols();
  index no_rows = S.get_num_rows();
  S.compute_col_last();
  
  vec<index> quotientBasis;
  
  for(index i = 0; i < no_rows; i++){
    if(S.col_last_vec[i].empty()){
      quotientBasis.push_back(i);
    } else {
      // Check if matrix is really reduced completly
      assert(S.col_last_vec[i].size() == 1);
    }
  }

  auto indexMap = shiftIndicesMap(quotientBasis);
  auto truncS = S;
  deleteLastEntry(truncS.A);
  transformMatrix(truncS.A, indexMap, true);
  index newRows = quotientBasis.size();
  index newCols = no_rows;
  sparseMatrix result(newCols, newRows);
  index j = 0;
  for(index i = 0; i < newCols; i++){
    // Construct Identity Matrix on the generators which descend to the quotient basis. 
    if(j < quotientBasis.size() && quotientBasis[j] == i){
      result.A[i].push_back(j);
      j++;
    } else {
      // Locate the unqiue column with the last entry at i.
      index colForPivot = *S.col_last_vec[i].begin();
      result.A[i] = truncS.A[colForPivot];
    }
  }
  assert(j == quotientBasis.size() && "Not all quotient basis elements were used");
  if(basisLift){
    *basisLift = quotientBasis;
  }
  return std::move(result);
}

/**
 * @brief Returns the scalar product of two sparse vectors over F_2
 * 
 * @param v 
 * @param w 
 * @return true 
 * @return false 
 */
bool scalarProduct(vec<index>& v, vec<index>& w){

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

void testScalarProduct(){
  vec<index> v = {1, 3, 7, 8, 9};
  vec<index> w = {3, 4, 6};

  bool result = scalarProduct(v, w);
  assert(result == true);
}

/**
 * @brief Computes the transpose of M, then multiplies the columns.
 * 
 * @param M 
 * @param N 
 * @return product M*N over F_2 
 */
sparseMatrix multiply(sparseMatrix& M, sparseMatrix& N){
  sparseMatrix result(N.get_num_cols(), M.get_num_rows());
  sparseMatrix transpose = M.transposedCopy();
  for(index i = 0; i < N.get_num_cols(); i++){
    for(index j = 0; j < transpose.get_num_cols(); j++){
      if(scalarProduct(transpose.A[j], N.A[i])){ 
      result.A[i].push_back(j);
      }
    }
  }
  return result;
}

void testSparseMatrixMultiplication(){
  
  sparseMatrix M(4, 3);  
  M.A[0] = {0, 1};
  M.A[1] = {0};
  M.A[2] = {2};
  M.A[3] = {0, 2};

  sparseMatrix N(2, 4);  
  N.A[0] = {0, 2};
  N.A[1] = {1, 3};
 
  sparseMatrix MN(2,3);
  MN.A[0] = {0, 1, 2};
  MN.A[1] = {2};

  sparseMatrix MNtest = multiply(M, N);
  bool success = MN.equals(MNtest);
  assert(success);
}

/**
 * @brief Test the cokernel computation.
 * 
 */
void testCokernel(){
  sparseMatrix S(4,7);
  S.A[0] = {0,1,2, 4, 6};
  S.A[1] = {1,2, 5, 6};
  S.A[2] = {2, 3};
  S.A[3] = {0, 3, 4, 5};
  S.print();
  sparseMatrix T = coKernel(S);
  T.print();
  sparseMatrix zero = multiply(T, S);
  zero.print();
}


/**
 * @brief Returns a sparseMatrix with the columns of S at the indices given in colIndices.
 * 
 * @param S 
 * @param colIndices 
 * @return sparseMatrix 
 */
sparseMatrix restrictedDomain(sparseMatrix& S, vec<index>& colIndices){
  for(index i : colIndices){
    assert(i < S.get_num_cols());
  }
  sparseMatrix result(colIndices.size(), S.no_rows);
  result.A.clear(); //unclear if needed
  for(auto it = colIndices.begin(); it!= colIndices.end(); ++it){
    result.A.push_back(S.A[*it]);
  }
  return result;
}

void testMatrixFunctionality(sgMatrix &A){
  std::cout << "Number of columns: " << A.get_num_cols() << std::endl;
  std::cout << "Number of rows: " << A.get_num_rows() << std::endl;
  std::cout << "Number of entries: " << A.number_of_entries() << std::endl;
  degreeList support = A.discreteSupport();
  for(int i = 0; i < 10; i++){
    std::cout << "Support: " << support[i].first << " " << support[i].second << std::endl;
  }
  std::cout << "size of discrete support: " << support.size() << std::endl;
  degree big = support[support.size()-3];
  degree small = support[20];
  std::cout << "small: " << small.first << " " << small.second << std::endl;
  sparseMatrix S;
  vec<index> rowIndices;
  std::tie(S, rowIndices) = A.map_at_degree(small);
  normaliseEntries(S);
  std::cout << "number of rows in derived matrix " << S.no_rows << std::endl;
  std::cout << "number of cols in derived matrix " << S.no_cols << std::endl;
  S.print();
  // sparseMatrix T = coKernel(S);
  std::cout << "computed cokernel. Reduced matrix is: " << std::endl;
  S.print();
  std::cout << "Its cokernel is: " << std::endl;
  //T.print();

  // std::pair<Point,Point> bb = A.bounding_box();
  // std::cout << "Bounding box: " << bb.first.x << " " << bb.first.y << " - " << bb.second.x << " " << bb.second.y << std::endl;
  // A.random_shuffle(100);
  // A.print();
  // std::cout << "Number of entries: " << A.number_of_entries() << std::endl;
  // A.col_reduce_complete_sparse();
  
}

/**
 * @brief Get all indices for which an element in the first vector is contained in the second vector. Both inputs should be ordered.
 * 
 * @param target 
 * @param subset 
 * @return vec<index> 
 */
vec<index> getIndicatorVector(vec<index> target, vec<index> subset){
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
vec<T> extractElements(const vec<T>& target, const vec<index>& indices) {
    vec<T> result;
    result.reserve(indices.size());
    for (index i : indices) {
        assert(i >= 0 && i < target.size() && "Index out of bounds");
        result.push_back(target[i]);
    }
    return result;
}

/**
 * @brief The first vector in each argument is a set of row indices of the original matrix telling us which generators form the domain of the cokernel
 * The second vector is a set of column indices of the cokernel matrix defining a section of the cokernel.
 * 
 * @param source 
 * @param target 
 * @return vec<index> 
 */
vec<index> basisLifting(std::pair<vec<index>, vec<index>>& source, std::pair<vec<index>, vec<index>>& target){
  //sanity check, comment out in real run
  vec<index> subsetIndicator1 = getIndicatorVector(target.first, source.first);
  // Probably this doesnt need to be its own function
  vec<index> result = getIndicatorVector(target.first, extractElements(source.first, source.second));
  return result;
}

void createFolder(const std::string& folderPath) {
    // Use std::filesystem::create_directories for recursive folder creation
    std::filesystem::create_directories(folderPath);
}

void writeQPAFile(const Graph& G, std::vector<index>& dimensionVector, std::map<Graph::edge_descriptor, sparseMatrix>& edgeMatrices, const std::string& outfileName, const std::string& folderName) {
   // Extract the folder path and file name
    size_t lastSlashPos = outfileName.find_last_of('/');
    std::string folderPath = (lastSlashPos != std::string::npos) ? outfileName.substr(0, lastSlashPos) : "";
    std::string fileName = (lastSlashPos != std::string::npos) ? outfileName.substr(lastSlashPos + 1) : outfileName;

    // Check if the filename has a file extension
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos) {
        fileName = fileName.substr(0, dotPos);
    }

    // Build the full folder path
    std::string fullFolderPath = (folderPath.empty()) ? std::filesystem::current_path().string() + "/" + folderName : folderPath + "/" + folderName;

    // Create the folder
    std::filesystem::create_directories(fullFolderPath);

    // Build the full file path including the folder
    std::string fullFilePath = fullFolderPath + "/" + fileName + ".g";

    std::ofstream outfile(fullFilePath);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << fileName << " at path: " << fullFilePath << std::endl;
        return;
    }

    // Write header information -  May need to be changed
    outfile << "LoadPackage(\"qpa\");\n\n";

    // Define the quiver
    outfile << "Q := Quiver(" << boost::num_vertices(G) << ", [";
    

    // Write arrows (edges)
    for (auto e : boost::make_iterator_range(edges(G))) {
        auto source = boost::source(e, G);
        auto target = boost::target(e, G);

        // Write arrow
        outfile << "["  << source << "," << target << ", \"E" << e << "\"" << "],";

        
    }

    outfile.seekp(-1, std::ios::end);  // Remove trailing comma
    outfile << "]);" << std::endl;

    // Define path algebra
    outfile << "A := PathAlgebra(Q, GF(2));" << std::endl;

    // Define module
    // syntax: N := RightModuleOverPathAlgebra( A, [1,2,2], [ .. ["b", [[1,0], [-1,0]] ], ..] )
    outfile << "M := RightModuleOverPathAlgebra(A, [";
    for (auto v : boost::make_iterator_range(vertices(G))) {
        outfile << dimensionVector[v] << ",";
    }
    outfile.seekp(-1, std::ios::end);  // Remove trailing comma
    outfile << "], [";

    for (auto e : boost::make_iterator_range(edges(G))) {
        auto source = boost::source(e, G);
        auto target = boost::target(e, G);
        // Easier to work with the transpose, because gap works with row vectors
        sparseMatrix edgeMatrix = edgeMatrices[e].transposedCopy();

        // Check dimensions
        if (edgeMatrix.get_num_rows() != dimensionVector[source]) {
            throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
        }
        if (edgeMatrix.get_num_cols() != dimensionVector[target]) {
            throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
        }

        if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
          continue;
        }
        // Write matrix at arrow; where each vector is a row

        
        outfile << "[\"E" << e << "\", [";
        for (index i = 0; i < edgeMatrix.get_num_cols(); ++i) {
            outfile << "[";
            auto it = edgeMatrix.A[i].begin();
            for (index j = 0; j < edgeMatrix.get_num_rows(); ++j) {
                if (it != edgeMatrix.A[i].end() && *it == j) {
                    outfile << 1 << ",";
                    ++it;
                } else {
                    outfile << 0 << ",";
                }
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "],";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "]], ";
    }

    outfile.seekp(-1, std::ios::end);  // Remove trailing comma
    outfile << "]);" << std::endl;

    std::cout << "Quiver file written to: " << fullFilePath << std::endl;
}


/**
 * @brief This function computes the quiver representation from a given sparse presentation matrix.
 * Not working proerply right now
 * 
 * @param M The presentation
 * @param filename Indicate the path to the file where presentation was located
 */
std::pair<index, index> computeQuiverRepresentation(sgMatrix &M, const std::string& filename, index kMax){
  degreeList discreteSupport = M.discreteSupport();
  Graph G = createMinimalGraphFromDegreeList(discreteSupport, kMax);
  auto vert = boost::num_vertices(G);
  auto edg = boost::num_edges(G);
  // std::cout << "Created Hasse-Diagram of the Partial Order on the support of size " << vert << " V "<< edg << "E" << std::endl;
  // For each degree we want to store the cokernel, 
  // the row-indices of the generators which form its domain 
  // and a section of the cokernel given by column indices which are mapped to a basis
  std::unordered_map<degree, std::pair< sparseMatrix , std::pair<vec<index>, vec<index>> > > pointwise_Presentations;
  pointwise_Presentations.reserve(vert);
  vec<index> dimensionVector;
  dimensionVector.reserve(vert);
  for (auto vertexIt = vertices(G).first; vertexIt != vertices(G).second; ++vertexIt) {
    auto v = *vertexIt;  // 'v' is a vertex descriptor
    degree d = G[v];     // Access the degree property of the vertex 'v'
    sparseMatrix S;
    vec<index> gens;
    std::tie(S, gens) = M.map_at_degree(d);
    normaliseEntries(S);
    vec<index> basisLift;
    auto T = coKernel(S, false, &basisLift);
    pointwise_Presentations[d]=std::make_pair(T, std::make_pair(gens, basisLift));
    dimensionVector[v] = basisLift.size();
  }
  // std::cout << "computed " << pointwise_Presentations.size() << " many quotient spaces." <<std::endl;
  std::map<Graph::edge_descriptor, sparseMatrix> pathActions;
  for (auto edgeIt = boost::edges(G).first; edgeIt != boost::edges(G).second; ++edgeIt) {
        auto edge = *edgeIt;
        degree sourceDegree = G[boost::source(edge, G)];
        degree targetDegree = G[boost::target(edge, G)];
        auto sourceBasis = pointwise_Presentations[sourceDegree].second;
        auto targetBasis = pointwise_Presentations[targetDegree].second;
        assert(pointwise_Presentations[sourceDegree].first.get_num_cols() == sourceBasis.first.size());
        assert(pointwise_Presentations[targetDegree].first.get_num_cols() == targetBasis.first.size());
        auto lift_of_basis = basisLifting(sourceBasis, targetBasis);
        sparseMatrix actionMap = restrictedDomain(pointwise_Presentations[targetDegree].first, lift_of_basis);
        pathActions[edge] = actionMap;
    }
  // std::cout << "Computed all " << pathActions.size() << "matrices of the representation" << std::endl;
  //sanity check
  for (auto v : boost::make_iterator_range(vertices(G))) {
    index dim = pointwise_Presentations[G[v]].first.get_num_rows();
        for (auto e : boost::make_iterator_range(boost::out_edges(v, G))) {
          if(dim != pathActions[e].get_num_cols()){
            throw std::runtime_error("Dimension mismatch in path action");
          }
        }

    }
  //Print the quiver to a file
  auto outfileName = filename + "_asQuiverRep";
  writeQPAFile(G, dimensionVector, pathActions, outfileName, "quiver_representations");
  return std::make_pair(vert, edg);
}

namespace fs = std::filesystem;

std::tuple<double, index, index, index, index, index> processFile(std::string& filePath) {
   

    std::cout << "Processing file: " << filePath << std::endl;
    sgMatrix A(filePath);
    A.print();
    index c = A.get_num_cols();
    index r = A.get_num_rows();
    index kMax = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    // auto graphSize = computeQuiverRepresentation(A, filePath, kMax);
    std::pair<index, index> graphSize = std::make_pair(0,0);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration<double>(endTime - startTime).count();
    return std::make_tuple(elapsedTime, c, r, graphSize.first, graphSize.second, kMax);
}

void iterateOverFiles(const std::string& folderPath) {
  // Open a text file for writing
    std::ofstream resultFile("runtime_for_pres_to_quiver.txt");
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


}    // namespace pres_to_quiver




int main() {
  
    bool test = false;
    if(test){
    
    }
    using index = pres_to_quiver::index;
    std::string toyExample = "/home/wsljan/generalized_persistence/code/toy_examples/small_pres_with_nonzero_k_min_pres.firep";
    std::string folderPath = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets";
    std::string smallNontrivial = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets/k_fold_46_10_1_min_pres.firep";
    std::string largeExample = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets/k_fold_96_10_1_min_pres.firep";
    std::string smallExample = "/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_awith_rivet_datasets/points_on_sphere_7500_1_min_pres.firep";
    // pres_to_quiver::iterateOverFiles(folderPath);

    std::cout << "Enter the file path: ";
    std::string filePath;
    // std::getline(std::cin, filePath);


    std::tuple<double, index, index, index, index, index> runTime = pres_to_quiver::processFile(smallExample);
    std::cout << "Elapsed Time: " << std::get<0>(runTime) << " seconds" << std::endl;
    std::cout << "Matrix Dimensions: " << std::get<1>(runTime) << " x " << std::get<2>(runTime) << " and kMax of " << std::get<5>(runTime) << std::endl;
    std::cout << "Graph Dimensions: " << std::get<3>(runTime) << " V " << std::get<4>(runTime) << " E" << std::endl;
    return 0;
}
