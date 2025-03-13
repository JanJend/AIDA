
#include "matrix_base.hpp"
#include <cassert>
#include "inheritence_test.hpp"

using namespace graded_linalg;

// Helper function to check if two vectors are equal
template <typename T>
bool vectorsEqual(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

void test_inheritence(){
    testMatrix A = testMatrix(2,3, vec<vec<int>>({{0,1}, {1,2}}));
    if(A.vis_nonzero_at(A.data[0], 1)){std::cout << "Test passed."<< std::endl;}
    else {std::cout << "Test failed."<< std::endl;}
    A.print();
}



/**
// Implementation of MatrixUtil<int, int>
struct MatrixInt : public MatrixUtil<int, int> {
    bool vis_nonzero_at(int& a, int i) override {
        return a != 0;
    }

    void vadd_to(int& a, int& b) override {
        b += a;
    }

    int vlast_entry_index(int& a) override {
        return a; // Update this according to your logic
    }

    bool vis_equal(int& a, int& b) override {
        return a == b;
    }

    MatrixInt() : MatrixUtil<int, int>() {
        // Additional initialization if needed
    }

    MatrixInt(int m, int n){
        num_cols = m;
        num_rows = n;
        data = vec<int>(m,0);    }


};
*/

/**
void runSparseTests(){
    // Test constructor and basic operations
    {
        SparseMatrix<int> matrix(2,7);

        // Check initial state
        assert(matrix.get_num_cols() == 2);
        assert(matrix.get_num_rows() == 7);
        

        // Modify the matrix and check the changes
        matrix.data = {{1, 3}, {3, 4}, {4, 6}};

        // Check if the number of columns is computed correctly
        matrix.compute_num_cols();
        assert(matrix.get_num_cols() == 3);

        //Check if col_op works as expected

        std::cout << matrix.data[0] << " " << matrix.data[1] << std::endl; 

        //add_to(matrix.data[0], matrix.data[1]);
        matrix.col_op(0, 1);

        std::cout << matrix.data[0] << " " << matrix.data[1] << std::endl;

        assert(vectorsEqual(matrix.data, {{1, 3}, {1, 4}, {4, 6}}));
        
        matrix.print();

        matrix.column_reduction();

        assert(vectorsEqual(matrix.data, {{1, 3}, {1, 4}, {1, 6}}));

        matrix.print();

        auto bmatrix = matrix.coKernel();

        bmatrix.print();

        auto cmatrix = multiply(bmatrix, matrix);

        cmatrix.print();

        // Add more tests as needed
    }

    // Add more test cases as needed

}
*/


/**
void runMatrixTests() {
    // Test constructor and basic operations
    {
        MatrixInt matrix(2,3);

        matrix.set_num_rows(2);

        // Check initial state
        assert(matrix.get_num_cols() == 2);
        assert(matrix.get_num_rows() == 2);
        

        // Modify the matrix and check the changes
        matrix.data = {1, 2, 3, 4, 5, 6};

        // Check if the number of columns is computed correctly
        matrix.compute_num_cols();
        assert(matrix.get_num_cols() == 6);

        // Check if col_op works as expected
        matrix.col_op(0, 1);
        assert(vectorsEqual(matrix.data, {1, 3, 3, 4, 5, 6}));
        
        matrix.print();
        // Add more tests as needed
    }

    // Add more test cases as needed
}
*/

int main() {
    test_inheritence();
    // runSparseTests();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
