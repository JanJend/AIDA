#include "grlina/graded_linalg.hpp"  // Replace with the actual name of your header file
#include <iostream>
#include <cassert>

using namespace graded_linalg; 


void testTransformedRestrictedCopy() {
    SparseMatrix<int> M(3, 3);
    M.data = {{0}, {0, 1}, {0, 1, 2}};
    DenseMatrix B(1, 2);
    boost::dynamic_bitset<> c(2, 3);
    print_bitset(c);
    B.data = {c};
    std::vector<int> v = {1, 2};
    auto result = M.transformed_restricted_copy(B, v);
    result.print();
    // Add test for transformed_restricted_copy function
}


int main() {
    // Run individual test functions
    testTransformedRestrictedCopy();


    return 0;
}
