
#include "graded_linalg.hpp"

#include <cassert>
#include <boost/timer/timer.hpp>

using namespace graded_linalg;
using namespace boost::timer;

// Helper function to check if two vectors are equal
template <typename T>
bool vectorsEqual(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}


bool runSparseTests(){
    bool result = true;

    vec<vec<int>> data = {{0, 2},{1, 3},{0,3}};

    SparseMatrix<int> A(3, 4, data);
    auto AT = A.transposed_copy();
    if(AT.get_num_cols() != 4){result = false;}

    AT.print();
    return result;   
}

void run_row_test(){
    vec<vec<int>> data = {{0, 1, 2}, {1, 3, 5, 7}, {0, 2, 3, 6, 7}, {2, 4, 5, 6, 7}};
    SparseMatrix<int> A(4, 8, data);
    A.compute_revrows();
    A.print();
    A.print_rows();
    A.fast_rev_row_op(3, 4);
    A.print_rows();
    A.print();
}

bool run_vectors_test(){
    bool passed = true;
    vec<int> v1 = {1, 2, 3, 2, 2, 1, 3};
    std::sort(v1.begin(), v1.end());
    convert_mod_2(v1);
    std::cout << v1 << std::endl;
    vec<int> v1_compare = {2};
    passed &= vectorsEqual(v1, v1_compare);
    return passed;
}

void compare_vec_set_test(){
    vec<int> test_sizes = {5000};
    for(int n : test_sizes){
        cpu_timer vec_timer;
        cpu_timer set_timer;
        int perc = 20;
        auto M = SparseMatrix<int>(n, n, "Random");
        auto N = SparseMatrix_set(M);

        vec_timer.start();
        M.column_reduction_triangular();
        vec_timer.stop();
        std::cout << "Vector time for n = " << n << ": " << vec_timer.format() << std::endl;


        set_timer.start();
        N.column_reduction_triangular();
        set_timer.stop();
        std::cout << "Set time for n = " << n << ": " << set_timer.format() << std::endl;
    }
}

bool test_kernel(){
    int n = 8;
    int perc = 20;
    auto M = SparseMatrix<int>(n, n, "Random", perc);
    std::cout << "M is nonzero: " << !M.is_zero() << std::endl;
    M.print();
    auto copy = M;
    auto K = copy.get_kernel();
    std::cout << "K is nonzero: " << !K.is_zero() << ". Size:" << K.get_num_cols() <<  std::endl;
    K.print();
    auto Zero = M.multiply_right(K);
    Zero.print();
    return Zero.is_zero(); 
}

int main() {
    compare_vec_set_test();
    auto passed = true;
    // passed &= test_kernel();
    if(passed){std::cout << "All tests passed!" << std::endl;}
    else{std::cout << "Some tests failed." << std::endl;}
    return 0;
}
