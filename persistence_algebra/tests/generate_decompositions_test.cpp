#include <iostream>
#include <cassert>
#include <vector>
#include <dense_matrix.hpp>
#include <matrix_base.hpp>
#include "generate_decompositions.cpp"

using namespace graded_linalg;

void test_transitions(int n){
    DecompTree tree = generateDecompositions(n);
    std::vector<transition> transitions = generateTransitions(n);
    DenseMatrix last = DenseMatrix(n, "Identity");
    auto it = transitions.begin();
    for(auto& [pivots, branch] : tree){
        for(int i=0; i< branch.size(); i++){
            for(auto& [first, second] : branch[i]){
                it->first.print();
                print_bitset(it->second);
                first.print();
                second.print();
                DenseMatrix nextMatrix = last.multiply(it->first);
                // nextMatrix.print();
                vec<int> restriction1;
                vec<int> restriction2;
                for(int j = 0; j < n; j++){
                    if(it->second.test(j)){
                        restriction1.push_back(j);
                    } else {
                        restriction2.push_back(j);
                    }
                }
                DenseMatrix nextMatrix1 = nextMatrix.restricted_domain_copy(restriction1);
                DenseMatrix nextMatrix2 = nextMatrix.restricted_domain_copy(restriction2);
                if(false && !compare_matrices(first, nextMatrix1)){
                    std::cout << "Error at position : " << std::distance(transitions.begin(), it) << std::endl;
                    std::cout << "expected is: "; first.print();
                    std::cout << "but restriction gives "; nextMatrix1.print();
                } 
                if(false && !compare_matrices(second, nextMatrix2)){
                    std::cout << "Error: " << std::endl;
                    std::cout << "expected is: "; second.print();
                    std::cout << "but restriction gives "; nextMatrix2.print();
                }
                it++;
            }
        }
    }
}

int main(){
    test_transitions(2);
    return 0;
}