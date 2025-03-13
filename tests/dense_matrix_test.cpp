#include <iostream>
#include <cassert>
#include <vector>
#include <dense_matrix.hpp>
#include <matrix_base.hpp>
#include <filesystem>

using namespace graded_linalg;


void test_algebra(){
    DenseMatrix D_test = DenseMatrix(3, "Identity");
    std::swap(D_test.data[0], D_test.data[2]);
    DenseMatrix A_test = DenseMatrix(3, "Identity");
    A_test.data[0] = bitset(3, 2);
    A_test.data[1] = bitset(3, 7);
    DenseMatrix D_A_test = DenseMatrix(A_test);
    D_A_test.data[2] = bitset(3, 1);
    DenseMatrix D_for_E_test = DenseMatrix(3, "Identity");
    D_for_E_test.data[0] = bitset(3, 2);
    D_for_E_test.data[1] = bitset(3, 6);
    D_for_E_test.data[2] = bitset(3, 7);

    DenseMatrix A = DenseMatrix(3, "Identity");
    DenseMatrix B = DenseMatrix(A);
    DenseMatrix C = DenseMatrix(3, 3);
    C.data[0] = bitset(3, 7);
    C.data[1] = bitset(3, 4);
    C.data[2] = bitset(3, 1);
    DenseMatrix E = DenseMatrix(C);
    E.data[0] = bitset(3, 5);
    E.data[2] = bitset(3, 3);
    std::cout << "C/D: ";
    C.print();
    DenseMatrix D = A.multiply(C);
    assert(compare_matrices(D, C));
    D.column_reduction_with_memory(A);
    std::cout << "full reduction (D): ";
    D.print();
    assert(compare_matrices(D, D_test));
    std::cout << "performed column ops: ";
    A.print();
    assert(compare_matrices(A, A_test));
    auto D_A = D.multiply(A);
    std::cout << "D_A: ";
    D_A.print();
    assert(compare_matrices(D_A, D_A_test));

    DenseMatrix C_div_B_test = DenseMatrix(C);
    C.column_reduction_triangular_with_memory(B);
    std::cout << "triangular reduction: ";
    C.print();
    std::cout << "performed column ops (B): ";
    B.print();
    auto B_inv = B.get_inverse();
    auto C_div_B = C.multiply(B_inv);
    assert(compare_matrices(C_div_B, C_div_B_test));

    std::cout << "E: ";
    E.print();
    DenseMatrix D_for_E = DenseMatrix(D);
    std::cout << "storing input in: ";
    D_for_E.print();   
    E.column_reduction_with_memory(D_for_E);
    std::cout << "full reduction (E): ";
    E.print();
    std::cout << "performed column ops on input: ";
    D_for_E.print();
    assert(compare_matrices(D_for_E, D_for_E_test));

    auto E_inverse = E.get_inverse();
    std::cout << "inverse of E: ";
    E_inverse.print();
    auto E_transpose = E.transposed_copy();
    std::cout << "transpose of E: ";
    E_transpose.print();
    assert(compare_matrices(E_transpose, E_inverse));
    
    auto B_div_E = B.divide_right(E);
    auto B_div_E_test = B.multiply(E_inverse);
    B_div_E.print();
    assert(compare_matrices(B_div_E, B_div_E_test));
    
    
    std::cout << "Test passed." << std::endl;
}


void test_bitset(){
    bitset bs = bitset(5, 11);
    bitset bs2 = bitset(4, 11);
    print_bitset(bs);
    print_bitset(bs2);
    std::string bitsetString = bitsetToString_alt(bs);
    int length = bitsetString.length();
    std::cout << bitsetString << " of length " << length << std::endl;
    auto number = bs.to_ulong();
    auto number2 = bs2.to_ulong();
    std::cout << "Numbers: " << number << " " << number2 << std::endl;
    std::cout << bs.npos << " " << bs.size() << std::endl;
    int i = bs.find_first();
    int j = bs.find_next(bs.find_next(bs.find_next(bs.find_next(i))));
    std::cout << "First: " << i << " Next: " << j << std::endl;
    auto e = compute_standard_vectors(3);
    std::cout << "Standard vectors: " << e[0] << " " <<  e[1] << " " << e[2] << std::endl;
    for(auto it = bs.find_first(); it != bs.npos; it = bs.find_next(it)){
        std::cout << it << " ";
    }
    std::cout << std::endl;
    std::cout << bitset(1, 1) << std::endl;
    auto count_vec = compute_sum_of_standard_vectors(3);
    std::cout << "Full Bitsets: ";
    for(auto it = count_vec.begin(); it != count_vec.end(); it++){
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

void test_matrix_serialization(){
    DenseMatrix A = DenseMatrix(3, "Identity");
    std::ofstream out("test_matrix.bin", std::ios::binary);
    A.serialize(out);
    out.close();
    std::ifstream in("test_matrix.bin", std::ios::binary);
    DenseMatrix B = DenseMatrix(in);
    assert(A.equals(B));
}

void test_iofstream(){
    std::ofstream out("test.bin", std::ios::binary);
    int x = 5;
    int x2 = 6;
    out.write((char*)&x, sizeof(x));
    out.write((char*)&x2, sizeof(x2));
    std::cout << "Wrote: " << x << std::endl;
    out.close();
    std::ifstream in("test.bin", std::ios::binary);
    int y;
    in.read((char*)&y, sizeof(y));
    std::cout << "Read: " << y << std::endl;

}

void test_load_transition(){
    auto filename = "decompositions_4.bin";
    auto filename2 = "transitions_4.bin";
    if (std::filesystem::exists(filename2)) {
        std::cout << "File '" << filename2 << "' exists." << std::endl;
    } else {
        std::cout << "Error, file '" << filename2 << "' does not exist." << std::endl;
    }
    DecompTree tree = loadDecompTree(filename);
    vec<transition> matrices = load_transition_list(filename2);
}



int main(){
    test_bitset();

    return 0;
}