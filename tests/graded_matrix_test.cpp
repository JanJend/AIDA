

#include "grlina/r2graded_matrix.hpp"  
#include "grlina/r3graded_matrix.hpp"  
#include <iostream>
#include <cassert>

using namespace graded_linalg; 


void test_order() {
    R2GradedSparseMatrix<int> M("/home/wsljan/AIDA/persistence_algebra/test_presentations/full_rips_size_1_instance_5_min_pres.scc");
    M.print_graded();
    M.sort_columns_lexicographically();
    M.print_graded();
    array<int> edges = M.get_column_graph();
    std::cout << "Edges: " << std::endl;
    print_edge_list(edges);
    array<int> scc;
    auto edge_map = convert_to_map(edges);
    std::set<int> vertices;
    for(int i = 0; i < M.num_cols; i++){
        vertices.insert(i);
    }
    array<int> cond = condensation(vertices, edge_map, scc);
    std::cout << "Condensation has strongly connected components: " << std::endl;
    print_edge_list(scc);
    std::cout << "and edges: " << std::endl;
    print_edge_list(cond);
}

void test_3d(){
    R3GradedSparseMatrix<int> M("/home/wsljan/AIDA/persistence_algebra/test_presentations/test_3d.scc", true, true);
    M.print_graded();
    std::cout << M.col_batches << std::endl;
    M.sort_columns_lexicographically();
    M.print_graded();
}

void test_boost_graphs(){
    R2GradedSparseMatrix<int> M("/home/wsljan/generalized_persistence/persistence_algebra/test_presentations/test_pres_with_nonzero_k_min_pres.firep");
    // M.print_graded();
    Degree_traits<degree> traits;
    auto degrees = M.col_degrees;
    auto edge_checker = [&degrees, &traits](const int& label1, const int& label2) -> bool {
        return traits.smaller_equal(degrees[label1], degrees[label2]); 
    };
    std::vector<int> vertices;
    for(int i = 0; i < M.num_cols; i++){
        vertices.push_back(i);
    }
    Graph g = construct_boost_graph<int>(vertices, edge_checker);

    print_graph(g);
    vec<std::string> labels = {"a", "b", "c", "d", "e", "f", "g"};
    print_graph_with_labels(g, labels);
    std::vector<int> component(boost::num_vertices(g));
    array<int> scc_vec;
    Graph condensation = compute_scc_and_condensation(g, component, scc_vec);

    std::cout << "SCCs: " << scc_vec << std::endl;
    std::cout << "Condensation: ";
    // boost::print_graph(condensation);
    std::cout << "component? " << component << std::endl;
    
    

}


void test_sorting() {
    std::cout << "Using Permutation" << std::endl;
    R2GradedSparseMatrix<int> N("/home/wsljan/generalized_persistence/persistence_algebra/test_presentations/test_pres_with_nonzero_k_min_pres.firep");
    N.print_graded();
    N.sort_columns_lexicographically();
    N.print_graded();
    N.sort_rows_lexicographically();
    N.print_graded();
    }

void test_graded_matrix() {
    R2GradedSparseMatrix<int> M("/home/wsljan/generalized_persistence/persistence_algebra/test_presentations/test_pres_with_nonzero_k_min_pres.firep");
    M.print_graded();
    if(M.is_admissible_column_operation(1, 0)){
        std::cout << "Admissable column operation from 0 to 1" << std::endl;
    }
}

void test_hom_spaces(){
    R2GradedSparseMatrix<int> M("/home/wsljan/generalized_persistence/persistence_algebra/test_presentations/toy_example_1.scc");
    R2GradedSparseMatrix<int> N("/home/wsljan/generalized_persistence/persistence_algebra/test_presentations/toy_example_6.scc");
    N.compute_rows_forward();
    M.compute_rows_forward();
    assert(N._rows.size() == 1);
    auto [Qs, positions] = hom_space(N, M);
    std::cout << "Hom spaces: " << std::endl;
    Qs.print();
    std::cout << "Positions: " << positions << std::endl;
}

int main() {
    test_3d();
    return 0;
}
