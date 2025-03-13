/**
 * @file orders_and_graphs.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once

#ifndef DIGRAPHS_HPP
#define DIGRAPHS_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>
#include <set>
#include <iomanip>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <functional>
#include <map>

namespace graded_linalg {


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;


template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;

template <typename D>
struct Degree_traits {
    
    static bool equals (const D& lhs, const D& rhs);

    static bool smaller (const D& lhs, const D& rhs);

    static bool greater (const D& lhs, const D& rhs);

    static bool greater_equal (const D& lhs, const D& rhs);

    static bool smaller_equal (const D& lhs, const D& rhs);
    

    /**
     * @brief This can be any topolgical order on the degrees.
     * 
     * @param a 
     * @param b 
     * @return true 
     * @return false 
     */
    static bool lex_order(const D& a, const D& b);

    /**
     * @brief Lambda function to compare lexicographically for sorting.
     * 
     */
    static std::function<bool(const D&, const D&)> lex_lambda;

    /**
     * @brief Any Embedding of the degree poset into any R^n.
     * 
     * @param a 
     * @return vec<double> 
     */
    static vec<double> position(const D& a);

    static void print_degree(const D& a);

    // -> Theses should only need to be there if the degrees form a lattice. TO-DO
    static D join(const D& a, const D& b);

    static D meet(const D& a, const D& b);

    /**
     * @brief Writes the degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const D& a);

    /**
     * @brief Gets the degree from an input stream.
     */
    template <typename InputStream>
    static D from_stream(InputStream& iss);

}; // Degree_traits


template <typename D>
std::function<bool(const D&, const D&)> Degree_traits<D>::lex_lambda = [](const D& a, const D& b) {
    return Degree_traits<D>::lex_order(a, b);
};


/**
 * @brief Creates an induced subgraph on a subset of the vertices.
 * 
 * @tparam index 
 * @param g 
 * @param vertices 
 * @return Graph 
 */
template <typename index>
Graph induced_subgraph (const Graph& g, const vec<index>& vertices) {
    Graph subgraph;
    std::unordered_map<index, Vertex> vertex_map;

    // Add vertices to the subgraph and create a mapping
    for (auto v : vertices) {
        Vertex new_v = boost::add_vertex(subgraph);
        vertex_map[v] = new_v;
    }

    // Add edges to the subgraph
    for (auto v : vertices) {
        for (auto it = boost::out_edges(v, g); it.first != it.second; ++it.first) {
            Vertex u = boost::target(*it.first, g);
            if ( std::binary_search(vertices.begin(), vertices.end(), u) ) {
                boost::add_edge(vertex_map[v], vertex_map[u], subgraph);
            }
        }
    }

    return subgraph;
}

void delete_incoming_edges(Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    std::tie(in_begin, in_end) = boost::in_edges(v, g);
    for (auto it = in_begin; it != in_end; ++it) {
        boost::remove_edge(*it, g);
    }
}

// Function to print the graph with labels
template <typename T>
void print_graph_with_labels(const Graph& g, const std::vector<T>& labels) {

    for (Vertex v = 0; v < boost::num_vertices(g); ++v) {
        std::cout << labels[v] << " -> ";
        out_edge_iterator out_i, out_end;
        bool first = true;
        for (boost::tie(out_i, out_end) = boost::out_edges(v, g); out_i != out_end; ++out_i) {
            Vertex u = boost::target(*out_i, g);
            if (!first) {
                std::cout << ", ";
            }
            std::cout << labels[u];
            first = false;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief prints the content of a graph
 * 
 */
void print_graph(const Graph& g){
    boost::print_graph(g); //For now
}


/**
 * @brief Checks if a vertex has incoming edges.
 * 
 * @param g The graph.
 * @param v The vertex.
 * @return true if the vertex has incoming edges, false otherwise.
 */
bool has_incoming_edges(const Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    // Get the range of incoming edges for vertex v
    std::tie(in_begin, in_end) = boost::in_edges(v, g);

    // Check if the range is empty
    return in_begin != in_end;
}

template <typename index>
std::vector<index> incoming_edges(const Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    std::tie(in_begin, in_end) = boost::in_edges(v, g);
    std::vector<index> sources;
    for (auto it = in_begin; it != in_end; ++it) {
        sources.push_back(static_cast<index>(boost::source(*it, g)));
    }
    return sources;
}

/**
 * @brief Constructs a directed boost graph from a list of labels and a function EdgeChecker which returns true if there is a directed edge between two labels.
 * 
 * @tparam index 
 * @tparam EdgeChecker 
 * @param labels 
 * @param has_edge 
 * @param visualise 
 * @return Graph 
 */
template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const vec<index>& labels, EdgeChecker has_edge, bool visualise = false) {
    Graph g;

    for (index i = 0; i < labels.size(); i++) {
        boost::add_vertex(g);
    }
    for (index i = 0; i < labels.size(); i++) {
        for (index j = 0; j < labels.size(); j++) {
            if (has_edge(labels[i], labels[j]) && labels[i] != labels[j]) {
                boost::add_edge(i, j, g);
            }
        }
    }

    if(visualise) {
        print_graph(g);
    }

    return g;
}

template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const index& num_vertices, EdgeChecker has_edge){
    Graph g;
    for (index i = 0; i < num_vertices; i++) {
        boost::add_vertex(g);
    }
    for (index i = 0; i < num_vertices; i++) {
        for (index j = 0; j < num_vertices; j++) {
            if (has_edge(i, j) && i != j) {
                boost::add_edge(i, j, g);
            }
        }
    }
    return g;
}

/**
 * @brief Constructs a directed boost graph from a list of labels and a function EdgeChecker which returns true if there is a directed edge between two labels.
 * 
 * @tparam index 
 * @tparam EdgeChecker 
 * @param labels 
 * @param has_edge 
 * @param visualise 
 * @return Graph 
 */
template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const std::set<index>& labels, EdgeChecker has_edge, bool visualise = false) {
    Graph g;

    std::map<index, Graph::vertex_descriptor> label_to_vertex;
    
    for (auto label : labels) {
        Vertex v = boost::add_vertex(label, g);
        label_to_vertex[label] = v;
    }


    for (auto label1 : labels) {
        for (auto label2 : labels) {
            if (has_edge(label1, label2) && label1 != label2) {
                boost::add_edge(label_to_vertex[label1], label_to_vertex[label2], g);
            }
        }
    }

    if(visualise) {
        print_graph(g);
    }

    return g;
}



/**
 * @brief Given a graph G computes its condenstion.
 * 
 * @param g 
 * @param component will store the index of the component a vertex belongs to
 * @return std::pair<std::vector<std::set<int>>, Graph> returns a vector containing the components as sets and the condensation graph
 */
template <typename index>
Graph compute_scc_and_condensation(const Graph& g, std::vector<index>& component, std::vector<std::vector<index>>& scc_sets) {
    int num_vertices = boost::num_vertices(g);

    // Vector to store the component index of each vertex
    component.assign(num_vertices, -1);
    int num_components = boost::strong_components(g, boost::make_iterator_property_map(component.begin(), boost::get(boost::vertex_index, g)));

    // Create a vector of sets to represent SCCs
    scc_sets = vec<vec<index>>(num_components);
    for (int i = 0; i < num_vertices; ++i) {
        scc_sets[component[i]].push_back(i);
    }

    // Construct the condensation graph - This is super inefficient, in fact 
    // the condensation is implicitly computed by tarjans algorithm which is used by boost

    Graph condensation(num_components);

    for (int u = 0; u < num_vertices; ++u) {
        for (auto it = boost::out_edges(u, g); it.first != it.second; ++it.first) {
            int v = boost::target(*it.first, g);
            if (component[u] != component[v] && !boost::edge(component[u], component[v], condensation).second) {
                boost::add_edge(component[u], component[v], condensation);
            }
        }
    }
    return condensation;
}

/**
 * @brief Returns a topological order for the input directed graph.
 * 
 * @tparam index 
 * @param g 
 * @return std::vector<index> 
 */
template <typename index>
std::vector<index> compute_topological_order(const Graph& g) {
    std::vector<index> topological_order;
    boost::topological_sort(g, std::back_inserter(topological_order));
    return topological_order;
}

/**
 * @brief Computes the Hasse Diagram of a sorted list of degrees fomring a Poset
 * 
 * @tparam D needs to suport functions "smaller" and "equals"
 * @param degrees needs to be sorted in any linear extension of the order
 * @return edgeList 
 */
template <typename D, typename index>
array<index> minimal_directed_graph(vec<D>& degrees, vec<index> support = vec<index>()) {

    array<index> edges(degrees.size(), vec<index>());
    Degree_traits<D> D_traits;
    // writeDegreeListToCSV("degreeList.csv", vertices);
    
    // Add directed edges based on the relationships in the degreeList
    // This computes a transitive reduction of the full poset graph

    index batch_start = 0;
    bool in_batch = false;
    bool batch_ends = false;

    if(support.size() == 0) {
        // Don't check the last element
        for (index i = 0; i < degrees.size() - 1; i++) {

            D& d = degrees[i];
            // Skip batch of equal degrees
            if( D_traits.equals(d, degrees[i+1]) ) {
                if(!in_batch) {
                    in_batch = true;
                    batch_start = i;
                }
                edges[i].push_back(i+1);
                continue;
            } else if (in_batch) {
                batch_ends = true;
                in_batch = false;
                edges[i].push_back(batch_start);
            }

            for (index j = i + 1; j < degrees.size(); j++) {
                bool comesAfterSuccessor = false;
                D& nextd = degrees[j];
                if( D_traits.smaller(d, nextd) ){
                    // If a batch has ended, then the first element in edges is the first element of the batch
                    // and so has the same degree, always making comesAfterSuccessor true
                    for(index k = !batch_ends ? 0 : 1; k < edges[i].size(); k++){
                        if( D_traits.smaller(degrees[edges[i][k]], nextd) ) {
                            comesAfterSuccessor = true;
                            break;
                        }
                    }
                    if(!comesAfterSuccessor){
                        edges[i].push_back(j);
                    }
                }
            }

            batch_ends = false;
        }
    } else {
        // Don't check the last element
        for (index s = 0; s < support.size() - 1; s++) {

            D d = degrees[support[s]];

            // Skip batch of equal degrees
            if( D_traits.equals(d, degrees[support[s+1]]) ) {
                if(!in_batch) {
                    in_batch = true;
                    batch_start = s;
                }
                edges[s].push_back(s+1);
                continue;
            } else if (in_batch) {
                in_batch = false;
                edges[s].push_back(batch_start);
            }

            for (index t = s + 1; t < support.size(); t++) {
            bool comesAfterSuccessor = false;
            D& nextd = degrees[support[t]];
                if( D_traits.smaller(d, nextd) ){
                    // TO-DO: This is a bit inefficient, but it should be fine for non-very large lists
                    for(auto u : edges[s]){
                        if( D_traits.smaller(degrees[support[u]], nextd) ) {
                            comesAfterSuccessor = true;
                            break;
                        }
                    }
                    if(!comesAfterSuccessor){
                        edges[s].push_back(t);
                    }
                }
            }
        }
    }

    return edges;
}

template <typename D>
void writeDegreeListToCSV(const std::string& filename, const vec<D>& degrees) {

    Degree_traits<D> D_traits;
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
        const vec<std::string>& pos = D_traits.position(element);
        for (const std::string& coord : pos) {
            outfile << coord << ",";
        }
        outfile << std::endl;
    }

    std::cout << "Degree list written to: " << filename << std::endl;
}

/**
 * @brief Writes a list of edges to a CSV file
 * 
 * @tparam D 
 * @param filename 
 * @param degrees 
 */
template <typename index>
void writeEdgeListToCSV(const std::string& filename, const array<index>& edges) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header if needed
    outfile << "Source: Targets" << std::endl;

    // Write integer pairs to the file
    for (index source = 0; source < edges.size(); source++) {
        outfile << source << ": " << std::endl;
        for(auto target : edges[source]){
            outfile << target << ", " << std::endl;
        }
    }

    std::cout << "Integer pairs written to: " << filename << std::endl;
}

/**
 * @brief Print the edges of a directed graph.
 * 
 * @tparam index 
 * @param edges 
 */
template <typename index>
void print_edge_list(const array<index>& edges){
    for (index source = 0; source < edges.size(); source++) {
        std::cout << source << " -> ";
        for(auto target : edges[source]){
            std::cout << target << ", ";
        }
        std::cout << std::endl;
    }
}



// TO-DO: This is AI generated, make sure it works. Maybe change to Kosaraju's algorithm.
// Function to perform DFS and find SCCs using Tarjan's algorithm
template <typename index>
void tarjanDFS(index u, const std::unordered_map<index, vec<index>>& outgoing_edges, vec<index>& disc, vec<index>& low, std::stack<index>& st, vec<bool>& inStack, array<index>& scc, index& time) {
    disc[u] = low[u] = ++time;
    st.push(u);
    inStack[u] = true;

    if (outgoing_edges.find(u) != outgoing_edges.end()) {
        for (int v : outgoing_edges.at(u)) {
            if (disc[v] == -1) {
                tarjanDFS(v, outgoing_edges, disc, low, st, inStack, scc, time);
                low[u] = std::min(low[u], low[v]);
            } else if (inStack[v]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    }

    if (low[u] == disc[u]) {
        vec<index> component;
        while (st.top() != u) {
            int v = st.top();
            st.pop();
            inStack[v] = false;
            component.push_back(v);
        }
        st.pop();
        inStack[u] = false;
        component.push_back(u);
        scc.push_back(component);
    }
}

template <typename index>
array<index> condensation(const set<index>& vertex_labels, const std::unordered_map<index, vec<index>>& outgoing_edges, array<index>& scc) {
    int n = vertex_labels.size();
    vec<int> disc(n, -1), low(n, -1);
    vec<bool> inStack(n, false);
    std::stack<int> st;
    int time = 0;

    for (int u : vertex_labels) {
        if (disc[u] == -1) {
            tarjanDFS(u, outgoing_edges, disc, low, st, inStack, scc, time);
        }
    }

    array<index> dag(scc.size());

    for (int u : vertex_labels) {
        if (outgoing_edges.find(u) != outgoing_edges.end()) {
            for (int v : outgoing_edges.at(u)) {
                int scc_u = -1, scc_v = -1;
                for (int i = 0; i < scc.size(); ++i) {
                    if (find(scc[i].begin(), scc[i].end(), u) != scc[i].end()) scc_u = i;
                    if (find(scc[i].begin(), scc[i].end(), v) != scc[i].end()) scc_v = i;
                }
                if (scc_u != -1 && scc_v != -1 && scc_u != scc_v) {
                    dag[scc_v].push_back(scc_u);  // Store predecessors
                }
            }
        }
    }

    return dag;
}


/**
 * @brief Copies an array into an analogous hash map.
 * 
 * @tparam index 
 * @param adj_matrix 
 * @return std::unordered_map<index, vec<index>> 
 */
template <typename index>
std::unordered_map<index, vec<index>> convert_to_map(const array<index>& arr) {
    std::unordered_map<index, vec<index>> map;
    for (index i = 0; i < arr.size(); ++i) {
        map[i] = arr[i];
    }
    return map;
}


// Adapted from https://stackoverflow.com/questions/69763576/sort-container-based-on-another-using-custom-iterator/69767844#69767844

    template <typename D, typename DataT>
    struct Value
    {
        D Order;
        DataT Data;
    };

    template <typename D, typename DataT>
    struct ValueReference
    {
        D* Order;
        DataT* Data;
        

        ValueReference& operator=(ValueReference&& r) noexcept
        {
            *Order = std::move(*r.Order);
            *Data = std::move(*r.Data);
            return *this;
        }

        ValueReference& operator=(Value<D, DataT>&& r)
        {
            *Order = std::move(r.Order);
            *Data = std::move(r.Data);
            return *this;
        }

        friend void swap(ValueReference a, ValueReference b)
        {
            std::swap(a.Order, b.Order);
            std::swap(a.Data, b.Data);
        }

        operator Value<D, DataT>()&&
        {
            return { std::move(*Order), std::move(*Data) };
        }
    };

    //TO-DO: Should not need to use an operator where we have to invoke the Degree_traits class every time.
    template <typename D, typename DataT>
    bool operator<(const ValueReference<D, DataT>& a, const Value<D, DataT>& b)
    {
        Degree_traits<D> D_traits;
        return D_traits.lex_order(*a.Order, b.Order);
    }

    template <typename D, typename DataT>
    bool operator<(const Value<D, DataT>& a, const ValueReference<D, DataT>& b)
    {   
        Degree_traits<D> D_traits;
        return D_traits.lex_order(a.Order, *b.Order);
    }

    template <typename D, typename DataT>
    bool operator<(const ValueReference<D, DataT>& a, const ValueReference<D, DataT>& b)
    {   
        Degree_traits<D> D_traits;
        return D_traits.lex_order(*a.Order, *b.Order);
    }

    template <typename D, typename DataT>
    struct ValueIterator
    {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = size_t;
        using value_type = Value<D, DataT>;
        using pointer = value_type*;
        using reference = ValueReference<D, DataT>;

        D* Order;
        DataT* Data;
        Degree_traits<D> D_traits;

        bool operator==(const ValueIterator& r) const
        {
            return Order == r.Order;
        }
        bool operator!=(const ValueIterator& r) const
        {
            return Order != r.Order;
        }
        bool operator<(const ValueIterator& r) const
        {
            return D_traits.lex_order(*Order, *r.Order);
        }

        ValueIterator& operator+=(difference_type i) {
            Order += i;
            Data += i;
            return *this;
        }

        ValueIterator operator+(difference_type i) const
        {
            return { Order + i, Data + i };
        }
        ValueIterator operator-(difference_type i) const
        {
            return { Order - i, Data - i };
        }

        difference_type operator-(const ValueIterator& r) const
        {
            return Order - r.Order;
        }

        ValueIterator& operator++()
        {
            ++Order;
            ++Data;
            return *this;
        }
        ValueIterator& operator--()
        {
            --Order;
            --Data;
            return *this;
        }

        ValueReference<D, DataT> operator*() const
        {
            return { Order, Data };
        }
    };


/**
 * @brief Sort data according to the function "lex_order" on degrees.
 * 
 * @param degrees 
 * @param data 
 */
template <typename D, typename T>
void sort_simultaneously( vec<D>& degrees, vec<T>& data) {
    std::sort(ValueIterator<D, T>{ degrees.data(), data.data() }, 
    ValueIterator<D, T>{ degrees.data() + degrees.size(), data.data() + data.size() });
}


/**
 * @brief Sort data according to the function "smaller" on degrees.
 * 
 * @param degrees 
 * @param data 
 */
template <typename D, typename T, typename Compare>
void sort_simultaneously_custom( vec<D>& degrees, vec<T>& data, Compare comp) {
    std::sort(ValueIterator<D, T>{ degrees.data(), data.data() }, 
    ValueIterator<D, T>{ degrees.data() + degrees.size(), data.data() + data.size() },
    [comp](const ValueIterator<D, T>& a, const ValueIterator<D, T>& b) {
                            return comp(*a.Order, *b.Order);
                        });
}

/**
 * @brief Does not sort the input vector, but returns a vector of indices representing the permutation which can be used to sort it.
 * 
 * @tparam T 
 * @param vec 
 * @return std::vector<index> 
 */
template <typename T, typename index>
vec<index> get_permutation_from_sort(const std::vector<T>& vec) {
    // Initialize indices
    std::vector<index> indices(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on comparing values in vec
    std::stable_sort(indices.begin(), indices.end(), [&vec](index i1, index i2) {
        return vec[i1] < vec[i2];
    });

    return indices;
}

/**
 * @brief Sorts the input vector and returns a vector of indices representing the permutation used to sort.
 * 
 * @tparam T Type of the elements in the vector.
 * @param vec The vector to be sorted.
 * @return std::vector<index> A vector of indices representing the permutation used to sort.
 */
template <typename T, typename index>
vec<index> sort_and_get_permutation(std::vector<T>& vec) {
    // Initialize indices
    std::vector<index> indices(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on comparing values in vec
    std::stable_sort(indices.begin(), indices.end(), [&vec](index i1, index i2) {
        return vec[i1] < vec[i2];
    });

    // Create a copy of the original vector
    std::vector<T> sorted_vec(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        sorted_vec[i] = vec[indices[i]];
    }

    // Assign the sorted vector back to the original vector
    vec = sorted_vec;

    return indices;
}

/**
 * @brief Sorts the input vector using a custom comparator and returns a vector of indices representing the permutation used to sort.
 * 
 * @tparam T Type of the elements in the vector.
 * @tparam index Type of the indices.
 * @param vec The vector to be sorted.
 * @param comp The comparison function to use for sorting.
 * @return std::vector<index> A vector of indices representing the permutation used to sort.
 */
template <typename T, typename index>
std::vector<index> sort_and_get_permutation(std::vector<T>& vec, std::function<bool(const T&, const T&)> comp) {
    // Initialize indices
    std::vector<index> indices(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on the custom comparator
    std::stable_sort(indices.begin(), indices.end(), [&vec, &comp](index i1, index i2) {
        return comp(vec[i1], vec[i2]);
    });

    // Create a copy of the original vector
    std::vector<T> sorted_vec(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        sorted_vec[i] = vec[indices[i]];
    }

    // Assign the sorted vector back to the original vector
    vec = sorted_vec;

    return indices;
}

} // namespace graded_linalg

#endif // DIGRAPHS_HPP

