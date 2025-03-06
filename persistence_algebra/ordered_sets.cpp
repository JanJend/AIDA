/** @file
 *  @brief poset Structure Definition
 *
 *  This file defines a struct called `poset`, which represents a vector of objects
 *  of a template type `T`. The objects of type `T` are required to have the operators
 *  `==` and `<` defined for comparison.
 *
 *  @author Jan Jendrysiak
 *  @date 15.01.2024
 */

#include <algorithm>
#include <vector>
#include <concepts>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

template <typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::same_as<bool>;
    { a == b } -> std::same_as<bool>;
};

/**
 * @brief Struct representing a vector of comparable objects of type T.
 *
 * The `poset` struct is a container for objects of type `T`, and it requires that
 * the template type `T` supports the comparison operators `==` and `<`.
 *
 * @tparam T The type of objects stored in the poset.
 */
template <Comparable T>
struct poset {
    std::vector<T> elements; /**< Vector to store elements of type T. */

    poset(std::vector<T> elems) : elements(std::move(elems)) {}
};

/**
 * @brief Orders the elements of a poset and removes duplicates.
 *
 * The function takes a reference to a poset and removes duplicate instances of elements.
 * Elements are considered equal if the `==` operator returns true for them.
 *
 * @tparam T The type of objects stored in the poset.
 * @param poset A reference to the poset to be modified.
 */
template <Comparable T>
void reduce(poset<T>& P) {
    std::sort(P.elements.begin(), P.elements.end());
    auto last = std::unique(P.elements.begin(), P.elements.end());
    P.elements.erase(last, P.elements.end());
}

/**
 * @brief Alias for the graph type used to represent a poset.
 * @tparam T The type of elements in the poset.
 */
template <Comparable T>
using poset_graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;

/**
 * @brief Convert a poset to a directed graph.
 * @tparam T The type of elements in the poset.
 * @param myPoset The input poset.
 * @return The directed graph representation of the poset.
 */
template <typename T>
poset_graph<T> posetToGraph(const poset<T>& myPoset) {
    poset_graph<T> g(myPoset.elements.size()); /**< The resulting directed graph. */

    for (std::size_t i = 0; i < myPoset.elements.size(); ++i) {
        for (std::size_t j = i + 1; j < myPoset.elements.size(); ++j) {
            if (myPoset.elements[i] < myPoset.elements[j]) {
                boost::add_edge(i, j, g);
            }
        }
    }

    return g;
}

using real = long double;
using degree = std::pair<real, real>;

bool operator==(const degree& lhs, const degree& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

bool operator<(const degree& lhs, const degree& rhs) {
    return (lhs.first < rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
}


/**
 * @brief Generates a poset of degrees with random entries.
 * @param n The size of the poset.
 * @param numDuplicates The number of entries to appear doubly.
 * @return The generated poset.
 */
poset<degree> generateRandomPoset(int n, int numDuplicates) {
    std::vector<degree> posetElements;

    // Seed for random number generation
    std::srand(std::time(0));

    // Generate random degrees
    for (int i = 0; i < n - numDuplicates; ++i) {
        real randomFirst = static_cast<real>(std::rand()) / static_cast<real>(RAND_MAX);
        real randomSecond = static_cast<real>(std::rand()) / static_cast<real>(RAND_MAX);
        posetElements.emplace_back(randomFirst, randomSecond);
    }

    // Ensure some entries appear doubly
    std::unordered_set<degree> duplicates;
    while (duplicates.size() < static_cast<std::size_t>(numDuplicates)) {
        real randomFirst = static_cast<real>(std::rand()) / static_cast<real>(RAND_MAX);
        real randomSecond = static_cast<real>(std::rand()) / static_cast<real>(RAND_MAX);
        duplicates.insert(std::make_pair(randomFirst, randomSecond));
    }

    // Add duplicates to the poset
    posetElements.insert(posetElements.end(), duplicates.begin(), duplicates.end());

    // Sort the poset
    std::sort(posetElements.begin(), posetElements.end());

    return poset<degree>(posetElements);
}

int main() {
    int n = 10;             // Specify the size of the poset
    int numDuplicates = 3;  // Specify the number of entries to appear doubly

    poset<degree> myPoset = generateRandomPoset(n, numDuplicates);

    // Display the generated poset
    for (const auto& deg : myPoset.elements) {
        std::cout << "(" << deg.first << ", " << deg.second << ") ";
    }

    return 0;
}