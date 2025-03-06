#include <iostream>
#include <utility>
#include <concepts>
#include <vector>
#include <algorithm>

template <typename T>
concept HasOutputOperator = requires(std::ostream& os, const T& obj) {
    { os << obj } -> std::same_as<std::ostream&>;
};


template <typename T>
concept Degree = requires(T a, T b) {
    { a == b } -> std::same_as<bool>;
    { a < b } -> std::same_as<bool>;
};

template <typename T>
using vec = std::vector<T>;

template <Degree D>
bool lex_order(D a, D b);

using degree = std::pair<double, double>;

bool lex_order(const degree& a, const degree& b) {
    if (a.first != b.first) {
        return a.first < b.first;
    }
    return a.second < b.second;
}

template <Degree D>
void delete_multiples(vec<D>& degrees) {
    std::sort(degrees.begin(), degrees.end(), [](const D& a, const D& b) {
        return lex_order(a, b);
    });
}

std::ostream& operator<< (std::ostream& ostr, const degree& deg) {
    ostr << "(" << deg.first << ", " << deg.second << ")";
    return ostr;
}

int main() {
    // Test the concept with std::pair<double, double>
    std::pair<double, double> p1{1.0, 2.0};
    std::pair<double, double> p2{3.0, 4.0};
    std::pair<double, double> p3{2.0, 5.0};
    if(p1 < p2){
        std::cout << "p1 < p2" << std::endl;
        
    }
    if(p3 < p2){
        std::cout << "p3 < p2" << std::endl;
    }
    static_assert(Degree<decltype(p1)>);
    
    static_assert(HasOutputOperator<decltype(p1)>);
    vec<degree> discreteSupport = {std::make_pair(0.2, 0.2), std::make_pair(0.1, 0.3), p1, p2, p3};
    delete_multiples(discreteSupport);
    std::cout << "Sorted discrete support: " << std::endl;
    for (const auto& d : discreteSupport) {
        std::cout << "(" << d.first << ", " << d.second << ")" << std::endl;
    }

    return 0;
}