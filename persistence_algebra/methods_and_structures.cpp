#include <algorithm>
#include <vector>



template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;

template <typename T, typename S>
struct testMatrix {
    vec<T> data;
    vec<S> degrees;

    testMatrix() {
        data = vec<T>();
        degrees = vec<S>();
    }
};

template <typename T>
struct derivedMatrix : public testMatrix<T, vec<T>> {

    derivedMatrix() : testMatrix<T, vec<T>>() {
        // Additional initialization if needed
    }

};


int main() {
    
    derivedMatrix<int> M;
    M.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    M.degrees = {{1, 2, 3}};
    return 0;
}
