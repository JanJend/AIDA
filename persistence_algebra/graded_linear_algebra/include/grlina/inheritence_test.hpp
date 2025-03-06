// inheritence_test.hpp

#pragma once

#ifndef INHERITENCE_TEST_HPP
#define INHERITENCE_TEST_HPP

namespace graded_linalg{


using testMatrix = MatrixUtil<vec<int>, int>;


std::ostream& operator<<(std::ostream &os, const vec<int>& data) {
        for (int i = 0; i < data.size(); i++) {
            os << data[i]<< " ";
        }
        return os;
}

template<>
bool MatrixUtil<vec<int>, int>::vis_nonzero_at(vec<int>& a, int i){
    return std::binary_search(a.begin(), a.end(), i);
}



} // namespace graded_linalg

#endif // INHERITENCE_TEST_HPP
