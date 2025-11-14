#ifndef LANDSCAPE_HPP
#define LANDSCAPE_HPP

#include <vector>
#include <string>

struct GridPoint {
    int i, j;
    double x, y;
};


struct Bar {
    double theta;
    double length;
};

struct GridData {
    int n_x, n_y;
    double start_x, start_y, end_x, end_y, step_x, step_y;
    double slope;
    std::vector<std::vector<std::vector<Bar>>> bars;
};

// Function declarations
GridData bars_from_sky(const std::string& filename);

int get_diagonal_index(int i, int j, const GridData& data);

void compute_landscape(const GridData& data, const std::string& output_filename, 
                      double theta, int k);

#endif // LANDSCAPE_HPP