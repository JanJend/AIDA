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
    double r1, r2;
};

struct GridData {
    int n_x, n_y;
    double start_x, start_y, step_x, step_y;
    std::vector<std::vector<Bar>> bars;
};

// Function declarations
GridData read_sky_file(const std::string& filename, double theta_threshold);

void compute_landscape(const GridData& data, const std::string& output_filename, 
                      double theta, int k);

#endif // LANDSCAPE_HPP