#include "filt_landscape.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cassert>

GridData bars_from_sky(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    GridData result;
    std::string line;
    
    // Line 1: Must be "HNF"
    std::getline(file, line);
    if (line.find("HNF") == std::string::npos) {
        throw std::runtime_error("First line must be 'HNF'");
    }
    
    // Line 2: Grid dimensions
    std::getline(file, line);
    std::sscanf(line.c_str(), "%d,%d", &result.n_x, &result.n_y);
    
    // Line 3: Lattice info - extract coordinates
    std::getline(file, line);
    std::regex coord_regex(R"(\((-?[0-9.eE+-]+),\s*(-?[0-9.eE+-]+)\))");
    auto coords_begin = std::sregex_iterator(line.begin(), line.end(), coord_regex);
    auto coords_end = std::sregex_iterator();
    
    std::vector<std::pair<double, double>> coords;
    for (auto it = coords_begin; it != coords_end; ++it) {
        double x = std::stod((*it)[1]);
        double y = std::stod((*it)[2]);
        coords.push_back({x, y});
    }
    
    if (coords.size() < 3) {
        throw std::runtime_error("Expected at least 3 coordinate pairs");
    }
    
    // Extract lattice vectors
    result.start_x = coords[0].first;
    result.start_y = coords[0].second;
    result.end_x = coords[1].first;
    result.end_y = coords[1].second;
    result.step_x = coords[2].first;
    result.step_y = coords[2].second;
    result.slope = result.step_y / result.step_x;
    
    // Initialize bars grid
    result.bars.resize(result.n_x, std::vector<std::vector<Bar>>(result.n_y));
    
    // Process grid points and stable modules
    std::pair<double, double> current_position;
    int i = -1, j = -1;
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        if (line.substr(0, 2) == "G,") {
            // Grid point line
            std::regex grid_regex(R"(G,(\d+),(\d+),\s*\(([^,]+),([^)]+)\))");
            std::smatch match;
            if (std::regex_match(line, match, grid_regex)) {
                i = std::stoi(match[1]);
                j = std::stoi(match[2]);
                current_position.first = std::stod(match[3]);
                current_position.second = std::stod(match[4]);
            }
        } else {

            std::istringstream iss(line);
            std::string token;
            
            // First token is slope
            std::getline(iss, token, ',');
            double theta = std::stod(token);
            
            // Parse relations
            std::vector<std::pair<double, double>> relations;
            std::regex rel_regex(R"(\(([^;]+);([^)]+)\))");
            
            while (std::getline(iss, token, ',')) {
                std::smatch rel_match;
                if (std::regex_search(token, rel_match, rel_regex)) {
                    double x = std::stod(rel_match[1]);
                    double y = std::stod(rel_match[2]);
                    relations.push_back({x, y});
                }
            }
            
            if (relations.empty()) continue;
            
            // Convert to relative coordinates and find intersections
            std::vector<std::pair<double, double>> candidates;
            for (const auto& [x, y] : relations) {
                double rel_x = x - current_position.first;
                double rel_y = y - current_position.second;
                
                double int_x, int_y;
                if (rel_y <= result.slope * rel_x) {
                    int_x = rel_x;
                    int_y = result.slope * rel_x;
                } else {
                    int_y = rel_y;
                    int_x = rel_y / result.slope;
                }
                candidates.push_back({int_x, int_y});
            }
            
            // Choose minimum candidate
            auto min_candidate = *std::min_element(candidates.begin(), candidates.end());
            
            // Calculate length (magnitude)
            double length = min_candidate.first;
            result.bars[i][j].push_back({theta, length});
        }
    }
    result.n_y -= 30; // Adjust for extra grid points
    return result;
}


// obsolete
int get_diagonal_index(int i, int j, const GridData& data) {
    // Grid point coordinates
    double x = data.start_x + i * data.step_x;
    double y = data.start_y + j * data.step_y;
    double S = data.slope;
    
    // Line through (x, y) with slope S: Y = S*X + (y - S*x)
    double intercept = y - S * x;
    
    // Check intersection with left edge (X = start_x)
    double y_at_left = S * data.start_x + intercept;
    
    // Check intersection with bottom edge (Y = start_y)
    double x_at_bottom = (data.start_y - intercept) / S;
    
    // Grid boundaries
    double max_x = data.start_x + (data.n_x - 1) * data.step_x;
    double max_y = data.start_y + (data.n_y - 1) * data.step_y;
    
    // Determine which boundary the diagonal starts from
    bool intersects_left = (y_at_left >= data.start_y && y_at_left <= max_y);
    bool intersects_bottom = (x_at_bottom >= data.start_x && x_at_bottom <= max_x);
    
    if (intersects_bottom) {
        // Diagonal starts from bottom edge
        // Find which grid column index this corresponds to
        int col_idx = std::round((x_at_bottom - data.start_x) / data.step_x);
        return col_idx;
    } else if (intersects_left) {
        // Diagonal starts from left edge (excluding bottom-left corner)
        // Find which grid row index this corresponds to
        int row_idx = std::round((y_at_left - data.start_y) / data.step_y);
        return data.n_x + row_idx - 1;
    } else {
        throw std::runtime_error("Grid point does not lie on any valid diagonal");
    }
}

void compute_landscape(const GridData& data, const std::string& output_filename,
                      double theta_min, int k) {

    // Initialize landscape array
    std::vector<std::vector<double>> landscape(data.n_x, std::vector<double>(data.n_y, 0.0));

    // Process each grid point
    for (int i = 0; i < data.n_x; i++) {
        for (int j = 0; j < data.n_y; j++) {
            auto bars = data.bars[i][j];

            // Filter out bars with theta < theta_min
            bars.erase(std::remove_if(bars.begin(), bars.end(),
                       [theta_min](const Bar& bar) { return bar.theta < theta_min; }),
                       bars.end());

            if (bars.size() < k) {
                continue;
            }
            std::sort(bars.begin(), bars.end(),
                      [](const Bar& a, const Bar& b) { return a.length > b.length; });
            const Bar& chosen = bars[k - 1];
            double length = chosen.length;
            double d = length / 2.0;
            
            // Update landscape along the diagonal
            for (int t = 0; i + t < data.n_x && j + t < data.n_y; t++) {
                double value = std::max(0.0, d - std::abs(d - t * data.step_x));
                if (value > landscape[i + t][j + t]) {
                    landscape[i + t][j + t] = value;
                }
            }
        }
    }
    
    // Write output
    std::ofstream out(output_filename);
    out << std::fixed << std::setprecision(14);
    out << "Sky Landscape " << data.n_x << " " << data.n_y << " " << k << " " << theta_min << "\n";
    for (int i = 0; i < data.n_x; i++) {
        for (int j = 0; j < data.n_y; j++) {
            out << landscape[i][j] << " ";
        }
        out << "\n";
    }
}
