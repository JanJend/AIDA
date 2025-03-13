#include <iostream>
#include <fstream>
#include <filesystem>
#include <utility>
#include <vector>
#include <bitset>
#include <algorithm>
#include <list>
#include <random>
#include <ctime>
#include <boost/dynamic_bitset.hpp>
#include <unordered_map>
#include <functional> 

using bitset = boost::dynamic_bitset<>;

struct BitsetHash {
    std::size_t operator()(const boost::dynamic_bitset<>& bitset) const {
        // Safe to use to_ulong() if the bitset size is guaranteed to be <= 64
        return static_cast<std::size_t>(bitset.to_ulong());
    }
};

// asks if a < b in order of the entries (reverse of standard comparison)
bool compareBitsets(const boost::dynamic_bitset<>& a, const boost::dynamic_bitset<>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Bitsets must be of the same size for comparison.");
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.test(i) != b.test(i)) {
            return a.test(i) < b.test(i);
        }
    }

    return false; // The bitsets are equal
}


void printBitsetReverse(const boost::dynamic_bitset<>& bitset) {
    std::cout << bitset << std::endl;
}
void printBitset(const boost::dynamic_bitset<>& bitset) {
    for (int i = 0; i < bitset.size(); i++) {
        std::cout << bitset[i];
    }
    std::cout << std::endl;
}

void addTo(boost::dynamic_bitset<> &a, boost::dynamic_bitset<> &b) {
    // Check that the bitsets are of equal size before performing the operation.
    if (a.size() != b.size()) {
        std::cerr << "Error: Bitsets must be of the same size to perform addition." << std::endl;
        return;
    }

    // Add 'a' to 'b' using the XOR operation.
    b ^= a;
}

// For a given bitset returns all non-set positions in a row-echelon matrix whose pivots are given by the input.
std::vector<std::pair<int, int>> getEchelonPositions(const boost::dynamic_bitset<> &bitset) {
    size_t countOnes = 0;
    std::vector<std::pair<int, int>> positions;

    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i) {
        if (!bitset.test(i)) {
            for (int j = 0; j < countOnes; j++){
                positions.push_back(std::make_pair(j, i));
            }
        } else {
            countOnes++;
        }
    }
    return positions;
}

void reOrderPositions ( std::vector<std::pair<int, int>> &positions){
    std::sort(positions.begin(), positions.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.first != b.first){    
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    });
}

// Will write this if needed.
std::vector<boost::dynamic_bitset<>> indexToRows(std::vector<std::pair<int, int>> &positions, int &index ){
    return std::vector<boost::dynamic_bitset<>>();
}

/**
 * @brief Represents Dense Matrices over F_2 columnwise.
 * 
 */
struct BitMatrix {
    int row;
    int col;
    bool rowReduced = false;
    bool completeRowReduced = false;
    boost::dynamic_bitset<> pivots;
    std::vector<boost::dynamic_bitset<>> data;

    BitMatrix() : row(0), col(0) {}

    BitMatrix(int r, int c) : row(r), col(c), data(r, boost::dynamic_bitset<>(c)) {}

    BitMatrix(const std::vector<boost::dynamic_bitset<>>& bitsetVector) {
        if (bitsetVector.empty()) {
            row = 0;
            col = 0;
            return;
        }

        col = bitsetVector[0].size();
        for (const boost::dynamic_bitset<>& bs : bitsetVector) {
            if (bs.size() != col) {
                throw std::invalid_argument("All bitsets must have the same length");
            }
        }

        row = bitsetVector.size();
        data = bitsetVector;
    }

    // Constructor using a pivot bitset
    BitMatrix(const boost::dynamic_bitset<>& i_pivots)
        : row(i_pivots.count()),  // Number of 1s in i_pivots is the number of rows
          col(i_pivots.size()),   // Number of bits in i_pivots is the number of columns
          rowReduced(true),       // By construction, the matrix is in reduced form
          pivots(i_pivots),       // Set the pivots bitset
          data(row, boost::dynamic_bitset<>(col)) { // Initialize all rows

        // Populate the matrix to form an identity matrix on the pivot positions
        size_t pivotIndex = 0; // To keep track of which pivot we're on
        for (size_t r = 0; r < row; ++r) {
            // Find the next set bit in i_pivots, which will be our next pivot
            while (!i_pivots[pivotIndex]) {
                ++pivotIndex; // Skip over unset bits
            }

            // Set the pivot position to 1
            data[r][pivotIndex] = 1;
            ++pivotIndex; // Move to the next pivot for the next row
        }
    }


    void addRow(int row1, int row2) {
        data[row1] ^= data[row2];
    }

    void addCol(int col1, int col2) {
        for (int i = 0; i < row; i++) {
            data[i][col1] = data[i][col1] ^ data[i][col2];
        }
    }

    void swapRows(int row1, int row2) {
        std::swap(data[row1], data[row2]);
    }

    void swapCols(int col1, int col2) {
        for (int i = 0; i < row; i++) {
            bool temp = data[i][col1];
            data[i][col1] = data[i][col2];
            data[i][col2] = temp;
        }
    }


    void rowReduce(bool complete = false) {
        int lead = 0;
        pivots = boost::dynamic_bitset<>(col);
        for (int r = 0; r < row; ++r) {
            if (lead >= col) {
                break; // No more columns to work on, exit the loop
            }

            int i = r;
            // Find the first non-zero entry in the column (potential pivot)
            while (i < row && !data[i][lead]) {
                ++i;
            }

            if (i < row) {
                // Found a non-zero entry, so this column does have a pivot
                // If the pivot is not in the current row, swap the rows
                if (i != r) {
                    swapRows(i, r);
                }
                pivots[lead] = true; // Mark this column as having a pivot after confirming pivot

                // Eliminate all non-zero entries below this pivot
                for (int j = r + 1; j < row; ++j) {
                    if (data[j][lead]) {
                        data[j] ^= data[r];
                    }
                }

                if (complete) {
                    // Eliminate all non-zero entries above this pivot
                    for (int j = 0; j < r; ++j) {
                        if (data[j][lead]) {
                            data[j] ^= data[r];
                        }
                    }
                }

                ++lead; // Move to the next column
            } else {
                // No pivot in this column, so we move to the next column without incrementing the row index
                ++lead;
                --r; // Stay on the same row for the next iteration
            }
        }
        rowReduced = true; 
        if(complete){completeRowReduced = true;}
    }


    bool isRowReduced(bool complete = false)  {
        int lastPivotCol = -1; // Start with -1 to indicate no pivots have been found yet

        for (size_t rowIndex = 0; rowIndex < data.size(); ++rowIndex) {
            const auto& row = data[rowIndex];

            // Find the first non-zero element (pivot) in the current row
            int pivotCol = row.find_first();
            
            if (pivotCol == boost::dynamic_bitset<>::npos) {
                // If there are no non-zero elements, then this row is all zeros, which is fine for echelon form
                continue;
            }

            if (pivotCol <= lastPivotCol) {
                // If the current pivot is not to the right of the previous pivot, the matrix is not in echelon form
                rowReduced = false;
                return false;
            }

            // Check that all elements in this column below the current row are zero
            for (size_t i = rowIndex + 1; i < data.size(); ++i) {
                if (data[i][pivotCol]) {
                    // Found a non-zero element below a pivot, not in echelon form
                    rowReduced = false;
                    return false;
                }
            }

            if (complete) {
                // If complete is true, also check that all elements in this column above the current row are zero
                for (size_t i = 0; i < rowIndex; ++i) {
                    if (data[i][pivotCol]) {
                        // Found a non-zero element above a pivot, not in reduced row-echelon form
                        completeRowReduced = false;
                        return false;
                    }
                }
            }

            // Update the last pivot column
            lastPivotCol = pivotCol;
        }

        // If we've made it here, the matrix is in the correct form
        if(complete){completeRowReduced = true; rowReduced = true;}
        if(!complete){rowReduced = true;}
        return true;
    }

    
    void addOutsideRow(int i, const boost::dynamic_bitset<>& bitset) {
        data[i] ^= bitset; // XOR the i-th row with the bitset
    }



    void appendCol(const boost::dynamic_bitset<>& newCol) {
        if (newCol.size() != row) {
            throw std::invalid_argument("The length of the new column must be equal to the number of rows.");
        }

        for (int i = 0; i < row; i++) {
            data[i].push_back(newCol[i]);
        }

        col++;
    }

    void appendRow(const boost::dynamic_bitset<>& newRow) {
        if (newRow.size() != col) {
            throw std::invalid_argument("The length of the new row must be equal to the number of columns.");
        }

        data.push_back(newRow);
        row++;
    }

    // Prints the matrix
    void print(){
        for (int i = 0; i < row; i++) {
            printBitsetReverse(data[i]);
            /**
            for (int j = 0; j < col; j++) {
                std::cout << data[i][j] << ' ';
            }
            std::cout << '\n';
            */
        }
    }
    
    void printPivotVector() const {
        std::cout << "Pivots vector: ";
        // Iterate over the bits in forward order
        for (std::size_t i = 0; i < pivots.size(); ++i) {
            // Use the test method to print each bit
            std::cout << pivots.test(i);
        }
        std::cout << std::endl;
    }

     // Helper function to compare pairs by their first element
    static bool comparePairs(const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.first != b.first){    
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

    // Function to set entries based on vector of pairs. This may not be super effcient if the vector has many entries.
    void setEntries(std::vector<std::pair<int, int>>& positions, bool target) {
        // Sort the vector by the row index
        std::sort(positions.begin(), positions.end(), comparePairs);

        // Iterate over the sorted vector and set entries
        for (const auto& position : positions) {
            int rowIndex = position.first;
            int colIndex = position.second;

            // Check if the indices are within the bounds of the matrix
            if (rowIndex >= 0 && rowIndex < row && colIndex >= 0 && colIndex < col) {
                data[rowIndex][colIndex] = target; // Set the value at the position to the target
            } else {
                // Handle the error for indices out of bounds if necessary
                std::cerr << "Index out of bounds: (" << rowIndex << ", " << colIndex << ")" << std::endl;
            }
        }
    }

    bool isEqualTo(BitMatrix& other)  {
        if (row != other.row || col != other.col) {
            return false; // Different dimensions mean they can't be equal
        }
        for (int i = 0; i < row; ++i) {
            if (data[i] != other.data[i]) {
                return false; // If any corresponding bitsets are not equal, matrices are not equal
            }
        }
        return true; // All corresponding bitsets are equal
    }

    bool comparePivots(const BitMatrix& other) const {
        return pivots == other.pivots;
    }
    
};

using BitDecomp = std::pair<BitMatrix, BitMatrix>;

using BitMatrixBranch = std::unordered_map<size_t, std::vector<BitDecomp>>;

using BitMatrixTree = std::unordered_map<boost::dynamic_bitset<>, BitMatrixBranch , BitsetHash>;



// Right now it copies the matrix, but there should be an option to work directly on the matrix.
std::pair<bool, boost::dynamic_bitset<>> Solve(BitMatrix &A, const boost::dynamic_bitset<> &b) {
    // Check for proper dimensions
    if (b.size() != A.row) {
        std::cerr << "Cannot solve the linear system because the target vector b does not have the correct length." << std::endl;
        exit(1);
    }

    // Create an augmented matrix by appending b as the last column
    BitMatrix M = A;
    M.appendCol(b);
    M.print();
    M.rowReduce();
    M.print();
    // Check if there is a pivot in the last column, indicating an inconsistent system
    if (M.pivots.test(M.col - 1)) {
        // Inconsistent system, no solution exists
        return std::make_pair(false, boost::dynamic_bitset<>());
    }

    // Otherwise, the system is consistent and we can find a solution
    boost::dynamic_bitset<> x(M.col - 1); // Solution vector of size col-1 (excluding augmented part)

    // Back substitution
    for (int i = M.row - 1; i >= 0; --i) {
        // If the row is all zeros, there is nothing to do
        if (M.data[i].none()) continue;

        // Find the pivot in the current row
        for (int j = 0; j < M.col - 1; ++j) {
            if (M.data[i][j]) {
                // Assign the corresponding value from the augmented column to the solution
                x[j] = M.data[i][M.col - 1];

                // Update the rest of the rows
                for (int k = 0; k < i; ++k) {
                    if (M.data[k][j]) {
                        M.data[k][M.col - 1] ^= x[j];
                    }
                }
                break; // Move to the next row
            }
        }
    }
    printBitset(x);
    
    // Return the pair (true, solution_vector)
    return std::make_pair(true, x);
}

// is it possible to make this more efficient?
size_t getIndex(const BitMatrix &M, std::vector<std::pair<int,int>>& positions){  
    if (positions.empty()){
        positions = getEchelonPositions(M.pivots);
    }
    size_t n = positions.size();
    int index = 0;
    for (int i = 0; i < positions.size(); i++){
        if(M.data[positions[i].first][positions[i].second]==1){
            index |= (1 << i);
        }
    }
    return index;
}

// Generate a list of all matrices in row-echelon form with the pivots given by the input.
std::vector<std::pair<BitMatrix, size_t> > pivotsToEchelon(const boost::dynamic_bitset<> &bitset, std::vector<std::pair<int,int>> &positions ){
    std::vector<std::pair<BitMatrix, size_t> > reducedMatrices;
    
    size_t n = bitset.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        BitMatrix current = BitMatrix(bitset);
        std::vector<std::pair<int,int>> subset;
        // This could be done faster, but I think it wont matter.
        for (int j=0; j < mul; j++){
            if (i & (1 << j)) {
                current.data[positions[j].first][positions[j].second] = 1;
            }
        }
        reducedMatrices.push_back(std::make_pair(current, i));
    }
    return reducedMatrices;
}

// returns the unique number corresponing to a completely reduced row-echelon matrix. 
// This is an inverse to the function in pivotsToEchelon.



//This should implement an algorihtm which solves a linear system by column reduction. 
// We are using the transpose, because in our model the row operations are fast
bool isInRowspace(BitMatrix &A, boost::dynamic_bitset<> &b) {
    if (b.size() != A.col) {
        std::cerr << "The vector" << b << " is not of the correct length" << std::endl;
        exit(1);
    }
    
// First, perform row reduction on matrix A
    BitMatrix M = A;
    M.rowReduce();

    // Now, try to express b as a linear combination of the rows of A.
    // If b is in the row space of A, there exists a set of coefficients (possibly a bitset)
    // such that a linear combination of the rows of A equals b.

    // We'll need a method to iterate over the non-zero rows of A and attempt
    // to construct b as a linear combination of these rows.
    for (size_t i = 0; i < M.row; ++i) {
        const auto& row = M.data[i];
        if (!row.none()) {  // Check if the row is not all zeroes
            // Check if this row can be used in the linear combination to construct b
            // This would involve bit manipulation logic specific to your application
        }
    }

    
    return false;  
}


std::list<BitMatrix> generateMatrixCombinations(int n, int k) {
    if (k > n) {
        throw std::invalid_argument("k should not be greater than n");
    }

    if (k == 0) {
        return std::list<BitMatrix>{BitMatrix(n, 0)};
    }

    std::list<BitMatrix> combinations = generateMatrixCombinations(n, k - 1);
    std::list<BitMatrix> currentCombinations;

    for (int i = 0; i < (1 << n); i++) {
        boost::dynamic_bitset<> rowBits(n, i);
        
        for (BitMatrix& matrix : combinations) {
            BitMatrix newMatrix(matrix);
            newMatrix.appendRow(rowBits);
            currentCombinations.push_back(newMatrix);
        }
    }

    return currentCombinations;
}





void generateCombinations(boost::dynamic_bitset<> &bitset, int offset, int k, std::vector<boost::dynamic_bitset<>> &combinations) {
    if (k == 0) {
        combinations.push_back(bitset);
        return;
    }

    for (int i = offset; i <= bitset.size() - k; ++i) {
        bitset.set(i);
        generateCombinations(bitset, i + 1, k - 1, combinations);
        bitset.reset(i);
    }
}

std::vector<boost::dynamic_bitset<>> generateAllBitsetsWithKOnes(int n, int k) {
    std::vector<boost::dynamic_bitset<>> combinations;
    boost::dynamic_bitset<> bitset(n, 0);
    generateCombinations(bitset, 0, k, combinations);
    return combinations;
}

std::vector<boost::dynamic_bitset<>> generateBitsets(int n) {
    std::vector<boost::dynamic_bitset<>> bitsets;

    // Start k from the largest possible value less than n/2 (if n is even) or less than n/2 rounded down (if n is odd)
    int startK = (n % 2 == 0) ? (n / 2 - 1) : (n / 2);

    // Generate and append bitsets with fewer ones
    for (int k = startK; k > 0; --k) {
        std::vector<boost::dynamic_bitset<>> additionalBitsets = generateAllBitsetsWithKOnes(n, k);
        bitsets.insert(bitsets.end(), additionalBitsets.begin(), additionalBitsets.end());
    }

    return bitsets;
}

std::vector<boost::dynamic_bitset<>> generateHalfBitsets(int n) {
    // Check if n is even
    if (n % 2 != 0 || n <= 0) {
        throw std::invalid_argument("n must be a positive even number.");
    }

    // Generate all bitsets of length n-1 with n/2 - 1 bits set
    std::vector<boost::dynamic_bitset<>> bitsets = generateAllBitsetsWithKOnes(n - 1, n / 2 - 1);

    // Prepend a '1' to each bitset
    for (auto& bitset : bitsets) {
        bitset.resize(n);
        bitset <<= 1;
        bitset[0]=1;
    }

    return bitsets;
}

// finds all decomposition mates and appends the pair <input,mate> them to a vector which it outputs
std::vector<BitDecomp> getComplementsToMatrix(boost::dynamic_bitset<> pivots, const BitMatrix &echelonMatrix, bool removeDoubles = false, const BitMatrixBranch &branch = BitMatrixBranch(), std::vector<std::pair<int,int>> positions = std::vector<std::pair<int,int>>()){
    std::vector<BitDecomp> matrixPairs;
    auto index = getIndex(echelonMatrix, positions);
    boost::dynamic_bitset<> complement = pivots;
    complement.flip();
    int n = pivots.size();
    int k = pivots.count();
    int l = n-k;
    unsigned long long totalCombinations = 1ULL << (k * l);
    for (unsigned long long combination = 0; combination < totalCombinations ; ++combination) { 
        BitMatrix firstMatrix = BitMatrix(echelonMatrix);
        BitMatrix secondMatrix = BitMatrix(complement);
        for (int i = 0; i < k; ++i) { 
            for (int j = 0; j < l; ++j) { 
                if ((combination >> (i * l + j)) & 1) {
                    secondMatrix.addOutsideRow(j, firstMatrix.data[i]);
                }
            }
        }
        // make sure that rowReduction get echelon form not only triangular.
        secondMatrix.rowReduce(true);
        if(removeDoubles){
            if (compareBitsets(pivots, secondMatrix.pivots)){  
                continue;
            } else if( secondMatrix.pivots == pivots){
                auto index2 = getIndex(secondMatrix, positions);
                if( index2 < index ){
                    continue;
                } else {
                    matrixPairs.push_back(std::make_pair(firstMatrix, secondMatrix));
                }
            } else {
                matrixPairs.push_back(std::make_pair(firstMatrix, secondMatrix));
            }
        } else {
            matrixPairs.push_back(std::make_pair(firstMatrix, secondMatrix));    
        }
        
    }
    return matrixPairs;
}



BitMatrixBranch getComplements(const boost::dynamic_bitset<>& pivots, bool removeDoubles){   
    
    BitMatrixBranch matrixPairsForPivots;

    // Calculate all matrices with theses pivots
    std::vector<std::pair<int,int>> positions = getEchelonPositions(pivots);
    std::vector<std::pair<BitMatrix, size_t>> allEchelonMatrices = pivotsToEchelon(pivots, positions);
    // For every reduced matrix we build all matrices representing a decomposition
    for (std::pair<BitMatrix, size_t>& matrix : allEchelonMatrices){
        matrixPairsForPivots[matrix.second] = getComplementsToMatrix(pivots, matrix.first, removeDoubles, matrixPairsForPivots, positions);
    }
    return matrixPairsForPivots;
}





// Returns a list of pairs of BitMatrices giving all decompositions of GF(2)^n into two subspaces
BitMatrixTree generateDecompositions(int n) {
    // std::vector<std::pair<BitMatrix, BitMatrix>> matrixPairs;
    // bool isIn = false;
    BitMatrixTree sortedMatrixPairs;
    try {
        if (n % 2 == 0) {
            std::vector<boost::dynamic_bitset<>> halfBitsets = generateHalfBitsets(n);
            int counter = 0;
            
            for (boost::dynamic_bitset<> &pivots : halfBitsets) {
                sortedMatrixPairs[pivots] = getComplements( pivots, true);
            }
        } 
        std::vector<boost::dynamic_bitset<>> bitsets = generateBitsets(n);
        for (auto &pivots : bitsets) {
            sortedMatrixPairs[pivots] = getComplements( pivots, false);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << '\n';
        // Handle the memory allocation failure (e.g., by returning an empty vector)
    }

    return sortedMatrixPairs;
    
}

void removeDoubles(BitMatrixTree &decomps){
    for(auto& [pivots, BitMatrixBranch] : decomps){
        auto positions = getEchelonPositions(pivots);
        for(auto& [index, vec] : BitMatrixBranch){
            for(auto it = vec.begin(); it != vec.end();){
                if( compareBitsets(pivots, it->second.pivots) ){
                    it = vec.erase(it);
                } else if( it->second.pivots == pivots){
                    auto compareIndex = getIndex( it->second, positions );
                    if (compareIndex < index){
                        it = vec.erase(it);
                    } else {
                        ++it;
                    }
                } else {
                    ++it;
                }
            }
        }
    }
}


void computeTransitionMatrices(int n){
    BitMatrixTree decomps = generateDecompositions(n);
    for(auto& [pivots, branch] : decomps){
        for(auto& [index, matrixPairs] : branch){
            for(auto& [first, second] : matrixPairs){
            }
        }
    }
}


boost::dynamic_bitset<> generateRandomBitset(size_t n) {
    boost::dynamic_bitset<> bitset(n);
    std::random_device rd; // Non-deterministic generator
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, 1);

    for (size_t i = 0; i < n; ++i) {
        bitset[i] = distrib(gen); // Randomly set to 0 or 1
    }

    return bitset;
}


boost::dynamic_bitset<> generateRandomBitsetWithKNonZero(size_t k, size_t n) {
    if (k > n) {
        throw std::invalid_argument("k cannot be greater than n.");
    }

    // Initialize a bitset of length n with all zeros
    boost::dynamic_bitset<> bitset(n);

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, n - 1);

    std::vector<size_t> positions(n);
    std::iota(positions.begin(), positions.end(), 0); // Fill with 0, 1, ..., n-1

    // Shuffle and pick the first k positions to set to 1
    std::shuffle(positions.begin(), positions.end(), gen);
    for (size_t i = 0; i < k; ++i) {
        bitset.set(positions[i]);
    }

    return bitset;
}


BitMatrix RandomBitMatrix(int rows, int cols) {
    BitMatrix randomMatrix(rows, cols);

    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Fill the matrix with random data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            randomMatrix.data[i][j] = std::rand() % 2;
        }
    }

    return randomMatrix;
}

//Tests:



void testPrint(int n){
    boost::dynamic_bitset<> x = generateRandomBitset( n);
    printBitset(x);
    printBitsetReverse(x);
}

void testCombinations() {
    int n = 4; // Length of bitsets
    int k = 2; // Number of bitsets in each combination

    std::list<BitMatrix> combinations = generateMatrixCombinations(n, k);

    int combinationCount = 1;
    for (const BitMatrix& matrix : combinations) {
        std::cout << "Combination " << combinationCount << ":\n";
        combinationCount++;

        for (int i = 0; i < matrix.row; i++) {
            for (int j = 0; j < matrix.col; j++) {
                std::cout << matrix.data[i][j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

// Function to generate a random BitMatrix and test row reduction
void testRowReduction(int rows, int cols) {
    // Create a random BitMatrix
    BitMatrix randomMatrix = RandomBitMatrix(rows, cols);

    // Print the random matrix
    randomMatrix.print();
    std::cout << "row reduction makes:";

    // Perform row reduction
    randomMatrix.rowReduce();

    // Print the row-reduced matrix
    randomMatrix.print();
    randomMatrix.printPivotVector();

    // Change the matrix a bit
    const boost::dynamic_bitset<> newCol =  generateRandomBitset(rows);
    const boost::dynamic_bitset<> newRow =  generateRandomBitset(cols +1);
    randomMatrix.appendCol( newCol);
    randomMatrix.appendRow( newRow );
    randomMatrix.print();
    // Perform row reduction again 
    randomMatrix.rowReduce();

    randomMatrix.print();
    randomMatrix.printPivotVector();
}

bool testPivotCreation(int length, int entries){
    boost::dynamic_bitset<> pivots = generateRandomBitsetWithKNonZero(length, entries);
    printBitset(pivots);
    BitMatrix testMatrix = BitMatrix(pivots);
    testMatrix.print();
    testMatrix.rowReduce();
    return (testMatrix.pivots == pivots);
}


void testEchelonFormer(int n){
    boost::dynamic_bitset<> pivots = generateRandomBitset(n);
    printBitset(pivots);
    auto positions = getEchelonPositions(pivots);
    std::vector<std::pair<BitMatrix, size_t>> matrices = pivotsToEchelon(pivots, positions);
    for (std::pair<BitMatrix, size_t>& matrix : matrices) {
            matrix.first.print();
    }
}


void testGenerateBitsets(int n){    
    std::vector<boost::dynamic_bitset<>> pivotlist = generateBitsets(n);
    for (boost::dynamic_bitset<>& pivots : pivotlist){
        printBitset(pivots);
    }
}

bool testSolver(int rows, int cols) {
    // Generate a random matrix A and a random solution vector x
    BitMatrix A = RandomBitMatrix(rows, cols);
    boost::dynamic_bitset<> x(cols); // solution vector should have 'cols' bits
    for (std::size_t i = 0; i < cols; ++i) {
        x[i] = std::rand() % 2;
    }
    A.print();
    printBitset(x);
    // Calculate b = Ax
    boost::dynamic_bitset<> b(rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (A.data[i][j]) {
                b[i] ^= x[j];
            }
        }
    }
    printBitset(b);
    std::cout << "Solver comes";

    // Use the Solve function to solve the system Ax = b
    auto [success, solution] = Solve(A, b);

    boost::dynamic_bitset<> bprime(rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (A.data[i][j]) {
                b[i] ^= solution[j];
            }
        }
    }

    // Check if the solution returned is successful and equals the original solution vector x
    bool testPassed = success && (b == bprime);

    // Output the result of the test
    std::cout << "Test " << (testPassed ? "PASSED" : "FAILED") << std::endl;

    // Return true if the test passed, false otherwise
    return testPassed;
}




void testGetComplement(int n){
    boost::dynamic_bitset<> pivots = generateRandomBitset(n);
    printBitset(pivots);
    BitMatrix subSpace = BitMatrix(pivots);
    subSpace.print();
    std::vector<std::pair<BitMatrix, BitMatrix>> matrixPairs = getComplementsToMatrix(pivots, subSpace);
    std::cout << "There are " << matrixPairs.size() << " decompositions with this initial subspace" << std::endl;
    for (std::pair<BitMatrix, BitMatrix>& pair : matrixPairs){
        std::cout << "first" << std::endl;
        pair.first.print();
        std::cout << "second" << std::endl;
        pair.second.print();

    }
}



// Helper function to create BitMatrix for testing
BitMatrix createTestMatrix(unsigned long a, unsigned long b) {
    boost::dynamic_bitset<> row1(4, a); // 4 is the number of bits
    boost::dynamic_bitset<> row2(4, b);
    std::vector<boost::dynamic_bitset<>> rows = {row1, row2};
    return BitMatrix(rows);
}

// Test function for removeReverseDoubles
void testRemoveDoubles() {
    // Create a list of unique 2x4 matrices
    std::vector<BitMatrix> matrices = {
        createTestMatrix(0b0010, 0b0100),
        createTestMatrix(0b0011, 0b0100),
        createTestMatrix(0b1010, 0b0110),
        createTestMatrix(0b1100, 0b0011),
        createTestMatrix(0b1110, 0b0111)
    };

    // Create a vector with some pairs and intentionally add reverse doubles
    std::vector<std::pair<BitMatrix, BitMatrix>> matrixPairs = {
        {matrices[0], matrices[1]},
        {matrices[2], matrices[3]},
        {matrices[1], matrices[0]}, // Reverse of the first pair
        {matrices[4], matrices[2]},
        {matrices[3], matrices[2]}  // Reverse of the second pair
    };

    // Size before removing reverse doubles
    size_t sizeBefore = matrixPairs.size();

    // Call the function to remove reverse doubles
    // removeDoubles(matrixPairs);

    // Size after removing reverse doubles
    size_t sizeAfter = matrixPairs.size();

    // There should be two fewer pairs after the function call
    if (sizeBefore - sizeAfter != 2) {
        std::cerr << "Test failed: Expected to remove 2 pairs, but " << sizeBefore - sizeAfter << " were removed." << std::endl;
        return;
    }

    // Additional checks can be added here to ensure that no reverse duplicates remain
    // ...

    std::cout << "Test passed: removeReverseDoubles appears to work correctly." << std::endl;
}

// Helper function to calculate binomial coefficients
size_t binomialCoefficient(size_t n, size_t k) {
    if (k > n) {
        return 0;
    }
    size_t result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Test function
void testGenerateBitsetsWithHalfOnes(int n) {
    auto bitsets = generateHalfBitsets(n);
    boost::dynamic_bitset<> lastbitset;
    size_t expectedCount = binomialCoefficient(n-1, n / 2-1);
    if (bitsets.size() != expectedCount) {
        std::cerr << "Test failed: Incorrect number of bitsets generated." << expectedCount << " versus " << bitsets.size() << std::endl;
        return;
    }

    for (const auto& bitset : bitsets) {
        printBitset(bitset);
        if (bitset.size() != static_cast<size_t>(n)) {
            std::cerr << "Test failed: Bitset size is incorrect." << std::endl;
            return;
        }
        if (bitset.count() != static_cast<size_t>(n / 2)) {
            std::cerr << "Test failed: Bitset does not have n/2 ones." << std::endl;
            return;
        }
        if (!bitset.test(0)) {
            std::cerr << "Test failed: The first bit of bitset is not set to 1." << std::endl;
            return;
        }
        if (! lastbitset.size() == 0 && !compareBitsets(bitset, lastbitset) ) {
            std::cerr << "Test failed: The bitsets do not appear in order:" << std::endl;
            printBitset(lastbitset);
            printBitset(bitset);
            return;
        }
        auto lastbitset = bitset;
    }

    std::cout << "Test passed: All generated bitsets are valid." << std::endl;
}

void testIndexFetching(int n){
    auto pivots = generateRandomBitset(n);
    printBitset(pivots);
    auto positions = getEchelonPositions(pivots);
    std::vector<std::pair<BitMatrix, size_t>> allEchelonMatrices = pivotsToEchelon(pivots, positions);
    for(auto& [matrix, index] : allEchelonMatrices){
        std::cout << "Index is " << index << std::endl;
        matrix.print();
        assert(index == getIndex(matrix, positions));
    }
}

size_t tree_size(BitMatrixTree& tree){
    size_t size = 0;
    for(auto& [pivots, branch] : tree){
        for(auto& [index, matrixPairs] : branch){
            for(auto& [first, second] : matrixPairs){
                size++;
            }
        }
    }
    return size;
}

void testDecomp(int n){
    BitMatrixTree decomps = generateDecompositions(n);
    // for(std::pair<BitMatrix, BitMatrix>& decomp : decomps){
    //    std::cout << "Next Decomposition is " << decomp.first.row << "," << decomp.second.row  << std::endl;
    //    decomp.first.print();
    //    decomp.second.print();
    // }
    std::cout << "Calculation finished. Now counting: \n";
    size_t counter = tree_size(decomps);
    std::cout << "There are " << counter << " non-trivial decompositions of the " << n << " dimensional vector space over GF(2)" << std::endl;
}



void testBitsetFunctioning(int n){
    boost::dynamic_bitset<> bitset1(3);
    bitset1[0] = 1; 

    
    boost::dynamic_bitset<> bitset2(3);
    bitset2[1] = 1;

    boost::dynamic_bitset<> bitset3(3);
    bitset3[2] = 1;

    printBitset(bitset1);
    printBitset(bitset2);
    printBitset(bitset3);
    std::cout << (bitset1 < bitset2) << std::endl;
    std::cout << (bitset2 < bitset3) << std::endl;
}

std::string bitsetToString(const boost::dynamic_bitset<>& bitset) {
    std::string result;
    result.reserve(bitset.size());
    for (size_t i = 0; i < bitset.size(); ++i) {
        result.push_back(bitset.test(i) ? '1' : '0');
    }
    return result;
}

void serializeDynamicBitset(const boost::dynamic_bitset<>& bitset, std::ofstream& file) {
    std::string bitsetString = bitsetToString(bitset);
        size_t length = bitsetString.length();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(bitsetString.c_str(), length);
}

// Deserialization function for dynamic_bitset
boost::dynamic_bitset<> deserializeDynamicBitset(std::ifstream& file) {
    size_t length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    std::string bitsetString(length, '\0');
    file.read(&bitsetString[0], length);
    return boost::dynamic_bitset<>(bitsetString);
}

void serializeBitMatrix(const BitMatrix& matrix, std::ofstream& file) {
     // Write the size of the vector
    size_t vectorSize = matrix.data.size();
    file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

    // Serialize each dynamic_bitset in the vector
    for (const auto& bitset : matrix.data) {
        serializeDynamicBitset(bitset, file);
    }
}

BitMatrix deserializeBitMatrix(std::ifstream& file) {
    size_t vectorSize;
    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));
    BitMatrix matrix;
    matrix.row = vectorSize;
    matrix.data.reserve(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        matrix.data.emplace_back(deserializeDynamicBitset(file));
    }
    matrix.col = matrix.data[0].size();
    return matrix;
}

void saveBitMatrixTree(const BitMatrixTree& tree, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing.");
    }

    // Write BitMatrixTree size
    size_t treeSize = tree.size();
    file.write(reinterpret_cast<const char*>(&treeSize), sizeof(treeSize));

    for (const auto& [key, branch] : tree) {
        serializeDynamicBitset(key, file);

        // Write BitMatrixBranch size
        size_t branchSize = branch.size();
        file.write(reinterpret_cast<const char*>(&branchSize), sizeof(branchSize));

        for (const auto& [branchKey, bitDecomps] : branch) {
            // Write branch key
            file.write(reinterpret_cast<const char*>(&branchKey), sizeof(branchKey));

            // Write vector size
            size_t vectorSize = bitDecomps.size();
            file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

            // Serialize and write each BitDecomp
            for (const auto& bitDecomp : bitDecomps) {
                // Serialize and write each BitMatrix in the BitDecomp pair
                // Assuming serializeBitMatrix is a function you implement
                serializeBitMatrix(bitDecomp.first, file);
                serializeBitMatrix(bitDecomp.second, file);
            }
        }
    }

    file.close();
}

BitMatrixTree loadBitMatrixTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for reading.");
    }

    size_t treeSize;
    file.read(reinterpret_cast<char*>(&treeSize), sizeof(treeSize));

    BitMatrixTree tree;
    for (size_t i = 0; i < treeSize; ++i) {
        boost::dynamic_bitset<> key = deserializeDynamicBitset(file);

        size_t branchSize;
        file.read(reinterpret_cast<char*>(&branchSize), sizeof(branchSize));

        BitMatrixBranch branch;
        for (size_t j = 0; j < branchSize; ++j) {
            size_t branchKey;
            file.read(reinterpret_cast<char*>(&branchKey), sizeof(branchKey));

            size_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

            std::vector<BitDecomp> bitDecomps;
            bitDecomps.reserve(vectorSize);
            for (size_t k = 0; k < vectorSize; ++k) {
                bitDecomps.emplace_back(deserializeBitMatrix(file), deserializeBitMatrix(file));
            }

            branch.emplace(branchKey, std::move(bitDecomps));
        }

        tree.emplace(std::move(key), std::move(branch));
    }

    file.close();
    return tree;
}

void saveDecompositions(int k) {
    const std::string folderName = "/home/wsljan/generalized_persistence/code/listsof_decompositions";
    std::filesystem::create_directory(folderName);

    for (int n = 1; n <= k; ++n) {
        BitMatrixTree tree = generateDecompositions(n);
        std::string filename = folderName + "/decomposition_" + std::to_string(n) + ".bin";
        saveBitMatrixTree(tree, filename);
    }
}

void printBitsetTree(BitMatrixTree& tree){
    for(auto& [pivots, branch] : tree){ 
            std::cout << "Pivots: ";
            printBitsetReverse(pivots);
            for(auto& [index, matrixPairs] : branch){
                std::cout << "index: ";
                std::cout << index << std::endl;
                for(auto& [second, first] : matrixPairs){
                    std::cout << "first" << std::endl;
                    first.print();
                    std::cout << "second" << std::endl;
                    second.print();
                }
            }
        }
}

int main() {
    

    // Set the number of rows and columns for the random BitMatrix
    int rows = 2;
    int cols = 4;
    int k = 4;
   


    testGenerateBitsetsWithHalfOnes(k);


    // saveDecompositions(k);
    //testGenerateBitsetsWithHalfOnes(n);
    
    
    // BitMatrixTree tree = loadBitMatrixTree("/home/wsljan/OneDrive/persistence_algebra/listsof_decompositions/decomposition_3.bin");
    // size_t counter = tree_size(tree);    
    //std::cout << "size of loaded tree is " << counter << std::endl;


    
    
    
}

