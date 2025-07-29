/**
 * @file generate_decompositions.cpp
 * @author Jan Jendrysiak
 * @brief Generates all vector space decompositions of finite vectorspaces over F_2.
 * @version 0.1
 * @date 2024-10-07
 * 
 * @copyright ?
 * 
 */

#include<grlina/dense_matrix.hpp>
#include<filesystem>

using namespace graded_linalg; 

namespace fs = std::filesystem;

// For a given bitset returns all non-set positions in a col-echelon matrix whose pivots are given by the input.
std::vector<std::pair<int, int>> getEchelonPositions_local(const boost::dynamic_bitset<> &bitset) {
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


std::string bitsetToString(const boost::dynamic_bitset<>& bitset) {
    std::string result;
    result.reserve(bitset.size());
    for (size_t i = 0; i < bitset.size(); ++i) {
        result.push_back(bitset.test(i) ? '1' : '0');
    }
    return result;
}

// Fetches the int which belongs to this pivot structure. is it possible to make this more efficient? 
size_t getIndex(const DenseMatrix &M, std::vector<std::pair<int,int>>& positions){  
    if (positions.empty()){
        positions = getEchelonPositions(M.pivot_vector);
    }
    size_t n = positions.size();
    int ind = 0;
    for (int i = 0; i < positions.size(); i++){
        if(M.data[positions[i].first][positions[i].second]==1){
            ind |= (1 << i);
        }
    }
    return ind;
}

// finds all decomposition mates and appends the pair <input,mate> them to a vector which it outputs
std::vector<VecDecomp> getComplementsToMatrix(boost::dynamic_bitset<> pivots, const DenseMatrix &echelonMatrix, 
                                            bool removeDoubles = false, const DecompBranch &branch = DecompBranch(), 
                                            std::vector<std::pair<int,int>> positions = std::vector<std::pair<int,int>>(), 
                                            bool onlyOneComplement = false){
    std::vector<VecDecomp> matrixPairs;
    auto index = getIndex(echelonMatrix, positions);
    boost::dynamic_bitset<> complement = pivots;
    complement.flip();
    if(onlyOneComplement){
        matrixPairs.push_back(std::make_pair(echelonMatrix, DenseMatrix(complement)));
        return matrixPairs;
    }
    int n = pivots.size();
    int k = pivots.count();
    int l = n-k;
    unsigned long long totalCombinations = 1ULL << (k * l);
    for (unsigned long long combination = 0; combination < totalCombinations ; ++combination) { 
        DenseMatrix firstMatrix = DenseMatrix(echelonMatrix);
        DenseMatrix secondMatrix = DenseMatrix(complement);
        for (int i = 0; i < k; ++i) { 
            for (int j = 0; j < l; ++j) { 
                if ((combination >> (i * l + j)) & 1) {
                    secondMatrix.addOutsideRow(j, firstMatrix.data[i]);
                }
            }
        }
        // make sure that rowReduction get echelon form not only triangular.
        secondMatrix.colReduce(true);
        if(removeDoubles){
            if (compareBitsets(pivots, secondMatrix.pivot_vector)){  
                continue;
            } else if( secondMatrix.pivot_vector == pivots){
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


std::vector<DenseMatrix > pivotsToEchelon_local(const boost::dynamic_bitset<> &pivots, std::vector<std::pair<int,int>> &positions ){
    std::vector<DenseMatrix> reducedMatrices;
    
    size_t n = pivots.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        reducedMatrices.emplace_back( DenseMatrix(pivots, positions, i) );
    }
    return reducedMatrices;
}



DecompBranch getComplements(const boost::dynamic_bitset<>& pivots, bool removeDoubles, bool onlyOneComplement = false){   
    
    DecompBranch matrixPairsForPivots;

    // Calculate all matrices with theses pivots
    std::vector<std::pair<int,int>> positions = getEchelonPositions(pivots);
    std::vector<DenseMatrix> allEchelonMatrices = pivotsToEchelon(pivots, positions);
    // For every reduced matrix we build all matrices representing a decomposition
    for (auto& matrix : allEchelonMatrices){
        matrixPairsForPivots.emplace_back( getComplementsToMatrix(pivots, matrix, removeDoubles, matrixPairsForPivots, positions, onlyOneComplement));
    }
    return matrixPairsForPivots;
}


void generateCombinations_local(boost::dynamic_bitset<> &bitset, int offset, int k, std::vector<boost::dynamic_bitset<>> &combinations) {
    if (k == 0) {
        combinations.push_back(bitset);
        return;
    }

    for (int i = offset ; i < bitset.size(); i++) {
        bitset.set(i);
        generateCombinations_local(bitset, i + 1, k - 1, combinations);
        bitset.reset(i);
    }
}

/**
 * @brief Recursively generates all bitsets of length n with k bits set to 1.
 * 
 * @param n 
 * @param k 
 * @return std::vector<boost::dynamic_bitset<>> 
 */
std::vector<boost::dynamic_bitset<>> generateAllBitsetsWithKOnes_local(int n, int k) {
    std::vector<boost::dynamic_bitset<>> combinations;
    boost::dynamic_bitset<> bitset(n, 0);
    generateCombinations(bitset,  0, k, combinations);
    return combinations;
}

/**
 * @brief Generates all bitsets of length n with n/2 bits set to 1, where the first bit is set to 1.
 * 
 * @param n 
 * @return std::vector<boost::dynamic_bitset<>> 
 */
std::vector<boost::dynamic_bitset<>> generateHalfBitsets_local(int n) {
    // Check if n is even
    if (n % 2 != 0 || n <= 0) {
        throw std::invalid_argument("n must be a positive even number.");
    }

    // Generate all bitsets of length n-1 with n/2 - 1 bits set
    std::vector<boost::dynamic_bitset<>> bitsets = generateAllBitsetsWithKOnes_local(n - 1, n / 2 - 1);

    // Prepend a '1' to each bitset
    for (auto& bitset : bitsets) {
        bitset.resize(n);
        bitset <<= 1;
        bitset[0]=1;
    }

    return bitsets;
}

std::vector<boost::dynamic_bitset<>> generateBitsets_local(int n) {
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

void printProgressBar(int current, int total, std::string message) {
    int barWidth = 70;
    float progress = (float)current / total;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % " << message << "\r";
    std::cout.flush();
}

/**
 * @brief Returns a list of pairs of DenseMatrices giving all decompositions of GF(2)^n into two subspaces. 
 * Alternatively it returns a cover - i.e. giving 
 * Every "pivot" bitset points to all decompositions, where the first subspace has the given pivot structure.
 * The number of decompositions is:
 * 1/2 * sum_{k= 1.. n-1} (n choose k)_2 * 2^(k(n-k))
 * Concretly: 1, 3, 28, 400, 10416, 525792, ... for n = 1, 2, 3, 4, 5, 6, ...
 * whereas if onlyOneComplement is set to true, then the number of decompositions is:
 * 1/2 * sum_{k= 1.. n-1} (n choose k)_2 if n is odd and
 * 1/2 * sum_{k= 1.. n-1} (n choose k)_2 + 1/2 * (n choose n/2)_2 - (n-1 choose n/2)_2 if n is even. 
 * Concretly: 1, 2, 7, 43, 186, 1965, 14605, 297181, ... for n = 1, 2, 3, 4, 5, 6, 7, 8, ...
 * @param n Dimension of the vector space.
 * @param onlyOneComplement Toggles from all decompositions to a cover.
 * @return DecompTree: bitset (pivots of reduced matrix) -> integer (Pluecker coord of first subspace) -> vector of decompositions
 */
DecompTree generateDecompositions(int n, bool onlyOneComplement = false) {
    // std::vector<std::pair<DenseMatrix, DenseMatrix>> matrixPairs;
    // bool isIn = false;
    DecompTree sortedMatrixPairs;
    try {
        int totalIterations = 0;

        std::vector<boost::dynamic_bitset<>> halfBitsets;
        if (n % 2 == 0) {
            halfBitsets = generateHalfBitsets(n);
            totalIterations += halfBitsets.size();
        }

        std::vector<boost::dynamic_bitset<>> bitsets = generateBitsets(n);
        totalIterations += bitsets.size();
        int currentIteration = 0;

        if(n % 2 == 0) {
            for (boost::dynamic_bitset<> &pivots : halfBitsets) {
                sortedMatrixPairs.emplace(pivots, getComplements( pivots, true, onlyOneComplement));
                currentIteration++;
                printProgressBar(currentIteration, totalIterations, "Constructing Matrices");
            }
        } 
        
        for (auto &pivots : bitsets) {
            sortedMatrixPairs.emplace(pivots, getComplements( pivots, false, onlyOneComplement));
            currentIteration++;
            printProgressBar(currentIteration, totalIterations, "Constructing Matrices");
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << '\n';
        // Handle the memory allocation failure (e.g., by returning an empty vector)
    }

    return sortedMatrixPairs;
    
}

vec<transition> compute_transitions(DecompTree& tree, int n){
    DenseMatrix lastMatrix = DenseMatrix(n, "Identity");
    vec<transition> transitions;

    int totalIterations = 0;
    for (auto& [pivots, branch] : tree) {
        totalIterations += branch.size();
    }

    int currentIteration = 0;
    for(auto& [pivots, branch] : tree){
        for(int pluecker = 0; pluecker < branch.size(); pluecker++){
            currentIteration++;
            printProgressBar(currentIteration, totalIterations, "Computing Transitions");
            for(auto& [first, second] : branch[pluecker]){
                int k_1 = first.get_num_cols();
                first.append_matrix(second);
                DenseMatrix T = first.divide_left(lastMatrix);
                vec<int> permutation = T.rectify_invertible();
                T.reorder_columns(permutation);

                assert(T.test_diagonal_entries());
                
                first.reorder_columns(permutation);
                lastMatrix = first;
                bitset subspace = bitset(n, 0);
                for(int j=0; j<k_1; j++){
                    subspace.set(permutation[j]);
                }
                transitions.emplace_back(std::make_pair(T, subspace));
            }
        }
    }
    return transitions;
}

std::vector<transition> generateTransitions(int n, bool onlyOneComplement = false) {
    DecompTree tree = generateDecompositions(n, onlyOneComplement);
    return compute_transitions(tree, n);
}

void save_decomposition(int n, const std::string& folderName, bool onlyOneComplement = false) {
    std::string filename = folderName + "/decompositions_" + (onlyOneComplement ? "reduced_" : "") + std::to_string(n) + ".bin";
    saveDecompTree(generateDecompositions(n, onlyOneComplement), filename);
}

void save_transitions(int n, const std::string& folderName, bool onlyOneComplement = false) {
    std::string filename = folderName + "/transitions_" + (onlyOneComplement ? "reduced_" : "") + std::to_string(n) + ".bin";
    save_transition_list(generateTransitions(n, onlyOneComplement), filename);
}

int tree_size(DecompTree& tree){
    int size = 0;
    for(auto& [pivots, branch] : tree){
        for(int i=0; i< branch.size(); i++){
            std::cout<< "Pivot: " << pivots << "Index: " << i << "size: " << branch[i].size() << std::endl;
            size += branch[i].size();
        }
    }
    return size;
}



void load_files(int dim, bool reduced){
    std::vector<DecompTree> trees;
    for(int i=1; i<dim; i++){
        // std::cout << "loading half decompositions for " << i << std::endl;
        std::string filename = std::string("/home/wsljan/OneDrive/persistence_algebra/listsof_decompositions/decompositions_") + (reduced ?  "reduced_" : "") + std::to_string(i) + ".bin";
        trees.emplace_back(loadDecompTree(filename));
        std::cout << i << " " << tree_size(trees.back()) << std::endl;
    }
}



enum class Option { UNSET, FIRST, SECOND };

struct Options {
    Option mode = Option::UNSET;
    Option coverage = Option::UNSET;
    Option target = Option::UNSET;
    int number = -1;
};

// Function to parse command line arguments and populate Options struct
Options parseArguments(int argc, char* argv[]) {
    Options options;
    bool dimensionProvided = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-transitions") {
            if (options.mode == Option::UNSET) {
                options.mode = Option::FIRST;
            } else {
                std::cerr << "Error: Conflicting mode options with '-transitions'. Either 'transitions' or 'decompositions' already selected." << std::endl;
                exit(1);
            }
        } else if (arg == "-decompositions") {
            if (options.mode == Option::UNSET) {
                options.mode = Option::SECOND;
            } else {
                std::cerr << "Error: Conflicting mode options with '-decompositions'. Either 'transitions' or 'decompositions' already selected." << std::endl;
                exit(1);
            }
        } else if (arg == "-all") {
            if (options.coverage == Option::UNSET) {
                options.coverage = Option::FIRST;
            } else {
                std::cerr << "Error: Conflicting coverage options with '-all'. Either 'cover' or 'all' already selected." << std::endl;
                exit(1);
            }
        } else if (arg == "-cover") {
            if (options.coverage == Option::UNSET) {
                options.coverage = Option::SECOND;
            } else {
                std::cerr << "Error: Conflicting coverage options with '-cover'. Either 'cover' or 'all' already selected." << std::endl;
                exit(1);
            }
        } else if (arg == "-until") {
            if (options.target == Option::UNSET) {
                options.target = Option::FIRST;
            } else {
                std::cerr << "Error: Conflicting target options with '-until'.  Either 'until' or 'at' already selected." << std::endl;
                exit(1);
            }
        } else if (arg == "-at") {
            if (options.target == Option::UNSET) {
                options.target = Option::SECOND;
            } else {
                std::cerr << "Error: Conflicting target options with '-at'. Either 'until' or 'at' already selected." << std::endl;
                exit(1);
            }
        } else {
            try {
                int num = std::stoi(arg);
                if (!dimensionProvided) {
                    options.number = num;
                    dimensionProvided = true;
                } else {
                    std::cerr << "Error: Conflicting dimension options. Dimension already set to " << options.number << "." << std::endl;
                    exit(1);
                }
            } catch (...) {
                std::cerr << "Error: Invalid argument: " << arg << std::endl;
                exit(1);
            }
        }
    }

    if (!dimensionProvided) {
        std::cout << "Please enter the dimension number: ";
        std::cin >> options.number;
    }

    // Output chosen options
    std::cout << "Options chosen:" << std::endl;
    std::cout << "- Mode: ";
    if (options.mode == Option::UNSET) {
        std::cout << "Defaulted to transitions" << std::endl;
        options.mode = Option::FIRST;
    } else {
        std::cout << (options.mode == Option::FIRST ? "transitions" : "decompositions") << std::endl;
    }
    std::cout << "- Coverage: ";
    if (options.coverage == Option::UNSET) {
        std::cout << "Defaulted to cover" << std::endl;
        options.coverage = Option::SECOND;
    } else {
        std::cout << (options.coverage == Option::FIRST ? "all" : "cover") << std::endl;
    }
    std::cout << "- Target: ";
    if (options.target == Option::UNSET) {
        std::cout << "Defaulted to at" << std::endl;
        options.target = Option::SECOND;
    } else {
        std::cout << (options.target == Option::FIRST ? "until" : "at") << std::endl;
    }
    std::cout << "- Dimension: " << options.number << std::endl;

    return options;
}

/**
 * @brief This tests if the transition matrices correctly compute the subspaces given by 
 * the decomposition algorithm. Careful, this test takes a long time for n>6, 
 * because it computes the column space of each matrix via Gauss elimination, 
 * although in practice, if the algorithm works,
 * then the columns of the expected and computed matrices are only permuted.
 * 
 * @param n 
 * @param print_all 
 */
void test_transitions(int n, bool mode = false, bool print_all = false){
    DecompTree tree = generateDecompositions(n, mode);
    std::vector<transition> transitions = generateTransitions(n, mode);
    DenseMatrix last = DenseMatrix(n, "Identity");
    auto it = transitions.begin();
    bool test_passed = true;
    for(auto& [pivots, branch] : tree){
        for(int i=0; i< branch.size(); i++){
            for(auto& [target_first, target_second] : branch[i]){
                if(print_all){
                    std::cout << "Transition:   ";
                    it->first.print();
                    std::cout << "Subspace:   ";
                    print_bitset(it->second);
                    std::cout << "Expected decomposition:   ";
                    target_first.print();
                    target_second.print();
                }
                DenseMatrix nextMatrix = last.multiply_right(it->first);
                // nextMatrix.print();
                vec<int> restriction1;
                vec<int> restriction2;
                for(int j = 0; j < n; j++){
                    if(it->second.test(j)){
                        restriction1.push_back(j);
                    } else {
                        restriction2.push_back(j);
                    }
                }
                DenseMatrix nextMatrix1 = nextMatrix.restricted_domain_copy(restriction1);
                DenseMatrix nextMatrix2 = nextMatrix.restricted_domain_copy(restriction2);
                if(test_passed && !compare_col_space<DenseMatrix>(target_first, nextMatrix1)){
                    std::cout << "First error at position : " << std::distance(transitions.begin(), it) << std::endl;
                    std::cout << "Last matrix was: "; last.print();
                    std::cout << "Transition is: "; it->first.print();
                    std::cout << "Their product is: "; nextMatrix.print();
                    std::cout << "Associated first subspace is: "; print_bitset(it->second);
                    std::cout << "corresponding indices are: " << restriction1 << std::endl;
                    std::cout << "Multiplication and restriction gives "; nextMatrix1.print();
                    std::cout << "But the first subspace needs to be"; target_first.print();
                    test_passed = false;
                } 
                if(!target_second.equals(nextMatrix2, true)){
                    std::cout << "Error: " << std::endl;
                    std::cout << "expected is: "; target_second.print();
                    std::cout << "but restriction gives "; nextMatrix2.print();
                }
                last = nextMatrix;
                it++;
            }
        }
    }
    if(test_passed){
        std::cout << "Test for computing transition matrices passed." << std::endl;
    }
}

void test_save_load(int n, bool mode = false){
    save_decomposition(n, "test_decompositions", mode );
    DecompTree tree_original = generateDecompositions(n, mode);
    save_transitions(n, "test_decompositions", mode );
    vec<transition> transitions_original = generateTransitions(n, mode);
    std::cout << "Test files saved in folder test_decompositions." << std::endl;
    std::string file_ending = (mode ? "reduced_" : "") + std::to_string(n) + ".bin";
    DecompTree tree = loadDecompTree("test_decompositions/decompositions_" + file_ending);
    vec<transition> transitions = load_transition_list("test_decompositions/transitions_" + file_ending);
    
    if(true){
        if(tree.size() != tree_original.size()){
            std::cout << "tree sizes don't match: " << std::endl;
            std::cout << "Expected: " << tree_original.size() << std::endl;
            std::cout << "But got: " << tree.size() << std::endl;
            throw std::runtime_error("Error loading tree.");
        }
        assert(transitions.size() == transitions_original.size());
    for(auto& [pivots, branch] : tree){
        for(int i=0; i< branch.size(); i++){
            for(int j = 0; j < branch[i].size(); j++){
                auto [first, second] = branch[i][j];
                auto [first_original, second_original] = tree_original[pivots][i][j];
                if(!first.equals(first_original) || !second.equals(second_original)){
                    std::cout << "Error at pivot-pluecker-entry triple: " << pivots << " x " << i << " x " << j << std::endl;
                    std::cout << "Expected: "; first_original.print();
                    std::cout << "But got: "; first.print();
                    std::cout << "Expected: "; second_original.print();
                    std::cout << "But got: "; second.print();
                }
            }
        }
    }
    for(int i = 0; i < transitions.size(); i++){
        auto [T, subspace] = transitions[i];
        auto [T_original, subspace_original] = transitions_original[i];
        if(!T.equals(T_original) || subspace != subspace_original){
            std::cout << "Error: " << std::endl;
            std::cout << "Expected: "; T_original.print();
            std::cout << "But got: "; T.print();
            std::cout << "Expected: "; print_bitset(subspace_original);
            std::cout << "But got: "; print_bitset(subspace);
        }
    }
    }
}

void test_sizes_of_decomposition(int n, bool mode = false){
    DecompTree tree = generateDecompositions(n, mode);
    std::cout << "Size of tree: " << std::endl;
    std::cout << tree_size(tree) << std::endl;
}

std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

std::string getExecutableDir() {
    std::string execPath = getExecutablePath();
    std::cout << execPath << std::endl;
    return execPath.substr(0, execPath.find_last_of("/\\"));
}

std::string findDecompositionsDir() {
    std::string base_path = getExecutableDir();
    std::string relative_path_1 = "/../lists_of_decompositions";
    std::string relative_path_2 = "/lists_of_decompositions";

    std::string full_path_1 = base_path + relative_path_1;
    std::string full_path_2 = base_path + relative_path_2;

    if (fs::exists(full_path_1)) {
        return full_path_1;
    } else if (fs::exists(full_path_2)) {
        return full_path_2;
    } else {
        throw std::runtime_error("Could not find the lists_of_decompositions directory. "
                                 "Searched in the following locations:\n" +
                                 full_path_1 + "\n" + full_path_2 + "\n"
                                 "Ensure that the directory exists in one of these locations.");
    }
}

int main(int argc, char* argv[]) {
    bool test_transition_computation = false;
    bool test_save_and_load = false;
    bool test_sizes = false;
    int dim = 5;
    if(test_transition_computation){
        test_transitions(dim, true);
    }
    if(test_save_and_load){
        test_save_load(dim, true);
    }
    if(test_sizes){
        test_sizes_of_decomposition(dim);
    }
    if(test_save_and_load || test_transition_computation || test_sizes){
        return 0;
    }

    const std::string folderName = findDecompositionsDir();

    Options options = parseArguments(argc, argv);
    bool onlyOneComplement = options.coverage == Option::SECOND;

    if (options.mode == Option::FIRST) {
        if (options.target == Option::FIRST) {
            for (int i = 2; i <= options.number; ++i) {
                save_transitions(i, folderName, onlyOneComplement);
            }
        } else {
            save_transitions(options.number, folderName, onlyOneComplement);
        }
    } else {
        if (options.target == Option::FIRST) {
            for (int i = 2; i <= options.number; ++i) {
                save_decomposition(i, folderName, onlyOneComplement);
            }
        } else {
            save_decomposition(options.number, folderName, onlyOneComplement);
        }
    }
    
    return 0;

}

