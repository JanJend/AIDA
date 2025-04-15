# AIDA

**AIDA** is a library for decomposing multiparameter persistence modules, along with a library **Persistence-Algebra** for computations involving (graded) matrices which present mpm over F_2.

**Author:** Jan Jendrysiak  
**Version:** 0.2  
**Last Updated:** 2025-04-15  
**License:** GNU  General Public License v3+

---

## 📄 Input Requirements

Make sure that the input file is a (sequence of) `.scc` or `.firep` presentations that are **minimised**.

---

## 🚀 How to install AIDA

1. Create a `build` folder in the root directory of the project:
   ```
   mkdir build && cd build
   ```
2. Run CMake with Release mode:
   ```
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```
3. Compile:
   ```
   make
   ```
4. Run AIDA:
   ```
   ./aida <path_to_file> -options
   ```

---

## ⚙️ Command-Line Options


### General Options

- `-h`, `--help`  
  Display a help message.

- `-v`, `--version`  
  Show version information.

- `-p`, `--progress`  
  Show progress bar.

- `-l`, `--less_console`  
  Suppress most console output.
  
- `-t`, `--statistics`  
  Show statistics about indecomposable summands.

- `-r`, `--runtime`  
  Show runtime statistics and timers.

---
  
### Output Options

- `-o`  
  Writes the output to `<input_file_name>_decomposition.scc` in the current directory.

- `-o <output_file>`  
  Writes the output to a specified file (absolute or relative path).

- `-o <output_directory>`  
  Writes the output to the given directory, with the file named `<input_file_name>_decomposition.scc`.

- `-o<argument>`  
  Same as above, but without a space between `-o` and the argument.

---

### Options changing the algorithm

- `-b`, `--bruteforce`  
  Disables the computation of homomorphisms completly and all further subroutines which rely on it. 
  Implies using the exhaustive alpha-decomposition

- `-s`, `--sort`  
  Lexicographically sort the relations in the input.

- `-e`, `--exhaustive`  
  Always iterate over all decompositions of a batch.

- `-c`, `--basechange`  
  Save base change data.

- `-f`, `--alpha`  
  Enable computation of alpha-homs.

- `-j`, `--no_hom_opt`  
  Disable optimized hom-space calculations.

---

### Testing & Debugging Options

- `-m`, `--compare_b`  
  Compare with `-b` at runtime, then rerun with only `-b` and compare results.

- `-a`, `--compare_e`  
  Compare exhaustive and brute-force strategies at runtime.

- `-i`, `--compare_hom`  
  Compare optimized and non-optimized hom-space calculations.

- `-w`, `--no_col_sweep`  
  Disable column sweep optimization.

- `-x`, `--test_files`  
  Run the algorithm on example test files.

---


## License

© 2025 Jan Jendrysiak / TU Graz  
This file is part of the AIDA library.  
You can redistribute it and/or modify it under the terms of the  
GNU General Public License as published by the Free Software Foundation,  
either version 3 of the License, or (at your option) any later version.
