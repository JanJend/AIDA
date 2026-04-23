# AIDA

**A**utomorphism **I**nvariant **D**ecomposition **A**lgorithm

AIDA is a C++ library and tool for decomposing finitely presented multiparameter persistence modules into indecomposable summands. It implements the algorithm described in [Dey, Jendrysiak, Kerber (2025)](https://arxiv.org/abs/2504.08119) with significant optimizations for practical computation.

## Key Features

- Decomposes all finitely presented multiparameter persistence modules (not limited to distinctly graded generators/relations)
- Fixed-parameter tractable with respect to the maximal number of relations of the same degree
- Optimized hom-space calculations for faster computation for modules with low layer-thickness (pointwise dimension of indecomposable summands)
- Runtime and statistical analysis tools

- Missing features: O(n³) algorithm for interval-decomposable modules 

## Installation

### Prerequisites

- **CMake** ≥ 3.12
- **C++17** compatible compiler
- **Boost** libraries (timer, chrono)
- **[Persistence-Algebra](https://github.com/JanJend/Persistence-Algebra/)** library

### Building from Source

1. Clone both repositories into the same parent directory:
```bash
git clone https://github.com/JanJend/AIDA.git
git clone https://github.com/JanJend/Persistence-Algebra.git
```

Your directory structure should look like:
```
parent-dir/
├── AIDA/
└── Persistence-Algebra/
```

2. Build AIDA:
```bash
cd AIDA
mkdir build && cd build
cmake ..
make
```

3. (Optional) Install system-wide:
```bash
sudo make install
```

This installs:
- Static library `libaida.a` to `/usr/local/lib`
- Executables to `/usr/local/bin`

> **Note for library users:** If you want to use AIDA as a library in another CMake project, the install step above is **required** — not optional. Without it, `libaida.a` only exists inside the build directory and other programs will not be able to find it. The recommended way to install is:
> ```bash
> sudo cmake --install build
> ```
> Run this from the `AIDA/` directory after building. You can also specify a custom install prefix:
> ```bash
> cmake --install build --prefix /your/custom/path
> ```
> After installing to a custom prefix, make sure to add the `lib/` subdirectory to your linker search path (e.g. via `LD_LIBRARY_PATH` or your project's `CMAKE_PREFIX_PATH`).

### Platform Support

AIDA should compile on Linux, Windows, and macOS. It has been primarily tested on Linux and Mac.

## Usage

### Basic Usage

```bash
aida <input_file> [options]
```

**Input format**: Minimized presentation in `.scc` (scc2020) or `.firep` format

**Example**:
```bash
aida tests/test_presentations/two_circles.scc -o 
```

saves the output as two_circles.sccsum or use 

```bash
aida tests/test_presentations/two_circles.scc -o path/output.sccsum
```

to indicate the output path yourself

### Executables

- **`aida`**: Main decomposition tool (optimized, no statistics)
- **`aida_with_stats`**: Same with indecomposable statistics enabled
- **`generate_decompositions`**: Utility for generating decomposition lists

### Command-Line Options

#### General Options
- `-h, --help` - Display help message
- `-v, --version` - Display version information
- `-o, --output [path]` - Write output to file (default: stdout)
- `-p, --progress` - Show progress bar
- `-l, --less_console` - Suppress console output

#### Algorithm Options
- `-b, --bruteforce` - Use brute-force mode (stops hom-space optimization)
- `-s, --sort` - Lexicographically sort input relations
- `-e, --exhaustive` - Iterate over all decompositions
- `-c, --basechange` - Save base change matrix
- `-f, --alpha` - Enable alpha-homs computation

#### Analysis Options
- `-t, --statistics` - Show indecomposable summand statistics
- `-r, --runtime` - Display runtime statistics and timers

#### Optimization Options
- `-j, --no_hom_opt` - Disable optimized hom-space calculation
- `-w, --no_col_sweep` - Disable column sweep optimization

#### Debug Options
- `-m, --compare_b` - Compare with brute-force at runtime
- `-a, --compare_e` - Compare exhaustive vs brute-force
- `-i, --compare_hom` - Compare optimized vs non-optimized hom calculation

### Output Format

AIDA outputs `.sccsum` files with the following structure:

```
sccsum
<number_of_indecomposables>

<type>
<summand_1_in_scc_format>

<type>
<summand_2_in_scc_format>
...
```

**Indecomposable types**:
- `free`: Module with one generator and no relations
- `cyclic`: One generator, not free
- `interval`: Non-cyclic interval module
- `non-interval`: All other modules

### Example Workflow

```bash
# With statistics and runtime info and a progress bar
./aida_with_stats tests/test_presentations/two_circles.scc -t -r -p
```

## Library Usage

AIDA can be used as a static library. Link against `libaida.a` and include headers from `include/` and `src/` (!) in future versions you should only have to link the include folder:

```cpp
#include <aida_interface.hpp>
// Your code here
```

**Important:** Before using AIDA as a library in another project, make sure you have run `sudo cmake --install build` as described in the [Installation](#installation) section above. This places `libaida.a` and the headers in system-wide locations so your project's build system can find them.

## Testing

Test presentations are available in `tests/test_presentations/`. Any `.scc` file can be used for testing:

```bash
./aida tests/test_presentations/davids_annulus_min.scc
```

## Citation


```bibtex
@software{Jendrysiak_AIDA_2024,
  author  = {Jendrysiak, Jan},
  title   = {{AIDA}: Automorphism Invariant Decomposition Algorithm},
  year    = {2024},
  doi     = {10.4230/artifacts.23282},
  url     = {https://github.com/JanJend/AIDA},
  version = {0.2},
  license = {LGPL-3.0-or-later}
}
```

If you use AIDA in your research, please also cite the accompanying paper:

```bibtex
@InProceedings{djk25,
  author =	{Dey, Tamal K. and Jendrysiak, Jan and Kerber, Michael},
  title =	{{Decomposing Multiparameter Persistence Modules}},
  booktitle =	{41st International Symposium on Computational Geometry (SoCG 2025)},
  pages =	{41:1--41:19},
  series =	{Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN =	{978-3-95977-370-6},
  ISSN =	{1868-8969},
  year =	{2025},
  volume =	{332},
  editor =	{Aichholzer, Oswin and Wang, Haitao},
  publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address =	{Dagstuhl, Germany},
  doi =		{10.4230/LIPIcs.SoCG.2025.41},
}
```

**Software DOI**: [10.4230/artifacts.23282](https://doi.org/10.4230/artifacts.23282)

## License

AIDA is licensed under the [LGPL-3.0-or-later](LICENSE) license.

## Authors

- Jan Jendrysiak (TU Graz) - [ORCID](https://orcid.org/0000-0002-3761-3463)


## Version

Current version: 0.2.1 (Released: 2024-10-01)

---

For questions, issues, or contributions, please open an issue on the repository.
