# Faster CPU two-scan AST construction

This is a performance variant of the project's first-hit and witness construction.
It keeps the same two suffix scans and AST link rules. The main change is an exact
simplification of the witness transitions: a continuing transition always keeps
its input state. That permits a compact mask representation, removes dependent
lookups from the second scan, and avoids storing a full suffix table per operator.
The proof is in [WITNESS_MASKS.md](WITNESS_MASKS.md).

The native library contains both the previous table-based implementation and the
optimized implementation, plus an efficient serial stack baseline. See
[RESULTS.md](RESULTS.md) for measured performance and its limits. This is CPU SIMD
and OpenMP code, supporting at most 16 distinct right binding powers.

## Build and try it

Tested on Linux x86-64 with GCC 14.2 and OpenMP. Build on the machine that will run
the program because `-march=native` selects that machine's instructions. Windows
users can use WSL2 with GCC and OpenMP.

```bash
python3 build.py
OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=TRUE GOMP_SPINCOUNT=0 ./peak_bench --verify
python3 -B verify_reference.py
python3 -B verify_concurrency.py
python3 simd_parser.py --threads 8 "while (x) { if (y) x = x - 1; else x = 0; }"
```

The Python command exercises the original language front end, native structural
core, and AST export. It includes bridge copies and tuple construction. Native
benchmark timings exclude those stages.

Reproduce the full experiment suite, including a build and correctness checks:

```bash
python3 -B run_experiments.py
```

`--quick` reduces sizes and repetitions. `--skip-build --skip-verify` reuses an
already verified executable. All benchmarks run sequentially; do not run other
benchmarks or compiler jobs alongside them. The runner records every native
sample, commands, CPU, affinity, compiler version, environment, and source hashes
in `results/benchmark.json` and the individual CSV files.

## Changes within the construction

1. The first scan computes each operator's pop event as before. While its suffix
   table is already in a register, it also computes two witness masks.
2. Each operator stores one `uint32_t` containing the two masks, instead of `W`
   integers containing the full first-scan suffix table. Stop takes priority over
   witness when both bits are set.
3. The second scan applies those masks. Composition becomes a pointwise choice
   between a terminal answer and the next table's same lane.
4. Parent defaults are written when the pop event becomes available. Same-event
   overrides still wait for all witnesses. Child and operand link rules are the
   same as in the reference.
5. A single worker evaluates the same two suffix scans directly, right to left.
   It skips block summaries and OpenMP team setup. Multiple workers use the
   blocked parallel scans. Requested teams are capped by the number of blocks;
   one operator has the fixed plan implied by two empty exclusive suffixes.

The scan implementation does not switch to a stack parser. The stack appears only
as a separate benchmark and verification baseline. The optimization changes the
native representation and execution schedule, preserving the first-hit/witness
functions and all six plan fields.

## Native API

```cpp
#include "parser.hpp"

peak::Input input;
input.states = {20, 31, 1000000000};
input.L = {21, 30, 10};
input.R = {20, 31, 1000000000};
input.rank = {0, 1, 2};
input.validate();

peak::Graph graph;
graph.resize(input.n());
peak::Scan parser(input);          // optimized implementation by default
parser.parse(input, graph, 8);     // requested maximum worker count
parser.parse(input, graph, 8);     // reuse the workspace and output buffers
```

This example supplies the normalized operator stream `= + ;`, with four operand
slots. The result is `parent`, `left`, `right`, and `root`. Operators are numbered
in source order; nonnegative children refer to operators, and child `-(k+1)`
refers to original operand `k`. Parent `-1` identifies the root. Empty groups have
root `-1` and empty arrays. The diagnostic arrays `parser.pop` and `parser.next`
contain pop events and same-event witnesses.

`states` must be strictly increasing, and `states[rank[i]] == R[i]`. The native
reduction comparison is strict: `R[top] > L[incoming]`. Bindings and indices use
32-bit signed integers, and `n <= INT_MAX-2`. The caller supplies an already
tokenized, grouped, normalized stream and encoded operator descriptors. Semantic
actions, type checking, and expression evaluation are separate stages.

The API checks workspace dimensions in `parse`; call `input.validate()` before
using a new input. A workspace may process different inputs with the same sizes
and state count. Each simultaneous call needs a separate workspace and graph.

To select the retained table-based mode:

```cpp
peak::Scan previous(input, peak::Backend::Auto, 4096, 128,
                    false, true, false);  // final argument disables witness masks
```

The C ABI and `simd_parser.py` provide convenience integration. They allocate,
validate/copy input, and copy results; use the reusable C++ API for kernel speed.
The Python wrapper still uses the frozen four-module reference under `reference/`
for normalization, local type checks, and AST export.

## SIMD, storage, and bounds

`auto` selects a padded width `W` of 4, 8, or 16 from the supplied state count `C`.
Four lanes use AVX/SSE4.1, eight lanes AVX2, and sixteen lanes AVX-512F. Scalar
implementations remain available. On AVX-512 machines, masked scatter also writes
independent AST links. Forcing a wider backend can increase costs.

Let `S` be the block count padded to a power of two. Without precomputed
transitions, scratch payload is:

| Implementation | 32-bit scratch words |
|---|---:|
| Previous table-based scans | `n*W + 2*n + 4*S*W` |
| Witness masks | `3*n + 4*S*W` |

Both counts exclude input and the three `n`-entry graph arrays, allocator
bookkeeping, and rounding allocations to 64-byte boundaries. Precomputed
transitions add `n*W` words to either version. At width eight, large-input scratch
payload drops by approximately 70%.

The general work bound remains `O(C*n)`. For a fixed operator table, this is
linear in `n`. The blocked parallel algorithm has logarithmic ideal span for
fixed block size, state count, and top-tree cutoff. Actual time includes OpenMP
synchronization, memory traffic, and finite worker counts. No `C`-independent
bound or historical novelty claim follows from these optimizations.

## What the benchmarks measure

The primary contract is prepared numerical input to `parent/left/right/root`.
Input generation, encoding, validation, lexing, grouping, Python copies, and AST
tuple export are outside the timers. All variants receive the same input and
optimization flags. Input family names are used only by the generator; the
parsers do not receive them or select family-specific paths.

The packed serial stack links the tree directly, stores right power and index in
one 64-bit stack entry, and computes no extra diagnostic arrays in timed runs.
The scans compute pop and witness arrays because their construction needs them.
The baseline does not do that extra work just to match diagnostics.

- `small_latency.csv`: one fixed small expression parsed repeatedly, buffers
  reused, 512 calls per timed sample. This is favorable to branch prediction.
- `batches.csv`: 1,024 independently generated small expressions per batch, a
  reusable workspace per worker, and separate output buffers per group. Stack
  and both scan versions receive the same 1/4/8-worker group-level parallelism.
  This parallelism is available to all three algorithms.
- `large.csv`: six families at 65,536, 1,048,576, and 4,194,304 operators, with
  requested teams of 1/4/8. Statement streams use one fewer operator to end at a
  terminator. Main comparisons supply the language's fixed eight-state table.
- `small_used_states.csv` and `large_used_states.csv`: only distinct right powers
  present in the stream. Table encoding is excluded and is not a free operation
  if a deployment must compute it.
- `allocation.csv`: new output and scratch allocation, parse, and scratch release
  at about one million operators. Returned graph release is outside the timer.

Reuse mode does not imply all data fits in cache. Each configuration has two
warmups and eleven timed samples. Variant order is shuffled each repetition;
every timed output is checked outside the timer. CSV files retain individual
samples. The experiments include repeated `+`, repeated `^`, alternating
negation/exponentiation, statement sequences, random conventional `+ - * / ^`
arithmetic (`mixed`), and independent numerical bindings (`random8`). The last
family exercises the broader numeric contract rather than typical source code.

The release build disables automatic loop/SLP vectorization for consistent scalar
comparisons, while explicit SIMD intrinsics remain enabled. Both the stack and
scans use those same flags. No link-time optimization is used.

The runner sets `OMP_WAIT_POLICY=PASSIVE`, `OMP_PROC_BIND=TRUE`, and
`GOMP_SPINCOUNT=0`, and records them. These are benchmark settings, not universally
optimal runtime settings. GCC documents the last option's spin behavior in its
[libgomp manual](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html).

## Verification

The optimized and retained modes pass an independent C++ two-stack oracle on
8,133 input groups and 66,744 scalar/SIMD/thread configurations. All pop events,
witnesses, parent/left/right arrays, and roots are compared. Coverage includes
ties, independent powers, empty groups, 1–16 states, partial blocks, padded scan
trees, multiple team sizes, precomputed transitions, and adversarial chains.

Python checks compare 10,175 normalized groups across 10,684 native calls, plus
1,000 generated or mutated programs against recursive descent (384 accepted,
616 rejected). Concurrent-call checks use four callers, 24 calls, and two
requested native workers per call. These tests support equivalence; they do not
replace a proof or constitute a race-detector run.

The full C++ suite also passed UBSan and AddressSanitizer/UBSan builds:

```bash
python3 build.py --sanitize
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ./peak_bench_ubsan --verify
python3 build.py --address-sanitize
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ASAN_OPTIONS=detect_leaks=0 ./peak_bench_asan --verify
```

Leak detection is disabled because of the execution environment's tracing
restriction. Address and undefined-behavior checks remain enabled.

## Files

| File | Purpose |
|---|---|
| `parser.hpp`, `parser.cpp` | Native scans, optimized witness masks, stack baseline, C ABI |
| `bench.cpp` | Independent oracle, verification, single-group benchmarks |
| `batch_bench.cpp` | Varied small-expression throughput benchmarks |
| `build.py` | Release and sanitizer builds |
| `run_experiments.py` | Sequential reproducible experiment runner |
| `simd_parser.py` | Original language integration |
| `verify_reference.py`, `verify_concurrency.py` | Python and concurrent-call checks |
| `WITNESS_MASKS.md` | Exact transition simplification proof |
| `RESULTS.md` | Measured results and interpretation |
| `results/` | Native samples, metadata, and verification logs |
| `reference/` | Frozen Python reference and runtime dependencies |
