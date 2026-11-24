# Two-Scan Parsing

**Construct operator-precedence ASTs with two associative suffix scans.**

Two-Scan Parsing reconstructs the tree induced by left and right binding powers
on a normalized operator stream. A running-minimum first-hit scan finds reduction
boundaries; a witness scan supplies the information needed to recover parent
links. The resulting tree can be a deep chain while its links are discovered
through a parallel scan schedule.

This repository includes a compact Python reference, a complete example-language
frontend, correctness arguments, differential tests, and a CPU SIMD/OpenMP C++
implementation with reproducible benchmarks.

**[Read the full write-up: intuition, derivation, implementation, and correctness](docs/two-scan-parsing.md)**

[Python core](two_scan.py) · [Native implementation](native/README.md) ·
[Performance details](native/RESULTS.md) · [Witness-mask optimization](native/WITNESS_MASKS.md)

## Quick start

The Python implementation requires **Python 3.10+** and only the standard library.
From the repository root:

```bash
python3 parser.py "x = a + b * c;"
python3 parser.py "while (x) { if (y) x = x - 1; else x = 0; }"
```

On Windows, use `python` if that is the name of your Python executable.

```python
from parser import parse

tree = parse("x = a + b * c;")
assert tree == (
    "block",
    ("expr", ("=", "x", ("+", "a", ("*", "b", "c")))))
```

The structural algorithm also works independently of the example language:

```python
from two_scan import discover

# a + b * c - d
plan = discover(lbp=[30, 40, 30], rbp=[31, 41, 31])

assert plan.pop == [2, 2, 3]
assert plan.next_same == [1, -1, -1]
assert plan.parent == [2, 0, -1]
assert plan.root == 2
```

The root is subtraction, whose left child is addition, whose right child is
multiplication. The resulting expression is `(a + (b * c)) - d`.

## How it works

The specification is a stack parser with the strict reduction rule
`R_top > L_incoming`. Equal powers do not trigger that reduction. Left and right
powers may be chosen independently.

1. **First-hit scan.** An intervening operator can lower the binding threshold
   that controls a later reduction. Finite-state transitions track this running
   minimum. Their exclusive suffix compositions determine `pop[i]`: the input
   position that eventually removes operator `i` from the conceptual stack.
2. **Witness scan.** A second family of transitions finds `next_same[i]`: the
   next operator in source order removed at that same event, when one exists.
3. **Independent linking.** Pop events and same-event successors determine
   operator parents and operand attachments. Nodes use stable integer references,
   so writing a parent link does not require first materializing its child tree.

Associative function composition makes both scans parallelizable. The native
implementation evaluates them using local block summaries and Blelloch scans of
those summaries. On one worker, it evaluates the same suffix functions directly.

The native witness optimization uses a property of the second transition family:
a continuing transition keeps its input state. Two masks encode each transition
exactly, reducing storage and removing dependent lookups from witness composition.
The [derivation](native/WITNESS_MASKS.md) explains why this preserves the functions.

## Input and output

The core receives one normalized group: `n` operators with left/right binding
powers and `n+1` implicit operand slots. The example frontend handles tokenization,
grouping, contextual operator classification, and virtual-operand insertion.
Prefix and postfix syntax is normalized into this input model.

`discover()` returns `pop`, `next_same`, `parent`, `left`, `right`, and `root`.
Nonnegative children refer to operator indices. Child `-(k+1)` refers to original
operand slot `k`. The Python core uses root `None` for an empty group; the native
API uses `-1`.

The example language supports identifiers, integer tokens, parentheses, prefix
minus, infix `+ - * / ^ =`, expression statements, blocks, `if`/`else`, and `while`.
Its frontend and nested-tuple exporter execute sequentially. Integer leaves remain
token strings. Calls, strings, comments, lvalue checking, source spans, and richer
diagnostics would require frontend extensions.

## Complexity and parallel scope

For one normalized group, let `n` be the number of operators and `C` the number of
encoded right-binding-power states. With the encoding supplied, the reference
construction uses **`O(Cn)` work and space** and has **`O(log(n+1))` ideal span**
with independent table cells. For a fixed language table, `C` is constant, giving
linear work and space.

The Python reference executes the scan schedule sequentially. The C++ backend
runs scans and link-writing phases with OpenMP and SIMD; its blocked schedule
has logarithmic ideal span for fixed state count, block size, and top-tree cutoff.
This is a bound on structural discovery. Lexing, grouping, encoding when not
supplied, semantic processing, and tuple export retain their separate costs.

The current native backend supports at most **16 distinct right binding powers**
and uses 32-bit signed powers and indices. Its witness masks reduce scratch
storage; they do not remove `C` from the general work bound.

## CPU performance

Measured on **November 23, 2026**, using an **Intel Xeon Platinum 8370C**, nine
logical CPUs available, and **GCC 14.2**. Stack, previous scans, and optimized
scans are C++ implementations built together with the same flags.

**Timing scope:** prepared normalized numerical input to `parent`, `left`,
`right`, and `root`. Timings exclude input generation, state encoding, validation,
lexing, grouping, Python copies, and AST tuple export. The stack baseline links
the tree directly and does not compute extra diagnostic arrays during timing.

### Large streams

Median milliseconds from eleven samples after two warmups, using reused buffers
and the same fixed eight-state table. Streams contain 4,194,304 operators, or
4,194,303 for statements. Lower is faster.

| Workload | Stack, 1 worker | Previous scan, 8 workers | Optimized scan, 1 worker | Optimized scan, 8 workers | Stack / optimized, 8 workers |
|---|---:|---:|---:|---:|---:|
| Repeated addition | 16.502 | 24.186 | 41.760 | 17.535 | 0.94× |
| Repeated exponentiation | 19.528 | 27.263 | 38.451 | 15.797 | 1.24× |
| Alternating negation/exponentiation | 19.322 | 22.238 | 40.254 | 15.817 | 1.22× |
| Statement sequences | 21.434 | 32.305 | 40.387 | 20.111 | 1.07× |
| Mixed arithmetic | 49.800 | 20.130 | 40.292 | 16.059 | 3.10× |
| Independent numerical bindings | 55.855 | 24.651 | 41.163 | 21.036 | 2.66× |

Ratios above one favor the optimized scan. Mixed arithmetic randomly chooses
conventional `+ - * / ^` operators. Independent numerical bindings are a
synthetic test of the broader numerical contract, rather than representative
source code. The parsers do not receive the generator's workload label.

Because this virtualized host showed timing variability, both mixed families
were measured again with **31 samples** and unchanged native source/build flags:

| Workload | Operators | Stack, 1 worker | Previous scan, 8 workers | Optimized scan, 8 workers | Stack / optimized |
|---|---:|---:|---:|---:|---:|
| Mixed arithmetic | 1,048,576 | 12.688 ms | 7.188 ms | 6.272 ms | 2.02× |
| Mixed arithmetic | 4,194,304 | 49.616 ms | 24.446 ms | 19.200 ms | 2.58× |
| Numerical bindings | 1,048,576 | 13.421 ms | 7.312 ms | 5.899 ms | 2.28× |
| Numerical bindings | 4,194,304 | 58.865 ms | 24.490 ms | 17.819 ms | 3.30× |

The four-million mixed optimized samples ranged from 14.37 to 36.36 ms in the
confirmation run. All samples are retained. These are wall-time gains using more
CPU resources than the serial stack; small differences within noisy results
should not be treated as established wins.

### Small expressions and varied batches

For one fixed mixed-arithmetic expression parsed repeatedly, one worker,
reused buffers, median microseconds per expression:

| Operators | Stack | Previous scan | Optimized scan |
|---|---:|---:|---:|
| 16 | 0.071 µs | 2.044 µs | 0.152 µs |
| 64 | 0.275 µs | 2.639 µs | 0.640 µs |
| 256 | 1.071 µs | 4.543 µs | 1.898 µs |
| 1,024 | 4.298 µs | 13.942 µs | 8.158 µs |

Repeated fixed inputs favor branch prediction. A second experiment uses 1,024
different mixed-arithmetic expressions per batch. The table reports milliseconds
for the entire batch, with the same group-level parallelism available to both
algorithms and one reusable workspace per worker:

| Operators per expression | Stack, 1 worker | Optimized, 1 worker | Stack, 8 workers | Optimized, 8 workers |
|---|---:|---:|---:|---:|
| 16 | 0.215 | 0.225 | 0.517 | 0.627 |
| 64 | 0.775 | 0.721 | 0.720 | 0.963 |
| 256 | 3.209 | 3.179 | 1.372 | 1.336 |
| 1,024 | 11.423 | 10.005 | 2.697 | 3.201 |

Small fixed expressions favor the stack. Varied batches are competitive in some
cases, with no consistent overall advantage established. Multiworker batch
timings were particularly variable.

### Scratch storage

For the eight-state configuration, replacing full first-scan suffix tables with
witness masks reduces requested scan workspace payload by approximately 70%:

| Operators | Previous scan | Optimized scan | Reduction |
|---|---:|---:|---:|
| 1,048,576 | 41,975,808 bytes | 12,615,680 bytes | 69.95% |
| 4,194,304 | 167,903,232 bytes | 50,462,720 bytes | 69.95% |

These figures exclude input, graph arrays, allocation rounding, and allocator
bookkeeping. They are not total process memory measurements.

See [the full performance report](native/RESULTS.md) for allocation costs,
state-table experiments, and interpretation. All raw samples, machine metadata,
commands, and source digests are under [native/results/](native/results/).
Download and open the [self-contained interactive report](docs/benchmark-results.html)
to filter all 510 recorded configurations and inspect 5,802 timing samples.

**Application scope remains open:** the demonstrated speedups are on synthetic
large streams. A large codebase composed of many small expressions is a different
workload. Benefits on real generated formulas and complete source-to-AST pipelines
have not yet been established.

## Build and reproduce the native results

Tested on Linux x86-64 with GCC 14.2 and OpenMP. Windows users can use WSL2 with
GCC/OpenMP. Build on the target machine: `-march=native` selects its instructions.
The backend supports scalar, four-lane, AVX2, and AVX-512 paths as available.

```bash
cd native
python3 build.py
OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=TRUE GOMP_SPINCOUNT=0 ./peak_bench --verify
python3 -B verify_reference.py
```

Run the full main experiment suite, including build and verification:

```bash
python3 -B run_experiments.py
```

Use `--quick` for a smaller run. The runner writes results into `native/results/`,
replacing the corresponding files in your checkout; retain a clean checkout if
you want to preserve the recorded results alongside your own measurements.
It requests up to eight workers. The native executable accepts custom worker
counts through `--threads`; see [native/README.md](native/README.md).

Reproduce the separate 31-sample confirmation from the `native` directory:

```bash
OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=TRUE GOMP_SPINCOUNT=0 ./peak_bench --sizes 1048576,4194304 --threads 8 --repeats 31 --families mixed,random8 --csv results/large_repeat.csv
```

Run timed experiments without other benchmarks or compiler jobs competing for
the same CPUs. Source encoding and front-end integration costs need separate
measurement in an application.

## Verification

From the repository root:

```bash
python3 -B tests/verify.py
```

The Python verification covers 18,175 normalized groups, 20,000 whole programs,
front-end correspondence on 20,000 programs, arbitrary transition sequences,
supplied state encodings, invalid inputs, and deep chains through 16,384
repetitions. Whole-program checks accepted 7,951 programs and rejected 12,049.

Native verification checks 8,133 groups across 66,744 backend/thread/configuration
comparisons against an independent oracle. Pop events, witnesses, parents,
children, and roots are compared. The same suite passed UBSan and combined
AddressSanitizer/UBSan builds. Separate Python/native checks cover 10,175 groups
and 1,000 programs; concurrent-call checks exercise independent workspaces.

Recorded verification output is in [results/](results/) and
[native/results/](native/results/). These checks support the correctness argument;
they are not an independent proof review or a race-detector run. Sanitizer and
concurrency commands are documented in [native/README.md](native/README.md).

## Repository guide

| Path | Purpose |
|---|---|
| [two_scan.py](two_scan.py) | Standalone numerical core and scan schedule |
| [lexer.py](lexer.py) | Tokenization for the example language |
| [frontend.py](frontend.py) | Operators, grouping, contextual classification, normalization |
| [parser.py](parser.py) | Public parse API, type checks, and AST export |
| [tests/](tests/) | Differential verification and frozen historical baselines |
| [docs/two-scan-parsing.md](docs/two-scan-parsing.md) | Full blog-style explanation and correctness argument |
| [native/](native/) | CPU implementation, bridge, benchmarks, and native documentation |
| [native/WITNESS_MASKS.md](native/WITNESS_MASKS.md) | Exact witness-transition simplification |
| [native/RESULTS.md](native/RESULTS.md) | Detailed benchmark methodology and results |

## Status and related work

This is an experimental research implementation. Parallel expression-tree
construction and associative state-transition scans have substantial prior art.
The specific formulation is documented for examination; historical priority has
not been established. See the write-up's [related-work discussion](docs/two-scan-parsing.md#26-relationship-to-earlier-algorithms).

Feedback is welcome on the correctness argument, equivalent prior constructions,
and realistic expression workloads for evaluation.

Created by **Jeter Pontes**.
