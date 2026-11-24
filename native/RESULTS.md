# CPU optimization results

Measured November 23, 2026, on an Intel Xeon Platinum 8370C at a reported
2.80 GHz, with nine logical CPUs available to the process. Compiler: GCC 14.2.0.
The comparison is between implementations built into the same executable on this
machine. Earlier project measurements on an AMD EPYC machine are not used for
before/after speedup claims.

The optimization produces a useful gain on large mixed operator streams while
preserving the two-scan construction. It does not establish a consistent advantage
for ordinary small expressions. For an eight-state table, it also removes about
70% of the scan workspace payload.

## What changed

The reference witness function has only three possible outcomes at each state:
stop, report the current operator as witness, or continue in that same state.
Its continuing transitions never change lanes. Two masks therefore encode an
operator's witness transition exactly, and witness composition becomes a
pointwise terminal selection. See [WITNESS_MASKS.md](WITNESS_MASKS.md) for the
derivation.

The optimized implementation computes those masks during first-scan suffix
expansion and stores four bytes per operator rather than the full eight-lane
table. It also writes parent defaults earlier and evaluates single-worker scans
directly, avoiding block-summary and team-setup work. One-operator groups use
their fixed plan. The measured gains are for this bundle of changes; they are
not an isolated measurement of the mask representation alone.

Both scans, pop events, witnesses, and AST linking semantics are retained. The
scan implementation never falls back to an operator stack. General work remains
`O(C*n)`; this is an improvement in instructions, memory traffic, and scheduling.

## Large streams: main run

All times below are median milliseconds across eleven samples, following two
warmups per configuration. Buffers are reused. All inputs use the fixed eight-state
language table. Old and new scan columns request eight workers; the stack uses
one. Smaller is faster. Ratios above one favor the optimized scan.

At 1,048,576 operators (1,048,575 for statements):

| Family | Serial stack | Previous scan, 8 workers | Masks, 8 workers | Stack / masks |
|---|---:|---:|---:|---:|
| Repeated addition | 4.120 | 7.960 | 7.644 | 0.54× |
| Repeated exponentiation | 4.703 | 6.950 | 5.868 | 0.80× |
| Alternating negation/exponentiation | 4.837 | 7.592 | 6.513 | 0.74× |
| Statement sequence | 5.116 | 6.479 | 6.026 | 0.85× |
| Mixed conventional arithmetic | 12.117 | 7.311 | 5.358 | 2.26× |
| Independent numerical bindings | 13.177 | 7.294 | 5.665 | 2.33× |

At 4,194,304 operators (4,194,303 for statements):

| Family | Serial stack | Previous scan, 8 workers | Masks, 8 workers | Previous / masks | Stack / masks |
|---|---:|---:|---:|---:|---:|
| Repeated addition | 16.502 | 24.186 | 17.535 | 1.38× | 0.94× |
| Repeated exponentiation | 19.528 | 27.263 | 15.797 | 1.73× | 1.24× |
| Alternating negation/exponentiation | 19.322 | 22.238 | 15.817 | 1.41× | 1.22× |
| Statement sequence | 21.434 | 32.305 | 20.111 | 1.61× | 1.07× |
| Mixed conventional arithmetic | 49.800 | 20.130 | 16.059 | 1.25× | 3.10× |
| Independent numerical bindings | 55.855 | 24.651 | 21.036 | 1.17× | 2.66× |

The independent numerical family is synthetic and is not representative source
code. The mixed family randomly chooses conventional `+ - * / ^` operators.
Neither algorithm receives the generator's family name.

These are wall-time improvements using more CPU resources than the serial stack.
They are not work-efficiency or energy-efficiency claims. Full rows also record
process CPU time. There is substantial host variability: the main four-million
mixed masks samples range from 13.19 to 29.98 ms. Small ratio differences, such as
the statement result, should not be treated as established wins.

## Additional large-stream repetition

Because of the variability, the two mixed families were measured again with
31 samples per configuration, without changing native source or build flags.
The original run is retained rather than replaced by selected samples.

| Family | Operators | Serial stack | Previous scan, 8 workers | Masks, 8 workers | Previous / masks | Stack / masks |
|---|---:|---:|---:|---:|---:|---:|
| Mixed arithmetic | 1,048,576 | 12.688 | 7.188 | 6.272 | 1.15× | 2.02× |
| Mixed arithmetic | 4,194,304 | 49.616 | 24.446 | 19.200 | 1.27× | 2.58× |
| Numerical bindings | 1,048,576 | 13.421 | 7.312 | 5.899 | 1.24× | 2.28× |
| Numerical bindings | 4,194,304 | 58.865 | 24.490 | 17.819 | 1.37× | 3.30× |

The repeat supports a real opportunity on these large mixed streams, with the
exact speedup dependent on the run. It is still one virtualized host, not a
multi-machine performance study. The four-million mixed mask samples range from
14.37 to 36.36 ms in this repeat. All 31 samples are retained.

Reproduce this additional comparison after building:

```bash
OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=TRUE GOMP_SPINCOUNT=0 ./peak_bench --sizes 1048576,4194304 --threads 8 --repeats 31 --families mixed,random8 --csv results/large_repeat.csv
```

The command, source hashes, runtime settings, machine details, and rows are in
`results/large_repeat_metadata.json`.

## Small expressions: latency and varied batches

For a repeatedly parsed fixed mixed-arithmetic expression, the optimized scan is
much faster than the previous scan, but the stack remains faster. Median time per
expression in microseconds, one worker, reused buffers, 512 calls per sample:

| Operators | Serial stack | Previous scan | Masks | Previous / masks |
|---|---:|---:|---:|---:|
| 16 | 0.071 | 2.044 | 0.152 | 13.44× |
| 64 | 0.275 | 2.639 | 0.640 | 4.12× |
| 256 | 1.071 | 4.543 | 1.898 | 2.39× |
| 1,024 | 4.298 | 13.942 | 8.158 | 1.71× |

Repeated fixed inputs are favorable to branch prediction. A second experiment
uses 1,024 different expressions per batch, with a workspace pool reused across
groups. This better exercises changing operator sequences. On one worker, median
milliseconds for the entire mixed-arithmetic batch were:

| Operators per expression | Serial stack | Previous scan | Masks | Stack / masks |
|---|---:|---:|---:|---:|
| 16 | 0.215 | 2.571 | 0.225 | 0.95× |
| 64 | 0.775 | 2.959 | 0.721 | 1.08× |
| 256 | 3.209 | 6.582 | 3.179 | 1.01× |
| 1,024 | 11.423 | 16.778 | 10.005 | 1.14× |

These are competitive results, not evidence of a dependable small-expression
advantage. The samples overlap and the multiworker batches are particularly
variable. At 1,024 operators per expression and eight workers, the mixed batch
takes 2.697 ms for the stack and 3.201 ms for masks. The table-based scan takes
4.584 ms. Both algorithms can parallelize independent expressions; this is not a
capability unique to two-scan parsing. Results for both numeric families and
1/4/8-worker batches are all in `results/batches.csv`.

## Scratch memory

The mask representation removes `W` first-scan suffix entries per operator and
replaces them with one 32-bit mask pair. This reduces scratch payload at width
eight from `10*n + 32*S` words to `3*n + 32*S` words, where `S` is the padded
number of blocks.

| Operators | Previous scratch | Mask scratch | Reduction |
|---|---:|---:|---:|
| 1,048,576 | 41,975,808 bytes | 12,615,680 bytes | 69.95% |
| 4,194,304 | 167,903,232 bytes | 50,462,720 bytes | 69.95% |

These are requested scratch payload bytes, excluding input, graph arrays,
allocation rounding, and allocator bookkeeping. They are not process RSS or a
70% reduction in total application memory.

## Allocation and state-table experiments

Fresh allocation is much more expensive than reuse on this host. At about one
million operators, allocating the output/workspace, parsing, and releasing
scratch gave these median milliseconds:

| Family | Serial stack | Previous scan, 8 workers | Masks, 8 workers |
|---|---:|---:|---:|
| Addition | 34.727 | 59.183 | 32.030 |
| Exponentiation | 66.765 | 56.998 | 24.471 |
| Alternating unary/exponentiation | 58.866 | 58.893 | 23.818 |
| Statements | 35.768 | 55.902 | 22.617 |
| Mixed arithmetic | 37.553 | 67.677 | 30.856 |
| Numerical bindings | 45.421 | 68.717 | 28.710 |

Graph release is outside these timers. These samples are also noisy; the table
does not isolate allocation from first-touch/page-management costs. A caller that
needs predictable kernel latency should reuse its workspace. Do not apply reuse
timings to callers that allocate and copy on every parse.

The `used_states` experiments narrow the state table to the powers present in
each stream. The outcome is not uniformly faster, and that encoding cost is
excluded. The main claims therefore use the same fixed eight-state table for
every family. At 65,536 operators, one-worker direct scans are faster than the
eight-worker scans on every tested family; requesting many workers is not always
helpful. Team selection remains an application-level tuning choice, not a
universal threshold established by this experiment.

## Verification and reproducibility

- C++: 8,133 input groups, 66,744 backend/thread/configuration comparisons; every
  pop event, witness, graph array, and root matches an independent two-stack
  oracle. Both retained and optimized modes are checked.
- The same C++ suite passes UBSan and combined AddressSanitizer/UBSan. Leak
  detection is disabled for the environment's tracing restriction.
- Python: 10,175 normalized groups, 10,684 native calls, and 1,000 language
  programs compared with recursive descent; 384 accepted and 616 rejected.
- Concurrent calls: 24 calls from four Python callers, each requesting two native
  workers. Each caller owns separate workspaces. This is not a race-detector run.
- All timed outputs are checked outside the timer against an independent oracle.

The native build uses C++17, `-O3 -march=native -fopenmp`, and disables automatic
loop/SLP vectorization while retaining explicit SIMD intrinsics. The same flags
apply to the packed-stack baseline. There is no link-time optimization. All
source digests in the main metadata match the delivered source.

Runtime settings were `OMP_WAIT_POLICY=PASSIVE`, `OMP_PROC_BIND=TRUE`, and
`GOMP_SPINCOUNT=0`. Two preliminary bounded-spin settings were tried; those tuning
samples are retained separately and are not mixed into the final result tables.
No other benchmarks or compiler jobs were run alongside a timed experiment.

`results/benchmark.json` contains all six main experiments, eleven samples per
configuration, commands, environment, CPU affinity, compiler, and source hashes.
Each experiment also has its native CSV and log. `large_repeat_metadata.json`
records the separate 31-sample confirmation. Verification and sanitizer logs are
included. Preliminary implementation timings are omitted from this package so
they cannot be confused with measurements of the delivered source.

## Practical conclusion

This is a faster and smaller implementation of the existing mechanism, with a
measured native advantage on large mixed streams. It is not yet a general
replacement for a serial expression parser on ordinary source workloads. The
native speedups exclude lexing, grouping, encoding, Python integration, and AST
export, so they are not whole-compiler speedups. A deployment should first measure
its actual group sizes and the share of runtime spent in tree construction.
