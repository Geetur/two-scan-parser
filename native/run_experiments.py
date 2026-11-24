"""Rebuild, verify, then benchmark both scan variants and the serial stack."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--quick", action="store_true")
cli.add_argument("--skip-build", action="store_true")
cli.add_argument("--skip-verify", action="store_true")
cli.add_argument("--spin-count", default="0")
args = cli.parse_args()
env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", OMP_PROC_BIND="TRUE",
           GOMP_SPINCOUNT=args.spin_count)
commands = []


def run(command, log):
    commands.append(command)
    print(" ".join(command), flush=True)
    with (RESULTS / log).open("w") as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


if not args.skip_build:
    run(["python3", "build.py"], "build.txt")
if not args.skip_verify:
    run(["./peak_bench", "--verify"], "cpp_verification.txt")
    run(["python3", "-B", "verify_reference.py"], "python_verification.txt")
    run(["python3", "-B", "verify_concurrency.py"], "concurrency.txt")

repeats = "5" if args.quick else "11"
teams = "1,8" if args.quick else "1,4,8"
large = "65536,1048576" if args.quick else "65536,1048576,4194304"
small = "16,256" if args.quick else "1,4,16,64,256,1024"
experiments = [
    ("small_latency", ["--sizes", small, "--threads", "1", "--inner", "512"]),
    ("small_used_states", ["--sizes", small, "--threads", "1", "--inner", "512", "--used-states"]),
    ("large", ["--sizes", large, "--threads", teams]),
    ("large_used_states", ["--sizes", "1048576", "--threads", teams, "--used-states"]),
    ("allocation", ["--sizes", "1048576", "--threads", "1,8", "--allocation"]),
]
rows = {}
for name, options in experiments:
    path = f"results/{name}.csv"
    run(["./peak_bench", *options, "--repeats", repeats, "--csv", path], name + ".txt")
    with (ROOT / path).open() as source:
        rows[name] = list(csv.DictReader(source))
    for row in rows[name]:
        actual, requested = int(row["actual_threads"]), int(row["threads"])
        assert 1 <= actual <= requested

run(["./peak_batch", "--groups", "256" if args.quick else "1024",
     "--repeats", repeats, "--csv", "results/batches.csv"], "batches.txt")
with (RESULTS / "batches.csv").open() as source:
    rows["batches"] = list(csv.DictReader(source))

sources = ["parser.cpp", "parser.hpp", "bench.cpp", "batch_bench.cpp", "build.py",
           "simd_parser.py", "verify_reference.py", "verify_concurrency.py", "run_experiments.py"]
cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")), "unknown")
result = dict(
    date=datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
    cpu=cpu, affinity=sorted(os.sched_getaffinity(0)),
    compiler=subprocess.check_output([env.get("CXX", "g++"), "--version"], text=True).splitlines()[0],
    environment={key: env[key] for key in ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "GOMP_SPINCOUNT")},
    quick=args.quick, commands=commands,
    scope="Prepared normalized numeric inputs to parent/left/right/root; excludes input generation/encoding, lexing, grouping, Python, and tuple export.",
    hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    rows=rows)
(RESULTS / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
print("Saved native samples and metadata to results/benchmark.json", flush=True)
