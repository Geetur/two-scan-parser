"""Check independent simultaneous bridge calls, each with its own workspace."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import random
import simd_parser as native

fields = ("pop", "next_same", "parent", "left", "right", "root")
inputs = []
expected = []
for seed in range(4):
    rng = random.Random(1431 + seed)
    ops = [native.P.Op("x", native.P.INFIX, rng.randrange(20), rng.randrange(8))
           for _ in range(10000)]
    vals = ["x"] * (len(ops) + 1)
    inputs.append((vals, ops))
    plan = native.L.discover(vals, ops)
    expected.append(tuple(getattr(plan, field) for field in fields))


def check(i):
    plan = native.discover(*inputs[i % 4], threads=2)
    assert tuple(getattr(plan, field) for field in fields) == expected[i % 4]
    return True


with ThreadPoolExecutor(max_workers=4) as pool:
    assert all(pool.map(check, range(24)))
result = dict(passed=True, calls=24, callers=4, requested_workers_per_call=2,
              operators_per_call=10000)
print(json.dumps(result))
(Path(__file__).parent / "results" / "concurrency.json").write_text(
    json.dumps(result, indent=2) + "\n")
