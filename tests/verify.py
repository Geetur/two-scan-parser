"""Compare the readable reference with the frozen original and stack/RD oracles."""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tests" / "baseline"))

import parser as New
import frontend as F
import lexer
import two_scan as Core
spec = importlib.util.spec_from_file_location(
    "original_linear_scan", ROOT / "tests" / "baseline" / "linear_scan.py"
)
Old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = Old
spec.loader.exec_module(Old)
P, S, R = Old.P, Old.S, Old.R


def links(plan):
    return tuple(getattr(plan, name) for name in
                 ("pop", "next_same", "parent", "left", "right", "root"))


def stack_oracle(ops):
    n = len(ops)
    pop, parent = [n] * n, [-1] * n
    left, right, stack, values = [None] * n, [None] * n, [], [-1]

    def fire(event):
        i = stack.pop()
        b, a = values.pop(), values.pop()
        pop[i], left[i], right[i] = event, a, b
        if a >= 0:
            parent[a] = i
        if b >= 0:
            parent[b] = i
        values.append(i)

    for j, op in enumerate(ops):
        while stack and ops[stack[-1]].rbp > op.lbp:
            fire(j)
        stack.append(j)
        values.append(-(j + 2))
    while stack:
        fire(n)
    next_same, last = [-1] * n, {}
    for i in reversed(range(n)):
        next_same[i] = last.get(pop[i], -1)
        last[pop[i]] = i
    return pop, next_same, parent, left, right, values[0] if n else None


def ast_result(function):
    try:
        return P.flatten_tree(function())
    except (P.ParseError, New.ParseError):
        return "ERR"



def new_inputs(vals, ops):
    """Copy the two modules' equivalent Box/Op classes without sharing code.

    reverse maps new opaque leaves to their original identities for exact AST
    comparisons, including repeated uses of the same unresolved group Box.
    """
    boxes, reverse = {}, {}

    def copy(value):
        if isinstance(value, P.Box):
            if value not in boxes:
                boxes[value] = box = F.Box(value.t)
                reverse[box] = value
                box.value = copy(value.value)
            return boxes[value]
        if isinstance(value, tuple):
            return tuple(copy(child) for child in value)
        return value

    values = [copy(value) for value in vals]
    operators = [F.Op(op.name, op.kind, op.lbp, op.rbp, copy(op.args)) for op in ops]
    return values, operators, reverse


def frontend_result(language, tokens):
    """Compare grouping and normalization before either reducer runs."""
    try:
        groups = language.build_groups(tokens)
    except language.ParseError as error:
        return "error", str(error)
    boxes = {group.box: i for i, group in enumerate(groups)}

    def describe(value):
        if isinstance(value, language.Box):
            return "box", boxes[value], value.t
        if isinstance(value, language.Op):
            return ("op", value.name, value.kind, value.lbp, value.rbp,
                    describe(value.args))
        if isinstance(value, (tuple, list)):
            return tuple(describe(child) for child in value)
        return value

    result = []
    for group in groups:
        try:
            flat = "ok", describe(language.build_flat(group.items))
        except language.ParseError as error:
            flat = "error", str(error)
        result.append((group.kind, group.box.t, group.cond,
                       describe(group.items), tuple(group.open_ifs),
                       group.pending, flat))
    return result


def run():
    start = time.perf_counter()
    rng = random.Random(73127)
    counts = {"scan_sequences": 0, "normalized_groups": 0,
              "random_groups": 0, "supplied_encodings": 0,
              "programs": 0, "accepted": 0, "rejected": 0,
              "frontend_programs": 0, "source_programs": 0}

    for _ in range(2000):
        C, n = rng.randint(1, 8), rng.randint(0, 30)
        tables = [tuple(rng.randrange(C + 5) for _ in range(C))
                  for _ in range(n)]
        expected = Old.suffix_after(tables, C, Old.Cost())
        assert Core.suffix_after(tables) == expected
        for i in range(n):
            direct = tuple(range(C))
            for table in tables[i + 1:]:
                direct = tuple(table[x] if x < C else x for x in direct)
            assert expected[i] == direct
        counts["scan_sequences"] += 1
    print("Scan equivalence: 2,000 arbitrary table sequences passed", flush=True)

    def checked(vals, ops):
        new_vals, new_ops, reverse = new_inputs(vals, ops)
        old, new = Old.discover(vals, ops), New.plan_group(new_vals, new_ops)
        assert links(new) == links(old) == stack_oracle(ops)
        if counts["normalized_groups"] % 17 == 0:
            states = sorted({op.rbp for op in ops} | {-100, 7, 999999999, P.INF})
            assert links(New.plan_group(new_vals, new_ops, states)) == links(
                Old.discover(vals, ops, states=states)) == links(new)
            counts["supplied_encodings"] += 1
        for order in (None, lambda n: reversed(range(n)),
                      lambda n: rng.sample(range(n), n)):
            def original_ast():
                root, arena = Old.construct(vals, ops, old, order)
                return S.flatten_seq(arena.materialize(root))
            expected = ast_result(original_ast)
            actual = ast_result(lambda: New.build_ast(new_vals, new_ops, new, order))
            if actual != "ERR":
                actual = [reverse[x] if isinstance(x, F.Box) else x for x in actual]
            assert actual == expected
        counts["normalized_groups"] += 1

    for items in R.enumerate_flats(6):
        try:
            vals, ops = P.build_flat(items)
        except P.ParseError:
            continue
        checked(vals, ops)
    assert counts["normalized_groups"] == 9175
    print("Exhaustive groups: 9,175 exact plans and AST/rejection results passed",
          flush=True)

    for _ in range(300):
        infix = [P.Op("i" + str(i), P.INFIX, rng.randint(0, 80), rng.randint(0, 80))
                 for i in range(rng.randint(1, 6))]
        prefix = [P.Op("neg", P.PREFIX, P.INF, rng.randint(0, 80))
                  for _ in range(rng.randint(0, 3))]
        postfix = [P.Op(";", P.POSTFIX, rng.randint(0, 80), P.INF)]
        for _ in range(30):
            vals, ops = R.random_flat(rng, (infix, prefix, postfix), rng.randint(1, 40))
            checked(vals, ops)
            counts["random_groups"] += 1
    print("Random mixed fixities and independent powers: 9,000 groups passed",
          flush=True)

    for i in range(20000):
        tokens = P.gen_program(rng)
        if i % 3:
            tokens = P.mutate(tokens, rng)
        assert frontend_result(F, tokens) == frontend_result(P, tokens), " ".join(tokens)
        counts["frontend_programs"] += 1
        expected = P.outcome(lambda t: P.RD(t).program(), tokens)
        old = ast_result(lambda: P.parse(tokens, contractor=Old.contractor_for()))
        stats = {}
        new = ast_result(lambda: New.parse_tokens(tokens, stats))
        assert new == old == expected, " ".join(tokens)
        if i % 29 == 0:
            source = ("\t" if i % 2 else "\n").join(tokens)
            assert lexer.tokenize(source) == P.tokenize(source) == tokens
            assert ast_result(lambda: New.parse(source)) == expected
            counts["source_programs"] += 1
        if new != "ERR":
            work = sum(len(P.build_flat(g.items)[1])
                       for g in P.build_groups(tokens) if g.items)
            assert stats["work"] == work
        counts["programs"] += 1
        counts["accepted"] += new != "ERR"
        counts["rejected"] += new == "ERR"
    print("Whole programs: 20,000 match original and recursive descent", flush=True)

    for repeats in (50, 400, 3000, 16384):
        tokens = S.bad_tokens(repeats)
        old = ast_result(lambda: P.parse(tokens, contractor=Old.contractor_for()))
        new = ast_result(lambda: New.parse_tokens(tokens))
        assert old == new
    print("Deep chains through 16,384 repetitions passed", flush=True)

    # Invalid metadata and the strict-tie/blocking cases retain their behavior.
    metadata_cases = [
        ([], [], None),
        (["x", "y"], [], None),
        (["x", "y"], [P.Op("+", P.INFIX, 30, 31)], []),
        (["x", "y"], [P.Op("+", P.INFIX, 30, 31)], [31, 31]),
        (["x", "y"], [P.Op("+", P.INFIX, 30, 31)], [32]),
        ([None, "x"], [P.Op("neg", P.PREFIX, 1, 45)], None),
        (["x", None], [P.Op(";", P.POSTFIX, 10, 5)], None),
        (["x", "y"], [P.Op("+", P.INFIX, P.INF, 31)], None),
        (["x", "y"], [P.Op("+", P.INFIX, 30, P.INF + 1)], None),
    ]
    for vals, ops, states in metadata_cases:
        outcomes = []
        for function in (lambda: Old.discover(vals, ops, states=states),
                         lambda: New.plan_group(vals, ops, states)):
            try:
                outcomes.append(("ok", links(function())))
            except (ValueError, KeyError) as error:
                outcomes.append(("error", type(error).__name__))
        assert outcomes[0] == outcomes[1]
    for source in ("", "if(x)y=1;else y=-2;", "α = 12;", "a_b23\t=003;",
                   "x!=3;", "// comment", "while (x) { if (y) x--; }"):
        assert lexer.tokenize(source) == P.tokenize(source)
        expected = P.outcome(lambda t: P.RD(t).program(), P.tokenize(source))
        assert ast_result(lambda: New.parse(source)) == expected
    assert Core.discover([0, 11, 5], [10, 1, 0]).pop == [3, 3, 3]
    assert Core.discover([0, 7], [7, 0]).pop == [2, 2]  # Strict > tie.
    counts.update(passed=True, metadata_cases=len(metadata_cases),
                  deep_chain_repetitions=[50, 400, 3000, 16384],
                  elapsed_seconds=round(time.perf_counter() - start, 3))
    print(json.dumps(counts), flush=True)
    return counts


if __name__ == "__main__":
    sys.setrecursionlimit(20000)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "verification.json").write_text(json.dumps(run(), indent=2) + "\n")
