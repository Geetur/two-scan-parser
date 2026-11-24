#!/usr/bin/env python3
"""
Composable chunk summaries for the peak parser. Plugs into peak_statements.py
through the existing `contractor` hook (parse(..., contractor=...)); the original
file is not modified.

    python peak_summaries.py            (demo, exactness tests, edge case, bench)

QUESTION
    Can independently parsed chunks of a prefix/infix/postfix token stream be
    merged by an associative operation that reproduces the exact AST?

SETUP
    A flat group is  v0 o0 v1 o1 ... o(n-1) vn  (EPS fills empty slots).
    Write it as items (o, u): each operator with the operand on its RIGHT, and
    put a sentinel START in front of v0 and END after vn:

        (START, v0) (o0, v1) ... (o(n-1), vn) (END)

    START is LT every operator and every operator is GT END, so the whole input
    contracts to (START, tree) (END). A "chunk" is any contiguous slice of items.

SUMMARY OF A CHUNK
    Fire every peak whose two neighbouring operators are both INSIDE the chunk.
    A peak is rel(prev, op) = LT and rel(op, next) = GT, as in peak_statements.py.
    What is left has no interior peak, so the rel sequence between consecutive
    operators is  GT^a LT^b  (a "V"):

        left chain   o1 > o2 > ... : waits for a LEFT neighbour (left context)
        valley       the last of the left chain / first of the right chain
        right chain  ... < om      : waits for a RIGHT neighbour (right context)

    This is the "operators already reduced along the left edge, operators still
    pending on the right" picture, except that nothing on the left edge is
    reduced early: a left-chain operator may be pre-empted by an operator
    to the left, so it stays pending, with its right operand attached.

COMBINE(A, B)
    Concatenate, then fire peaks created at the junction. Only two operators
    can newly become peaks: A's last operator and B's first. Each firing exposes
    the next candidate on that side, so the merge is two monotone lists being
    interleaved, like merging two sorted lists:

        A.top fires (right operand = junction value) if rel(A.prev, A.top) = LT
              and rel(A.top, B.front) = GT
        B.front fires (left operand = junction value) if rel(A.top, B.front) = LT
              and rel(B.front, B.next) = GT

    The result is again V-shaped.

WHY IT IS EXACT
    Firing a peak never changes whether another peak is a peak (two peaks are
    never adjacent, and a peak's neighbours stay put until it fires), so the
    set of firings reaching a normal form is unique (diamond property, then
    Newman's lemma). Every firing made inside a chunk is a firing of the full
    sequence. Hence any bracketing of combine() reaches the same normal form,
    which is the one the sequential contract() reaches. This is also tested
    (associativity, identity, equality with contract() and with recursive
    descent on valid and mutated programs).

COST, STATED CAREFULLY
    Every operator fires exactly once overall, so total firings = number of
    operators for any bracketing. A combine's firings form a sequence in which
    each firing uses the previous firing's result as one child and a fixed
    operand as the other. A maximal run of firings from the same side has
    index-determined children (a right fold for A-side, a left fold for B-side)
    and is one parallel step if nodes are mutable records, the same assumption
    fold() makes for plateaus. So the depth proxy here is

        depth = critical path over the combine tree of (1 + #same-side runs)

    NOT the number of firings. What is NOT done: a parallel merge for runs
    that interleave sides. The proxy charges one step per run; a run count that
    grows with n would need a parallel merge, which is not implemented here.
    No bound on runs is proved; the adversarial search below measures it.
    The Python lists copied inside combine() cost O(n log n) in total for a
    balanced tree and are not part of the proxy.
"""
import math
import random
import sys

from peak_statements import (
    INF, INFIX, LT, GT, EPS, Op, ParseError, RD, _opstr, _show, build_flat,
    build_groups, contract, flatten_tree, gen_program, make, mutate, need, outcome,
    parse, rel, shapes, tokenize,
)

START = Op("<start>", INFIX, -INF, -INF)    # LT every operator
END = Op("<end>", INFIX, -INF, -INF)        # every operator is GT END


class Summary:
    """o_1 u_1 o_2 u_2 ... o_m u_m: each operator with the operand on its right."""
    __slots__ = ("ops", "vals")

    def __init__(self, ops, vals):
        self.ops, self.vals = ops, vals


IDENTITY = Summary([], [])


def fire(op, a, b):
    """One operator firing. seq stays a constant-size BINARY node here; the n-ary
    flatten that fold() does is deferred to flatten_seq(), once per group. (Flattening
    at every firing re-copies the accumulated list: quadratic on x; x; x; ...)"""
    if op.name != "seq":
        return make(op, a, b)
    need(a, "s", "seq")
    need(b, "s", "seq")
    return ("seq", a, b)


def flatten_seq(v):
    """Binary seq nodes -> one n-ary ("seq", ...) in a single pass. Only a group's
    top node can be a seq: seq is looser than every operator that takes a statement."""
    if not (isinstance(v, tuple) and v[0] == "seq"):
        return v
    out, st = [], [v]
    while st:
        t = st.pop()
        if isinstance(t, tuple) and t[0] == "seq":
            st.append(t[2])
            st.append(t[1])
        else:
            out.append(t)
    return ("seq", *out)


def leaves(vals, ops):
    out = [Summary([START], [vals[0]])]
    out += [Summary([o], [v]) for o, v in zip(ops, vals[1:])]
    out.append(Summary([END], [None]))
    return out


def combine(A, B, info=None, sides=None):
    """Associative merge. info (a list) receives (firings, same-side runs);
    sides (a list) receives "A" or "B" for each firing, in order."""
    ops, vals = list(A.ops), list(A.vals)
    bo, bv = B.ops, B.vals
    b = fired = runs = 0
    side = None
    while True:
        if (len(ops) >= 2 and b < len(bo)
                and rel(ops[-2], ops[-1]) == LT and rel(ops[-1], bo[b]) == GT):
            node = fire(ops[-1], vals[-2], vals[-1])
            ops.pop()
            vals.pop()
            vals[-1] = node
            s = "A"
        elif (ops and b + 1 < len(bo)
                and rel(ops[-1], bo[b]) == LT and rel(bo[b], bo[b + 1]) == GT):
            vals[-1] = fire(bo[b], vals[-1], bv[b])
            b += 1
            s = "B"
        else:
            break
        fired += 1
        if sides is not None:
            sides.append(s)
        if s != side:
            runs += 1
            side = s
    ops.extend(bo[b:])
    vals.extend(bv[b:])
    if info is not None:
        info.append((fired, runs))
    return Summary(ops, vals)


def reduce_items(items, shape="balanced", rng=None, log=None):
    """Bracket the items with the given shape. Returns (Summary, depth proxy)."""
    if shape in ("left", "right"):
        depth = 0
        if shape == "left":
            acc = items[0]
            for it in items[1:]:
                info = []
                acc = combine(acc, it, info)
                depth += 1 + info[0][1]
                if log is not None:
                    log.append(info[0])
        else:
            acc = items[-1]
            for it in reversed(items[:-1]):
                info = []
                acc = combine(it, acc, info)
                depth += 1 + info[0][1]
                if log is not None:
                    log.append(info[0])
        return acc, depth

    def rec(lo, hi):
        if hi - lo == 1:
            return items[lo], 0
        mid = (lo + hi) // 2 if shape == "balanced" else rng.randint(lo + 1, hi - 1)
        a, da = rec(lo, mid)
        b, db = rec(mid, hi)
        info = []
        s = combine(a, b, info)
        if log is not None:
            log.append(info[0])
        return s, max(da, db) + 1 + info[0][1]

    return rec(0, len(items))


def make_contractor(shape="balanced", rng=None, log=None):
    """A contractor for peak_statements.parse(): same signature as contract()."""
    def contractor(vals, ops, plateaus=True, trace=None):
        local = []
        final, depth = reduce_items(leaves(vals, ops), shape, rng, local)
        if len(final.ops) != 2 or final.ops[0] is not START or final.ops[1] is not END:
            raise RuntimeError("summary did not reduce to a single tree")
        if log is not None:
            log.extend(local)
        value = flatten_seq(final.vals[0])
        if trace is not None:
            trace.append(_show(value))
        return value, depth, sum(1 + f for f, _ in local)
    return contractor


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def fmt(S):
    return "  ".join(f"[{_opstr(o)} {_show(v) if o is not END else ''}]".replace(" ]", "]")
                     for o, v in zip(S.ops, S.vals))


def flat_of(src):
    groups = build_groups(tokenize(src))
    return build_flat(groups[0].items)


def demo(src="x = - a ^ - b ^ c + d ;", cuts=(3, 6)):
    vals, ops = flat_of(src)
    items = leaves(vals, ops)
    parts = [items[:cuts[0]], items[cuts[0]:cuts[1]], items[cuts[1]:]]
    sums = [reduce_items(p, "left")[0] for p in parts]
    print("source:", src, "  (_ = empty operand slot)")
    for name, s in zip("ABC", sums):
        print(f"  chunk {name}: {fmt(s)}")
    ab = combine(sums[0], sums[1])
    bc = combine(sums[1], sums[2])
    print(f"  (A.B)   : {fmt(ab)}")
    print(f"  (B.C)   : {fmt(bc)}")
    print(f"  (A.B).C : {fmt(combine(ab, sums[2]))}")
    print(f"  A.(B.C) : {fmt(combine(sums[0], bc))}")


# --------------------------------------------------------------------------
# Exactness tests
# --------------------------------------------------------------------------
def flat_groups(toks):
    try:
        gs = build_groups(toks)
        return [build_flat(g.items) for g in gs if g.items]
    except ParseError:
        return None


def sig(S):
    return (tuple(id(o) for o in S.ops), [flatten_tree(v) for v in S.vals])


def try_sig(f):
    try:
        return sig(f())
    except ParseError:
        return "ERR"


def test_monoid(trials=6000, seed=1):
    rng = random.Random(seed)
    checked = both_err = 0
    for _ in range(trials):
        toks = gen_program(rng)
        if rng.random() < 0.3:
            toks = mutate(toks, rng)
        gs = flat_groups(toks)
        if not gs:
            continue
        vals, ops = rng.choice(gs)
        items = leaves(vals, ops)
        if len(items) < 3:
            continue
        i = rng.randint(1, len(items) - 2)
        j = rng.randint(i + 1, len(items) - 1)
        try:
            A, B, C = (reduce_items(p, rng.choice(["left", "right", "random"]), rng)[0]
                       for p in (items[:i], items[i:j], items[j:]))
        except ParseError:
            continue
        l = try_sig(lambda: combine(combine(A, B), C))
        r = try_sig(lambda: combine(A, combine(B, C)))
        if l != r:
            print("ASSOCIATIVITY FAILURE on:", " ".join(toks), (i, j))
            return False
        for X in (A, B, C):
            if sig(combine(IDENTITY, X)) != sig(X) or sig(combine(X, IDENTITY)) != sig(X):
                print("IDENTITY FAILURE on:", " ".join(toks))
                return False
        checked += 1
        both_err += l == "ERR"
    print(f"monoid: associativity and identity hold on {checked} random chunk triples "
          f"({both_err} where both bracketings raise)")
    return True


def test_equivalence(valid=3000, mutated=15000, seed=2):
    rng = random.Random(seed)
    oracle = lambda t: RD(t).program()
    kinds = ["balanced", "left", "right", "random"]
    for _ in range(valid):
        toks = gen_program(rng)
        want = outcome(oracle, toks)
        seq = outcome(lambda t: parse(t, plateaus=False), toks)
        if seq != want or want == "ERR":
            print("sequential contract mismatch:", " ".join(toks))
            return False
        for k in kinds:
            c = make_contractor(k, random.Random(rng.random()))
            got = outcome(lambda t: parse(t, contractor=c), toks)
            if got != want:
                print(f"MISMATCH ({k}) on valid program:", " ".join(toks))
                return False
    print(f"equivalence: {valid} valid programs x {len(kinds)} bracketings "
          f"equal sequential contract() and recursive descent")
    acc = rej = 0
    for _ in range(mutated):
        toks = mutate(gen_program(rng), rng)
        want = outcome(oracle, toks)
        for k in kinds:
            c = make_contractor(k, random.Random(rng.random()))
            got = outcome(lambda t: parse(t, contractor=c), toks)
            if got != want:
                print(f"MISMATCH ({k}) on mutated program:", " ".join(toks))
                print("  summaries:", "ERR" if got == "ERR" else "accepted",
                      " oracle:", "ERR" if want == "ERR" else "accepted")
                return False
        acc += want != "ERR"
        rej += want == "ERR"
    print(f"equivalence: {mutated} mutated programs x {len(kinds)} bracketings agree with "
          f"recursive descent on accept/reject and on the tree ({acc} accepted, {rej} rejected)")
    return True


# --------------------------------------------------------------------------
# The edge case, the benchmark shapes, and an adversarial search
# --------------------------------------------------------------------------
def bad_tokens(reps):
    return ["x", "="] + ["-", "a", "^"] * reps + ["b", ";"]


def rd_deep(toks):
    """Recursive-descent oracle on a deep input: big stack, high recursion limit."""
    import threading
    box = {}

    def work():
        try:
            box["tree"] = RD(toks).program()
        except BaseException as e:              # surfaced below, not swallowed
            box["err"] = e
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    threading.stack_size(0)
    sys.setrecursionlimit(old)
    if "err" in box:
        raise box["err"]
    return box["tree"]


def run(toks, shape="balanced"):
    log, st = [], {}
    tree = parse(toks, st, contractor=make_contractor(shape, log=log))
    return tree, st, log


def edge_case():
    print("\nEDGE CASE  x = -a ^ -a ^ ... ^ b;")
    print(f"{'repeats':>8}{'ops':>8}{'existing rounds':>17}{'summary depth':>15}"
          f"{'levels':>8}{'max runs':>10}{'max firings':>13}")
    for reps in (50, 400, 3000):
        toks = bad_tokens(reps)
        ours, st, log = run(toks)
        want = rd_deep(toks)
        assert flatten_tree(ours) == flatten_tree(want), reps
        if reps <= 400:
            s0 = {}
            base = parse(toks, s0)
            assert flatten_tree(base) == flatten_tree(want)
            old = str(s0["rounds"])
        else:
            old = "(skipped)"
        nops = 2 * reps + 2 + 1
        print(f"{reps:>8}{nops:>8}{old:>17}{st['rounds']:>15}"
              f"{math.ceil(math.log2(nops + 2)):>8}{max(r for _, r in log):>10}"
              f"{max(f for f, _ in log):>13}")
    print("existing rounds = the plateau contraction in peak_statements.py")
    print("summary depth   = critical path of (1 + same-side runs) over the combine tree")


def bench():
    small, big = 2000, 50000
    print(f"\n{'shape':<26}{'n':>8}{'groups':>9}{'plateau':>9}{'summary':>9}{'max runs':>10}")
    for (label, toks), (_, btoks) in zip(shapes(small), shapes(big)):
        s2 = {}
        t2 = parse(toks, s2)
        t3, st, log = run(toks)
        assert flatten_tree(t2) == flatten_tree(t3), label
        print(f"{label:<26}{small:>8}{s2['groups']:>9}{s2['rounds']:>9}{st['rounds']:>9}"
              f"{max((r for _, r in log), default=0):>10}")
        _, sb, lb = run(btoks)
        print(f"{'':<26}{big:>8}{sb['groups']:>9}{'-':>9}{sb['rounds']:>9}"
              f"{max((r for _, r in lb), default=0):>10}")


OPS = "+-*/^="


def toks_from(spec):
    out = ["x", "="]
    for neg, op in spec:
        out += (["-"] if neg else []) + ["a", op]
    return out + ["b", ";"]


def score(spec):
    _, st, _ = run(toks_from(spec))
    return st["rounds"]


def adversary(lengths=(32, 64, 128, 256), iters=1500, seed=3):
    rng = random.Random(seed)
    print("\nADVERSARIAL SEARCH (hill-climb on which operators and which negations)")
    print(f"{'ops':>6}{'random best':>13}{'climbed':>9}{'log2(ops)':>11}")
    for n in lengths:
        best = -1
        for _ in range(200):
            spec = [(rng.random() < 0.5, rng.choice(OPS)) for _ in range(n)]
            best = max(best, score(spec))
        spec = [(rng.random() < 0.5, rng.choice(OPS)) for _ in range(n)]
        cur = score(spec)
        for _ in range(iters):
            k = rng.randrange(n)
            cand = list(spec)
            cand[k] = (rng.random() < 0.5, rng.choice(OPS))
            s = score(cand)
            if s >= cur:
                spec, cur = cand, s
        print(f"{n:>6}{best:>13}{cur:>9}{math.log2(n):>11.1f}")
    print("Evidence only, not a bound: the climber maximises the depth proxy at fixed size.")


if __name__ == "__main__":
    demo()
    print()
    if test_monoid() and test_equivalence():
        edge_case()
        bench()
        adversary()