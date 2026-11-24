#!/usr/bin/env python3
"""
Ranked merge reference: keyed2 rank plan + independently linked AST nodes, checked
against the sequential combine() in peak_summaries.py.

    python ranked_merge_reference.py            # standard verification suite
    python ranked_merge_reference.py --full     # larger sweeps

WHAT THIS FILE IS
    A correctness prototype for milestones 1-3 of NEXT_STEPS.md. It runs
    sequentially. Binary searches run in Python loops: this SIMULATES independent
    per-operator searches, it is not parallel execution. No work or depth bound is
    claimed for the whole merge tree (see "COST ACCOUNTING" below).

THE THREE STAGES OF ONE MERGE
    plan   = discover_merge(A, B)        decide every firing and its rank, from numbers only
    nodes  = write_rank(plan, r) for r   link each preallocated node independently
    result = assemble_summary(...)       surviving boundaries + new junction operand

CORRESPONDENCE TO THE ACTUAL PARSER (proved in CORRESPONDENCE.md, tested below)
    L1  rel(a, b) == GT  <=>  a.rbp > b.lbp, for every operator the parser builds.
        Prefix lbp and postfix rbp are INF, so the fixity overrides are already
        numeric. Nothing in the rank plan calls rel() on a (row, column) pair.
    L2  Every Summary is V-shaped: rel along ops is GT^a LT^b. The valley is
        ops[a]. It is the BARRIER row on the A side and the last COLUMN on the B
        side. Rows = A.ops[a+1:] read from the right; columns = B.ops[:a_B + 1].
    L3  combine()'s loop is exactly the grid walk below, with these stop rules:
          A row i fires if rel(row_i, col_j) == GT
          else B column j fires if j < a_B  (the valley column NEVER B-fires)
          the barrier row never A-fires: GT at the barrier stops the merge
        Note the difference from monotonicity_experiment.py: there a postfix valley
        column may fire (lastS); the real combine() never fires it, because a peak
        needs both neighbours inside the chunk.
    L4  PROOF.md (prefix minima preserve the walk), applied with keyed2.

THE RANK PLAN
    rows    R_i = rbp of the i-th row from the top, then the barrier's rbp
    columns L_j = lbp of B.ops[j], j = 0..a_B (K = a_B + 1 columns)
    alpha_i = min(R_0..R_i)           (barrier row: min(R_bar, alpha_last))   keyed2
    beta_j  = min(L_0..L_j)
    G(i, j) = alpha_i > beta_j        false* true* in j, true* false* in i... (monotone)
    t_i     = first j with G(i, j), else K
    A row i fires  <=>  t_i < K  <=>  alpha_i > beta_{K-1}       (binary search on i)
    f       = number of A firings
    J       = a_B if f < p else min(a_B, t_bar)                  (t_bar: barrier row)
    pos(A_i) = i + t_i        pos(B_j) = j + #{i < f : t_i <= j}

COST ACCOUNTING (honest)
    The Cost object counts reads and comparisons per merge: valley scans (rel calls),
    prefix-minimum reads, binary-search comparisons, rank-position searches, node
    writes, entries in the result summaries (output_entries), and every entry
    written into a new list by assemble_summary, intermediates included
    (list_entries_written). Allocations in discovery (alpha, beta, t, positions,
    order) are not copying of the summaries and are not in list_entries_written;
    they are bounded by the reads in prefix_min_reads plus the firing count. The
    valley scans and prefix-minimum construction are O(boundary) per merge and the
    survivor copies are O(boundary):
    NO linear-total-work claim follows from this file. A parallel-depth claim would
    need a parallel scan, a persistent boundary structure, and a global analysis.
"""
import argparse
import contextlib
import itertools
import random
import sys
import time

import peak_statements as P
import peak_summaries as S
from peak_statements import (
    EPS, GT, INF, INFIX, LT, POSTFIX, PREFIX, STMT_HEADS, Box, Op, ParseError,
    flatten_tree, rel, type_of,
)
from peak_summaries import END, START


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
class Cost:
    FIELDS = ("merges", "firings", "valley_rel", "prefix_min_reads", "row_search_cmps",
              "threshold_cmps", "rank_cmps", "node_writes", "output_entries",
              "list_entries_written")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def as_dict(self):
        return {f: getattr(self, f) for f in self.FIELDS}


# ---------------------------------------------------------------------------
# Arena (nodes with stable ids) and the ranked summary
# ---------------------------------------------------------------------------
class Ref:
    """A child pointer to an arena slot. Carries no content."""
    __slots__ = ("i",)

    def __init__(self, i):
        self.i = i


def head_of(op):
    """Head symbol of the node make()/fire() would build for op: a function of op only."""
    n = op.name
    if op.kind == INFIX:
        return n
    if n == "neg":
        return "neg"
    if n == ";":
        return "expr"
    if n in ("if", "ifelse"):
        return "if"
    return "while"


def out_type(op):
    return "s" if head_of(op) in STMT_HEADS else "e"


def needs(op):
    """(required type of left child, required type of right child); None = ignored."""
    n = op.name
    if n == "seq":
        return ("s", "s")
    if op.kind == INFIX:
        return ("e", "e")
    if n == "neg":
        return (None, "e")
    if n == ";":
        return ("e", None)
    return (None, "s")


def shape(op, a, b):
    """Same tuple shape as make()/fire(), without validation."""
    n = op.name
    if op.kind == INFIX:
        return (n, a, b)
    if n == "neg":
        return ("neg", b)
    if n == ";":
        return ("expr", a)
    if n == "if":
        return ("if", op.args[0], b)
    if n == "ifelse":
        return ("if", op.args[0], op.args[1], b)
    return ("while", op.args[0], b)


def used_children(op, a, b):
    n = op.name
    if op.kind == INFIX:
        return (a, b)
    if n == "neg" or n not in (";",):
        return (b,)
    return (a,)


class Arena:
    __slots__ = ("op", "a", "b", "ty")

    def __init__(self):
        self.op, self.a, self.b, self.ty = [], [], [], []

    def allocate(self, m):
        base = len(self.op)
        self.op.extend([None] * m)
        self.a.extend([None] * m)
        self.b.extend([None] * m)
        self.ty.extend([None] * m)
        return base

    def type_of_value(self, x):
        return self.ty[x.i] if isinstance(x, Ref) else type_of(x)

    def node(self, i):
        return shape(self.op[i], self.a[i], self.b[i])

    def materialize(self, root):
        """Nested tuples, built iteratively (deep chains). Not part of the merge."""
        memo = {}
        stack = [root]
        while stack:
            x = stack[-1]
            if not isinstance(x, Ref) or x.i in memo:
                stack.pop()
                continue
            op, a, b = self.op[x.i], self.a[x.i], self.b[x.i]
            todo = [c for c in used_children(op, a, b)
                    if isinstance(c, Ref) and c.i not in memo]
            if todo:
                stack.extend(todo)
                continue
            sub = lambda c: memo[c.i] if isinstance(c, Ref) else c
            memo[x.i] = shape(op, sub(a), sub(b))
            stack.pop()
        return memo[root.i] if isinstance(root, Ref) else root

    def flat(self, root):
        """Signature identical to flatten_tree(materialize(root)), without materializing."""
        out, st = [], [root]
        while st:
            t = st.pop()
            if isinstance(t, Ref):
                t = self.node(t.i)
            if isinstance(t, tuple):
                out.append(t[0])
                out.append(len(t) - 1)
                st.extend(reversed(t[1:]))
            else:
                out.append(t)
        return out


class RSummary:
    """Same layout as Summary: o_1 u_1 ... o_m u_m; each u is a leaf value or a Ref."""
    __slots__ = ("ops", "vals")

    def __init__(self, ops, vals):
        self.ops, self.vals = ops, vals


def ranked_leaves(vals, ops):
    out = [RSummary([START], [vals[0]])]
    out += [RSummary([o], [v]) for o, v in zip(ops, vals[1:])]
    out.append(RSummary([END], [None]))
    return out


# ---------------------------------------------------------------------------
# Stage 1: rank discovery (numbers only)
# ---------------------------------------------------------------------------
class MergePlan:
    __slots__ = ("nA", "nB", "vA", "vB", "p", "K", "f", "J", "t", "pos_a", "pos_b",
                 "order", "stop", "m")

    def __init__(self):
        self.nA = self.nB = self.vA = self.vB = self.p = self.K = 0
        self.f = self.J = self.m = 0
        self.t, self.pos_a, self.pos_b, self.order = [], [], [], []
        self.stop = "empty"


def valley(ops, cost):
    """Index of the valley = number of leading GT relations. (L2: ops is GT^a LT^b.)
    A scan for now; maintaining it as summary metadata is future work."""
    i, n = 0, len(ops)
    while i + 1 < n:
        cost.valley_rel += 1
        if rel(ops[i], ops[i + 1]) != GT:
            break
        i += 1
    return i


def first_true(a, beta, cost):
    """min{j : a > beta[j]}, or len(beta). beta is nonincreasing, so the predicate is
    false* true*. One independent binary search."""
    lo, hi = 0, len(beta)
    while lo < hi:
        mid = (lo + hi) // 2
        cost.threshold_cmps += 1
        if a > beta[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def count_le(t, j, cost):
    """#{i : t[i] <= j} for nondecreasing t."""
    lo, hi = 0, len(t)
    while lo < hi:
        mid = (lo + hi) // 2
        cost.rank_cmps += 1
        if t[mid] <= j:
            lo = mid + 1
        else:
            hi = mid
    return lo


def discover_merge(A, B, cost):
    plan = MergePlan()
    plan.nA, plan.nB = nA, nB = len(A.ops), len(B.ops)
    if nA == 0 or nB == 0:
        return plan
    plan.vA = vA = valley(A.ops, cost)
    plan.vB = vB = valley(B.ops, cost)
    plan.p = p = nA - 1 - vA          # firable rows; row i is A.ops[nA - 1 - i]
    plan.K = K = vB + 1               # columns B.ops[0..vB]; only j < vB can B-fire

    # keyed2 keys. (A parallel version would use scans; this is a sequential loop.)
    alpha, m_ = [], INF + 1
    for i in range(p):
        m_ = min(m_, A.ops[nA - 1 - i].rbp)
        alpha.append(m_)
    a_bar = min(A.ops[vA].rbp, m_)    # barrier row keeps the running minimum (keyed2)
    beta, m_ = [], INF + 1
    for j in range(K):
        m_ = min(m_, B.ops[j].lbp)
        beta.append(m_)
    cost.prefix_min_reads += p + 1 + K

    # f = first row that no column accepts: alpha_i <= beta_{K-1}. alpha nonincreasing.
    lo, hi = 0, p
    while lo < hi:
        mid = (lo + hi) // 2
        cost.row_search_cmps += 1
        if alpha[mid] <= beta[-1]:
            hi = mid
        else:
            lo = mid + 1
    plan.f = f = lo
    plan.t = t = [first_true(alpha[i], beta, cost) for i in range(f)]   # independent searches

    if f < p:
        plan.J = J = vB
        plan.stop = "rows_blocked"
    else:
        t_bar = first_true(a_bar, beta, cost)
        plan.J = J = min(vB, t_bar)
        plan.stop = "barrier_gt" if t_bar <= vB else "valley"

    plan.m = m = f + J
    plan.pos_a = [i + t[i] for i in range(f)]
    plan.pos_b = [j + count_le(t, j, cost) for j in range(J)]
    order = [None] * m
    for i, pos in enumerate(plan.pos_a):
        assert order[pos] is None, "rank collision (A)"
        order[pos] = ("A", i)
    for j, pos in enumerate(plan.pos_b):
        assert order[pos] is None, "rank collision (B)"
        order[pos] = ("B", j)
    assert None not in order, "rank gap"
    plan.order = order
    return plan


# ---------------------------------------------------------------------------
# Stage 2: independent node construction
# ---------------------------------------------------------------------------
def op_at(plan, A, B, r):
    side, idx = plan.order[r]
    return A.ops[plan.nA - 1 - idx] if side == "A" else B.ops[idx]


def write_rank(plan, A, B, arena, base, r):
    """Write arena slot base+r. Reads: the plan, the summaries, and (for rank 0) the
    junction value. Never reads another rank's slot, so ranks may be written in any
    order. Returns an error message or None (validation is per rank, too)."""
    side, idx = plan.order[r]
    if side == "A":
        op = A.ops[plan.nA - 1 - idx]
        fixed = A.vals[plan.nA - 2 - idx]
    else:
        op = B.ops[idx]
        fixed = B.vals[idx]
    if r == 0:
        prev, prev_ty = A.vals[plan.nA - 1], arena.type_of_value(A.vals[plan.nA - 1])
    else:
        prev, prev_ty = Ref(base + r - 1), out_type(op_at(plan, A, B, r - 1))
    fixed_ty = arena.type_of_value(fixed)
    a, b = (fixed, prev) if side == "A" else (prev, fixed)
    a_ty, b_ty = (fixed_ty, prev_ty) if side == "A" else (prev_ty, fixed_ty)
    na, nb = needs(op)
    err = None
    if na is not None and a_ty != na:
        err = f"{op.name}: left operand must be {'a statement' if na == 's' else 'an expression'}"
    elif nb is not None and b_ty != nb:
        err = f"{op.name}: right operand must be {'a statement' if nb == 's' else 'an expression'}"
    n = base + r
    arena.op[n], arena.a[n], arena.b[n], arena.ty[n] = op, a, b, out_type(op)
    return err


# ---------------------------------------------------------------------------
# Stage 3: surviving summary
# ---------------------------------------------------------------------------
def assemble_summary(plan, A, B, base, cost):
    """Build the surviving summary. cost.output_entries counts the entries of the
    result (ops + vals). cost.list_entries_written counts every entry this function
    writes into a new list, intermediates included, exactly as the code below does."""
    if plan.nA == 0 or plan.nB == 0:
        src = B if plan.nA == 0 else A
        n = len(src.ops)
        cost.output_entries += 2 * n
        cost.list_entries_written += 2 * n                  # list(ops) + list(vals)
        return RSummary(list(src.ops), list(src.vals))
    keep = plan.nA - plan.f
    tail = plan.nB - plan.J
    out = keep + tail
    junction = Ref(base + plan.m - 1) if plan.m else A.vals[plan.nA - 1]
    ops = A.ops[:keep] + B.ops[plan.J:]
    vals = A.vals[:keep - 1] + [junction] + B.vals[plan.J:]
    cost.output_entries += 2 * out
    # ops : slice(keep) + slice(tail) + concatenation(out)
    # vals: slice(keep-1) + [junction](1) + slice(tail) + first concat(keep) + second concat(out)
    cost.list_entries_written += (keep + tail + out) + ((keep - 1) + 1 + tail + keep + out)
    return RSummary(ops, vals)


def ranked_combine(A, B, arena, cost=None, plan_out=None, write_order=None):
    """Merge two ranked summaries. plan_out (a list) receives the plan before
    validation, so plans can be compared even when the merge is rejected."""
    cost = cost if cost is not None else Cost()
    cost.merges += 1
    plan = discover_merge(A, B, cost)
    if plan_out is not None:
        plan_out.append(plan)
    base = arena.allocate(plan.m)
    ranks = range(plan.m) if write_order is None else write_order(plan.m)
    errors = []
    for r in ranks:
        e = write_rank(plan, A, B, arena, base, r)
        cost.node_writes += 1
        if e is not None:
            errors.append((r, e))
    if errors:
        raise ParseError(min(errors)[1])
    out = assemble_summary(plan, A, B, base, cost)
    cost.firings += plan.m
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def non_validating_fire():
    """Run the real combine() with firing stubbed out: gives its firing sequence and
    survivors even where types would make it raise."""
    old = S.fire
    S.fire = lambda op, a, b: ("node", op.name)
    try:
        yield
    finally:
        S.fire = old


def v_shaped(ops):
    seen_lt = False
    for x, y in zip(ops, ops[1:]):
        r = rel(x, y)
        if r == LT:
            seen_lt = True
        elif seen_lt:
            return False
    return True


class Disagreement(Exception):
    pass


def check_lemma1():
    """L1 over every operator the parser can build (plus both sentinels)."""
    box = Box("e")
    universe = list(P.BIN_OPS.values()) + [
        P.NEG_OP, P.SEMI_OP, P.SEQ_OP, START, END,
        Op("if", PREFIX, INF, P.HDR_RBP, (box,)),
        Op("while", PREFIX, INF, P.HDR_RBP, (box,)),
        Op("ifelse", PREFIX, INF, P.HDR_RBP, (box, box))]
    for a, b in itertools.product(universe, repeat=2):
        assert (rel(a, b) == GT) == (a.rbp > b.lbp), (a.name, b.name)
    return len(universe) ** 2


def compare_merge(a_s, a_r, b_s, b_r, arena, cost, rng, tally):
    """One merge, sequential vs ranked. Raises Disagreement; returns (seq, ranked)
    summaries, or None when both reject."""
    sides = []
    seq_err = rank_err = False
    try:
        s_res = S.combine(a_s, b_s, sides=sides)
    except ParseError:
        seq_err = True
    plan_box = []
    shuffle = lambda m: rng.sample(range(m), m)
    try:
        r_res = ranked_combine(a_r, b_r, arena, cost, plan_out=plan_box, write_order=shuffle)
    except ParseError:
        rank_err = True
    plan = plan_box[0]

    # discovery: firing sequence and survivors, compared on the type-free run
    with non_validating_fire():
        sides0 = []
        ref = S.combine(a_s, b_s, sides=sides0)
    got = [side for side, _ in plan.order]
    if got != sides0:
        raise Disagreement(f"firing order: plan {''.join(got)} vs sequential {''.join(sides0)}")
    if (plan.f, plan.J) != (sides0.count("A"), sides0.count("B")):
        raise Disagreement("firing counts")
    keep = plan.nA - plan.f
    surv = (a_r.ops[:keep] + b_r.ops[plan.J:]) if plan.nA and plan.nB else (a_r.ops + b_r.ops)
    if [id(o) for o in surv] != [id(o) for o in ref.ops]:
        raise Disagreement("survivors")
    if plan.nA and plan.nB:
        top, nxt = a_r.ops[keep - 1], b_r.ops[plan.J]
        if plan.stop == "rows_blocked":
            ok = plan.f < plan.p and rel(top, nxt) == LT and plan.J == plan.vB
        elif plan.stop == "barrier_gt":
            ok = plan.f == plan.p and rel(top, nxt) == GT
        else:
            ok = plan.f == plan.p and rel(top, nxt) == LT and plan.J == plan.vB
        if not ok:
            raise Disagreement(f"stop reason {plan.stop}")
    tally["plans"] += 1
    tally["firings"] += plan.m
    tally[plan.stop] += 1
    runs = sum(1 for i, x in enumerate(got) if i == 0 or x != got[i - 1])
    tally["interleaved_3plus_runs"] += runs >= 3
    tally["max_runs"] = max(tally["max_runs"], runs)

    # rejection contract and AST
    if seq_err != rank_err:
        raise Disagreement(f"rejection: sequential={'ERR' if seq_err else 'ok'} "
                           f"ranked={'ERR' if rank_err else 'ok'}")
    if seq_err:
        tally["both_reject"] += 1
        return None
    if [id(o) for o in r_res.ops] != [id(o) for o in s_res.ops]:
        raise Disagreement("result ops")
    if not v_shaped(r_res.ops):
        raise Disagreement("L2: result is not V-shaped")
    for u, v in zip(s_res.vals, r_res.vals):
        if flatten_tree(u) != arena.flat(v):
            raise Disagreement("AST")
    tally["asts"] += 1
    return s_res, r_res


def lockstep_reduce(vals, ops, shape_, rng, arena, cost, tally):
    """Bracket the items; at every internal node run sequential and ranked and compare.
    Returns (seq, ranked) for the whole group, or None if both reject somewhere."""
    seq_items = S.leaves(vals, ops)
    rk_items = ranked_leaves(vals, ops)

    def rec(lo, hi):
        if hi - lo == 1:
            return seq_items[lo], rk_items[lo]
        if shape_ == "left":
            mid = hi - 1
        elif shape_ == "right":
            mid = lo + 1
        elif shape_ == "balanced":
            mid = (lo + hi) // 2
        else:
            mid = rng.randint(lo + 1, hi - 1)
        left = rec(lo, mid)
        right = rec(mid, hi)
        if left is None or right is None:
            return None
        return compare_merge(left[0], left[1], right[0], right[1], arena, cost, rng, tally)

    out = rec(0, len(seq_items))
    if out is not None and not (len(out[1].ops) == 2 and out[1].ops[0] is START
                                and out[1].ops[1] is END):
        raise Disagreement("did not reduce to START END")
    return out


def new_tally():
    from collections import Counter
    return Counter()


def groups_from_tokens(toks):
    try:
        gs = P.build_groups(toks)
        return [P.build_flat(g.items) for g in gs if g.items]
    except ParseError:
        return []


def test_programs(valid, mutated, seed, cost, tally):
    """Programs from the existing generators, every bracketing shape, every group."""
    rng = random.Random(seed)
    shapes = ("balanced", "left", "right", "random")
    n = 0
    for k in range(valid + mutated):
        toks = P.gen_program(rng)
        if k >= valid:
            toks = P.mutate(toks, rng)
        for flat in groups_from_tokens(toks):
            for sh in shapes:
                arena = Arena()
                lockstep_reduce(flat[0], flat[1], sh, rng, arena, cost, tally)
                n += 1
    return n


def contractor_for(shape_, rng, cost):
    """A contractor for peak_statements.parse(): the full ranked pipeline."""
    def contractor(vals, ops, plateaus=True, trace=None):
        arena = Arena()
        items = ranked_leaves(vals, ops)
        firings_before = cost.firings       # parse() sums work over groups: report this call only

        def rec(lo, hi):
            if hi - lo == 1:
                return items[lo], 0
            mid = {"left": hi - 1, "right": lo + 1, "balanced": (lo + hi) // 2}.get(
                shape_) or rng.randint(lo + 1, hi - 1)
            a, da = rec(lo, mid)
            b, db = rec(mid, hi)
            return ranked_combine(a, b, arena, cost), max(da, db) + 1

        final, depth = rec(0, len(items))
        if len(final.ops) != 2 or final.ops[0] is not START or final.ops[1] is not END:
            raise RuntimeError("ranked summary did not reduce to a single tree")
        value = S.flatten_seq(arena.materialize(final.vals[0]))
        if trace is not None:
            trace.append(P._show(value))
        return value, depth, cost.firings - firings_before
    return contractor


def test_language(valid, mutated, seed, cost):
    """Complete programs against recursive descent (accept/reject and exact tree)."""
    rng = random.Random(seed)
    oracle = lambda t: P.RD(t).program()
    acc = rej = 0
    for k in range(valid + mutated):
        toks = P.gen_program(rng)
        if k >= valid:
            toks = P.mutate(toks, rng)
        want = P.outcome(oracle, toks)
        for sh in ("balanced", "left", "right", "random"):
            c = contractor_for(sh, random.Random(rng.random()), cost)
            got = P.outcome(lambda t: P.parse(t, contractor=c), toks)
            if got != want:
                raise Disagreement(f"language ({sh}): {' '.join(toks)}")
        acc += want != "ERR"
        rej += want == "ERR"
    return acc, rej


# -- exhaustive short normalized sequences over the actual operator alphabet ------
def enumerate_flats(max_tokens):
    """All token sequences build_flat accepts, as item lists, up to max_tokens items."""
    infix = [P.BIN_OPS[s] for s in "+-*/^="]
    prefix = [P.NEG_OP, Op("if", PREFIX, INF, P.HDR_RBP, (Box("e"),))]
    stmt = Box("s")

    def rec(items, state):
        if state != "E" and items:
            yield list(items)
        if len(items) == max_tokens:
            return
        choices = []
        choices += [(("val", "x", "e"), "Ae"), (("val", stmt, "s"), "As")]
        choices += [(("op", o), "E") for o in prefix]
        if state == "Ae":
            choices += [(("op", o), "E") for o in infix]
            choices += [(("op", P.SEMI_OP), "As")]
        for it, nxt in choices:
            items.append(it)
            yield from rec(items, nxt)
            items.pop()

    yield from rec([], "E")


def test_exhaustive(max_tokens, cost, tally, seed=5):
    rng = random.Random(seed)
    seqs = pairs = 0
    for items in enumerate_flats(max_tokens):
        try:
            vals, ops = P.build_flat(items)
        except ParseError:
            continue
        seqs += 1
        n = len(ops) + 2
        for cut in range(1, n):
            # every two-chunk split; chunks folded left, sequentially (reference side)
            sl, sr = S.leaves(vals, ops), ranked_leaves(vals, ops)
            with non_validating_fire():
                A0 = S.reduce_items(sl[:cut], "left")[0]
                B0 = S.reduce_items(sl[cut:], "left")[0]
            # the ranked inputs for the type-free plan check carry dummy values
            a_r = RSummary(list(A0.ops), list(A0.vals))
            b_r = RSummary(list(B0.ops), list(B0.vals))
            with non_validating_fire():
                sides0 = []
                ref = S.combine(A0, B0, sides=sides0)
            plan = discover_merge(a_r, b_r, Cost())
            if [s for s, _ in plan.order] != sides0:
                raise Disagreement("exhaustive plan: " + " ".join(map(str, ops)))
            keep = plan.nA - plan.f
            surv = (a_r.ops[:keep] + b_r.ops[plan.J:]) if plan.nA and plan.nB else a_r.ops + b_r.ops
            if [id(o) for o in surv] != [id(o) for o in ref.ops]:
                raise Disagreement("exhaustive survivors")
            pairs += 1
        # AST + rejection at every merge of three bracketings
        for sh in ("left", "right", "balanced"):
            lockstep_reduce(vals, ops, sh, rng, Arena(), cost, tally)
    return seqs, pairs


# -- random real operator tables (ties, odd binding powers) -------------------------
def random_table(rng):
    infix = []
    for k in range(rng.randint(2, 5)):
        L = rng.randint(1, 6)
        infix.append(Op(f"i{k}", INFIX, L, max(0, L + rng.choice((-1, 0, 1)))))
    prefix = [Op("neg", PREFIX, INF, rng.randint(1, 7)) for _ in range(rng.randint(0, 2))]
    postfix = [Op(";", POSTFIX, rng.randint(1, 6), INF)] if rng.random() < 0.6 else []
    return infix, prefix, postfix


def random_flat(rng, table, n_items):
    infix, prefix, postfix = table
    vals, ops, want = [], [], "E"
    while len(ops) < n_items or want == "E":
        if want == "E":
            if prefix and rng.random() < 0.4:
                vals.append(EPS)
                ops.append(rng.choice(prefix))
            else:
                vals.append("x")
                want = "A"
        else:
            if postfix and rng.random() < 0.2:
                ops.append(postfix[0])
                vals.append(EPS)
            else:
                ops.append(rng.choice(infix))
                want = "E"
            if want == "A" and len(ops) >= n_items:
                break
        if len(ops) >= n_items and want == "A":
            break
    return vals, ops


def test_random_tables(tables, per_table, max_items, seed, cost, tally):
    rng = random.Random(seed)
    n = 0
    for _ in range(tables):
        table = random_table(rng)
        for _ in range(per_table):
            vals, ops = random_flat(rng, table, rng.randint(1, max_items))
            if len(vals) != len(ops) + 1:       # postfix-last shape: vals = ops + 1 required
                continue
            for sh in ("balanced", "left", "right", "random"):
                lockstep_reduce(vals, ops, sh, rng, Arena(), cost, tally)
                n += 1
    return n


# -- long alternating unary/exponentiation chain ---------------------------------------
def test_chain(reps_list, cost):
    out = []
    for reps in reps_list:
        toks = S.bad_tokens(reps)
        groups = P.build_groups(toks)
        flat = P.build_flat(groups[0].items)
        tally = new_tally()
        res = lockstep_reduce(flat[0], flat[1], "balanced", random.Random(reps), Arena(), cost, tally)
        assert res is not None
        c = contractor_for("balanced", random.Random(1), Cost())
        got = P.parse(toks, contractor=c)
        want = S.rd_deep(toks)
        assert P.flatten_tree(got) == P.flatten_tree(want), reps
        out.append((reps, tally["plans"], tally["firings"]))
    return out


# -- independence of ranks ---------------------------------------------------------------
def test_rank_independence(seed, cost):
    """Same merge, ranks written in ascending, descending and shuffled order: identical
    arena contents. (Shows no rank reads another rank's slot; it is not parallel.)"""
    rng = random.Random(seed)
    checked = 0
    for _ in range(300):
        toks = P.gen_program(rng)
        for vals, ops in groups_from_tokens(toks):
            items = ranked_leaves(vals, ops)
            snaps = []
            for order in (lambda m: list(range(m)), lambda m: list(range(m))[::-1],
                          lambda m: rng.sample(range(m), m)):
                arena = Arena()
                acc = items[0]
                try:
                    for it in items[1:]:
                        acc = ranked_combine(acc, it, arena, cost, write_order=order)
                except ParseError:
                    snaps.append("ERR")
                    continue
                snaps.append(arena.flat(acc.vals[0]))
            assert snaps[0] == snaps[1] == snaps[2]
            checked += 1
    return checked


def test_accounting(seed=23, trials=400):
    """(1) parse() sums contractor work over groups: that sum must equal the number of
    operators (each fires exactly once), not a running total repeated per group.
    (2) assemble_summary's counters against sizes recomputed from the plan."""
    rng = random.Random(seed)
    multi = checked = 0
    for _ in range(trials):
        toks = P.gen_program(rng)
        want_ops = 0
        try:
            groups = P.build_groups(toks)
            for g in groups:
                if g.items:
                    want_ops += len(P.build_flat(g.items)[1])
        except ParseError:
            continue
        cost = Cost()
        stats = {}
        try:
            P.parse(toks, stats, contractor=contractor_for("random", random.Random(rng.random()), cost))
        except ParseError:
            continue
        assert stats["work"] == want_ops == cost.firings, (stats["work"], want_ops, cost.firings)
        multi += stats["groups"] > 1
        checked += 1
    # (2) one explicit merge: A = [START][x =], B = [a +][b END] style pieces
    vals, ops = P.build_flat(P.build_groups(P.tokenize("x = a + b * c ;"))[0].items)
    items = ranked_leaves(vals, ops)
    left = items[0]
    arena, cost = Arena(), Cost()
    for it in items[1:-1]:
        before = cost.as_dict()
        plan_box = []
        left = ranked_combine(left, it, arena, cost, plan_out=plan_box)
        pl = plan_box[0]
        keep, tail = pl.nA - pl.f, pl.nB - pl.J
        out = keep + tail
        assert cost.output_entries - before["output_entries"] == 2 * out
        assert (cost.list_entries_written - before["list_entries_written"]
                == (keep + tail + out) + (2 * keep + tail + out))
    assert cost.list_entries_written >= cost.output_entries
    return checked, multi


def scaling(sizes=(256, 1024, 4096)):
    """Counter growth per source token, by input family and bracketing. Measurement only:
    counters, not timings, and no bound is inferred from three sizes."""
    fams = {
        "x = a + a + ... (left-assoc plateau)": lambda n: ["x", "="] + ["a", "+"] * n + ["b", ";"],
        "x = a ^ a ^ ... (right-assoc, long stack)": lambda n: ["x", "="] + ["a", "^"] * n + ["b", ";"],
        "x = -a ^ -a ^ ... (alternating chain)": lambda n: S.bad_tokens(n),
        "x; x; x; ... (statements)": lambda n: ["x", ";"] * n,
    }
    keys = ("valley_rel", "prefix_min_reads", "threshold_cmps", "rank_cmps",
            "list_entries_written")
    print(f"{'family':<44}{'shape':<9}{'n':>6}" + "".join(f"{k:>22}" for k in keys) + "   (per source token)")
    for name, mk in fams.items():
        for sh in ("balanced", "left"):
            for n in sizes:
                toks = mk(n)
                flat = P.build_flat(P.build_groups(toks)[0].items)
                cost = Cost()
                items = ranked_leaves(*flat)
                arena = Arena()
                if sh == "left":
                    acc = items[0]
                    for it in items[1:]:
                        acc = ranked_combine(acc, it, arena, cost)
                else:
                    def rec(lo, hi):
                        if hi - lo == 1:
                            return items[lo]
                        mid = (lo + hi) // 2
                        return ranked_combine(rec(lo, mid), rec(mid, hi), arena, cost)
                    rec(0, len(items))
                tot = len(items)
                print(f"{name:<44}{sh:<9}{n:>6}" + "".join(
                    f"{getattr(cost, k) / tot:>22.1f}" for k in keys))


def run(full=False):
    t0 = time.perf_counter()
    cost, tally = Cost(), new_tally()
    report = {}

    n = check_lemma1()
    print(f"L1: rel(a,b)==GT <=> a.rbp > b.lbp on all {n} ordered pairs of the operator universe")

    max_tok = 6 if full else 5
    seqs, pairs = test_exhaustive(max_tok, cost, tally)
    print(f"exhaustive: {seqs} normalized sequences (<= {max_tok} source items, real operator "
          f"alphabet, types included), {pairs} two-chunk splits plan-checked, "
          f"3 bracketings AST/rejection-checked at every merge")

    progs = test_programs(*((1500, 6000) if full else (300, 1200)), seed=11, cost=cost, tally=tally)
    print(f"programs: {progs} (group, bracketing) lockstep reductions from the generators")

    nt = test_random_tables(*((300, 40, 14) if full else (80, 25, 12)), seed=13, cost=cost, tally=tally)
    print(f"random real tables (ties, mixed fixity): {nt} lockstep reductions")

    acc, rej = test_language(*((1500, 6000) if full else (300, 1500)), seed=17, cost=cost)
    print(f"language: ranked reducer equals recursive descent on {acc} accepted and {rej} rejected "
          f"programs x 4 bracketings")

    chain = test_chain((50, 400) if not full else (50, 400, 3000), cost)
    print("alternating unary/exponentiation chain (plans, firings per lockstep run):", chain)

    ch, multi = test_accounting()
    print(f"accounting: parse() work equals total operators on {ch} programs ({multi} with "
          f"several groups); per-merge copy counters match plan-derived sizes")

    ind = test_rank_independence(19, cost)
    print(f"rank independence: {ind} reductions identical under ascending/descending/shuffled writes")

    print()
    print("merges compared in lockstep:", tally["plans"], " rejected by both:", tally["both_reject"],
          " ASTs compared:", tally["asts"], " firings planned:", tally["firings"])
    print("stop reasons:", {k: tally[k] for k in ("rows_blocked", "barrier_gt", "valley")})
    print("merges whose firings interleave A and B (3+ same-side runs):",
          tally["interleaved_3plus_runs"], " maximum runs in one merge:", tally["max_runs"])
    print("cost counters over the lockstep + language runs:", cost.as_dict())
    print(f"ALL CHECKS PASSED in {time.perf_counter() - t0:.1f}s. Sequential simulation: "
          f"no parallel timing, work bound, or depth bound is claimed.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--scaling", action="store_true",
                    help="print counter growth per source token and exit")
    sys.setrecursionlimit(10000)
    args = ap.parse_args()
    if args.scaling:
        scaling()
        raise SystemExit(0)
    try:
        run(args.full)
    except Disagreement as e:
        print("DISAGREEMENT:", e)
        raise SystemExit(1)