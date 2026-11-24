#!/usr/bin/env python3
"""
Peak (plateau) contraction extended from expressions to a whole toy language:

    program := stmt*
    stmt    := '{' stmt* '}' | 'if' '(' e ')' stmt ['else' stmt]
             | 'while' '(' e ')' stmt | e ';'
    e       := ints, identifiers, ( e ), infix + - * / ^ =, prefix -

Self-contained. Run:  python peak_statements.py            (trace, fuzz, bench)
                      python peak_statements.py "x = 1; if (x) y = -x ^ 2;"

THE IDEA, UNCHANGED
    Split the input into bracket groups (each group is an independent flat
    problem; nested groups are opaque operands). Inside a group, repeatedly
    reduce every "peak": an operator that must fire before both neighbours.
    A maximal run of same-class operators bounded by looser neighbours fires
    in ONE round (a "plateau"), since associativity fixes its shape.

WHAT IS NEW HERE (three moves)
    1. Operators get a fixity (infix / prefix / postfix) and two binding
       powers. rel(a, b) says which of two adjacent operators fires first.
       A peak is an operator with rel(prev, op) = LT and rel(op, next) = GT.
       Empty operand slots (EPS) make prefix/postfix fit the same
       operand-op-operand scheme; the relation keeps infix ops away from EPS.
    2. Statements are just operators:
          ';'              postfix, looser than every expression operator
          if/while header  prefix, looser than ';' (its operand is a statement)
          invisible 'seq'  loosest infix, inserted between adjacent
                           statements, collapsing a run into one n-ary node
    3. Dangling else is bracket matching. In the grouping phase an 'else'
       matches the nearest unmatched 'if' in the same group (matching with
       unmatched openers allowed, as in parallel parenthesis matching). The
       then-branch becomes its own group, and 'if(c) then else' becomes an
       ordinary prefix header over the else-branch. The grouping phase here
       is a sequential stack scan; the parallel claims are about 'rounds'.

ROUND BOUND (corrected)
    If every operator in a group is infix, rel() is acyclic on classes, the
    highest class always fires in a round, and rounds <= number of classes
    (this is the setting of peak_contraction.py). That argument FAILS once
    prefix operators are present: an operator preceding a prefix always yields to that
    prefix, so rel(^, neg) = LT even though rel(neg, ^) = LT as well.
    That cycle gives right-nested chains that alternate classes, e.g.

        x = -a ^ -a ^ -a ^ ... ^ b;

    which takes 2n+2 rounds for n repeats (checked), with only two classes
    involved and no plateau longer than one operator. So for this language the
    bound is NOT constant in general. It holds for ordinary programs and for
    all the shapes benchmarked below, but it is not proved, and the benchmark
    includes the known bad shape. A likely fix (untested): collapse a whole
    ascending run in one step when the operator after it is looser than every
    operator in the run, found with a range-minimum query over the run. That
    is the stack-pop loop of shunting yard done by nearest-smaller lookup.
"""
import random
import re
import sys

INF = 10 ** 9
INFIX, PREFIX, POSTFIX = "infix", "prefix", "postfix"
LT, GT = "<", ">"      # a LT b: b fires first.  a GT b: a fires first.
KEYWORDS = {"if", "else", "while"}
STMT_HEADS = {"expr", "block", "if", "while", "seq"}
EPS = None             # empty operand slot of a prefix/postfix operator


class ParseError(Exception):
    pass


class Box:
    """A bracket group's result, referenced by its parent as an opaque operand."""
    __slots__ = ("t", "value")

    def __init__(self, t):
        self.t = t          # 'e' expression or 's' statement
        self.value = None


class Op:
    __slots__ = ("name", "kind", "lbp", "rbp", "args", "cls")

    def __init__(self, name, kind, lbp, rbp, args=()):
        self.name, self.kind, self.lbp, self.rbp, self.args = name, kind, lbp, rbp, args
        self.cls = (kind, lbp, rbp)   # same cls => same plateau


BINARY = {"=": (21, 20), "+": (30, 31), "-": (30, 31),
          "*": (40, 41), "/": (40, 41), "^": (51, 50)}      # (lbp, rbp)
BIN_OPS = {s: Op(s, INFIX, *bp) for s, bp in BINARY.items()}
NEG_OP = Op("neg", PREFIX, INF, 45)
SEMI_OP = Op(";", POSTFIX, 10, INF)
SEQ_OP = Op("seq", INFIX, 1, 2)
HDR_RBP = 5            # if / ifelse / while headers


def rel(a, b):
    if b.kind == PREFIX:
        return LT
    if a.kind == POSTFIX:
        return GT
    return GT if a.rbp > b.lbp else LT


# --------------------------------------------------------------------------
# Types: only used to reject nonsense like  `{} + 1`  or  `x y;`
# --------------------------------------------------------------------------
def type_of(v):
    if isinstance(v, Box):
        return v.t
    if isinstance(v, tuple):
        return "s" if v[0] in STMT_HEADS else "e"
    return "e"


def need(v, t, what=""):
    if type_of(v) != t:
        raise ParseError(f"{what or 'operand'} must be {'a statement' if t == 's' else 'an expression'}")


def make(op, a, b):
    n = op.name
    if op.kind == INFIX:
        need(a, "e", n)
        need(b, "e", n)
        return (n, a, b)
    if n == "neg":
        need(b, "e", n)
        return ("neg", b)
    if n == ";":
        need(a, "e", ";")
        return ("expr", a)
    need(b, "s", n)
    if n == "if":
        return ("if", op.args[0], b)
    if n == "ifelse":
        return ("if", op.args[0], op.args[1], b)
    return ("while", op.args[0], b)


# --------------------------------------------------------------------------
# Phase 1: grouping (parallel in principle: bracket matching via prefix sums)
# --------------------------------------------------------------------------
def tokenize(src):
    return re.findall(r"[A-Za-z_]\w*|\d+|\S", src)


def is_operand(t):
    return t.isdigit() or (t.isidentifier() and t not in KEYWORDS)


class Frame:
    __slots__ = ("kind", "box", "cond", "items", "open_ifs", "pending")

    def __init__(self, kind, box, cond=None):
        self.kind, self.box, self.cond = kind, box, cond
        self.items, self.open_ifs, self.pending = [], [], None


def build_groups(tokens):
    root = Frame("stmts", Box("s"))
    frames, groups = [root], [root]

    def new_frame(kind, box, cond=None):
        f = Frame(kind, box, cond)
        groups.append(f)
        return f

    for tok in tokens:
        f = frames[-1]
        if f.pending is not None:                    # if/while must see '('
            if tok != "(":
                raise ParseError(f"expected '(' after {f.pending}")
            frames.append(new_frame("expr", Box("e"), f.pending))
            f.pending = None
        elif tok == "(":
            frames.append(new_frame("expr", Box("e")))
        elif tok == ")":
            if f.kind != "expr":
                raise ParseError("unmatched ')'")
            frames.pop()
            parent = frames[-1]
            if f.cond is None:
                parent.items.append(("val", f.box, "e"))
            else:                                    # keyword + condition = one header
                parent.items.append(("op", Op(f.cond, PREFIX, INF, HDR_RBP, (f.box,))))
                if f.cond == "if":
                    parent.open_ifs.append(len(parent.items) - 1)
        elif tok == "{":
            frames.append(new_frame("stmts", Box("s")))
        elif tok == "}":
            if f.kind != "stmts" or f is root:
                raise ParseError("unmatched '}'")
            frames.pop()
            frames[-1].items.append(("val", f.box, "s"))
        elif tok in ("if", "while"):
            f.pending = tok
        elif tok == "else":                          # dangling else = bracket matching
            if not f.open_ifs:
                raise ParseError("else without if")
            idx = f.open_ifs.pop()
            hdr = f.items[idx][1]
            then = new_frame("then", Box("s"))
            then.items = f.items[idx + 1:]
            f.items[idx:] = [("op", Op("ifelse", PREFIX, INF, HDR_RBP, (hdr.args[0], then.box)))]
        elif tok == ";":
            f.items.append(("op", SEMI_OP))
        elif is_operand(tok):
            f.items.append(("val", tok, "e"))
        elif tok == "-":                             # two-sided: prefix vs infix by left context
            prev = f.items[-1] if f.items else None
            infix = prev is not None and prev[0] == "val" and prev[2] == "e"
            f.items.append(("op", BIN_OPS["-"] if infix else NEG_OP))
        elif tok in BIN_OPS:
            f.items.append(("op", BIN_OPS[tok]))
        else:
            raise ParseError(f"bad token {tok!r}")

    if len(frames) != 1:
        raise ParseError("unclosed bracket")
    if root.pending:
        raise ParseError(f"dangling {root.pending}")
    return groups


def build_flat(items):
    """Items -> alternating operands/operators, inserting EPS slots and invisible seq."""
    vals, ops = [], []
    state = "E"                                      # E: want operand; Ae/As: after expr/stmt
    for it in items:
        if it[0] == "val":
            if state != "E":
                ops.append(SEQ_OP)
            vals.append(it[1])
            state = "Ae" if it[2] == "e" else "As"
        else:
            op = it[1]
            if op.kind == PREFIX:
                if state != "E":
                    ops.append(SEQ_OP)
                vals.append(EPS)
                ops.append(op)
                state = "E"
            elif op.kind == INFIX:
                if state != "Ae":
                    raise ParseError(f"unexpected {op.name}")
                ops.append(op)
                state = "E"
            else:                                    # postfix ';'
                if state != "Ae":
                    raise ParseError("unexpected ';'")
                ops.append(op)
                vals.append(EPS)
                state = "As"
    if state == "E":
        raise ParseError("expression ends early")
    return vals, ops


# --------------------------------------------------------------------------
# Phase 2: plateau contraction on one flat group
# --------------------------------------------------------------------------
def _show(v):
    if isinstance(v, Box):
        return "{..}" if v.t == "s" else "(..)"
    if v is EPS:
        return "_"
    if isinstance(v, tuple):
        return v[0] + "(" + ", ".join(_show(k) for k in v[1:]) + ")"
    return str(v)


def _opstr(op):
    if op.name == "seq":
        return "<seq>"
    return op.name + "()" if op.kind == PREFIX and op.args else op.name


def _snapshot(vals, ops):
    parts = [_show(vals[0])]
    for op, v in zip(ops, vals[1:]):
        parts += [_opstr(op), _show(v)]
    return " ".join(parts)


def fold(vals, ops, i, j, cur):
    """Reduce plateau ops[i:j] in one step. Every node's children depend only on
    its index in the run, so this is a chain build, not a sequential dependency."""
    op = ops[i]
    if op.name == "seq":                             # n-ary flatten
        items = []
        for v in [cur] + vals[i + 1:j + 1]:
            need(v, "s", "seq")
            if isinstance(v, tuple) and v[0] == "seq":
                items.extend(v[1:])
            else:
                items.append(v)
        return ("seq", *items)
    if op.kind == PREFIX:                            # right fold, EPS on the left
        acc = vals[j]
        for t in range(j - 1, i - 1, -1):
            acc = make(ops[t], EPS, acc)
        return acc
    if op.kind == POSTFIX:                           # left fold, EPS on the right
        acc = cur
        for t in range(i, j):
            acc = make(ops[t], acc, EPS)
        return acc
    if rel(op, op) == GT:                            # left-assoc infix
        acc = cur
        for t in range(i, j):
            acc = make(ops[t], acc, vals[t + 1])
        return acc
    acc = vals[j]                                    # right-assoc infix
    for t in range(j - 1, i - 1, -1):
        acc = make(ops[t], vals[t], acc)
    return acc


def contract(vals, ops, plateaus=True, trace=None):
    rounds = work = 0
    while ops:
        if trace is not None:
            trace.append(_snapshot(vals, ops))
        rounds += 1
        n = len(ops)
        work += n
        new_vals, new_ops = [], []
        cur = vals[0]
        i = 0
        while i < n:
            op = ops[i]
            j = i + 1
            if plateaus:
                while j < n and ops[j].cls == op.cls:
                    j += 1
            if (i == 0 or rel(ops[i - 1], op) == LT) and (j == n or rel(ops[j - 1], ops[j]) == GT):
                cur = fold(vals, ops, i, j, cur)
            else:
                for t in range(i, j):
                    new_vals.append(cur)
                    new_ops.append(ops[t])
                    cur = vals[t + 1]
            i = j
        new_vals.append(cur)
        vals, ops = new_vals, new_ops
    if trace is not None:
        trace.append(_show(vals[0]))
    return vals[0], rounds, work


def contract_group(g, plateaus, trace, contractor=None):
    if not g.items:
        if g.kind == "stmts":
            g.box.value = ("block",)
            return 0, 0
        raise ParseError("empty group")
    vals, ops = build_flat(g.items)
    reducer = contract if contractor is None else contractor
    v, rounds, work = reducer(vals, ops, plateaus, trace)
    is_seq = isinstance(v, tuple) and v[0] == "seq"
    if g.kind == "expr":
        need(v, "e", "parenthesised expression")
    elif g.kind == "then":
        need(v, "s", "then-branch")
        if is_seq:
            raise ParseError("then-branch must be a single statement")
    else:
        need(v, "s", "statement list")
        v = ("block", *v[1:]) if is_seq else ("block", v)
    g.box.value = v
    return rounds, work


def resolve(root):
    """Unwrap boxes into plain nested tuples. Iterative (deep trees)."""
    out = []
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        while isinstance(node, Box):
            node = node.value
        if isinstance(node, tuple):
            if done:
                k = len(node) - 1
                kids = out[len(out) - k:]
                del out[len(out) - k:]
                out.append((node[0], *kids))
            else:
                stack.append((node, True))
                for child in reversed(node[1:]):
                    stack.append((child, False))
        else:
            out.append(node)
    return out[0]


def parse(tokens, stats=None, trace=None, plateaus=True, contractor=None):
    groups = build_groups(tokens)
    max_rounds = total_work = 0
    for k, g in enumerate(groups):                   # groups are independent: all at once
        r, w = contract_group(g, plateaus, trace if k == 0 else None, contractor)
        max_rounds = max(max_rounds, r)
        total_work += w
    if stats is not None:
        stats.update(rounds=max_rounds, work=total_work, groups=len(groups))
    return resolve(groups[0].box)


# --------------------------------------------------------------------------
# Independent oracle: ordinary recursive descent
# --------------------------------------------------------------------------
class RD:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        t = self.peek()
        if t is None or (x is not None and t != x):
            raise ParseError(f"expected {x}")
        self.i += 1
        return t

    def program(self):
        out = []
        while self.peek() is not None:
            out.append(self.stmt())
        return ("block", *out)

    def stmt(self):
        t = self.peek()
        if t == "{":
            self.eat()
            out = []
            while self.peek() != "}":
                out.append(self.stmt())
            self.eat("}")
            return ("block", *out)
        if t in ("if", "while"):
            self.eat()
            self.eat("(")
            c = self.expr()
            self.eat(")")
            body = self.stmt()
            if t == "while":
                return ("while", c, body)
            if self.peek() == "else":
                self.eat()
                return ("if", c, body, self.stmt())
            return ("if", c, body)
        e = self.expr()
        self.eat(";")
        return ("expr", e)

    def expr(self):
        l = self.add()
        if self.peek() == "=":
            self.eat()
            return ("=", l, self.expr())
        return l

    def add(self):
        l = self.mul()
        while self.peek() in ("+", "-"):
            op = self.eat()
            l = (op, l, self.mul())
        return l

    def mul(self):
        l = self.unary()
        while self.peek() in ("*", "/"):
            op = self.eat()
            l = (op, l, self.unary())
        return l

    def unary(self):
        if self.peek() == "-":
            self.eat()
            return ("neg", self.unary())
        return self.pow()

    def pow(self):
        b = self.primary()
        if self.peek() == "^":
            self.eat()
            return ("^", b, self.unary())
        return b

    def primary(self):
        t = self.peek()
        if t == "(":
            self.eat()
            e = self.expr()
            self.eat(")")
            return e
        if t is not None and is_operand(t):
            return self.eat()
        raise ParseError("operand expected")


# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------
def flatten_tree(tree):
    out, st = [], [tree]
    while st:
        t = st.pop()
        if isinstance(t, tuple):
            out.append(t[0])
            out.append(len(t) - 1)
            st.extend(reversed(t[1:]))
        else:
            out.append(t)
    return out


def gen_expr(rng, ctr, max_ops=5):
    n = rng.randint(0, max_ops)
    toks, depth = [], 0
    for k in range(n + 1):
        while rng.random() < 0.15:
            toks.append("(")
            depth += 1
        while rng.random() < 0.15:
            toks.append("-")
        ctr[0] += 1
        toks.append(str(ctr[0]) if ctr[0] % 2 else f"v{ctr[0]}")
        while depth and (k == n or rng.random() < 0.3):
            toks.append(")")
            depth -= 1
        if k < n:
            toks.append(rng.choice("+-*/^="))
    return toks


def gen_stmt(rng, ctr, depth):
    r = rng.random()
    if depth <= 0 or r < 0.35:
        return gen_expr(rng, ctr) + [";"]
    if r < 0.50:
        out = ["{"]
        for _ in range(rng.randint(0, 3)):
            out += gen_stmt(rng, ctr, depth - 1)
        return out + ["}"]
    if r < 0.80:
        out = ["if", "("] + gen_expr(rng, ctr) + [")"] + gen_stmt(rng, ctr, depth - 1)
        if rng.random() < 0.6:                       # dangling-else cases arise naturally
            out += ["else"] + gen_stmt(rng, ctr, depth - 1)
        return out
    return ["while", "("] + gen_expr(rng, ctr) + [")"] + gen_stmt(rng, ctr, depth - 1)


def gen_program(rng, depth=4):
    ctr, out = [0], []
    for _ in range(rng.randint(0, 5)):
        out += gen_stmt(rng, ctr, depth)
    return out


VOCAB = ["if", "else", "while", "(", ")", "{", "}", ";", "+", "-", "*", "/", "^", "=", "v1", "7"]


def mutate(toks, rng):
    toks = list(toks)
    if not toks:
        return [rng.choice(VOCAB)]
    i = rng.randrange(len(toks))
    m = rng.randrange(4)
    if m == 0:
        del toks[i]
    elif m == 1:
        toks.insert(i, rng.choice(VOCAB))
    elif m == 2:
        toks[i] = rng.choice(VOCAB)
    elif i + 1 < len(toks):
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return toks


def outcome(fn, toks):
    try:
        return flatten_tree(fn(toks))
    except ParseError:
        return "ERR"


def fuzz(valid=6000, mutated=30000, seed=0):
    rng = random.Random(seed)
    for _ in range(valid):
        toks = gen_program(rng)
        a, b = outcome(parse, toks), outcome(lambda t: RD(t).program(), toks)
        if a != b or a == "ERR":
            print("MISMATCH on valid program:", " ".join(toks))
            return False
    print(f"fuzz: {valid} random valid programs match recursive descent")
    accepted = rejected = 0
    for _ in range(mutated):
        toks = mutate(gen_program(rng), rng)
        a, b = outcome(parse, toks), outcome(lambda t: RD(t).program(), toks)
        if a != b:
            print("MISMATCH on mutated program:", " ".join(toks))
            print("  mine:  ", "ERR" if a == "ERR" else "accepted")
            print("  oracle:", "ERR" if b == "ERR" else "accepted")
            return False
        accepted += a != "ERR"
        rejected += a == "ERR"
    print(f"fuzz: {mutated} mutated programs agree with recursive descent "
          f"on accept/reject and on the tree ({accepted} accepted, {rejected} rejected)")
    return True


# --------------------------------------------------------------------------
# Demo and benchmark
# --------------------------------------------------------------------------
def sexp(t):
    return str(t) if not isinstance(t, tuple) else "(" + t[0] + " " + " ".join(sexp(k) for k in t[1:]) + ")"


def pretty(t, ind=0):
    pad = "  " * ind
    if isinstance(t, tuple) and t[0] in ("block", "if", "while"):
        return "\n".join([pad + t[0]] + [pretty(k, ind + 1) for k in t[1:]])
    return pad + (sexp(t) if isinstance(t, tuple) else str(t))


def show(src):
    toks = tokenize(src)
    trace, stats = [], {}
    tree = parse(toks, stats, trace)
    print("source:", src)
    print("\nroot group, round by round   (_ = empty operand slot, <seq> = invisible separator)")
    for r, line in enumerate(trace[:-1]):
        print(f"  round {r}: {line}")
    print(f"  result : {trace[-1]}")
    print(f"\nwhole program: {stats['rounds']} rounds (max over {stats['groups']} groups)")
    print("\ntree:")
    print(pretty(tree))


def shapes(n):
    r = lambda k, *t: list(t) * k
    return [
        ("n statements", ["x", "=", "1", "+", "y", "*", "2", ";"] * n),
        ("else-if chain", ["if", "(", "c", ")", "x", ";", "else"] * n + ["y", ";"]),
        ("n nested ifs, no braces", ["if", "(", "c", ")"] * n + ["x", ";"]),
        ("dangling else pile", ["if", "(", "c", ")"] * n + ["x", ";"] + ["else", "y", ";"] * n),
        ("nested blocks", ["{"] * n + ["x", ";"] + ["}"] * n),
        ("nested parens", ["x", "="] + ["("] * n + ["1"] + [")"] * n + [";"]),
        ("long sum", ["x", "="] + ["1", "+"] * n + ["1", ";"]),
        # KNOWN BAD: alternating prefix/infix classes defeat plateaus (see docstring)
        ("-a ^ -a ^ ... (n/4)", ["x", "="] + ["-", "a", "^"] * (n // 4) + ["b", ";"]),
    ]


def bench():
    small, big = 2000, 50000
    print(f"\nPARALLEL DEPTH PROXY: rounds = max over groups of contraction rounds")
    print(f"{'shape':<26}{'n':>8}{'groups':>9}{'rake-only':>11}{'plateau':>9}")
    for (label, toks), (_, btoks) in zip(shapes(small), shapes(big)):
        s1, s2 = {}, {}
        t1 = parse(toks, s1, plateaus=False)
        t2 = parse(toks, s2, plateaus=True)
        assert flatten_tree(t1) == flatten_tree(t2), label
        print(f"{label:<26}{small:>8}{s2['groups']:>9}{s1['rounds']:>11}{s2['rounds']:>9}")
        if label.startswith("-a ^"):
            continue                      # quadratic here; the small row already shows it
        sb = {}
        parse(btoks, sb)
        print(f"{'':<26}{big:>8}{sb['groups']:>9}{'-':>11}{sb['rounds']:>9}")
    print("\nrake-only = the same grouped algorithm with plateaus switched off.")
    print("'-' = skipped (rake-only is O(n^2) work on these shapes).")
    print("Last row is the counterexample: plateaus do not help, rounds grow linearly.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        show("a = 1 + 2 * 3; if (c) if (d) b = -e ^ 2; else f = 4;")
        print()
        if fuzz():
            bench()