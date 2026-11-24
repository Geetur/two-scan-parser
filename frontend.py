"""Language rules, group formation, normalization, and group resolution."""
from dataclasses import dataclass, field
from lexer import is_operand

INF = 10 ** 9
INFIX, PREFIX, POSTFIX = "infix", "prefix", "postfix"
STMT_HEADS = {"expr", "block", "if", "while", "seq"}
EPS = None


class ParseError(Exception):
    pass


@dataclass(eq=False, slots=True)
class Box:
    """An opaque group operand; equality and hashing use object identity."""
    t: str                 # 'e' expression or 's' statement
    value: object = None


@dataclass(eq=False, slots=True)
class Op:
    name: str
    kind: str
    lbp: int
    rbp: int
    args: tuple = ()


BINARY = {"=": (21, 20), "+": (30, 31), "-": (30, 31),
          "*": (40, 41), "/": (40, 41), "^": (51, 50)}
BIN_OPS = {s: Op(s, INFIX, *bp) for s, bp in BINARY.items()}
NEG_OP = Op("neg", PREFIX, INF, 45)
SEMI_OP = Op(";", POSTFIX, 10, INF)
SEQ_OP = Op("seq", INFIX, 1, 2)
HDR_RBP = 5


def type_of(value):
    if isinstance(value, Box):
        return value.t
    return "s" if isinstance(value, tuple) and value[0] in STMT_HEADS else "e"


def need(value, category, what=""):
    if type_of(value) != category:
        expected = "a statement" if category == "s" else "an expression"
        raise ParseError(f"{what or 'operand'} must be {expected}")


@dataclass(eq=False, slots=True)
class Frame:
    kind: str
    box: Box
    cond: str | None = None
    items: list = field(default_factory=list)
    open_ifs: list = field(default_factory=list)
    pending: str | None = None


def build_groups(tokens):
    """Turn nested syntax into flat groups joined by typed Box references."""
    groups = []

    def new_frame(kind, cond=None):
        frame = Frame(kind, Box("e" if kind == "expr" else "s"), cond)
        groups.append(frame)
        return frame

    root = new_frame("stmts")
    frames = [root]

    for token in tokens:
        frame = frames[-1]
        if frame.pending is not None:                # if/while must see '('
            if token != "(":
                raise ParseError(f"expected '(' after {frame.pending}")
            frames.append(new_frame("expr", frame.pending))
            frame.pending = None
        elif token in ("(", "{"):
            frames.append(new_frame("expr" if token == "(" else "stmts"))
        elif token in (")", "}"):
            expected = "expr" if token == ")" else "stmts"
            if frame.kind != expected or frame is root:
                raise ParseError(f"unmatched {token!r}")
            frames.pop()
            parent = frames[-1]
            if frame.cond is None:
                parent.items.append(("val", frame.box, frame.box.t))
            else:                                    # condition becomes a prefix header
                header = Op(frame.cond, PREFIX, INF, HDR_RBP, (frame.box,))
                parent.items.append(("op", header))
                if frame.cond == "if":
                    parent.open_ifs.append(len(parent.items) - 1)
        elif token in ("if", "while"):
            frame.pending = token
        elif token == "else":                        # nearest unmatched if in this group
            if not frame.open_ifs:
                raise ParseError("else without if")
            index = frame.open_ifs.pop()
            header = frame.items[index][1]
            then = new_frame("then")
            then.items = frame.items[index + 1:]
            header = Op("ifelse", PREFIX, INF, HDR_RBP, (header.args[0], then.box))
            frame.items[index:] = [("op", header)]
        elif token == ";":
            frame.items.append(("op", SEMI_OP))
        elif is_operand(token):
            frame.items.append(("val", token, "e"))
        elif token in BIN_OPS:
            previous = frame.items[-1] if frame.items else None
            after_expr = previous is not None and previous[0] == "val" and previous[2] == "e"
            op = NEG_OP if token == "-" and not after_expr else BIN_OPS[token]
            frame.items.append(("op", op))
        else:
            raise ParseError(f"bad token {token!r}")

    if len(frames) != 1:
        raise ParseError("unclosed bracket")
    if root.pending:
        raise ParseError(f"dangling {root.pending}")
    return groups


def build_flat(items):
    """Insert virtual operands and sequence operators; return (vals, ops)."""
    vals, ops = [], []
    # None means the stream needs an operand; 'e'/'s' means it has one.
    state = None
    for item in items:
        is_value = item[0] == "val"
        op = None if is_value else item[1]
        if (is_value or op.kind == PREFIX) and state is not None:
            ops.append(SEQ_OP)
        if is_value:
            vals.append(item[1])
            state = "e" if item[2] == "e" else "s"
        elif op.kind == PREFIX:
            vals.append(EPS)
            ops.append(op)
            state = None
        elif op.kind == INFIX:
            if state != "e":
                raise ParseError(f"unexpected {op.name}")
            ops.append(op)
            state = None
        else:
            if state != "e":
                raise ParseError("unexpected ';'")
            ops.append(op)
            vals.append(EPS)
            state = "s"
    if state is None:
        raise ParseError("expression ends early")
    return vals, ops


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
                start = len(out) - len(node) + 1
                out[start:] = [(node[0], *out[start:])]
            else:
                stack.append((node, True))
                for child in reversed(node[1:]):
                    stack.append((child, False))
        else:
            out.append(node)
    return out[0]
