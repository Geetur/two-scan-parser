"""Public parser: source text -> normalized groups -> two scans -> AST tuples.

The two-scan numerical core is isolated in two_scan.py. Lexing, grouping,
normalization, type checking, and tuple export run sequentially here.
"""
import frontend as F
from frontend import ParseError
from lexer import tokenize
from two_scan import discover


def head_of(op):
    if op.kind == F.INFIX:
        return op.name
    return {"neg": "neg", ";": "expr", "if": "if", "ifelse": "if"}.get(
        op.name, "while"
    )


def output_type(op):
    return "s" if head_of(op) in F.STMT_HEADS else "e"


def required_types(op):
    if op.name == "seq":
        return "s", "s"
    if op.kind == F.INFIX:
        return "e", "e"
    if op.name == "neg":
        return None, "e"
    if op.name == ";":
        return "e", None
    return None, "s"


def node_shape(op, left, right):
    if op.kind == F.INFIX:
        return op.name, left, right
    if op.name == "neg":
        return "neg", right
    if op.name == ";":
        return "expr", left
    if op.name == "ifelse":
        return "if", op.args[0], op.args[1], right
    return head_of(op), op.args[0], right


def build_ast(vals, ops, plan, write_order=None):
    """Check all node types independently, then export nested tuples.

    The plan already supplies stable integer node references. Optional
    write_order(n) must enumerate every operator once; it changes only the
    order of independent checks, never the structural plan or tuple export.
    """
    if not ops:
        return vals[0]
    types = [output_type(op) for op in ops]

    def child_type(child):
        # Negative references encode original operand slots; nonnegative references are operators.
        return types[child] if child >= 0 else F.type_of(vals[-child - 1])

    order = range(len(ops)) if write_order is None else write_order(len(ops))
    invalid = False
    for i in order:
        for child, required in zip(
            (plan.left[i], plan.right[i]), required_types(ops[i])
        ):
            if required is not None and child_type(child) != required:
                invalid = True
    if invalid:
        raise F.ParseError("invalid operand types in direct scan AST")

    # Iterative postorder: child tuples are ready when their parent is built.
    # This avoids recursive materialization of deep chains.
    built, pending = [None] * len(ops), [(plan.root, False)]

    def value(child):
        return built[child] if child >= 0 else vals[-child - 1]

    while pending:
        i, ready = pending.pop()
        op, a, b = ops[i], plan.left[i], plan.right[i]
        if ready:
            built[i] = node_shape(op, value(a), value(b))
            continue
        pending.append((i, True))
        children = (a, b) if op.kind == F.INFIX else (a,) if op.name == ";" else (b,)
        pending.extend((child, False) for child in children if child >= 0)
    result = built[plan.root]

    # Flatten the group's top binary sequence once, exactly as before.
    if isinstance(result, tuple) and result[0] == "seq":
        statements, pending = [], [result]
        while pending:
            node = pending.pop()
            if isinstance(node, tuple) and node[0] == "seq":
                pending.extend((node[2], node[1]))
            else:
                statements.append(node)
        result = ("seq", *statements)
    return result


def plan_group(vals, ops, states=None):
    """Retain the original normalized-input and fixity checks."""
    if len(vals) != len(ops) + 1:
        raise ValueError("expected len(vals) == len(ops)+1")
    for op in ops:
        if op.kind == F.PREFIX and op.lbp != F.INF:
            raise ValueError("prefix sentinel convention violated")
        if op.kind != F.PREFIX and op.lbp >= F.INF:
            raise ValueError("non-prefix lbp must be below prefix sentinel")
        if op.kind == F.POSTFIX and op.rbp != F.INF:
            raise ValueError("postfix sentinel convention violated")
        if op.lbp > F.INF or op.rbp > F.INF:
            raise ValueError("binding powers exceed sentinel")
    return discover([op.lbp for op in ops], [op.rbp for op in ops], states)


def parse_group(group):
    """Fill one group's Box, returning its number of normalized operators."""
    if not group.items:
        if group.kind == "stmts":
            group.box.value = ("block",)
            return 0
        raise ParseError("empty group")
    vals, ops = F.build_flat(group.items)
    result = build_ast(vals, ops, plan_group(vals, ops))
    is_sequence = isinstance(result, tuple) and result[0] == "seq"
    if group.kind == "expr":
        F.need(result, "e", "parenthesised expression")
    elif group.kind == "then":
        F.need(result, "s", "then-branch")
        if is_sequence:
            raise ParseError("then-branch must be a single statement")
    else:
        F.need(result, "s", "statement list")
        result = ("block", *result[1:]) if is_sequence else ("block", result)
    group.box.value = result
    return len(ops)


def parse_tokens(tokens, stats=None):
    """Parse tokens, then resolve opaque group references into plain tuples."""
    groups = F.build_groups(tokens)
    work = sum(parse_group(group) for group in groups)
    if stats is not None:
        stats.update(work=work, groups=len(groups))
    return F.resolve(groups[0].box)


def parse(source, stats=None):
    """Parse source text; optionally record groups and normalized operators."""
    return parse_tokens(tokenize(source), stats)


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source")
    args = cli.parse_args()
    pprint(parse(args.source))
