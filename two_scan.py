"""Exact tree links from two exclusive suffix scans of binding-power tables.

Input:
    lbp[j]: left binding power of operator j
    rbp[j]: right binding power of operator j
    n operators imply n+1 implicit operand slots

Output:
    Plan.pop[j]: incoming operator index that removes operator j
    Plan.next_same[j]: next operator removed at the same event, or -1
    Plan.parent[j]: parent operator index, or -1 for the root
    Plan.left[j] / Plan.right[j]: child operator indices or operand slots
    operand slot k is encoded as -(k+1)
    Plan.root: root operator index, or None for an empty input

This Python runs sequentially. Its scan DAG has O(C*n) work/space and
O(log(n+1)) span with independent table cells and supplied state encoding.
"""
from dataclasses import dataclass


@dataclass
class Plan:
    pop: list[int]
    next_same: list[int]
    parent: list[int]
    left: list[int]
    right: list[int]
    root: int | None


def compose(first, second):
    """Apply first, then second.

    Codes below C are live states; codes at or above C are absorbing answers.
    """
    C = len(first)
    return tuple(second[code] if code < C else code for code in first)


def suffix_after(tables):
    """Blelloch scan: result[i] composes tables strictly AFTER position i.

    Each inner loop is one parallel level; table order is never reversed.
    Equal-width tables use live states 0..C-1 and absorbing codes >= C.
    """
    n = len(tables)
    if not n:
        return []
    size = 1 << (n - 1).bit_length()
    identity = tuple(range(len(tables[0])))
    tree = [identity] * (2 * size)
    tree[size:size + n] = tables

    width = size // 2
    while width:                              # Upsweep: interval summaries.
        for node in range(width, 2 * width):
            tree[node] = compose(tree[2 * node], tree[2 * node + 1])
        width //= 2

    # Reuse summaries as suffix carries; tables are immutable tuples.
    tree[1] = identity
    width = 1
    while width < size:                       # Downsweep: suffix carries.
        for node in range(width, 2 * width):
            carry = tree[node]
            tree[2 * node] = compose(tree[2 * node + 1], carry)
            tree[2 * node + 1] = carry
        width *= 2
    return tree[size:size + n]


def discover(lbp, rbp, states=None):
    """Find the exact binary tree for the strict stack rule R_top > L_next.

    states may supply a strictly increasing superset of all right powers.
    Otherwise a sorted set is built as preprocessing. EOF is position n.
    Operators are nodes 0..n-1; negative children denote original operands.
    """
    n = len(lbp)
    if len(rbp) != n:
        raise ValueError("expected equal numbers of left and right powers")
    if not n:
        return Plan([], [], [], [], [], None)
    states = sorted(set(rbp)) if states is None else tuple(states)
    if not states or any(a >= b for a, b in zip(states, states[1:])):
        raise ValueError("states must be strictly increasing and nonempty")
    C = len(states)
    index = {power: k for k, power in enumerate(states)}
    rank = [index[power] for power in rbp]

    def hit(code):
        return code - C if code >= C else n

    # 1. F_j: stop at j, or lower the running-minimum state and continue.
    first = [
        tuple(C + j if power > lbp[j] else min(k, rank[j])
              for k, power in enumerate(states))
        for j in range(n)
    ]
    after = suffix_after(first)
    pop = [hit(after[i][rank[i]]) for i in range(n)]

    # 2. G_j: return the first operator sharing the state's eventual hit.
    # after[j] ALREADY represents H_(j+1); do not use after[j+1].
    none = C + n
    second = []
    for j in range(n):
        table = []
        for k, power in enumerate(states):
            if power > lbp[j]:
                table.append(none)
            else:
                state = min(k, rank[j])
                event = hit(after[j][state])
                table.append(C + j if pop[j] == event else state)
        second.append(tuple(table))
    witnesses = suffix_after(second)
    next_same = []
    for i in range(n):
        code = witnesses[i][rank[i]]
        next_same.append(code - C if C <= code < none else -1)

    # 3. Default parents, then disjoint same-event overrides.
    parent = [event if event < n else -1 for event in pop]
    assigned = [False] * n
    for i, following in enumerate(next_same):
        if following >= 0:
            assert following > i and not assigned[following]
            parent[following] = i
            assigned[following] = True

    # 4. Operator edges, then the n+1 original operand edges.
    left, right = [None] * n, [None] * n

    def put(children, node, child):
        assert children[node] is None, "child slot assigned twice"
        children[node] = child

    root = None
    for i, p in enumerate(parent):
        if p < 0:
            assert root is None
            root = i
        else:
            put(left if i < p else right, p, i)
    put(left, 0, -1)
    put(right, n - 1, -(n + 1))
    for k in range(1, n):
        if pop[k - 1] < pop[k]:
            put(right, k - 1, -(k + 1))
        else:
            put(left, k, -(k + 1))
    assert root is not None and None not in left and None not in right
    return Plan(pop, next_same, parent, left, right, root)


if __name__ == "__main__":
    print(discover([30, 40, 30], [31, 41, 31]))  # a + b * c - d
