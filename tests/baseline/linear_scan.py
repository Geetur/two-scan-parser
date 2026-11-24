"""Direct parent links by two finite-state suffix scans.

Sequential simulation of an explicit work-efficient scan DAG. Work O(C*n),
span O(log(n+1)) with table cells processed independently, space O(C*n).
C = number of distinct rbp states. This is linear for a fixed operator table,
not a uniform linear bound for arbitrary unbounded tables. Normalized groups
only; lexing/grouping/export remain outside the parallel bound.
"""
from dataclasses import dataclass
import math
import peak_statements as P
import peak_summaries as S
import ranked_merge_reference as R


@dataclass
class Cost:
    operators: int = 0
    states: int = 0
    compositions: int = 0
    table_cells: int = 0
    plan_cells: int = 0
    node_writes: int = 0
    scan_levels: int = 0


def compose(first, second, C, cost):
    # Execute first, then second. Terminal codes >= C are absorbing.
    cost.compositions += 1
    cost.table_cells += C
    return tuple(second[x] if x < C else x for x in first)


def suffix_after(tables, C, cost):
    """Exclusive suffix scan, preserving noncommutative composition order.

    Upsweep and downsweep are organized in levels. The Python loops execute
    serially; each node at a level, and every cell of a table, is independent.
    Identity padding <2n, each internal node composed once per sweep.
    """
    n = len(tables)
    if not n:
        return []
    size = 1 << (n-1).bit_length()
    identity = tuple(range(C))
    tree = [identity] * (2*size)
    tree[size:size+n] = tables
    width = size//2
    while width:
        for i in range(width,2*width):
            tree[i] = compose(tree[2*i], tree[2*i+1], C, cost)
        width //= 2
    carry = [identity] * (2*size)
    width = 1
    while width < size:
        for i in range(width,2*width):
            carry[2*i] = compose(tree[2*i+1], carry[i], C, cost)
            carry[2*i+1] = carry[i]
        width *= 2
    cost.scan_levels += 2 * int(math.log2(size))
    return carry[size:size+n]


@dataclass
class Plan:
    pop: list
    next_same: list
    parent: list
    left: list
    right: list
    root: object
    cost: Cost


def discover(vals, ops, cost=None, states=None):
    n = len(ops)
    if len(vals) != n+1:
        raise ValueError('expected len(vals) == len(ops)+1')
    cost = cost if cost is not None else Cost()
    cost.operators += n
    if not n:
        return Plan([],[],[],[],[],None,cost)
    # This compression is preprocessing in the Python reference. For the fixed
    # language table it is a constant-sized mapping, not an unbounded sort claim.
    states = sorted({o.rbp for o in ops}) if states is None else tuple(states)
    if not states or any(a >= b for a,b in zip(states,states[1:])):
        raise ValueError('states must be a strictly increasing nonempty encoding')
    C = len(states)
    cost.states = max(cost.states,C)
    index = {v:i for i,v in enumerate(states)}
    r = [index[o.rbp] for o in ops]
    for o in ops:
        if o.kind == P.PREFIX and o.lbp != P.INF:
            raise ValueError('prefix sentinel convention violated')
        if o.kind != P.PREFIX and o.lbp >= P.INF:
            raise ValueError('non-prefix lbp must be below prefix sentinel')
        if o.kind == P.POSTFIX and o.rbp != P.INF:
            raise ValueError('postfix sentinel convention violated')
        if o.lbp > P.INF or o.rbp > P.INF:
            raise ValueError('binding powers exceed sentinel')
    # F_j(s): hit j if the current running minimum rbp beats lbp_j;
    # otherwise update minimum and continue through the suffix.
    first = [tuple(C+j if s > o.lbp else min(k,r[j])
                   for k,s in enumerate(states)) for j,o in enumerate(ops)]
    after = suffix_after(first,C,cost)
    pop = [after[i][r[i]]-C if after[i][r[i]]>=C else n for i in range(n)]
    # W_j(s): first operator in this suffix sharing the eventual hit of s.
    # This second finite-state scan gives next occurrence of each pop bucket
    # without sorting/bucketing atomics or nearest-smaller-value black boxes.
    none = C+n
    second=[]
    for j,o in enumerate(ops):
        table=[]
        for k,s in enumerate(states):
            if s > o.lbp:
                table.append(none)
            else:
                nxt=min(k,r[j])
                code=after[j][nxt]
                hit=code-C if code>=C else n
                table.append(C+j if pop[j]==hit else nxt)
        second.append(tuple(table))
    witnesses=suffix_after(second,C,cost)
    next_same=[]
    for i in range(n):
        code=witnesses[i][r[i]]
        next_same.append(code-C if C<=code<none else -1)
    # Default parent: the incoming operator that triggers this pop. Later
    # same-bucket operators instead attach to their preceding popped-stack item.
    parent=[p if p<n else -1 for p in pop]
    marked=[False]*n
    for i,j in enumerate(next_same):
        if j>=0:
            assert j>i and not marked[j]
            parent[j]=i
            marked[j]=True
    # Nonnegative children denote operators; -(k+1) denotes original operand k.
    left=[None]*n;right=[None]*n
    def put(i,child,on_left):
        target=left if on_left else right
        assert target[i] is None,('child collision',i,child)
        target[i]=child
    for i,p in enumerate(parent):
        if p>=0:put(p,i,i<p)
    root=None
    for i,p in enumerate(parent):
        if p<0:
            assert root is None
            root=i
    for k in range(n+1):
        if k==0:put(0,-1,True)
        elif k==n:put(n-1,-(k+1),False)
        elif pop[k-1]<pop[k]:put(k-1,-(k+1),False)
        else:put(k,-(k+1),True)
    assert root is not None and None not in left and None not in right
    cost.plan_cells += 2*C*n + 8*n+1
    return Plan(pop,next_same,parent,left,right,root,cost)


def construct(vals,ops,plan,write_order=None):
    if not ops:
        return vals[0],R.Arena()
    arena=R.Arena();arena.allocate(len(ops))
    errors=[]
    def value(child):
        return R.Ref(child) if child>=0 else vals[-child-1]
    def ty(child):
        return R.out_type(ops[child]) if child>=0 else P.type_of(vals[-child-1])
    order=range(len(ops)) if write_order is None else write_order(len(ops))
    for i in order:
        op=ops[i];a,b=plan.left[i],plan.right[i]
        na,nb=R.needs(op)
        if (na is not None and ty(a)!=na) or (nb is not None and ty(b)!=nb):
            errors.append(i)
        arena.op[i],arena.a[i],arena.b[i],arena.ty[i]=op,value(a),value(b),R.out_type(op)
        plan.cost.node_writes+=1
    if errors:
        raise P.ParseError('invalid operand types in direct scan AST')
    return R.Ref(plan.root),arena


def contractor_for(cost=None,write_order=None,states=None):
    cost=cost if cost is not None else Cost()
    def contractor(vals,ops,plateaus=True,trace=None):
        plan=discover(vals,ops,cost,states)
        root,arena=construct(vals,ops,plan,write_order)
        result=S.flatten_seq(arena.materialize(root))
        if trace is not None:trace.append(P._show(result))
        return result,0,len(ops)
    return contractor


if __name__ == '__main__':
    import argparse,json
    from dataclasses import asdict
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('--stats-only',action='store_true')
    args=ap.parse_args()
    cost=Cost()
    tree=P.parse(P.tokenize(args.source),contractor=contractor_for(cost))
    if not args.stats_only:print(P.pretty(tree))
    print(json.dumps(asdict(cost)))
    print('Sequential simulation; scan_levels sums scan stages, not program runtime or end-to-end parallel depth.')
