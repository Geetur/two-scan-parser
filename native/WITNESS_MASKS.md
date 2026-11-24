# Exact witness masks

This optimization evaluates the same first-hit and witness functions as the
two-scan reference. It changes their native representation and evaluation
schedule; it uses no nearest-smaller or Cartesian-tree replacement.

Let `r` be operator `j`'s right-power rank, and let `after[k]` describe the
first-hit suffix strictly after `j` when entered at state `k`. Define
`event(code)` to return the terminal position, or EOF `n` for a live code.
The first scan gives `pop[j] = event(after[r])`.

The reference witness transition first returns `NONE` if `states[k] > L[j]`.
Otherwise it sets `x = min(k, r)` and either returns witness `j` when
`event(after[x]) == pop[j]`, or continues at state `x`.

There are only two cases:

- If `k >= r`, then `x == r`. The equality with `pop[j]` holds by definition,
  so the transition returns a witness (unless the earlier `NONE` rule applied).
- If `k < r`, then `x == k`. A continuing transition leaves the state unchanged.

Consequently, **every witness transition is diagonal**: each input lane returns
`NONE`, witness `j`, or its own input state `k`. No continuation changes lanes.

An exact representation therefore consists of two masks:

```text
witness[k] = (k >= r) OR (event(after[k]) == pop[j])
stop[k]    = states[k] > L[j]

G_j[k] = NONE if stop[k], else witness(j) if witness[k], else k
```

Stop has priority when both bits are set. In the native encoding, `NONE` is
`~n` and witness `j` is `~j`. With at most sixteen states, both masks fit in
one `uint32_t`; this is the same state-count limit as the previous native
implementation.

Two diagonal tables compose pointwise:

```text
compose(A, B)[k] = A[k] if A[k] is terminal, otherwise B[k]
```

The masks are computed while the first scan already holds `after[j]` in a
register. This removes the need to store the full per-token first-scan suffix
table, reload it twice to create witness transitions, or permute live states
during the second scan. The second block-summary scan and suffix expansion
still run, with the same ordered transitions and absorbing answers.

Parent defaults can also be written when the first scan obtains `pop[j]`.
Same-event overrides remain deferred until all witness answers are complete.
The subsequent parent, child, operand, and root rules are unchanged.

On one worker, the same two suffix scans are evaluated directly, right to
left. This omits block summaries, inter-worker carries, and team setup. On
multiple workers, the blocked parallel scan remains in use; the team is
capped by the number of blocks. For one operator, both suffixes are empty
identities and the entire plan is a fixed base case.

The general construction remains `O(C*n)` work. This optimization reduces
instructions, synchronization for small groups, and memory traffic. It does
not establish a `C`-independent work bound or a new historical novelty claim.
