#!/usr/bin/env python3
"""Differential regression against the original Python algorithm and language."""
import json, random, sys, time
from pathlib import Path
import simd_parser as N
L,P=N.L,N.P
sys.path.insert(0,str(N.reference))
import ranked_merge_reference as R

def run():
    start=time.perf_counter();rng=random.Random(7301);groups=0;calls=0
    def checked(vals,ops):
        nonlocal groups,calls
        expected=L.discover(vals,ops)
        for threads in ([1,4] if groups%20==0 else [1]):
            got=N.discover(vals,ops,threads)
            for field in ('pop','next_same','parent','left','right','root'):
                assert getattr(got,field)==getattr(expected,field),(groups,threads,field)
            calls+=1
        groups+=1
    for items in R.enumerate_flats(6):
        try:vals,ops=P.build_flat(items)
        except P.ParseError:continue
        checked(vals,ops)
    for _ in range(1000):
        ops=[P.Op('o',P.INFIX,rng.randrange(-10,11),rng.randrange(-7,8)) for _ in range(rng.randrange(101))]
        checked(['x']*(len(ops)+1),ops)
    print('Python exact graph comparisons:',groups,calls,flush=True)
    accepted=rejected=0
    for i in range(1000):
        tokens=P.gen_program(rng)
        if i%3:tokens=P.mutate(tokens,rng)
        expected=P.outcome(lambda t:P.RD(t).program(),tokens)
        try:
            ast=P.parse(tokens,contractor=N.contractor_for(4,min_parallel=0))
            actual=P.flatten_tree(ast)
        except P.ParseError:actual='ERR'
        assert expected==actual,(' '.join(tokens),expected,actual)
        accepted+=actual!='ERR';rejected+=actual=='ERR'
    result=dict(passed=True,normalized_groups=groups,native_calls=calls,programs=1000,
                accepted=accepted,rejected=rejected,elapsed_seconds=time.perf_counter()-start)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    sys.setrecursionlimit(20000)
    result=run()
    (Path(__file__).resolve().parent/'results'/'python_verification.json').write_text(json.dumps(result,indent=2)+'\n')
