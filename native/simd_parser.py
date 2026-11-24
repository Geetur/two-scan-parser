#!/usr/bin/env python3
"""Convenience bridge; the C++ benchmark bypasses Python and these copies."""
import argparse, ctypes as C, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
reference=ROOT/'reference'
if not reference.exists():reference=ROOT.parent/'peak_parser_linear'
sys.path.insert(0,str(reference))
import linear_scan as L
import peak_statements as P
import peak_summaries as S

BACKENDS={'scalar':0,'simd4':1,'simd8':2,'simd16':3,'auto':4}
library=C.CDLL(str(ROOT/'libpeak_simd.so'))
ptr=C.POINTER(C.c_int)
library.peak_simd_graph.argtypes=[C.c_int,C.c_int,ptr,ptr,ptr,C.c_int,C.c_int,C.c_int,
                                ptr,ptr,ptr,ptr,ptr,ptr]
library.peak_simd_graph.restype=C.c_int
library.peak_simd_supported.restype=C.c_int


def discover(vals,ops,threads=1,backend='auto',block=4096):
    n=len(ops)
    if len(vals)!=n+1:raise ValueError('expected n + 1 operands')
    states=sorted({o.rbp for o in ops}) or [0]
    if len(states)>16:raise ValueError('SIMD implementation supports at most 16 distinct right powers')
    values=[*states,*[o.lbp for o in ops],*[o.rbp for o in ops]]
    if any(x<-(2**31) or x>=2**31 for x in values):raise ValueError('binding powers must fit int32')
    array=lambda xs:(C.c_int*len(xs))(*xs)
    q,left,right=array(states),array([o.lbp for o in ops]),array([o.rbp for o in ops])
    outputs=[(C.c_int*n)() for _ in range(5)];root=C.c_int()
    status=library.peak_simd_graph(n,len(states),q,left,right,threads,BACKENDS[backend],block,
                                  *outputs,C.byref(root))
    if status:raise RuntimeError(f'native SIMD graph error {status}')
    return L.Plan(*[list(a) for a in outputs],root.value if n else None,L.Cost(operators=n))


def contractor_for(threads=8,backend='auto',min_parallel=4096):
    def contractor(vals,ops,plateaus=True,trace=None):
        plan=discover(vals,ops,threads if len(ops)>=min_parallel else 1,backend)
        root,arena=L.construct(vals,ops,plan)
        result=S.flatten_seq(arena.materialize(root))
        if trace is not None:trace.append(P._show(result))
        return result,0,len(ops)
    return contractor


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source');ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--backend',choices=BACKENDS,default='auto')
    ap.add_argument('--min-parallel',type=int,default=4096)
    args=ap.parse_args();start=time.perf_counter()
    result=P.parse(P.tokenize(args.source),contractor=contractor_for(args.threads,args.backend,args.min_parallel))
    elapsed=time.perf_counter()-start
    print(P.pretty(result))
    print(json.dumps({'whole_python_pipeline_seconds':elapsed,
                     'scope':'includes sequential lexing, grouping, bridge copies and tuple export; use peak_bench for native measurements'}))
