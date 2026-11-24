#!/usr/bin/env python3
"""Build on the CPU where the executable will run (GCC + OpenMP)."""
import argparse, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sanitize',action='store_true')
p.add_argument('--address-sanitize',action='store_true')
args=p.parse_args();root=Path(__file__).resolve().parent
flags=['-std=c++17','-O3','-march=native','-fopenmp',
       '-fno-tree-vectorize','-fno-tree-slp-vectorize','-Wall','-Wextra',
       '-Wno-maybe-uninitialized']
suffix='_asan' if args.address_sanitize else '_ubsan' if args.sanitize else ''
if args.sanitize or args.address_sanitize:
    flags[1]='-O1'
    check='address,undefined' if args.address_sanitize else 'undefined'
    flags+=['-fsanitize='+check,'-fno-sanitize-recover=all','-fno-omit-frame-pointer','-g']
cxx=os.environ.get('CXX','g++')
for files,extra,out in [(['parser.cpp','bench.cpp'],[],f'peak_bench{suffix}'),
                        (['parser.cpp'],['-fPIC','-shared'],f'libpeak_simd{suffix}.so'),
                        (['parser.cpp','batch_bench.cpp'],[],f'peak_batch{suffix}')]:
    cmd=[cxx,*flags,*extra,*[str(root/f) for f in files],'-o',str(root/out)]
    print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
