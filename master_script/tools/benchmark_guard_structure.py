"""CPU-only synthetic v1/v2 extraction and full-runtime costs in fresh processes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import numpy as np
from master_script.core.config import implementation_fingerprint
from master_script.core.guard_features import FEATURE_SCHEMA, STRUCTURE_SCHEMA, FEATURE_NAMES, parameter_features
from master_script.core.guard_runtime import GuardSettings, prepare_guard, read_guard_events


def sample(mib, workers, schema, repeats):
    # Deterministic nonzero dense and sparse updates; fixed tensor count makes
    # comparisons at two sizes test scratch growth independently of layer count.
    count = mib*1024*1024//4//16
    prior=[np.full(count,1+i/16,dtype='float32') for i in range(16)]
    values=[p.copy() for p in prior]
    for i,v in enumerate(values):
        v[::(1 if i%2 else 97)] += np.float32(.001)
    parameter_features(values,prior,schema)  # warm-up, outside timed samples
    extraction=[]
    for _ in range(repeats):
        tick=time.perf_counter(); features=parameter_features(values,prior,schema)
        extraction.append(time.perf_counter()-tick)
    tracemalloc.start()
    parameter_features(values,prior,schema)
    _,scratch_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    with tempfile.TemporaryDirectory(prefix='guard-structure-') as tmp:
        policy=json.dumps({'schema':'client_guard_v1','observation_architecture':'causal_lm'})
        settings=GuardSettings('shadow','benchmark',sha256(policy.encode()).hexdigest(),policy,repeats,
            runtime_directory=tmp,diagnostic=True,validation_workers=workers,feature_schema=schema)
        runtime=prepare_guard(settings,'synthetic',prior,rounds=repeats)
        for i in range(repeats):
            snapshot=runtime.authorize(values,0,i+1)
            if snapshot is None:raise ValueError('Benchmark request unexpectedly rejected')
            del snapshot
        events=read_guard_events(tmp)
    if any(e['features']!=features for e in events):raise ValueError('Runtime feature mismatch')
    return {'mib':mib,'workers':workers,'feature_schema':schema,'tensor_count':16,
        'extraction_seconds':extraction,'median_extraction_seconds':statistics.median(extraction),
        'runtime_seconds':[e['seconds'] for e in events],
        'median_runtime_seconds':statistics.median(e['seconds'] for e in events),
        'extraction_tracemalloc_peak_bytes':scratch_peak,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'request_sha256':events[0]['request_sha256'],'reference_sha256':events[0]['reference_sha256'],
        'features':features,'last_event':events[-1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib',type=int,nargs='+',default=[64,256])
    parser.add_argument('--workers',type=int,nargs='+',default=[1,4])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output');parser.add_argument('--child',action='store_true')
    parser.add_argument('--schema',choices=[FEATURE_SCHEMA,STRUCTURE_SCHEMA],default=FEATURE_SCHEMA)
    args=parser.parse_args()
    if any(m<=0 or m>4096 for m in args.mib) or any(w not in (1,2,4) for w in args.workers) or not 1<=args.repeats<=10:
        parser.error('Use 1–4096 MiB, workers 1/2/4, repeats 1–10')
    if args.child:
        print(json.dumps(sample(args.mib[0],args.workers[0],args.schema,args.repeats)));return
    if not args.output:parser.error('--output required')
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    samples=[]
    for mib in args.mib:
        for workers in args.workers:
            for schema in [FEATURE_SCHEMA,STRUCTURE_SCHEMA]:
                command=[sys.executable,'-m',__spec__.name,'--child','--mib',str(mib),'--workers',str(workers),
                    '--schema',schema,'--repeats',str(args.repeats)]
                samples.append(json.loads(subprocess.check_output(command,text=True)))
    for mib in args.mib:
        subset=[s for s in samples if s['mib']==mib]; first=subset[0]
        for s in subset:
            if s['request_sha256']!=first['request_sha256'] or s['reference_sha256']!=first['reference_sha256']:
                raise ValueError('Schema/worker comparison changed exact parameter identity')
            if any(s['features'][k]!=first['features'][k] for k in FEATURE_NAMES):
                raise ValueError('Legacy feature projection changed')
    report={'schema':'guard_structure_cost_v1','implementation_fingerprint':implementation_fingerprint(),
        'environment':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__},
        'repeats':args.repeats,'samples':samples,'hashes_and_v1_projection_identical':True,
        'limits':['Synthetic CPU arrays; no full-model FL, GPU, utility or privacy measurement.',
            'Runtime includes owned request copy, exact hashes, reference read/hash, features and reservation; primary audit commit is separate.',
            'Fresh-process RSS peak includes input arrays and snapshot preparation; not incremental extractor memory.',
            'tracemalloc peak is measured on extraction only, not a complete RSS or native-library peak measurement.',
            'Fixed 16-tensor layout and warm caches do not predict production concurrency, disk I/O or FL overhead.']}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps([{k:s[k] for k in ['mib','workers','feature_schema','median_extraction_seconds','median_runtime_seconds','extraction_tracemalloc_peak_bytes','process_peak_rss_bytes']} for s in samples],indent=2))


if __name__=='__main__':main()
