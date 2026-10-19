"""Compare deterministic native resume, including RNG, occupancy and Adam state."""
import json, pathlib, sys
import numpy as np
a,b=map(pathlib.Path,sys.argv[1:3])
ma,mb=[json.loads((d/'model.json').read_text()) for d in (a,b)]
assert ma==mb, 'model manifests differ'
report={}
for p in ma['parameters']:
    for suffix in ('','.m','.v'):
        name=p['file']+suffix
        x=np.fromfile(a/name,dtype='<f4');y=np.fromfile(b/name,dtype='<f4')
        assert x.shape==y.shape
        error=float(np.abs(x-y).max(initial=0))
        report[name]=error
        assert np.allclose(x,y,rtol=2e-4,atol=2e-6), (name,error)
    name=p['file']+'.adam.json'
    assert (a/name).read_bytes()==(b/name).read_bytes(), name
sa,sb=[json.loads((d/'training.json').read_text()) for d in (a,b)]
assert sa['rng']==sb['rng'], 'RNG differs'
assert sa['occupancy']==sb['occupancy'], 'occupancy differs'
print(json.dumps(dict(max_parameter_or_moment_error=max(report.values()),rng='identical',occupancy='identical',step=ma['step']),indent=2))
