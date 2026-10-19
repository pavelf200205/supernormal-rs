"""Development-only exporter. Never required by the shipped Rust executable."""
import argparse, json, pathlib, sys, math
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SuperNormal_memory_optimized'))
from models.fields import SDFNetwork, SingleVarianceNetwork

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=pathlib.Path)
p.add_argument('--out', type=pathlib.Path, required=True)
p.add_argument('--points', type=int, default=128)
p.add_argument('--active', type=int)
a = p.parse_args()
torch.manual_seed(42)
np.random.seed(42)
cfg = dict(d_in=3,d_out=1,d_hidden=64,n_layers=1,skip_in=[-1],bias=.8,
           geometric_init=True,weight_norm=True,input_concat=True,
           encoding_config=dict(n_levels=14,n_features_per_level=2,log2_hashmap_size=19,
                                base_resolution=32,per_level_scale=1.3195079107728942))
model = SDFNetwork(**cfg).cuda()
variance = SingleVarianceNetwork(.5).cuda()
step = 0
if a.checkpoint:
    cp = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['sdf_network_fine'])
    variance.load_state_dict(cp['variance_network_fine'])
    step = cp['iter_step']
model.bindwidth = a.active if a.active is not None else min(14, max(1, math.ceil(step/2000)))
a.out.mkdir(parents=True, exist_ok=True)
manifest = dict(format_version=1, step=step, active_levels=model.bindwidth, parameters=[])
parameters = list(model.named_parameters()) + [('variance',variance.variance)]
optimizer_ids = cp['optimizer']['param_groups'][0]['params'] if a.checkpoint else []
for index, (name, param) in enumerate(parameters):
    arr = param.detach().float().cpu().numpy()
    shape = list(arr.shape)
    if len(shape)==0: shape=[1,1]
    elif len(shape)==1: shape=[1,shape[0]]
    fname = name.replace('.','_')+'.bin'
    arr.astype('<f4').tofile(a.out/fname)
    if a.checkpoint:
        state = cp['optimizer']['state'].get(optimizer_ids[index])
        if state:
            assert state['exp_avg'].shape == param.shape, (name,state['exp_avg'].shape,param.shape)
            state['exp_avg'].cpu().numpy().astype('<f4').tofile(a.out/(fname+'.m'))
            state['exp_avg_sq'].cpu().numpy().astype('<f4').tofile(a.out/(fname+'.v'))
            (a.out/(fname+'.adam.json')).write_text(json.dumps(int(state['step'])))
    manifest['parameters'].append(dict(name=name,shape=shape,file=fname))
(a.out/'model.json').write_text(json.dumps(manifest,indent=2))
x = torch.tensor(np.random.uniform(-1.1,1.1,(a.points,3)).astype('float32'),device='cuda')
y = model(x)
loss = y.square().mean()
loss.backward()
reference = dict(points=x.cpu().tolist(), sdf=y.detach().cpu().flatten().tolist(), loss=loss.item(),
                 active_levels=model.bindwidth, grad_norms={})
for name,param in model.named_parameters():
    if param.grad is not None:
        reference['grad_norms'][name] = float(param.grad.norm())
        param.grad.detach().cpu().numpy().astype('<f4').tofile(a.out/(name.replace('.','_')+'.grad.bin'))
(a.out/'reference.json').write_text(json.dumps(reference))
print(json.dumps(dict(step=step,active=model.bindwidth,loss=loss.item(),out=str(a.out))))
