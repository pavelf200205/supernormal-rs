"""Independent checks using the repository's actual renderer and marching cubes."""
import argparse, json, pathlib, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SuperNormal_memory_optimized'))
from models.fields import SDFNetwork, SingleVarianceNetwork
import models.renderer as renderer_module

p = argparse.ArgumentParser()
p.add_argument('--model', type=pathlib.Path, required=True)
p.add_argument('--fixture', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
p.add_argument('--mesh', type=pathlib.Path)
p.add_argument('--resolution', type=int, default=96)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((a.model/'model.json').read_text())
model = SDFNetwork(d_in=3,d_out=1,d_hidden=64,n_layers=1,skip_in=[-1],bias=.8,
    geometric_init=True,weight_norm=True,input_concat=True,
    encoding_config=dict(n_levels=14,n_features_per_level=2,log2_hashmap_size=19,
                         base_resolution=32,per_level_scale=1.3195079107728942)).cuda()
variance = SingleVarianceNetwork(.5).cuda()
params = dict(model.named_parameters()) | {'variance': variance.variance}
with torch.no_grad():
    for meta in manifest['parameters']:
        value = np.fromfile(a.model/meta['file'], dtype='<f4').copy()
        params[meta['name']].copy_(torch.from_numpy(value).reshape_as(params[meta['name']]))
model.bindwidth = manifest['active_levels']
f = json.loads((a.fixture/'render.json').read_text())
t = lambda v: torch.tensor(v, dtype=torch.float32, device='cuda')
patches = f['patches']; n = len(patches)
origins = t([v['origin'] for v in patches])[:,None,None,:].expand(n,3,3,3)
dirs = t([v['dirs'] for v in patches]).reshape(n,3,3,3)
plane = t([v['plane'] for v in patches])
inverse = t([v['inverse'] for v in patches]).reshape(n,3,3,3,3)
truth = t([v['normals'] for v in patches]).reshape(n,3,3,3)
mask = t([v['mask'] for v in patches]).reshape(n,3,3,1)
indices = torch.tensor([s['patch'] for s in f['samples']], device='cuda')
starts = t([s['start'] for s in f['samples']])[:,None]
ends = t([s['end'] for s in f['samples']])[:,None]
# Fix only the sampler so precisely the same samples feed the unmodified renderer.
original_march = renderer_module.ray_marching
renderer_module.ray_marching = lambda *args, **kw: (indices, starts, ends)
renderer = renderer_module.NeuSRenderer(model, variance, gradient_method='dfd')
result = renderer.render(origins,dirs,plane,t([0]*n),t([4]*n),inverse)
renderer_module.ray_marching = original_march
nl = (((result['comp_normal']-truth)*mask)**2).sum()/f['normal_den']
ml = F.binary_cross_entropy(result['weight_sum'].clamp(1e-5,1-1e-5),mask)
el = ((result['gradients'].norm(dim=-1)-1)**2).mean()
(nl+ml+el).backward()
f['loss'] = dict(normal=nl.item(), mask=ml.item(), eikonal=el.item())
f['normals'] = result['comp_normal'].detach().reshape(-1,3).cpu().tolist()
f['masks'] = result['weight_sum'].detach().flatten().cpu().tolist()
(a.out/'render.json').write_text(json.dumps(f))
for name,param in params.items():
    if param.grad is not None:
        param.grad.detach().cpu().numpy().astype('<f4').tofile(a.out/(name.replace('.','_')+'.step.grad.bin'))
optimizer = torch.optim.Adam(params.values(), lr=5e-4)
for meta in manifest['parameters']:
    state_path = a.model/(meta['file']+'.adam.json')
    if state_path.exists():
        param = params[meta['name']]
        optimizer.state[param] = dict(step=torch.tensor(float(json.loads(state_path.read_text()))),
            exp_avg=t(np.fromfile(a.model/(meta['file']+'.m'),dtype='<f4').copy()).reshape_as(param),
            exp_avg_sq=t(np.fromfile(a.model/(meta['file']+'.v'),dtype='<f4').copy()).reshape_as(param))
optimizer.step()
for name,param in params.items():
    param.detach().cpu().numpy().astype('<f4').tofile(a.out/(name.replace('.','_')+'.step.value.bin'))
report = dict(actual_renderer_loss=f['loss'])

if a.mesh:
    import mcubes, open3d as o3d
    from scipy.spatial import cKDTree
    # Restore weights before mesh comparison (optimizer above deliberately changed them).
    with torch.no_grad():
        for meta in manifest['parameters']:
            value = np.fromfile(a.model/meta['file'], dtype='<f4').copy()
            params[meta['name']].copy_(torch.from_numpy(value).reshape_as(params[meta['name']]))
        axis = np.linspace(-1,1,a.resolution,dtype=np.float32)
        grid = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        sdf = np.concatenate([model(t(c)).cpu().numpy().ravel() for c in np.array_split(grid, max(1,len(grid)//16384))])
    vertices,faces = mcubes.marching_cubes(-sdf.reshape((a.resolution,)*3),0.)
    vertices = vertices/(a.resolution-1)*2-1
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    labels,counts,_ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(labels)!=np.argmax(counts)); mesh.remove_unreferenced_vertices()
    native = o3d.io.read_triangle_mesh(str(a.mesh))
    v1,v2 = np.asarray(native.vertices),np.asarray(mesh.vertices)
    d1=cKDTree(v1).query(v2)[0];d2=cKDTree(v2).query(v1)[0]
    report['mesh'] = dict(native_vertices=len(v1),reference_vertices=len(v2),native_faces=len(native.triangles),
        reference_faces=len(mesh.triangles),symmetric_max=float(max(d1.max(),d2.max())),
        symmetric_mean=float((d1.mean()+d2.mean())/2))
    assert report['mesh']['symmetric_max'] < 1e-4, report['mesh']
    o3d.io.write_triangle_mesh(str(a.out/'python_reference.ply'),mesh)
(a.out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
