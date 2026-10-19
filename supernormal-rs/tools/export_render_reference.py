"""Capture real camera/patch fixtures and an independent PyTorch compositor oracle."""
import pathlib, sys, json, argparse
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SuperNormal_memory_optimized'))
from models.dataset_loader import Dataset
from pyhocon import ConfigFactory
p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--checkpoint',type=pathlib.Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
cfg=ConfigFactory.parse_string('data_dir="'+str(ROOT/'data/own_objects_normals/horse_960').replace('\\','/')+'"\nnormal_dir=normal_world_space_sdmunips\ncameras_name=cameras_sphere.npz\nexclude_views=[]\nupsample_factor=1')
d=Dataset(cfg)
torch.manual_seed(17);np.random.seed(17)
o,dirs,plane,inverse,normal,mask=d.gen_random_patches(3)
patches=[dict(origin=o[i,0,0].cpu().tolist(),dirs=dirs[i].reshape(9,3).cpu().tolist(),plane=plane[i].cpu().tolist(),inverse=inverse[i].reshape(9,3,3).cpu().tolist(),normals=normal[i].reshape(9,3).cpu().tolist(),mask=mask[i].flatten().cpu().tolist()) for i in range(3)]
# Also export a deterministic center patch for native image/camera ingestion verification.
v=0;cx=d.W//2;cy=d.H//2
xx,yy=torch.meshgrid(torch.arange(cx-1,cx+2,device='cuda'),torch.arange(cy-1,cy+2,device='cuda'),indexing='xy')
pix=torch.stack([xx,yy,torch.ones_like(xx)],-1).float()
ray=(d.intrinsics_all_inv[v,:3,:3]@pix[...,None])[...,0];ray=ray/ray.norm(dim=-1,keepdim=True);ray=(d.pose_all[v,:3,:3]@ray[...,None])[...,0]
camera=dict(width=d.W,height=d.H,origin=d.pose_all[v,:3,3].cpu().tolist(),dirs=ray.reshape(9,3).cpu().tolist(),normals=d.normals[v,cy-1:cy+2,cx-1:cx+2].float().reshape(9,3).tolist())
(a.out/'camera.json').write_text(json.dumps(camera))
samples=[dict(patch=k,start=2.1+j*.01,end=2.11+j*.01) for k,num in enumerate([3,1,4]) for j in range(num)]
points=[];dt=[]
for s in samples:
    pp=patches[s['patch']];dr=torch.tensor(pp['dirs']);pl=torch.tensor(pp['plane']);ratio=(dr[4]*pl).sum()/(dr*pl).sum(-1)
    points.extend((torch.tensor(pp['origin'])[None]+dr*(s['start']*ratio)[:,None]).tolist())
    points.extend((torch.tensor(pp['origin'])[None]+dr*(s['end']*ratio)[:,None]).tolist())
    dt.append((s['end']-s['start'])*ratio)
pts=torch.tensor(points,dtype=torch.float32).reshape(-1,2,3,3,3)
# Stable analytic SDF values keep opacity away from exact clipping for gradient comparisons.
sdf=(pts.norm(dim=-1)-.6).detach().requires_grad_()
variance=torch.tensor(.3,requires_grad=True)
model=None
if a.checkpoint:
    from models.fields import SDFNetwork
    model=SDFNetwork(d_in=3,d_out=1,d_hidden=64,n_layers=1,skip_in=[-1],bias=.8,geometric_init=True,weight_norm=True,input_concat=True,encoding_config=dict(n_levels=14,n_features_per_level=2,log2_hashmap_size=19,base_resolution=32,per_level_scale=1.3195079107728942)).cuda()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['sdf_network_fine']);model.bindwidth=min(14,max(1,int(np.ceil(checkpoint['iter_step']/2000))))
    # CPU compositor with autograd propagating to the CUDA field through the transfer.
    sdf=model(pts.reshape(-1,3).cuda()).cpu().reshape(-1,2,3,3);sdf.retain_grad()
    variance=checkpoint['variance_network_fine']['variance'].clone().requires_grad_()
inv=torch.exp(variance*10).clamp(1e-6,1e6);s0=sdf[:,0];s1=sdf[:,1]
c0=torch.sigmoid(s0*inv);c1=torch.sigmoid(s1*inv);alpha=((c0-c1+1e-5)/(c0+1e-5)).clamp(0,1)
dx=(pts[:,0,:,1:]-pts[:,0,:,:-1]).norm(dim=-1);dy=(pts[:,0,1:]-pts[:,0,:-1]).norm(dim=-1)
gx=torch.cat([(s0[:,:,1:2]-s0[:,:,:1])/dx[:,:,:1],(s0[:,:,2:]-s0[:,:,:-2])/(dx[:,:,:1]+dx[:,:,1:]),(s0[:,:,-1:]-s0[:,:,-2:-1])/dx[:,:,-1:]],2)
gy=torch.cat([(s0[:,1:]-s0[:,:-1])[:,:1]/dy[:,:1],(s0[:,2:]-s0[:,:-2])/(dy[:,:1]+dy[:,1:]),(s0[:,-1:]-s0[:,-2:-1])/dy[:,-1:]],1)
gt=(s1-s0)/torch.stack(dt).reshape(-1,3,3)
proj=torch.stack([gt,gx,gy],-1)
vinv=torch.tensor([patches[s['patch']]['inverse'] for s in samples]).reshape(-1,3,3,3,3)
grad=(vinv@proj[...,None])[...,0]
comps=[];masks=[];offset=0
for n in [3,1,4]:
    aa=alpha[offset:offset+n];t=torch.cat([torch.ones_like(aa[:1]),torch.cumprod((1-aa).clamp(min=1e-10),0)[:-1]],0);w=aa*t
    comps.append((w[...,None]*grad[offset:offset+n]).sum(0));masks.append(w.sum(0));offset+=n
comp=torch.stack(comps);cm=torch.stack(masks);truth=torch.tensor([p['normals'] for p in patches]).reshape(3,3,3,3);mask=torch.tensor([p['mask'] for p in patches]).reshape(3,3,3)
normal_den=mask.sum().item()+1e-5;pixel_den=27;sample_den=len(samples)*9
nl=(((comp-truth)*mask[...,None])**2).sum()/normal_den;ml=torch.nn.functional.binary_cross_entropy(cm.clamp(1e-5,1-1e-5),mask);el=((grad.norm(dim=-1)-1)**2).mean()
(nl+ml+el).backward()
fixture=dict(patches=patches,samples=samples,points=pts.reshape(-1,3).tolist(),sdf=sdf.detach().flatten().tolist(),variance=variance.item(),normal_den=normal_den,pixel_den=pixel_den,sample_den=sample_den,loss=dict(normal=nl.item(),mask=ml.item(),eikonal=el.item()),sdf_grad=sdf.grad.flatten().tolist(),variance_grad=variance.grad.item(),normals=comp.detach().reshape(-1,3).tolist(),masks=cm.detach().flatten().tolist())
(a.out/'render.json').write_text(json.dumps(fixture));print(json.dumps(fixture['loss']))
if model is not None:
    parameters=list(model.named_parameters())+[('variance',variance)]
    for name,param in parameters:
        if param.grad is not None:param.grad.detach().cpu().numpy().astype('<f4').tofile(a.out/(name.replace('.','_')+'.step.grad.bin'))
    optimizer=torch.optim.Adam([p for _,p in parameters],lr=5e-4);optimizer.step()
    for name,param in parameters:param.detach().cpu().numpy().astype('<f4').tofile(a.out/(name.replace('.','_')+'.step.value.bin'))

# Classic Lorensen-Cline lookup table distributed by scikit-image (BSD-3-Clause).
import base64
from skimage.measure import _marching_cubes_lewiner_luts as lut
table=np.frombuffer(base64.decodebytes(lut.CASESCLASSIC[1].encode()),dtype='int8').reshape(256,16)
dest=pathlib.Path(__file__).resolve().parents[1]/'src/mc_table.rs'
dest.write_text('// Classic Marching Cubes table from scikit-image (BSD-3-Clause). See THIRD_PARTY_NOTICES.md.\npub const TRIANGLES:[[i8;16];256]=[\n'+''.join(str(row.tolist())+',\n' for row in table)+'];\n')
