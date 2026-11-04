"""One capped local normal correction from dev stereo, with support gating and Laplacian regularization."""
import sys,json
import numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags,vstack
from scipy.sparse.linalg import lsqr
from common import *
ep=int(sys.argv[1]);name=sys.argv[2];d=load(ep,name);m=d['mesh'];V=m.vertices;F=m.faces;n=len(V);samples,face=trimesh.sample.sample_surface(m,80000,seed=0);tree=cKDTree(samples);N=m.vertex_normals;Ps=[];frameids=[];counts=[]
for f in frames(d,'dev'):
 P=object_points(d,f);counts.append({'f':f,'points':len(P)})
 if len(P)>400:P=P[np.random.default_rng(f).choice(len(P),400,False)]
 Ps.append(P);frameids.extend([f]*len(P))
P=np.concatenate(Ps);frameids=np.array(frameids);dist,ids=tree.query(P);fi=face[ids];closest=trimesh.triangles.closest_point(V[F[fi]],P);bc=trimesh.triangles.points_to_barycentric(V[F[fi]],closest);norm=m.face_normals[fi];target=np.sum((P-closest)*norm,axis=1);good=(dist<.008)&(abs(target)<.006);P=P[good];fi=fi[good];bc=bc[good];target=target[good];frameids=frameids[good];norm=norm[good]
coef=bc*np.einsum('nij,nj->ni',N[F[fi]],norm);A=coo_matrix((coef.ravel(),(np.repeat(np.arange(len(fi)),3),F[fi].ravel())),shape=(len(fi),n)).tocsr();support=np.bincount(F[fi].ravel(),minlength=n);fcount=np.zeros(n,int)
for i in range(n):fcount[i]=len(np.unique(frameids[(F[fi]==i).any(1)]))
allowed=(support>=10)&(fcount>=5);edges=m.edges_unique;L=coo_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),n)).tocsr();reg=diags(np.where(allowed,2.,1e6));w=np.ones(len(fi));sol=np.zeros(n)
for it in range(3):
 AA=vstack([diags(w)@A,3*L,reg]);bb=np.r_[w*target,np.zeros(len(edges)+n)];sol=lsqr(AA,bb,atol=1e-8,btol=1e-8,iter_lim=400)[0];err=A@sol-target;w=np.sqrt(np.minimum(1,.002/np.maximum(abs(err),1e-9)))
sol=np.clip(sol,-.003,.003);sol[~allowed]=0
for label,factor in [('normal_half',.5),('normal_full',1.)]:
 out=V+factor*sol[:,None]*N;np.savez_compressed(d['o']/f'{label}.npz',vertices=out,faces=F);trimesh.Trimesh(out,F,process=False).export(d['o']/f'{label}.ply')
report={'episode':ep,'name':name,'dev_frames':counts,'points_used':len(fi),'vertices':n,'supported_vertices':int(allowed.sum()),'max_displacement_mm':float(abs(sol).max()*1000),'median_displacement_mm':float(np.median(abs(sol))*1000),'face_area_observed_fraction':float(m.area_faces[np.unique(fi)].sum()/m.area),'fit_residual_mm':float(np.median(abs(A@sol-target))*1000),'baseline_residual_mm':float(np.median(abs(target))*1000),'candidate_hashes':{x:sha(d['o']/f'{x}.npz')for x in ['normal_half','normal_full']},'identity_body_transform':np.eye(4).tolist()};(d['o']/'FIT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='dev_frames'},indent=2),flush=True)
