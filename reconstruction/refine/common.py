"""Own staged observations only; no global campaign helper with implicit public inputs."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2,trimesh
R=Path('/data/recon')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(ep,name):
 o=R/f'e{ep}_{name}';info=json.load(open(o/'INPUTS.json'));p=Path(info['post']);reg=Path(info['regen']);cal=json.load(open(reg/'stage/calib/camera_calibration.json'))['cameras'];cams={};Es={};masks={}
 for c in 'abc':
  v=cal['ego_cam_'+c];cams[c]={'K':np.asarray(v['intrinsic_matrix'],float),'D':np.asarray(v['distortion_coefficients'],float),'size':tuple(v['resolution'])};Es[c]=np.asarray(v.get('extrinsics_to_stereo_left',np.eye(4)),float)
  z=np.load(reg/f'output/ep{ep:06d}/masks/ego_cam_{c}/masks_clean.npz');masks[c]=(int(z['H']),int(z['W']),{k[7:]:z[k]for k in z.files if k.startswith('packed_')})
 tim=json.load(open(reg/'stage/timing/timing.json'))['episodes'][str(ep)]
 if (p/'pts/points.npz').exists():
  z=np.load(p/'pts/points.npz');pt={k:z[k]for k in z.files};z=np.load(p/'track_final.npz');track={k:z[k]for k in z.files}
 else:
  # E8: fresh own-model FP, expressed in each frame's C camera; no stale public-frame rebase.
  z=np.load(reg/f'output/ep{ep:06d}/fp_own/{info["slot"]}_{name}/fp_poses.npz');ta=z['T_A_from_scan'];track={'world_T_obj':np.einsum('ij,njk->nik',np.linalg.inv(Es['c'])@Es['a'],ta),'flags':z['flags']};n=len(ta)
  pp=o/'fresh_points.npz'
  if not pp.exists():
   Ec=Es['c'];rc,rb,pc,pb,*_=cv2.stereoRectify(cams['c']['K'],cams['c']['D'],cams['b']['K'],cams['b']['D'],cams['c']['size'],Ec[:3,:3],Ec[:3,3:],flags=cv2.CALIB_ZERO_DISPARITY,alpha=0);pts=[];ff=[]
   for f in range(n):
    dz=np.load(reg/f'output/ep{ep:06d}/depth/frames/{f:06d}.npz')['depth_m'].astype(float);h,w=dz.shape;K=pc[:3,:3].copy();K[0]*=w/1280;K[1]*=h/800;y,x=np.indices(dz.shape);ok=np.isfinite(dz)&(dz>.2)&(dz<1.5);d=dz[ok];X=np.stack([(x[ok]-K[0,2])*d/K[0,0],(y[ok]-K[1,2])*d/K[1,1],d],1)@rc
    H,W,pack=masks['c'];mo=np.unpackbits(pack[f'obj{info["slot"]}'][f],count=H*W).reshape(H,W).astype('uint8');hand=np.unpackbits(pack['hand'][f],count=H*W).reshape(H,W).astype('uint8');mo=cv2.erode(mo,np.ones((5,5),np.uint8))&~cv2.dilate(hand,np.ones((9,9),np.uint8));uv=cv2.projectPoints(X,np.zeros(3),np.zeros(3),cams['c']['K'],cams['c']['D'])[0].reshape(-1,2);u=np.rint(uv).astype(int);valid=(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H);ids=np.flatnonzero(valid);keep=ids[mo[u[ids,1],u[ids,0]]>0];pts.append(X[keep].astype('float32'));ff.append(np.full(len(keep),f,dtype='int16'))
   np.savez_compressed(pp,ptsC=np.concatenate(pts),frame=np.concatenate(ff),world_T_C=np.repeat(np.eye(4)[None],n,axis=0))
  z=np.load(pp);pt={k:z[k]for k in z.files}
 z=np.load(o/'baseline_body.npz');mesh=trimesh.Trimesh(z['vertices']+info['centre_m'],z['faces'],process=True);mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices()
 return dict(o=o,info=info,reg=reg,cams=cams,Es=Es,masks=masks,tim=tim,pt=pt,track=track,mesh=mesh)
def mask(data,c,f,div=4):
 H,W,p=data['masks'][c];size=(W//div,H//div);return{k:cv2.resize(np.unpackbits(a[f],count=H*W).reshape(H,W).astype('uint8'),size,interpolation=cv2.INTER_NEAREST)>0 for k,a in p.items()}
def frames(d,part):
 for f,flag in enumerate(d['track']['flags']):
  if str(flag)not in ['static','track','register']:continue
  if part=='dev' and (f//20)%2!=0:continue
  if part=='heldout' and (f//20)%2!=1:continue
  yield f
def object_points(d,f):
 X=d['pt']['ptsC'][d['pt']['frame']==f];H=np.linalg.inv(d['track']['world_T_obj'][f])@d['pt']['world_T_C'][f];return X@H[:3,:3].T+H[:3,3]
def view(d,f,c,div=4):
 lag=d['tim']['a_lag_per_frame'][f]if c=='a'else 0;fi=f+lag
 H0,W,pa=d['masks'][c]
 if not 0<=fi<len(next(iter(pa.values()))):return None
 M=mask(d,c,fi,div);own=f'obj{d["info"]["slot"]}';block=M.get('hand',np.zeros_like(M[own])).copy()
 for k,v in M.items():
  if k.startswith('obj') and k!=own:block|=v&~M[own]
 block=cv2.dilate(block.astype('uint8'),np.ones((3,3),np.uint8))>0;obs=M[own]&~block
 H=np.linalg.inv(d['Es'][c])@d['Es']['c']@np.linalg.inv(d['pt']['world_T_C'][f])@d['track']['world_T_obj'][f]
 cam=dict(d['cams'][c]);cam['K']=cam['K'].copy();cam['K'][:2]/=div;cam['size']=(W//div,H0//div)
 return H,cam,obs,block,fi
def raster(mesh,H,cam,shape):
 X=mesh.vertices@H[:3,:3].T+H[:3,3];assert np.isfinite(X).all() and (X[:,2]>.05).all();uv=cv2.projectPoints(X,np.zeros(3),np.zeros(3),cam['K'],cam['D'])[0].reshape(-1,2);out=np.zeros(shape,'uint8')
 for tri in uv[mesh.faces]:cv2.fillConvexPoly(out,np.ascontiguousarray(np.rint(tri*16),dtype='int32'),1,shift=4)
 return out>0
