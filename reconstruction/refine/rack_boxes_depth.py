"""max own interior q95, all-reference spoon depth+2mm.
Input arrays: observed interior XYZ in C camera metres; own runtime reference
position[T,2,3] and quaternion_wxyz[T,2,4], rack slot0/spoon slot1;
stock spoon vN[vertices,3]/fN[triangles,3] in native body metres.
Output native stock-box centers[3], halfextents[3], rotation[3,3], same schema.
"""
from pathlib import Path
import json,hashlib,copy,numpy as np,trimesh
from scipy.spatial.transform import Rotation
R=Path('/data/recon');O=R/'work/rack_boxes_depth_v2';O.mkdir(exist_ok=True);assert not (O/"BOXES.json").exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bp=R/'work/rack_boxes/BOXES.json';b=json.load(open(bp));assert sha(bp)=='799ac27fbe7f7799a40e6edea69aa444afcb687cbd2ba83b7361c3663f69193a'
A=Path('/data/e16_cavity');rp=A/'gate_v4/cpu/ACTUAL_REFERENCE.npz';sp=A/'STOCK_IMPORTED_V3_wooden_spoon.npz';ep=R/'work/rack_cavity/EVIDENCE.npz';gp=R/'verification/e16_plastic_dish_rack.json'
z=np.load(ep);n=np.array(json.load(open(gp))['rows'][0]['plane_c'])[:3];height=float(np.quantile(z['rim_stereo']@n,.9));depths=height-z['inner_stereo']@n;q95=float(np.quantile(depths,.95))
ref=np.load(rp);pos=ref['position'];rot=Rotation.from_quat(ref['quaternion_wxyz'].reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(pos),2,3,3);assert pos.shape==(344,2,3)
s=np.load(sp);basis=np.array(b['native_from_aperture_rotation']);center=np.array(b['aperture_center_native_m']);a,bb=np.array(b['inner_aperture_mm'])/2000
# Clip every surface triangle to the infinite XY pocket prism; minimum z of
# clipped polygon is exact for planar triangle. No sampled vertices shortcut.
def clip(poly,axis,sign,lim):
 out=[]
 for i,p in enumerate(poly):
  prev=poly[i-1];dp=sign*p[axis]-lim;dq=sign*prev[axis]-lim
  if (dp<=0)!=(dq<=0):out.append(prev+(p-prev)*dq/(dq-dp))
  if dp<=0:out.append(p)
 return np.array(out)
rows=[]
for t in range(len(pos)):
 deepest=None;witness=None
 for k in range(int(s['n'])):
  v=s[f'v{k}'];world=v@rot[t,1].T+pos[t,1];native=(world-pos[t,0])@rot[t,0];local=(native-center)@np.linalg.inv(basis).T
  faces=s[f'f{k}'];tri=local[faces];possible=(tri[:,:,0].min(1)<=a)&(tri[:,:,0].max(1)>=-a)&(tri[:,:,1].min(1)<=bb)&(tri[:,:,1].max(1)>=-bb)
  for fi in np.flatnonzero(possible):
   poly=tri[fi]
   for axis,sign,lim in [(0,1,a),(0,-1,a),(1,1,bb),(1,-1,bb)]:
    poly=clip(poly,axis,sign,lim)
    if not len(poly):break
   if len(poly):
    pt=poly[np.argmin(poly[:,2])];dep=-float(pt[2])
    if deepest is None or dep>deepest:deepest=dep;witness=dict(part=k,face=int(fi),local_xyz_m=pt.tolist())
 rows.append(dict(control=t,native_phase=t*172/343,depth_m=deepest,witness=witness))
valid=[r for r in rows if r['depth_m']is not None];worst=max(valid,key=lambda r:r['depth_m']);depth=max(q95,worst['depth_m']+.002)
# Declared bound only. Do not silently extend outside the existing outer floor.
outer_floor_depth=-b['boxes'][0]['local_high_m'][2]
outer_floor_overlap=max(0.,depth+.002-outer_floor_depth)
b2=copy.deepcopy(b);b2.pop('median_visible_depth_prior_mm');b2['pocket_depth_prior_mm']=depth*1000;b2['depth_rule']='max(q95 own visible interior depth, deepest all344 own spoon reference surface inside XY aperture +2mm clearance)';b2['previous_candidate_sha256']=sha(bp);b2['limits']=[x.replace('median-depth','own-data-bound').replace('medvisible depth is not confirmedfloor','observed q95 is not confirmed floor; reference+2mm is an engineering free-space constraint')for x in b2['limits']]
for box in b2['boxes']:
 if not box['name'].startswith('pocket_'):continue
 low=np.array(box['local_low_m']);high=np.array(box['local_high_m']);low[2]=-depth-(.002 if box['name']=='pocket_floor'else 0)
 if box['name']=='pocket_floor':high[2]=-depth
 box.update(local_low_m=low.tolist(),local_high_m=high.tolist(),center_native_m=(center+basis@((low+high)/2)).tolist(),half_extents_m=((high-low)/2).tolist(),provenance='Own aperture, depth from frozen q95/reference+2mm rule; flat floor and 2 mm walls are engineering priors.')
payload={'n':np.array(len(b2['boxes']))};parts=[]
for k,box in enumerate(b2['boxes']):
 T=np.eye(4);T[:3,:3]=box['rotation_native_from_box'];T[:3,3]=box['center_native_m'];p=trimesh.creation.box(extents=2*np.array(box['half_extents_m']),transform=T);parts.append(p);payload[f'v{k}']=p.vertices;payload[f'f{k}']=p.faces
np.savez_compressed(O/'box_convex.npz',**payload);trimesh.util.concatenate(parts).export(O/'collision_only.obj');b2['convex_sha256']=sha(O/'box_convex.npz');b2['source_pins'].update({str(p):sha(p)for p in[bp,rp,sp,Path(__file__)]});(O/'BOXES.json').write_text(json.dumps(b2,indent=2)+'\n')
result=dict(status='DATA_BOUND_GEOMETRY_READY_NOT_DYNAMIC_PASS',observed_depth_quantiles_mm=(np.quantile(depths,[0,.5,.9,.95,1])*1000).tolist(),observation_count=len(depths),worst_reference=worst,reference_controls=len(rows),controls_with_pocket_intersection=len(valid),selected_depth_mm=depth*1000,outer_floor_depth_mm=outer_floor_depth*1000,outer_floor_overlap_depth_mm=outer_floor_overlap*1000,quantile_method="numpy linear",inside_predicate="abs(aperture_x)<=50.736681752268725mm and abs(aperture_y)<=19.785144900529045mm; no z cutoff",changed_boxes=[x["name"]for x in b2["boxes"]if x["name"].startswith("pocket_")],clearance_prior_mm=2,rows=rows,boxes_sha256=sha(O/'BOXES.json'),convex_sha256=b2['convex_sha256'],limits=['Own stereo interior samples include walls; q95 is a free-space depth bound, not identified floor.','Own reconstructed spoon/reference defines conditional clearance; errors in either remain possible.','All344 own reference controls tested; exact triangle clipping against aperture, not hidden score selection.','Same ten boxes and native frame, outer boxes unchanged. Floor thickness2mm and shape remain priors.'])
(O/'DEPTH_BOUND.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k!='rows'})
