#!/usr/bin/env python3
"""spoon-mesh: fuse the camera-C masked stereo clouds of all well-registered frames (spoon_register.py) into the spoon body
frame. Keeps frames with trimmed distance < --dmax mm and C IoU > --imin. Writes <out>/fused_ep<N>.npz (pts_body, frame,
view_dir_body = unit camera-to-point ray in body frame) for each episode. """
import argparse
from spoon_common import *
from spoon_register import frame_cloud
from multiprocessing import Pool

def run(args):
    ep, reg, out, dmax, imin = args
    z = np.load(f'{reg}/ep{ep}.npz'); T, d, iou = z['T_world_body'], z['dist_mm'], z['iou_c']
    cal = calib(); W_B, _ = cam_track(ep, 'cuvslam'); gC, _, _ = masks_clean(ep, 'c'); slot = spoon_slot(ep)
    P, F, Vd = [], [], []
    for f in np.where(np.isfinite(d) & (d < dmax) & (iou > imin))[0]:
        X, _ = frame_cloud(ep, f, slot, W_B, gC, cal)
        if X is None: continue
        Ti = np.linalg.inv(T[f]); Y = X @ Ti[:3, :3].T + Ti[:3, 3]
        cam = (W_B[f] @ cal['c']['T_B_from'])[:3, 3]; cb = Ti[:3, :3] @ cam + Ti[:3, 3]
        v = Y - cb; v /= np.linalg.norm(v, axis=1, keepdims=True)
        P.append(Y.astype(np.float32)); F.append(np.full(len(Y), f, np.int32)); Vd.append(v.astype(np.float16))
    np.savez_compressed(f'{out}/fused_ep{ep}.npz', pts_body=np.concatenate(P), frame=np.concatenate(F), view_dir_body=np.concatenate(Vd))
    return ep, len(set(np.concatenate(F).tolist())), sum(len(p) for p in P)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reg', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--eps', type=int, nargs='*', default=EPS); ap.add_argument('--dmax', type=float, default=2.5); ap.add_argument('--imin', type=float, default=0.7)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with Pool(5) as p:
        for r in p.imap_unordered(run, [(e, a.reg, a.out, a.dmax, a.imin) for e in a.eps]): print(r, flush=True)
