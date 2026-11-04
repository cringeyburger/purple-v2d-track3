#!/usr/bin/env python3
"""spoon-mesh: register a spoon model to every usable frame of the 10 eval spoon episodes, CAMERA C ONLY
(C SAM 3 mask + rectified-C FFS depth; cameras A and B are never read here, so they stay held out).
Per frame: masked C stereo cloud (C mask eroded 1 px, hand dilated 5 px removed) in the cuVSLAM world; starts = previous
frame's pose + 8 PCA starts; trimmed (85 %) point-to-plane ICP of cloud -> model; score = trimmed mean distance (mm) +
20 x (1 - C silhouette IoU outside the hand mask, 1/4 res). Writes <out>/ep<N>.npz (T_world_body per frame, dist_mm,
iou_c, ok) and <out>/summary.json. usage: spoon_register.py --model M.ply --out DIR [--eps ...]"""
import argparse, json, time
from spoon_common import *
from multiprocessing import Pool
from scipy.spatial import cKDTree
import trimesh

def canon(mesh):
    V = np.asarray(mesh.vertices); c = V.mean(0); w, E = np.linalg.eigh(np.cov((V - c).T)); E = E[:, ::-1]
    if np.linalg.det(E) < 0: E[:, 2] *= -1
    Y = (V - c) @ E
    w_ = lambda s: np.ptp(Y[s][:, 1])
    if w_(Y[:, 0] < Y[:, 0].min() + 0.08) > w_(Y[:, 0] > Y[:, 0].max() - 0.08): Y = Y * np.array([-1, -1, 1])
    Y -= (Y.max(0) + Y.min(0)) / 2
    return trimesh.Trimesh(Y, mesh.faces, process=False)

def frame_cloud(ep, f, slot, W_B, gC, cal):
    mc = gC(f'obj{slot}', f)
    if mc.sum() < 1500: return None, mc
    mce = cv2.erode(mc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    hc = cv2.dilate(gC('hand', f).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    X, _ = depth_points_C(load_depth(ep, f), zmax=1.5)
    uv, ok = project(X, cal['c']); ui = np.round(uv).astype(int).clip(0, [1279, 799])
    keep = ok & mce[ui[:, 1], ui[:, 0]] & ~hc[ui[:, 1], ui[:, 0]]
    if keep.sum() < 300: return None, mc
    T = W_B[f] @ cal['c']['T_B_from']
    return X[keep] @ T[:3, :3].T + T[:3, 3], mc

def small_rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def icp(T, P, tree, Mb, Nb, iters=25, trim=0.85):
    T = T.copy()
    for _ in range(iters):
        Q = (P - T[:3, 3]) @ T[:3, :3]            # world -> body
        d, i = tree.query(Q); k = d <= np.quantile(d, trim)
        q, m, nrm = Q[k], Mb[i[k]], Nb[i[k]]
        A = np.hstack([np.cross(q, nrm), nrm]); b = ((m - q) * nrm).sum(1)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # body-frame update applied to the cloud: q' = R q + t ; pose update T <- T * inv(dT)
        dT = np.eye(4); dT[:3, :3] = small_rot(x[:3]); dT[:3, 3] = x[3:]
        T = T @ np.linalg.inv(dT)
        if np.linalg.norm(x) < 1e-6: break
    Q = (P - T[:3, 3]) @ T[:3, :3]; d = tree.query(Q)[0]
    return T, float(1e3 * np.sort(d)[:int(len(d) * trim)].mean())

def sil_iou(T, Md, mc, hand, cal, s=4):
    Tc = np.linalg.inv(cal['c']['T_B_from']) @ np.linalg.inv(WB_CUR[0]) @ T
    X = Md @ Tc[:3, :3].T + Tc[:3, 3]; uv, ok = project(X, cal['c'])
    R_ = np.zeros((800 // s, 1280 // s), np.uint8); u = (uv[ok] / s).astype(int); R_[u[:, 1], u[:, 0]] = 1
    R_ = cv2.morphologyEx(R_, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    M_ = cv2.resize(mc.astype(np.uint8), (1280 // s, 800 // s), interpolation=cv2.INTER_AREA) > 0
    H_ = cv2.resize(hand.astype(np.uint8), (1280 // s, 800 // s), interpolation=cv2.INTER_AREA) > 0
    v = ~H_; return float((R_ & M_ & v).sum() / max((( R_ | M_) & v).sum(), 1))

WB_CUR = [None]
def run(args):
    ep, model_path = args
    mesh = trimesh.load(model_path, force='mesh'); cal = calib()
    Mb, fi = trimesh.sample.sample_surface_even(mesh, 8000, seed=0) if hasattr(trimesh.sample, 'sample_surface_even') else trimesh.sample.sample_surface(mesh, 8000)
    Nb = mesh.face_normals[fi]; Md = np.asarray(trimesh.sample.sample_surface(mesh, 25000, seed=1)[0]); tree = cKDTree(Mb)
    info = episode_info(ep); n = info['n_frames']; slot = spoon_slot(ep); W_B, _ = cam_track(ep, 'cuvslam'); WB_CUR[0] = W_B[0] @ np.linalg.inv(W_B[0])
    gC, _, _ = masks_clean(ep, 'c')
    WB_CUR[0] = np.eye(4)
    Tall = np.repeat(np.eye(4)[None], n, 0); dist = np.full(n, np.nan); iou = np.full(n, np.nan); prev = None; prevf = -9
    for f in range(n):
        P, mc = frame_cloud(ep, f, slot, W_B, gC, cal)
        if P is None: continue
        if len(P) > 5000: P = P[np.random.default_rng(f).choice(len(P), 5000, replace=False)]
        # world_T_C for this frame: put B's world pose into the IoU projection
        WB_CUR[0] = W_B[f]
        hand = cv2.dilate(gC('hand', f).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        c = P.mean(0); w, E = np.linalg.eigh(np.cov((P - c).T)); e1, e2, e3 = E[:, 2], E[:, 1], E[:, 0]
        starts = []
        if prev is not None and f - prevf <= 3: starts.append(prev)
        for sx in (1, -1):
            for zax in (e3, e2):
                for sz in (1, -1):
                    x = sx * e1; z = sz * zax; y = np.cross(z, x); T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = c; starts.append(T)
        best = None
        for k, T0 in enumerate(starts):
            T, d = icp(T0, P, tree, Mb, Nb, iters=20)
            i_ = sil_iou(T, Md, mc, hand, cal); sc = d + 20 * (1 - i_) - (0.5 if (k == 0 and prev is not None and f - prevf <= 3) else 0)
            if best is None or sc < best[0]: best = (sc, T, d, i_)
        T, d = icp(best[1], P, tree, Mb, Nb, iters=15); i_ = sil_iou(T, Md, mc, hand, cal)
        Tall[f], dist[f], iou[f] = T, d, i_
        if d < 3.0 and i_ > 0.6: prev, prevf = T, f
    np.savez_compressed(f'{OUT}/ep{ep}.npz', T_world_body=Tall, dist_mm=dist, iou_c=iou)
    ok = np.isfinite(dist)
    return dict(ep=ep, frames=int(ok.sum()), dist_med=float(np.nanmedian(dist)), iou_med=float(np.nanmedian(iou)),
                good=int(((dist < 2.5) & (iou > 0.75)).sum()))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--eps', type=int, nargs='*', default=EPS); ap.add_argument('--canon', action='store_true')
    a = ap.parse_args(); OUT = a.out; os.makedirs(OUT, exist_ok=True); model = a.model
    if a.canon:
        m = canon(trimesh.load(a.model, force='mesh')); model = f'{OUT}/model_canon.ply'; m.export(model)
    t0 = time.time()
    with Pool(len(a.eps)) as p: R = p.map(run, [(e, model) for e in a.eps])
    json.dump(dict(model=model, eps=R, sec=time.time() - t0), open(f'{OUT}/summary.json', 'w'), indent=1)
    for r in R: print(r)
