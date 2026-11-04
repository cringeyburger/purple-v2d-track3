"""spoon-mesh shared helpers (eval-only inputs: FFS stereo depth, SAM 3 masks, cuVSLAM camera track, rig calibration)."""
import os, sys, json
os.environ.setdefault('CAMPAIGN_ROOT', '/data/campaign_eval')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from mesh_common import (epdir, episode_info, calib, load_depth, depth_points_C, project, masks_clean, A_OFFSET)
from sfp_common import cam_track
ROOT = '/data/spoon-mesh'
EPS = [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]

def spoon_slot(ep):
    return episode_info(ep)['objects'].index('wooden_spoon')

def ev(ep):
    return json.load(open(f'{ROOT}/evidence/ep{ep}.json'))

def ev_pts(ep):
    z = np.load(f'{ROOT}/evidence/ep{ep}.npz'); return z['pts'].astype(np.float64), z['frame']

def table_plane(ep, frames, W_B, center, getC, slot, rad=0.25, ransac=300, thr=0.004, seed=0):
    """table plane (n, d) with n.z+d=0, n pointing to the camera side, from non-object non-hand depth points within rad of center (world)."""
    cal = calib(); EC = cal['c']['T_B_from']; rng = np.random.default_rng(seed); Q = []
    for f in frames:
        X, _ = depth_points_C(load_depth(ep, f), zmax=1.5)
        uv, ok = project(X, cal['c']); ui = np.round(uv).astype(int).clip(0, [1279, 799])
        bad = cv2.dilate((getC(f'obj{slot}', f) | getC('hand', f)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        for k in range(3):
            if f'obj{k}' != f'obj{slot}':
                try: bad |= cv2.dilate(getC(f'obj{k}', f).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                except KeyError: pass
        keep = ok & ~bad[ui[:, 1], ui[:, 0]]
        T = W_B[f] @ EC; Y = X[keep] @ T[:3, :3].T + T[:3, 3]
        Y = Y[np.linalg.norm(Y - center, axis=1) < rad]
        if len(Y) > 4000: Y = Y[rng.choice(len(Y), 4000, replace=False)]
        Q.append(Y)
    Q = np.concatenate(Q); best = (0, None)
    for _ in range(ransac):
        p = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); d = -n @ p[0]; inl = np.abs(Q @ n + d) < thr
        if inl.sum() > best[0]: best = (inl.sum(), inl)
    I = Q[best[1]]; c = I.mean(0); n = np.linalg.eigh(np.cov((I - c).T))[1][:, 0]
    camc = (W_B[frames[0]] @ EC)[:3, 3]
    if n @ (camc - c) < 0: n = -n
    res = (I - c) @ n
    return n, -n @ c, dict(n_inl=int(len(I)), rms_mm=float(1e3 * res.std()))

def table_frame(n, d, pts):
    """world->table frame: z = height above plane, x = principal axis of pts projected on the plane, origin = projected centroid."""
    P0 = pts - np.outer(pts @ n + d, n); c = P0.mean(0)
    ev_, V = np.linalg.eigh(np.cov((P0 - c).T)); x = V[:, 2] - (V[:, 2] @ n) * n; x /= np.linalg.norm(x); y = np.cross(n, x)
    R = np.stack([x, y, n]); t = -R @ c; T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T
