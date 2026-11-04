"""Scan-free pose (sfp) shared helpers. Inputs: eval campaign layout (CAMPAIGN_ROOT, default campaign_eval):
FFS stereo depth (rectified C), SAM 3 text-prompted masks (masks_clean.npz), cuVSLAM/PnP camera track
(camera/poses.csv, world = cam B at frame 0, static-feature PnP with hands/objects masked)."""
import os, json
os.environ.setdefault('CAMPAIGN_ROOT', '/data/campaign_eval')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
from mesh_common import (epdir, episode_info, calib, rect_intrinsics, load_depth, depth_points_C, project,
                         masks_clean, DW, DH, A_OFFSET)

OUT = Path('/data/scanfree-pose')


CAM_SOURCES = {'cuvslam': 'camera/cuvslam/poses.csv', 'pnp': 'camera/poses.csv'}

def read_camera_csv(path, n):
    """cuVSLAM-schema CSV (frame,t_capture_s,status,x,y,z,qx,qy,qz,qw = world_T_camB, world = cam B at
    frame 0) -> (n,4,4) world_T_B with SLERP/linear fill over frames whose status is not ok, and the
    list of filled frames."""
    import pandas as pd
    from scipy.spatial.transform import Slerp
    df = pd.read_csv(path).sort_values('frame')
    ok = df['status'].astype(str).str.lower().isin(['ok', 'tracked', 'valid', 'true', '1', 'good'])
    v = df[ok]
    T = np.repeat(np.eye(4)[None], n, 0)
    fr = np.clip(np.arange(n), v['frame'].min(), v['frame'].max())
    T[:, :3, :3] = Slerp(v['frame'].values, R.from_quat(v[['qx', 'qy', 'qz', 'qw']].values))(fr).as_matrix()
    for i, k in enumerate('xyz'):
        T[:, i, 3] = np.interp(fr, v['frame'].values, v[k].values)
    T = np.linalg.inv(T[0]) @ T
    return T, sorted(set(range(n)) - set(v['frame'].astype(int)))

def cam_track(ep, source=None):
    """world_T_camB per frame. source: 'cuvslam' (campaign default, used by the hand fits, gravity and the bridge) or 'pnp'
    (static-feature keyframe PnP). Default = $SFP_CAM or 'pnp' (rounds 1-3 point extraction / ICP); every BUNDLE-facing world
    track must use 'cuvslam' (one camera track for cup, hands and bridge)."""
    source = source or os.environ.get('SFP_CAM', 'pnp')
    n = episode_info(ep)['n_frames']
    W_B, filled = read_camera_csv(epdir(ep) / CAM_SOURCES[source], n)
    return W_B, filled


def T_B_C():
    return calib()['c']['T_B_from']


def frame_points_world(ep, f, slot, W_B, getC, getB, erode=1, hand_dilate=5, b_dilate=5, zmax=1.5):
    """masked stereo points of object `slot` at B/C frame f, in world coords (no object pose involved)."""
    cal = calib(); EC = cal['c']['T_B_from']
    mc = getC(f'obj{slot}', f)
    if mc.sum() < 200:
        return np.zeros((0, 3)), dict(c_px=int(mc.sum()))
    mc = cv2.erode(mc.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
    hc = cv2.dilate(getC('hand', f).astype(np.uint8), np.ones((2 * hand_dilate + 1,) * 2, np.uint8)) > 0
    X, px = depth_points_C(load_depth(ep, f), zmax=zmax)
    uv, ok = project(X, cal['c'])
    ui = np.round(uv).astype(int).clip(0, [1279, 799])
    keep = ok & mc[ui[:, 1], ui[:, 0]] & ~hc[ui[:, 1], ui[:, 0]]
    mb_raw = getB(f'obj{slot}', f)
    if mb_raw.sum() >= 200:
        XB = X @ EC[:3, :3].T + EC[:3, 3]
        uvb, okb = project(XB, cal['b'])
        ub = np.round(uvb).astype(int).clip(0, [1279, 799])
        mb = cv2.dilate(mb_raw.astype(np.uint8), np.ones((2 * b_dilate + 1,) * 2, np.uint8)) > 0
        hb = cv2.dilate(getB('hand', f).astype(np.uint8), np.ones((2 * hand_dilate + 1,) * 2, np.uint8)) > 0
        keep &= okb & mb[ub[:, 1], ub[:, 0]] & ~hb[ub[:, 1], ub[:, 0]]
    X = X[keep]
    T = W_B[f] @ EC                     # world_T_C
    return X @ T[:3, :3].T + T[:3, 3], dict(c_px=int(mc.sum()), pts=int(len(X)))


