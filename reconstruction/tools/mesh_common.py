"""Shared helpers for own-mesh reconstruction (own_mesh_ep0 task).

Inputs come from the recon campaign layout (/data/campaign/ep<NNNNNN>/):
  depth/frames/<f>.npz   Fast-FoundationStereo depth, rectified C (left) at 736x480, metres (float16)
  masks/ego_cam_<k>/masks_clean.npz  SAM 3 masks, packed bits, keys obj<slot>, hand
  fp/<slot>_<name>/fp_poses.npz      FoundationPose T_A_from_scan at B/C time f (A image = A(f+1))
Conventions: calibration is CAM_B-referenced; E_X = extrinsics_to_stereo_left maps camera-X coords into
camera B. Colour A is one frame late.
"""
import json
from pathlib import Path
import numpy as np, cv2

ROOT = Path('/data/project')
DATA = ROOT / 'resources/data/track3/public'
KIT = ROOT / 'resources/submission_kit/v2d_submission_kit'
import os
CAMP = Path(os.environ.get('CAMPAIGN_ROOT', '/data/campaign'))
MESHOUT = Path('/data/mesh')
DW, DH = 736, 480
A_OFFSET = 1


def epdir(ep):
    return CAMP / f'ep{ep:06d}'


def episode_info(ep):
    return json.loads((epdir(ep) / 'episode.json').read_text())


_CAL = {}
def calib():
    if _CAL:
        return _CAL
    c = json.loads((DATA / 'meta/camera_calibration.json').read_text())['cameras']
    for k in 'abc':
        e = c[f'ego_cam_{k}']
        _CAL[k] = dict(K=np.array(e['intrinsic_matrix'], float), D=np.array(e['distortion_coefficients'], float),
                       size=tuple(e['resolution']))
    EA = np.array(c['ego_cam_a']['extrinsics_to_stereo_left'], float)
    EC = np.array(c['ego_cam_c']['extrinsics_to_stereo_left'], float)
    _CAL['a']['T_B_from'] = EA; _CAL['b']['T_B_from'] = np.eye(4); _CAL['c']['T_B_from'] = EC
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(_CAL['c']['K'], _CAL['c']['D'], _CAL['b']['K'], _CAL['b']['D'], (1280, 800),
                                                EC[:3, :3], EC[:3, 3:], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    _CAL['rect'] = dict(R1=R1, P1=P1, baseline=abs(P2[0, 3] / P2[0, 0]))
    return _CAL


def rect_intrinsics():
    """pinhole K of the rectified-C depth map at model resolution."""
    P1 = calib()['rect']['P1']; sx, sy = DW / 1280, DH / 800
    return np.array([[P1[0, 0] * sx, 0, P1[0, 2] * sx], [0, P1[1, 1] * sy, P1[1, 2] * sy], [0, 0, 1.0]])


def load_depth(ep, f):
    with np.load(epdir(ep) / f'depth/frames/{f:06d}.npz') as z:
        return z['depth_m'].astype(np.float32)


def depth_points_C(depth, zmin=0.2, zmax=1.5):
    """rectified-C depth -> (N,3) points in camera-C (unrectified) coords, (N,2) model pixels (x,y)."""
    K = rect_intrinsics(); R1 = calib()['rect']['R1']
    yy, xx = np.indices(depth.shape)
    ok = np.isfinite(depth) & (depth > zmin) & (depth < zmax)
    z = depth[ok]
    X = np.stack(((xx[ok] - K[0, 2]) * z / K[0, 0], (yy[ok] - K[1, 2]) * z / K[1, 1], z), 1)
    return X @ R1, np.stack((xx[ok], yy[ok]), 1)


def project(X, cam):
    """camera-frame points -> distorted pixel coords, in-image flag."""
    z = X[:, 2]; ok = z > 0.05
    uv = np.full((len(X), 2), -1e4)
    if ok.any():
        uv[ok] = cv2.projectPoints(np.ascontiguousarray(X[ok]), np.zeros(3), np.zeros(3), cam['K'], cam['D'])[0].reshape(-1, 2)
    W, H = cam['size']
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < W - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 0.5)
    return uv, ok


def masks_clean(ep, cam):
    z = np.load(epdir(ep) / f'masks/ego_cam_{cam}/masks_clean.npz')
    H, W = int(z['H']), int(z['W'])
    arr = {k[len('packed_'):]: z[k] for k in z.files if k.startswith('packed_')}
    def get(k, f):
        return np.unpackbits(arr[k][f], count=H * W).reshape(H, W).astype(bool)
    return get, list(arr), (H, W)


def fp_poses(ep, slot, name, key='T_A_from_scan', path=None):
    """(n,4,4) object pose in camera A at B/C time f, and flags."""
    p = Path(path) if path else epdir(ep) / f'fp/{slot}_{name}/fp_poses.npz'
    z = np.load(p)
    return z[key], z['flags']




def kit_import():
    import sys
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
