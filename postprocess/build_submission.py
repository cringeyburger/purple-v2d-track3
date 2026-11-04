"""Build the Track 3 submission file from stored policy rollouts.

Steps (all automatic):
  1. Start from the packed base rollouts (one policy per episode, fixed world 0).
  2. E14, E16: take the rows of their retrained policies.
  3. E20: rotate the pitcher's frame-0 orientation toward the mean of reference frames 0-10 (rotation applied twice).
  4. E25: flip each board's body frame by 180 deg about its width axis (PCA axis 1), keeping the surface centroid fixed.
  5. Set checkpoint_sha256 from the shipped checkpoints and code_commit_url from this repository's HEAD.

Usage: python build_submission.py --inputs DIR --checkpoints DIR --output DIR [--expect FILE]
"""
import argparse, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Q

REPO_URL = "https://github.com/cringeyburger/purple-v2d-track3/commit/"
POS = [f"pos_{a}" for a in "xyz"]
QUAT = [f"quat_{a}" for a in "xyzw"]
POSE = POS + QUAT
ACH = [f"achieved_quaternion_{a}" for a in "xyzw"]
HERE = Path(__file__).resolve().parent
EXPORTER = HERE / "bridge_export.py"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export(variant, rollout, episode, pose, out):
    subprocess.run([sys.executable, "-B", str(EXPORTER), "frames", "--variant-dir", str(variant), "--rollout", str(rollout),
                    "--episode", str(episode), "--world", "0", "--mode", "interp", "--pose", pose, "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return pd.read_parquet(out)


def episode_rows(df, ep):
    return df.row_id.str.startswith(f"track_3/e{ep:06d}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect", type=Path)
    a = ap.parse_args()
    I, O = a.inputs, a.output
    O.mkdir(parents=True, exist_ok=False)

    repo = HERE.parent
    git = lambda *x: subprocess.check_output(["git", "-C", str(repo), *x], text=True).strip()
    assert not git("status", "--porcelain"), "commit all changes before building"
    commit = git("rev-parse", "HEAD")

    kit = I / "kit"
    spec = importlib.util.spec_from_file_location("auc", kit / "metric_code/track_3/AUC.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    sys.path.insert(0, str(kit))
    from v2dlb.submission_parquet import read_trajectory

    b = pd.read_parquet(I / "base.parquet")

    # 2a. E14 retrained policy.
    e14 = pd.read_parquet(I / "e14.parquet")
    m = episode_rows(b, 14)
    b.loc[m, POSE] = e14.loc[m, POSE].to_numpy()

    # 3. E20 frame-0 pitcher orientation.
    v20 = I / "e20/variant"
    ref = export(v20, I / "e20/rollout.parquet", 20, "reference", O / "e20_reference.parquet")
    ref = ref[ref.object_body_id == 1].sort_values("reference_frame")
    q = Q.from_quat(ref[ACH].to_numpy())
    d = q[:11].mean() * q[0].inv()
    delta = d * d
    align = next((v20 / "rg_intermediate/ego_recon/loaded").glob("*_bridge_align.json"))
    S = Q.from_matrix(np.array(json.loads(align.read_text())["R_align"]).T)
    tab = pq.read_table(I / "e20/rollout.parquet")
    w = tab.to_pandas()
    w = w[w.environment_id == 0].copy().astype({c: "float64" for c in ACH})
    sel = (w.step == 0) & (w.object_body_name == "water_pitcher")
    w.loc[sel, ACH] = (S.inv() * delta * S * Q.from_quat(w.loc[sel, ACH].to_numpy())).as_quat()
    pq.write_table(pa.Table.from_pandas(w, preserve_index=False).replace_schema_metadata(tab.schema.metadata), O / "e20_rollout.parquet")
    ach = export(v20, O / "e20_rollout.parquet", 20, "achieved", O / "e20_achieved.parquet")
    row = b.row_id == "track_3/e000020/w000/f000000/o1"
    b.loc[row, QUAT] = ach.loc[(ach.reference_frame == 0) & (ach.object_body_id == 1), ACH].to_numpy()

    # 4. E25 clean boards with body-frame flip.
    e25 = pd.read_parquet(I / "e25.parquet")
    m = episode_rows(b, 25)
    b.loc[m, :] = e25.loc[m, :]
    for slot in (0, 1):
        s = m & b.row_id.str.endswith(f"/o{slot}")
        v, f = M._decode_object_mesh(b.loc[s, "object_mesh_b64"].iloc[0])
        area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        cc = M._surface_moments(v, f)[0]
        P = M._sample_mesh_surface(v, f[area > 0], 6000, 0)
        _, U = np.linalg.eigh(np.cov((P - cc).T))
        F = Q.from_rotvec(np.pi * U[:, 1])
        qq = Q.from_quat(b.loc[s, QUAT].to_numpy(float))
        b.loc[s, QUAT] = (qq * F).as_quat()
        b.loc[s, POS] = b.loc[s, POS].to_numpy(float) + qq.apply(cc - F.apply(cc))

    # 2b. E16 retrained policy.
    export(I / "e16/variant", I / "e16/rollout.parquet", 16, "achieved", O / "episode_000016.parquet")
    tr = read_trajectory(O / "episode_000016.parquet")
    keys = [f"track_3/e000016/w{w_:03d}/f{f_:06d}/o{s_}" for w_, f_, s_ in zip(tr.world_index, tr.frame_index, tr.object_slot)]
    src = pd.DataFrame(tr.pose_xyzw, index=keys, columns=POSE)
    m = episode_rows(b, 16)
    b.loc[m, POSE] = src.loc[b.loc[m, "row_id"]].to_numpy()

    # 5. Provenance.
    ep = b.row_id.str.extract(r"/e0*(\d+)/")[0].astype(int)
    hashes = {e: sha(a.checkpoints / f"episode_{e:06d}.safetensors") for e in sorted(ep.unique())}
    b["checkpoint_sha256"] = ep.map(hashes)
    b["code_commit_url"] = REPO_URL + commit

    out = O / "track3_submission.parquet"
    b.to_parquet(out, index=False)
    for p in O.glob("e*_*.parquet"):
        if p != out:
            p.unlink()
    if a.expect is not None:
        x = pd.read_parquet(a.expect)
        keep = [c for c in b.columns if c not in ("checkpoint_sha256", "code_commit_url")]
        pd.testing.assert_frame_equal(pd.read_parquet(out)[keep], x[keep], check_exact=True)
    print(json.dumps({"output": str(out), "sha256": sha(out), "commit": commit, "e20_delta_deg": float(np.degrees(delta.magnitude()))}))


if __name__ == "__main__":
    main()
