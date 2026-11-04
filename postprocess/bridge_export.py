"""Bridge step 4 (export): FlashCHORD step-indexed object trajectories -> dataset-frame trajectories that
the kit packer (tools/pack_submission.py) reads, in the dataset world and the submitted mesh's body frame.

  reference  (FlashCHORD venv) -- REFERENCE-ONLY stand-in for evaluate_policy.py, no policy, no simulator:
             load the bridge's ManoSharpaData parquet exactly as the evaluator plays it
             (flash_chord.data.loader.load_reference(control_fps, motion_speed)) and write it with
             FlashCHORD's own write_object_trajectories_parquet, achieved = reference, one world.
             Output rows: step 0..S-1, reference_frame = step (what the real evaluator writes), sim frame.
  frames     (any venv with numpy/scipy/pyarrow) -- step parquet (reference mode output OR a real
             evaluate_policy.py rollout) -> one row per DATASET frame 0..N-1 and object slot:
               step(f) = f (S-1)/(N-1)   (the playback grid; bridge_meta.json timing)
               --mode interp  : lerp/slerp between the two neighbouring steps (default)
               --mode nearest : nearest step
               --mode raw     : step f (what the kit packer does with an unconverted file; a TRAP, kept to measure it)
             sim -> world: world = R_align^T (sim - t_align) (rg_intermediate/.../<seq>_bridge_align.json).
             --pose achieved|reference picks the column; --world picks the eval world.
             Writes reference_frame = dataset frame, so the packer's frame grid lines up 1:1.
Usage:
  python bridge_export.py reference --variant-dir V --out V/export/steps/episode_000000.parquet
  python bridge_export.py frames --variant-dir V --rollout V/export/steps/episode_000000.parquet \
         --episode 0 --out V/export/results/episode_000000.parquet [--mode interp] [--pose achieved]
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["reference", "frames"])
ap.add_argument("--variant-dir", type=Path, required=True)
ap.add_argument("--rollout", type=Path)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--mode", choices=["interp", "nearest", "raw"], default="interp")
ap.add_argument("--pose", choices=["achieved", "reference"], default="achieved")
ap.add_argument("--world", type=int, default=0)
ap.add_argument("--checkpoint-label", default="reference-only-no-policy",
                help="written as checkpoint_sha256 (the reference-only export has no checkpoint)")
ap.add_argument("--compare-objects", type=Path, nargs="*", help="input object CSVs: report export-vs-input error per slot")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

V = a.variant_dir
meta = json.loads((V / "clip/outputs/result/bridge_meta.json").read_text())
tim = meta["timing"]
N = meta["num_frames"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731

if a.cmd == "reference":
    from flash_chord.data.loader import load_reference
    from flash_chord.evaluation.rollout import CompletionOutcome, ObjectPoseRollout, write_object_trajectories_parquet

    pq_path = glob.glob(str(V / "assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet"))
    assert len(pq_path) == 1, pq_path
    ref = load_reference(pq_path[0], control_fps=tim["control_fps"], motion_speed=tim["motion_speed"])
    pos = np.asarray(ref.object_body_pos_w())  # (S, B, 3) sim frame
    wxyz = np.asarray(ref.object_body_quat_w())  # (S, B, 4)
    pose = np.concatenate([pos, wxyz[..., [1, 2, 3, 0]]], -1)  # xyzw
    S, B = pose.shape[:2]
    assert S == tim["rollout_steps"], (S, tim["rollout_steps"])
    one = np.zeros(1)
    roll = ObjectPoseRollout(achieved_pose_w=pose[:, None].astype(np.float32), reference_pose_w=pose.astype(np.float32),
                             tracking_error=np.zeros((S, 1, 0)), tracking_error_names=(), object_vertices_o=np.zeros((B, 0, 3)),
                             body_object_ids=np.arange(B), object_body_names=tuple(ref.object_body_names()),
                             completion=CompletionOutcome(one, one.astype(bool), one.astype(bool), one),
                             control_fps=float(tim["control_fps"]), start_frame=0, non_finite_worlds=())
    prov = dict(episode_index=a.episode, reference_parquet=pq_path[0], eval_script_sha256=sha(__file__),
                eval_code_sha256=sha(Path(__file__).with_name("bridge_common.py")), checkpoint_sha256=a.checkpoint_label,
                motion_speed=tim["motion_speed"], note="REFERENCE-ONLY export by bridge_export.py (no policy, no sim)")
    write_object_trajectories_parquet(a.out, roll, provenance=prov)
    print(json.dumps(dict(out=str(a.out), steps=S, bodies=list(ref.object_body_names()), control_fps=tim["control_fps"],
                          motion_speed=tim["motion_speed"])))
    sys.exit(0)

# ------------------------------------------------------------------------------------------- frames
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

align = json.loads(next((V / "rg_intermediate/ego_recon/loaded").glob("*_bridge_align.json")).read_text())
Ra, ta = np.asarray(align["R_align"]), np.asarray(align["t_align"])
t = pq.read_table(a.rollout)
md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
cols = t.to_pydict()
env = np.asarray(cols["environment_id"]); step = np.asarray(cols["step"]); body = np.asarray(cols["object_body_id"])
names_col = np.asarray(cols["object_body_name"], dtype=object)
sel = env == a.world
if not np.any(sel):
    raise ValueError(f"requested world {a.world} is absent")
if "environment_valid" not in cols:
    raise ValueError("rollout lacks environment_valid; cannot certify the selected world")
if not np.all(np.asarray(cols["environment_valid"], dtype=bool)[sel]):
    raise ValueError(f"world {a.world} contains invalid rollout rows; refusing to relabel them valid")
if "reference_frame" not in cols or not np.array_equal(
        np.asarray(cols["reference_frame"])[sel], step[sel]):
    raise ValueError("export requires frame-0 full playback: reference_frame must equal step")
S = int(step[sel].max()) + 1
if S != int(tim["rollout_steps"]):
    raise ValueError(f"rollout has {S} steps, expected complete playback of {tim['rollout_steps']}; refusing time stretch")
assert abs(float(md.get("flash_chord.control_fps", tim["control_fps"])) - tim["control_fps"]) < 1e-9
names = meta["objects"]
B = len(names)
P = np.zeros((S, B, 7))
for k, nm in enumerate(names):  # match bodies to dataset slots BY NAME
    m = sel & (names_col == nm)
    assert m.sum() == S, f"{nm}: {m.sum()} rows for world {a.world}, expected {S} (bodies {set(names_col)})"
    o = np.argsort(step[m])
    if not np.array_equal(step[m][o], np.arange(S)):
        raise ValueError(f"{nm}: duplicate or missing control steps")
    P[:, k] = np.stack([np.asarray(cols[f"{a.pose}_{c}"])[m][o] for c in
                        ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")], -1)

f = np.arange(N, dtype=float)
x = f * (S - 1) / (N - 1) if a.mode != "raw" else f
if a.mode == "nearest":
    x = np.round(x)
x = np.clip(x, 0, S - 1)
out = np.zeros((N, B, 7))
for k in range(B):
    out[:, k, :3] = np.stack([np.interp(x, np.arange(S), P[:, k, c]) for c in range(3)], -1)
    out[:, k, 3:] = Slerp(np.arange(S), R.from_quat(P[:, k, 3:]))(x).as_quat()
# sim -> world
out[..., :3] = (out[..., :3] - ta) @ Ra
out[..., 3:] = (R.from_matrix(Ra.T) * R.from_quat(out[..., 3:].reshape(-1, 4))).as_quat().reshape(N, B, 4)
assert np.isfinite(out).all()

rows_f = np.repeat(np.arange(N), B); rows_b = np.tile(np.arange(B), N)
flat = out.reshape(-1, 7).astype(np.float32)
table = {"environment_id": pa.array(np.zeros(N * B, np.int32)),
         "step": pa.array(np.round(x).astype(np.int32)[rows_f]),
         "reference_frame": pa.array(rows_f.astype(np.int32)),
         "time_s": pa.array((rows_f / tim["capture_rate_hz"]).astype(np.float64)),
         "object_id": pa.array(rows_b.astype(np.int32)), "object_body_id": pa.array(rows_b.astype(np.int32)),
         "object_body_name": pa.DictionaryArray.from_arrays(pa.array(rows_b.astype(np.int32)), pa.array(names)),
         "environment_valid": pa.array(np.ones(N * B, bool))}
for i, c in enumerate(("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")):
    table[f"achieved_{c}"] = pa.array(flat[:, i])
newmd = {k: v for k, v in md.items()}
newmd.update({"flash_chord.episode_index": str(a.episode), "flash_chord.step_count": str(N), "flash_chord.world_count": "1",
              "bridge.export_mode": a.mode, "bridge.pose_column": a.pose, "bridge.source_rollout": str(a.rollout),
              "bridge.source_world": str(a.world), "bridge.frame_of_step": f"step = frame*({S}-1)/({N}-1)",
              "bridge.capture_rate_hz": str(tim["capture_rate_hz"]), "bridge.world": "dataset world (sim->world undone)",
              "bridge.mesh_note": meta.get("mesh_note", "")})
tb = pa.table(table).replace_schema_metadata({k.encode(): str(v).encode() for k, v in newmd.items()})
a.out.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(tb, a.out)
cmp = {}
if a.compare_objects:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bridge_common as BC
    for k, (csvp, nm) in enumerate(zip(a.compare_objects, names)):
        p, q, v = BC.read_object_csv(csvp, N)
        pe = np.linalg.norm(out[:, k, :3] - p, axis=1) * 100
        re_ = np.degrees((R.from_quat(out[:, k, 3:]).inv() * R.from_quat(q)).magnitude())
        cmp[nm] = dict(pos_err_cm_mean=float(pe[v].mean()), pos_err_cm_max=float(pe[v].max()),
                       rot_err_deg_mean=float(re_[v].mean()), rot_err_deg_max=float(re_[v].max()))
    Path(str(a.out) + ".chain.json").write_text(json.dumps(cmp, indent=1))
print(json.dumps(dict(out=str(a.out), frames=N, bodies=names, steps_in=S, mode=a.mode, pose=a.pose,
                      x_first=x[:3].round(3).tolist(), x_last=float(x[-1]), export_vs_input=cmp)))
