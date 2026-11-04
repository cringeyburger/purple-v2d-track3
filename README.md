# Team Purple — NVIDIA V2D Challenge, Track 3

Egocentric video → object reconstruction → per-episode dexterous policy (FlashCHORD, Soft Actor-Critic) → policy rollout → submission file.

## Layout

- `reconstruction/` — object mesh fitting and refinement from the episode video (masks, stereo depth, tracked poses).
- `runtime/` — FlashCHORD training and evaluation runtime used for all policies.
- `training/e16_weight40.yaml` — the one non-default training setting (E16 continuation).
- `postprocess/` — builds the submission file from the stored rollouts:
  - `build_submission.py` — assembles all 20 episodes and applies the two automatic pose steps (E20 frame-0 orientation, E25 board-frame flip).
  - `bridge_export.py` — maps simulator control steps to video frames.

## Reproduce

The verification container holds the checkpoints, scenes, stored rollouts and the submission kit.

```bash
/submission/run.sh verify --output /output/verify
/submission/run.sh reproduce --mode stored-repack --output /output/pack   # byte-exact submission file
/submission/run.sh reproduce --mode fresh --output /output/fresh          # re-run all policies (NVIDIA GPU)
```

Fresh GPU rollouts are not bit-exact; the stored repack reproduces the submitted file exactly.

## License

Apache-2.0 (see `LICENSE`, `NOTICE`).
