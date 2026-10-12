## Summary

```diff
 simple_app.py
 ├── Reels
-│   └── 2–5 s, Lightning, one window
+│   ├── 2–5 s  → Lightning, one window
+│   └── 6–15 s → SVI 2 Pro, one job, windows ≤ 5 s anchored on the original image,
+│                prompt split per window (split_timeline)
+├── Reels Multi Image (new)
+│   └── 1–6 images, per-image duration + prompt → one Lightning job (SE + PW + [/duration])
 ├── Motion Control
-│   └── Animate 2, capped by a slider, 30 fps driving video
+│   ├── length = driving video unless "Giới hạn thời lượng" is ticked
+│   ├── driving video resampled to 16 fps (+ RIFE x2)
+│   └── model: Animate 2 distilled | Wan 2.2 Animate 14B + FusionX
+├── self-calibrating time estimate (simple_app_timings.json)
+├── Upscale ×2 (Lanczos) on every tab
+└── guide dialog (methods, Pha đầu / Pha cuối, LoRA sources)
```

Spec: `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` · Plan: `docs/superpowers/plans/2026-10-01-simple-app-v2.md`

## Evidence

- **Before:** Reels capped at 5 s; Motion Control cut to the slider (2–10 s).
  **After:** headless checks assert the output duration:

```text
--check-reel   Ryo.png 15                 → duration 15.09s (483 frames @ 32 fps), expected 15.00s   36 min
--check-motion Klein.png drv.mp4 animate 5 → duration 5.06s  (81 frames @ 16 fps),  expected 5.00s    17 min
--check-motion Klein.png drv.mp4           → duration 27.06s (433 frames @ 16 fps), expected 27.10s   1 h 50 min
python test_simple_app.py                  → ok
```

UI checked in the browser: three tabs, guide dialog open/close, estimate updates (15 s → "SVI 2 Pro · 3 cửa sổ × 8 bước"), Multi add/remove shifts slots, Motion limit checkbox enables the slider.

Not yet exercised: a Multi Image generation, Upscale ×2 on a real video.

## Merge Danger

**Door:** two-way

Only `simple_app.py`, its test, docs and `.gitignore` change; Wan2GP itself is untouched.

**Blast Radius:** simple_app

Motion Control now defaults to the whole driving video at 16 fps, so long driving videos take hours (the estimate line shows it before starting).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
