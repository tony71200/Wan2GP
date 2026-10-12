## Summary

Stacked on `20261001_simple_app_v2`.

```diff
 simple_app.py
 ├── UI
-│   └── orange Soft theme, peach page, orange header bar, monospace title, ⏱ ⚠ in text
+│   └── dark Base theme (zinc + one orange accent), Inter Tight / JetBrains Mono,
+│       one primary button per tab, hairline borders, no shadows, no emoji
 ├── Reels / Reels Multi Image
+│   └── "Giữ da & tóc": Detailz LoRA on the low-noise phase only (0;0.6)
+│       "Viết chi tiết" no longer zooms into the face
 ├── Motion Control
+│   ├── "Giữ âm thanh video mẫu": postprocess_audio="control"
+│   └── "Trang phục theo video mẫu": Animate 2 in UV mode (no fixed reference)
+├── keep_intermediate_sliding_windows = 0 (in memory) → only the Final Video stays
 └── harness
+    ├── --check-reel <img> <s> <prompt file> [--detail [N]] [--quality 720p] [--seed N]
+    ├── --check-multi <folder with story.txt>   (parse_story)
+    ├── --check-motion ... [--outfit]            (+ audio stream check)
+    └── logs/simple_app_harness.log + contact sheets
 GLOSSARY.md (new)
```

Spec: `docs/superpowers/specs/2026-10-03-simple-app-v3-design.md` · Plan: `docs/superpowers/plans/2026-10-03-simple-app-v3.md`

## Evidence

```text
python test_simple_app.py                                   → ok
--check-reel day77 10 prompt --seed 20261003                 → 10.09s, 24 min
--check-reel ... --detail                                    → 10.09s, 24 min (skin slightly more even)
--check-multi day_24_1/Ryo (3+4+4 s)                         → 11.03s, 13 min, passes every keyframe
--check-motion Klein drv animate2 16 --outfit                → 16.06s, audio in → audio out, 65 min
```

UI checked on the running app (1440 px screenshot): three tabs match the approved mockup; new checkboxes and guide sections present.

Known gap: "Trang phục theo video mẫu" does not make the characters take their shirts off (they mime it, shirts stay on). Detailz at 0.6 is a small improvement; a 720p + 1.0 run is recorded in the spec.

## Merge Danger

**Door:** two-way

Only `simple_app.py`, its test, docs, `GLOSSARY.md` and `.gitignore`. Wan2GP code and `wgp_config.json` are untouched; the intermediate-window setting is changed in memory by this app only.

**Blast Radius:** simple_app

Needs `loras/wan/detailz-wan.safetensors` (307 MB, not in git) for "Giữ da & tóc"; without it that option fails at job start.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
