from simple_app import (
    REELS_MODEL, EXTEND_MODEL, MOTION_MODEL, MOTION_DEFAULT_PROMPT,
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
    reels_settings, extend_settings, motion_settings,
    reel_plan, reel_windows, split_timeline, ideas_instructions, detail_instructions,
    multi_settings, aspect_mismatch, MAX_IMAGES,
    ANIMATE_MODEL, FUSIONX_LORA, MOTION_MODELS, MOTION_WINDOWS, motion_frames, sliding_windows,
    estimate_seconds, fmt_minutes, eta_text, load_calib, save_calib,
)


def test_pick_resolution():
    assert pick_resolution(1080, 1920) == "480x832"
    assert pick_resolution(1920, 1080) == "832x480"
    assert pick_resolution(1000, 1000) == "720x720"
    assert pick_resolution(1000, 1050) == "720x720"
    assert pick_resolution(1080, 1920, "720p") == "720x1280"
    assert pick_resolution(1920, 1080, "720p") == "1280x720"
    assert pick_resolution(1024, 1024, "720p") == "960x960"


def test_frames_for():
    assert frames_for(2) == 33
    assert frames_for(3) == 49
    assert frames_for(4) == 65
    assert frames_for(5) == 81
    assert frames_for(0.5) == 17
    assert frames_for(10) == 81


def test_round_frames():
    assert round_frames(150) == 149
    assert round_frames(65) == 65
    assert round_frames(68) == 65
    assert round_frames(3) == 5


def test_pair_loras():
    names = [
        "Twerking_I2V_high_noise.safetensors",
        "Twerking_I2V_low_noise.safetensors",
        "slow_motion.safetensors",
        "transition_v2.0_H.safetensors",
        "transition_v2.0_L.safetensors",
        "wan2.2_i2v_lownoise_pov.safetensors",
        "x_HN.safetensors",
    ]
    activated, mults = pair_loras(names, 1.0, 0.5)
    assert activated == names
    assert mults == "1;0 0;0.5 1;0.5 1;0 0;0.5 0;0.5 1;0"
    assert pair_loras([], 1, 0) == ([], "")
    assert pair_loras(["a_high.safetensors"])[1] == "1;0"
    assert pair_loras(["a_low.safetensors"])[1] == "0;0"


def test_parse_ideas():
    text = "1. Wind blows her hair\n2) She turns around\n\n- She smiles\n4. extra"
    assert parse_ideas(text) == ["Wind blows her hair", "She turns around", "She smiles"]
    assert parse_ideas("") == []
    assert parse_ideas(None) == []
    assert parse_ideas('"Only one idea"') == ["Only one idea"]


def test_reels_settings():
    s = reels_settings(
        {"num_inference_steps": 4, "prompt": "default", "model_type": "x"},
        image="a.png", prompt="p", resolution="480x832", seconds=5, seed=-1,
        loras=["x_high_noise.safetensors", "x_low_noise.safetensors"], high=1, low=0.3, smooth=True,
    )
    assert s["model_type"] == REELS_MODEL
    assert s["num_inference_steps"] == 4
    assert s["prompt"] == "p"
    assert s["image_prompt_type"] == "S" and s["image_start"] == "a.png"
    assert s["resolution"] == "480x832" and s["video_length"] == 81
    assert s["seed"] == -1
    assert s["activated_loras"] == ["x_high_noise.safetensors", "x_low_noise.safetensors"]
    assert s["loras_multipliers"] == "1;0 0;0.3"
    assert s["temporal_upsampling"] == "rife*2"
    assert reels_settings({}, image="a.png", prompt="p", resolution="480x832", seconds=2, seed=7, smooth=False)["temporal_upsampling"] == ""


def test_extend_settings():
    s = extend_settings(
        {"num_inference_steps": 8}, video="v.mp4", anchor="a.png", prompt="p",
        resolution="480x832", seconds=3, seed=5,
    )
    assert s["model_type"] == EXTEND_MODEL
    assert s["num_inference_steps"] == 8
    assert s["image_prompt_type"] == "V" and s["video_source"] == "v.mp4"
    assert s["video_prompt_type"] == "KI" and s["image_refs"] == ["a.png"]
    assert s["image_start"] is None
    assert s["video_length"] == 45 and s["seed"] == 5
    assert s["activated_loras"] == [] and s["loras_multipliers"] == ""
    assert s["temporal_upsampling"] == "rife*2"


def test_motion_settings():
    defaults = {"prompt": "Character appearance: default", "num_inference_steps": 10, "video_prompt_type": "UV", "image_prompt_type": "S"}
    s = motion_settings(defaults, image="c.png", video="d.mp4", description="", resolution="480x832", frames=149, seed=-1)
    assert s["model_type"] == MOTION_MODEL
    assert s["num_inference_steps"] == 10
    assert s["prompt"] == MOTION_DEFAULT_PROMPT
    assert s["image_prompt_type"] == "T" and s["image_start"] is None
    assert s["video_prompt_type"] == "UVI" and s["image_refs"] == ["c.png"]
    assert s["video_guide"] == "d.mp4" and s["force_fps"] == "16"
    assert s["video_length"] == 149 and s["resolution"] == "480x832"
    assert s["temporal_upsampling"] == ""
    assert motion_settings(defaults, image="c.png", video="d.mp4", description=" red hair ", resolution="480x832", frames=65, seed=1)["prompt"] == "red hair"

    a = motion_settings(
        {"num_inference_steps": 30, "guidance_scale": 5}, image="c.png", video="d.mp4", description="",
        resolution="480x832", frames=433, seed=2, model=ANIMATE_MODEL, smooth=True,
    )
    assert a["model_type"] == "animate"
    assert a["image_prompt_type"] == "" and a["video_prompt_type"] == "PVBKI"
    assert a["image_refs"] == ["c.png"] and a["video_guide"] == "d.mp4"
    assert a["activated_loras"] == [FUSIONX_LORA] and a["loras_multipliers"] == "1"
    assert a["num_inference_steps"] == 10 and a["guidance_phases"] == 1 and a["guidance_scale"] == 1
    assert a["sliding_window_size"] == 81 and a["force_fps"] == "16"
    assert a["video_length"] == 433 and a["temporal_upsampling"] == "rife*2"
    assert MOTION_MODELS == {"Animate 2 distilled": MOTION_MODEL, "Wan 2.2 Animate 14B": ANIMATE_MODEL}


def test_motion_frames():
    assert motion_frames(30, 813) == 433        # 27.1 s at 30 fps -> 27.06 s at 16 fps
    assert motion_frames(30, 813, 5) == 81
    assert motion_frames(30, 813, 40) == 433    # a limit above the video length changes nothing
    assert motion_frames(30, 60) == 33
    assert motion_frames(30, 3) == 5


def test_sliding_windows():
    assert sliding_windows(433, *MOTION_WINDOWS[MOTION_MODEL]) == [65] * 6 + [49]
    assert sliding_windows(433, *MOTION_WINDOWS[ANIMATE_MODEL]) == [81] * 5 + [53]
    assert sliding_windows(49, 65, 1) == [49]


def test_reel_plan():
    assert reel_plan(2) == (REELS_MODEL, [2])
    assert reel_plan(5) == (REELS_MODEL, [5])
    assert reel_plan(6) == (EXTEND_MODEL, [3, 3])
    assert reel_plan(7) == (EXTEND_MODEL, [4, 3])
    assert reel_plan(11) == (EXTEND_MODEL, [4, 4, 3])
    assert reel_plan(15) == (EXTEND_MODEL, [5, 5, 5])


def test_reel_windows():
    assert reel_windows(3) == (REELS_MODEL, [49])
    assert reel_windows(5) == (REELS_MODEL, [81])
    assert reel_windows(15) == (EXTEND_MODEL, [81, 81, 81])


def test_split_timeline():
    prompt = (
        "A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he stands)\n\n"
        "(at 4 seconds: he waves)\n"
        "(at 5 seconds: he smiles)\n"
        "(3-7 seconds: camera pushes in)"
    )
    first, second = split_timeline(prompt, [5, 5]).split("\n\n")
    assert first == (
        "[/duration=5s] A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he stands)\n(at 4 seconds: he waves)\n(3-5 seconds: camera pushes in)"
    )
    assert second == (
        "[/duration=5s] A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he smiles)\n(0-2 seconds: camera pushes in)"
    )
    # other time formats, Vietnamese unit, no parentheses
    assert split_timeline("(from 0 to 2 seconds: walks)\n(at 2s: stops)", [2, 1]) == (
        "[/duration=2s] (0-2 seconds: walks)\n\n[/duration=1s] (at 0 seconds: stops)"
    )
    assert split_timeline("0-2 giây: đi tới", [2]) == "[/duration=2s] (0-2 seconds: đi tới)"
    assert split_timeline("(at 0–2 seconds: turns)", [2]) == "[/duration=2s] (0-2 seconds: turns)"
    # no timed line: every window gets the whole prompt, blank lines removed
    assert split_timeline("he walks\n\nslowly", [3, 3]) == (
        "[/duration=3s] he walks\nslowly\n\n[/duration=3s] he walks\nslowly"
    )
    # a window with no timed line carries on the latest earlier action
    assert split_timeline("(at 0 seconds: x)\n(at 6 seconds: y)", [4, 4, 3]) == (
        "[/duration=4s] (at 0 seconds: x)\n\n[/duration=4s] (at 2 seconds: y)\n\n[/duration=3s] (at 0 seconds: y)"
    )


def test_reels_settings_long():
    s = reels_settings(
        {"num_inference_steps": 8}, image="a.png", prompt="(at 0 seconds: x)\n(at 6 seconds: y)",
        resolution="480x832", seconds=11, seed=3,
    )
    assert s["model_type"] == EXTEND_MODEL and s["num_inference_steps"] == 8
    assert s["multi_prompts_gen_type"] == "PW"
    assert s["image_prompt_type"] == "S" and s["image_start"] == "a.png"
    assert s["video_prompt_type"] == "KI" and s["image_refs"] == ["a.png"]
    assert s["video_length"] == 177
    assert s["prompt"] == split_timeline("(at 0 seconds: x)\n(at 6 seconds: y)", [4, 4, 3])
    assert s["seed"] == 3 and s["temporal_upsampling"] == "rife*2"


def test_instructions():
    assert "one continuous 15 second shot" in ideas_instructions(15)
    assert "from 0 to 14 seconds" in detail_instructions(15)
    assert "from 0 to 4 seconds" in detail_instructions()


def test_multi_settings():
    s = multi_settings(
        {"num_inference_steps": 4}, images=["a.png", "b.png", "c.png"], durations=[3, 2, 4],
        prompts=["walks in", "", "sits.\n\ndown"], common="A man in a red coat.",
        resolution="480x832", seed=1, loras=["x_high_noise.safetensors"], high=1, low=0, smooth=False,
    )
    assert s["model_type"] == REELS_MODEL and s["num_inference_steps"] == 4
    assert s["image_prompt_type"] == "SE"
    assert s["image_start"] == "a.png" and s["image_end"] == ["b.png", "c.png"]
    assert s["multi_prompts_gen_type"] == "PW"
    assert s["prompt"] == (
        "[/duration=3s] A man in a red coat. walks in\n\n"
        "[/duration=2s] A man in a red coat\n\n"
        "[/duration=4s] A man in a red coat. sits. down"
    )
    assert s["video_length"] == 145 and s["resolution"] == "480x832" and s["seed"] == 1
    assert s["loras_multipliers"] == "1;0" and s["temporal_upsampling"] == ""
    one = multi_settings({}, images=["a.png"], durations=[2], prompts=["waves"], common="", resolution="480x832", seed=-1)
    assert one["image_prompt_type"] == "S" and one["image_end"] is None
    assert one["prompt"] == "[/duration=2s] waves" and one["video_length"] == 33
    assert MAX_IMAGES == 6


def test_aspect_mismatch():
    assert aspect_mismatch([(1000, 1250), (800, 1000), (1000, 1000), None, (1080, 1920)]) == [3, 5]
    assert aspect_mismatch([None, (1000, 1000)]) == []
    assert aspect_mismatch([]) == []


def test_estimate_seconds():
    assert estimate_seconds(REELS_MODEL, [81], "720x720") == 120 + 4 * 97
    assert estimate_seconds(REELS_MODEL, [81], "720x720", {REELS_MODEL: 2}) == 2 * (120 + 4 * 97)
    assert estimate_seconds(REELS_MODEL, [81], "720x720", {EXTEND_MODEL: 2}) == 120 + 4 * 97
    assert estimate_seconds(EXTEND_MODEL, [81, 81], "720x720") == 120 + 2 * 8 * 106
    assert estimate_seconds(MOTION_MODEL, [65], "480x848") == 120 + 10 * 94
    assert estimate_seconds(ANIMATE_MODEL, [65], "480x848") == 120 + 10 * 94
    assert estimate_seconds(REELS_MODEL, [81], "960x960") > estimate_seconds(REELS_MODEL, [81], "720x720")


def test_fmt_minutes():
    assert fmt_minutes(419) == "7 phút"
    assert fmt_minutes(10) == "1 phút"
    assert fmt_minutes(6266) == "1 giờ 44 phút"


def test_eta_text():
    assert eta_text(EXTEND_MODEL, [81, 81, 81], "480x832") == "⏱ ≈ 35 phút (SVI 2 Pro · 3 cửa sổ × 8 bước)"
    assert eta_text(REELS_MODEL, [81], "480x832") == "⏱ ≈ 7 phút (Lightning v2 · 1 cửa sổ × 4 bước)"


def test_calib_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "timings.json"
        assert load_calib(p) == {}
        assert save_calib(REELS_MODEL, 1000, 500, p) == {REELS_MODEL: 2.0}
        assert save_calib(MOTION_MODEL, 300, 400, p) == {REELS_MODEL: 2.0, MOTION_MODEL: 0.75}
        assert load_calib(p) == {REELS_MODEL: 2.0, MOTION_MODEL: 0.75}
        p.write_text("not json")
        assert load_calib(p) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
