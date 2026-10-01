from simple_app import (
    REELS_MODEL, EXTEND_MODEL, MOTION_MODEL, MOTION_DEFAULT_PROMPT,
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
    reels_settings, extend_settings, motion_settings,
    reel_plan, reel_windows, split_timeline, ideas_instructions, detail_instructions,
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
    assert s["video_guide"] == "d.mp4" and s["force_fps"] == "control"
    assert s["video_length"] == 149 and s["resolution"] == "480x832"
    assert motion_settings(defaults, image="c.png", video="d.mp4", description=" red hair ", resolution="480x832", frames=65, seed=1)["prompt"] == "red hair"


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
