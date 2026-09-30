from simple_app import (
    REELS_MODEL, EXTEND_MODEL, MOTION_MODEL,
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
    reels_settings, extend_settings, motion_settings,
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
    assert s["video_length"] == 49 and s["seed"] == 5
    assert s["activated_loras"] == [] and s["loras_multipliers"] == ""
    assert s["temporal_upsampling"] == "rife*2"


def test_motion_settings():
    defaults = {"prompt": "Character appearance: default", "num_inference_steps": 10, "video_prompt_type": "UV", "image_prompt_type": "S"}
    s = motion_settings(defaults, image="c.png", video="d.mp4", description="", resolution="480x832", frames=149, seed=-1)
    assert s["model_type"] == MOTION_MODEL
    assert s["num_inference_steps"] == 10
    assert s["prompt"] == "Character appearance: default"
    assert s["image_prompt_type"] == "T" and s["image_start"] is None
    assert s["video_prompt_type"] == "UVI" and s["image_refs"] == ["c.png"]
    assert s["video_guide"] == "d.mp4" and s["force_fps"] == "control"
    assert s["video_length"] == 149 and s["resolution"] == "480x832"
    assert motion_settings(defaults, image="c.png", video="d.mp4", description=" red hair ", resolution="480x832", frames=65, seed=1)["prompt"] == "red hair"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
