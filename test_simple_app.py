from simple_app import (
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
