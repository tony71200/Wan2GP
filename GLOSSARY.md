# Simple App

The small three-tab video app (`simple_app.py`) built on Wan2GP: turn a photo of a person into a short social video while keeping that person recognisable.

## Language

### Tabs

**Reels**:
One image becomes one continuous video of 2–15 seconds.
_Avoid_: Reef

**Reels Multi Image**:
Up to six images of the same character become one continuous video that passes through every image in order.
_Avoid_: Reef multi image, slideshow

**Motion Control**:
A character image is animated with the motion of a Driving Video.
_Avoid_: Motion transfer

### Inputs

**Character**:
The person (or people) in the user's image whose face, hair and body must stay recognisable in the video.
_Avoid_: Subject, model

**Driving Video**:
The video whose motion (and, optionally, sound) Motion Control copies.
_Avoid_: Reference video, control video, video mẫu (in code)

**Keyframe**:
One image of Reels Multi Image; the video reaches it at the start of its Segment.
_Avoid_: Mốc, slide

**Segment**:
The stretch of a Reels Multi Image video that starts at a Keyframe and lasts that Keyframe's duration; the last Segment ends freely.
_Avoid_: Clip, shot

**Story**:
A text file listing Keyframes in order, each with a time range or duration and its prompt; used only by the Multi Image harness.

### Generation

**Window**:
One block of at most ~5 seconds that the model generates at a time; longer videos are a chain of Windows.
_Avoid_: Cửa sổ trượt, chunk

**Anchor**:
The original image that every Window of a long Reels video refers back to so the Character does not drift.
_Avoid_: Ảnh neo (in code), reference

**Intermediate Video**:
The file Wan2GP saves after each Window, holding every frame generated so far; superseded by the Final Video.
_Avoid_: Temp video, partial

**Final Video**:
The single file a successful job hands back to the user.

**Pha đầu / Pha cuối**:
The two denoising phases of Wan 2.2: the first decides layout and motion, the second decides fine detail such as skin and hair.
_Avoid_: High/low (in UI text)
