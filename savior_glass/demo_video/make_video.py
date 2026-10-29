"""Make the text-reading demo video from a result of run_ocr_demo.py. New, stand-alone file; changes nothing else.

    python demo_video/make_video.py demo_video/results/ocr_demo_pi5_<time>.json [--run glass_camera_640x480]
           [--glass-dir <savior_glass folder that holds models/piper>] [--ffmpeg <ffmpeg.exe>]

Nothing in the video is invented: the text shown and spoken is what the OCR mode produced in that result file, and
each line appears when the Pi produced it. Speech is rendered with the glass's own speech engine (utils.TTSEngine,
Piper voices), in the order the glass speaks: "পড়ছি" at capture, a soft tick every OCR_PROGRESS_TICK_S seconds while
nothing is being said, each line as soon as it is ready and the previous line has been spoken, then the closing
sentence. The waiting time before the first line is shown as a labelled time-lapse.
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
GREEN, DARK, PALE, GREY, RED = (0, 106, 78), (11, 77, 59), (238, 245, 241), (94, 107, 102), (215, 38, 61)


def font(size, bold=False):
    for name in (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def camera_frame(img, w=640, h=480):
    """Same as run_ocr_demo.camera_frame (copied so this script needs no OCR libraries): the page fitted into 640x480."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    frame = np.full((h, w, 3), 128, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    frame[y:y + small.shape[0], x:x + small.shape[1]] = small
    return frame


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- audio with the glass's speech engine
def render_audio(run, glass_dir):
    sys.path.insert(0, str(glass_dir))
    import config
    import utils
    tts = utils.TTSEngine()
    rate = 22050

    def say(text, lang="auto"):
        pieces = ([(s, l) for seg, l in utils.split_language_segments(text) for s in utils.split_sentences(seg)]
                  if lang == "auto" else [(text, lang)])
        out = []
        for s, l in pieces:
            if l != "tone" and not any(c.isalnum() for c in s):
                continue
            pcm, r = tts._render(s, l)
            a = np.frombuffer(pcm, np.int16).astype(np.float32)
            if r != rate and len(a):
                a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * rate / r)), np.arange(len(a)), a)
            lead = np.zeros(int(rate * (0 if l == "tone" else config.TTS_LEAD_IN_MS / 1000)), np.float32)
            gap = np.zeros(int(rate * (config.TTS_TAIL_MS / 1000 + config.TTS_INTER_UTTERANCE_PAUSE)), np.float32)
            out += [lead, a, gap]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    # the glass's order of events, in real seconds after the press (t = 0)
    events, t_busy = [], 0.0          # (start, samples, label, line index or None)
    clip = say("পড়ছি")
    events.append((0.0, clip, "cue", None))
    t_busy = len(clip) / rate
    total = float(run["seconds"])
    lines = run["spoken_lines"]
    tick = say("tick", "tone")
    tick_every = float(getattr(config, "OCR_PROGRESS_TICK_S", 4))
    next_tick, i = tick_every, 0
    while i < len(lines) or next_tick < total:
        t_line = lines[i]["t_s"] if i < len(lines) else float("inf")
        if next_tick < min(t_line, total) and next_tick >= t_busy:
            events.append((next_tick, tick, "tick", None))
            t_busy = next_tick + len(tick) / rate
            next_tick += tick_every
            continue
        if next_tick < min(t_line, total):        # a tick while something is said is skipped, as on the glass
            next_tick += tick_every
            continue
        if i < len(lines):
            start = max(t_line, t_busy)
            clip = say(lines[i]["text"])
            events.append((start, clip, "line", i))
            t_busy = start + len(clip) / rate
            i += 1
        else:
            break
    if run.get("returned_sentence"):
        start = max(total, t_busy)
        clip = say(run["returned_sentence"])
        events.append((start, clip, "end", None))
        t_busy = start + len(clip) / rate
    return events, rate, t_busy


# --------------------------------------------------------------------------- drawing
def frame_card(lines_big, lines_small, foot=""):
    im = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(im)
    y = 200
    for t in lines_big:
        d.text((90, y), t, font=font(46, True), fill="white")
        y += 64
    y += 24
    for t in lines_small:
        for ln in wrap(d, t, font(26), W - 180):
            d.text((90, y), ln, font=font(26), fill=(207, 226, 218))
            y += 38
        y += 8
    if foot:
        d.text((90, H - 70), foot, font=font(20), fill=(159, 194, 180))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--run", default="glass_camera_640x480")
    ap.add_argument("--glass-dir", default=str(HERE.parent))
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = json.loads(Path(a.result).read_text(encoding="utf-8"))
    run = res["runs"][a.run]
    photo = cv2.imread(str(HERE / "input" / res["photo"]))
    if a.run.startswith("glass"):
        view = camera_frame(photo)               # the same 640x480 frame the OCR mode was given
    else:
        view = photo
    events, rate, t_end = render_audio(run, Path(a.glass_dir))

    first_line = next((e[0] for e in events if e[2] == "line"), run["seconds"])
    lapse = max(1.0, first_line / 6.0)          # the wait before the first line takes at most about 6 s of video
    intro = 4.0 + 3.0                           # title card, then the photo and the camera frame

    def v_of_t(t):                              # real seconds after the press -> seconds in the video
        return intro + (t / lapse if t <= first_line else first_line / lapse + (t - first_line))

    dur = v_of_t(t_end) + 7.0                   # closing card
    n = int(dur * FPS)

    # audio track
    track = np.zeros(int(dur * rate) + rate, np.float32)
    for start, clip, kind, _ in events:
        s = int(v_of_t(start) * rate)
        if kind in ("cue", "tick") and start < first_line:   # cues during the time-lapse play at normal speed
            s = int(v_of_t(start) * rate)
        track[s:s + len(clip)] += clip[: max(0, len(track) - s)]
    track = np.clip(track, -32767, 32767).astype(np.int16)
    out = Path(a.out or HERE / "results" / f"SARA_text_reading_demo_{a.run}.mp4")
    wav, raw = out.with_suffix(".wav"), out.with_suffix(".noaudio.mp4")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(track.tobytes())

    # static parts
    dev = res["device"]
    date = res["created_utc"][:10]
    title = frame_card(["SARA: text-reading demo"],
                       [f"The glass's own text-reading mode (OCR) run on a photographed page, on {dev}, {date}.",
                        "The photo was given to the program; it was not held in front of the glass camera.",
                        "Everything you see and hear below is the program's real output, at the time it was produced."],
                       "Independent University, Bangladesh  ·  CSE Senior Project")
    vw, vh = 720, int(720 * view.shape[0] / view.shape[1])
    if vh > 540:
        vh, vw = 540, int(540 * view.shape[1] / view.shape[0])
    sx, sy = vw / view.shape[1], vh / view.shape[0]
    view_rgb = cv2.cvtColor(cv2.resize(view, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    spoken = run["spoken_lines"]
    boxes = run["boxes"]
    line_events = {e[3]: e[0] for e in events if e[2] == "line"}
    end_card = frame_card(["Result of this run"], [
        f"Frame given to the reader: {run['frame_size'][0]}×{run['frame_size'][1]} pixels ({a.run.replace('_', ' ')}).",
        f"First line spoken after {first_line:.1f} s; whole page read in {run['seconds']:.1f} s on {dev}.",
        (f"Character error rate against a hand transcription of the page: {100 * run['cer_vs_transcription']:.1f}%."
         if "cer_vs_transcription" in run else "No transcription was available for scoring."),
        "Demo on a photo: the glass camera itself was not used, and no user took part."], f"Result file: {Path(a.result).name}")

    vid = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    f_title, f_body, f_small = font(30, True), font(22), font(18)
    for k in range(n):
        t_v = k / FPS
        if t_v < 4.0:
            img = title
        elif t_v < 5.5 or t_v < intro:
            im = Image.new("RGB", (W, H), PALE)
            d = ImageDraw.Draw(im)
            if t_v < 5.5:
                p = Image.fromarray(photo_rgb)
                p.thumbnail((1100, 560))
                im.paste(p, ((W - p.width) // 2, 110))
                d.text((90, 40), "The page", font=f_title, fill=DARK)
            else:
                im.paste(Image.fromarray(view_rgb), ((W - vw) // 2, 110))
                d.text((90, 40), "What the reader is given" + (" — the glass camera's 640×480 frame" if a.run.startswith("glass") else ""),
                       font=f_title, fill=DARK)
            img = np.array(im)
        elif t_v < dur - 7.0:
            # real time after the press
            tv = t_v - intro
            t_real = tv * lapse if tv <= first_line / lapse else first_line + (tv - first_line / lapse)
            im = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(im)
            d.text((40, 24), "SARA reads the page", font=f_title, fill=DARK)
            reading = t_real < run["seconds"]
            chip = "Reading…" if reading else "Done"     # the glass says "পড়ছি" at the press; it is in the audio
            d.rounded_rectangle([900, 22, 1240, 66], 12, fill=GREEN if reading else GREY)
            d.text((918, 30), chip, font=font(22, True), fill="white")
            # camera frame with the boxes found so far
            fr = Image.fromarray(view_rgb.copy())
            fd = ImageDraw.Draw(fr)
            done = sum(1 for i in line_events if line_events[i] <= t_real)
            show = boxes if not reading else boxes[: int(len(boxes) * done / max(1, len(spoken)))]
            for b in show:
                fd.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=GREEN if b[5] else (160, 160, 160), width=2)
            im.paste(fr, (40, 90))
            # what has been said so far
            x0, y0 = 790, 96
            d.text((x0, y0), "Heard from the glass:", font=font(22, True), fill=DARK)
            said = [spoken[i]["text"] for i in sorted(line_events) if line_events[i] <= t_real]
            rows = []
            for j, s in enumerate(said):
                for ln in wrap(d, s, f_body, W - x0 - 40):
                    rows.append((ln, j == len(said) - 1))
            y = y0 + 40
            for ln, cur in rows[-17:]:
                d.text((x0, y), ln, font=f_body, fill=DARK if cur else GREY)
                y += 30
            lap = t_real < first_line
            d.text((40, H - 46), f"t = {t_real:5.1f} s after the press, on {dev}" + (f"   ·   time-lapse ×{lapse:.1f} until the first line" if lap else "   ·   real time"),
                   font=f_small, fill=RED if lap else GREY)
            img = np.array(im)
        else:
            img = end_card
        vid.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vid.release()
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    raw.unlink()
    wav.unlink()
    print("video:", out, f"{dur:.1f} s")


if __name__ == "__main__":
    main()
