"""SARA text-reading demo: run the glass's own OCR mode on a photo and save everything needed for a demo video.

New, stand-alone file: it imports the application's modules exactly as main.py does and changes none of them.
Run it on the Raspberry Pi 5 (inside savior_glass/, with the usual virtual environment):

    python demo_video/run_ocr_demo.py demo_video/input/page_vertical_farming.jpg

It reads the photo twice, as the glass camera would deliver it (fitted into a 640x480 frame) and at the photo's
own size, and writes demo_video/results/ocr_demo_<device>_<time>.json with: the spoken sentences and the moment
each was produced (pages are spoken line by line, as on the glass), the text boxes, the time taken, the settings
used, and the character error rate against demo_video/input/<photo>.txt when that transcription exists.
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # savior_glass/, as when main.py runs

import cv2                                 # noqa: E402
import numpy as np                         # noqa: E402

import config                              # noqa: E402
from modes.ocr_mode import OCRMode         # noqa: E402


def device():
    try:
        return Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00 \n")
    except OSError:
        return platform.node() or platform.platform()


def camera_frame(img, w=640, h=480):
    """The photo as the glass camera would hand it over: one 640x480 picture, the page fitted inside it."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    frame = np.full((h, w, 3), 128, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    frame[y:y + small.shape[0], x:x + small.shape[1]] = small
    return frame


def cer(hyp, ref):
    """Character error rate: edit distance / reference length, on lower-cased text with single spaces."""
    a, b = " ".join(hyp.lower().split()), " ".join(ref.lower().split())
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


def run(mode, frame):
    spoken, t0 = [], time.perf_counter()
    mode.speak_line = lambda text: spoken.append({"t_s": round(time.perf_counter() - t0, 3), "text": text})
    result = mode.process_frame(frame)
    total = time.perf_counter() - t0
    return {
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "seconds": round(total, 3),
        "spoken_lines": spoken,                 # each line as it was produced (page streaming)
        "returned_sentence": result,            # what process_frame returned (spoken after the lines)
        "stored_text": mode._stored_text,       # what READ would repeat
        "streamed": bool(mode.streamed),
        "boxes": [[int(v) if i < 4 else (float(v) if i == 4 else bool(v)) for i, v in enumerate(b)]
                  for b in getattr(mode, "last_boxes", [])],   # x0, y0, x1, y1, confidence, in the main text block
    }


def main():
    photo = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "input" / "page_vertical_farming.jpg").resolve()
    img = cv2.imread(str(photo))
    if img is None:
        sys.exit(f"cannot read {photo}")
    truth_file = photo.with_suffix(".txt")
    truth = truth_file.read_text(encoding="utf-8") if truth_file.exists() else None

    mode = OCRMode()
    t = time.perf_counter()
    mode.activate()                         # loads EasyOCR, the ONNX text detector and the blended recogniser
    load_s = time.perf_counter() - t

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device(),
        "photo": photo.name,
        "photo_size": [int(img.shape[1]), int(img.shape[0])],
        "model_load_seconds": round(load_s, 2),
        "settings": {k: str(getattr(config, k, None)) for k in
                     ["OCR_PIPELINE", "OCR_DETECTOR", "OCR_RECOGNIZER_PATH", "OCR_READ_WIDTH", "OCR_V2_DECODER",
                      "OCR_STREAM_PAGES", "OCR_STREAM_MIN_LINES", "OCR_V2_MIN_CONF"]},
        "runs": {},
    }
    for name, frame in [("glass_camera_640x480", camera_frame(img)), ("full_photo", img)]:
        r = run(mode, frame)
        if truth and r["stored_text"]:
            r["cer_vs_transcription"] = round(cer(r["stored_text"], truth), 4)
        out["runs"][name] = r
        print(f"{name}: {r['seconds']:.1f} s, {len(r['spoken_lines'])} lines spoken, "
              f"CER {r.get('cer_vs_transcription', 'n/a')}")
        print("   ", (r["stored_text"] or "")[:200])

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    tag = "pi5" if "Raspberry Pi 5" in out["device"] else "other"
    path = res / f"ocr_demo_{tag}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cv2.imwrite(str(res / "frame_glass_camera_640x480.png"), camera_frame(img))
    print("saved", path)


if __name__ == "__main__":
    main()
