# SARA: Savior Glass

A Bangla-first assistive smart glass for blind and low-vision people in Bangladesh, built on a Raspberry Pi 5.
It reads Bangla and English text aloud, names objects in front of the user, recognises all nine Bangladeshi Taka
notes with a cautious counterfeit check, and names the facial expression of the person in front. Everything
except the optional online description runs on the device, without the internet.

Senior project, Department of Computer Science and Engineering, Independent University, Bangladesh.
MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti. Supervisor: Mohammad Shidujaman, PhD.

## Modes

| Mode | What it does | Model |
|---|---|---|
| Text reading | Reads the main block of text in Bangla or English and speaks it | EasyOCR (CRAFT + CRNN) |
| Objects | Names objects in front of the user every 1.5 s | YOLOv8 (COCO) |
| Currency | Names the Taka note; for 500 and 1,000 Taka says "likely genuine" or "check by hand", never "counterfeit" | YOLOv8s trained on 22,333 synthetic images, multi-view authenticator, watermark check |
| Emotion | Names the expression of the nearest face | YuNet + EfficientNet-V2-S |
| Online | Describes the scene (needs internet and an API key) | Cloud vision model |

Five buttons: MODE, ACTION, READ, VOLUME UP, VOLUME DOWN. Every change is spoken.

## Run on a Raspberry Pi 5

```bash
cd ~
git clone https://github.com/aflam330/SARA.git
cd SARA/savior_glass
sh scripts/deploy_pi5.sh                  # system packages, Python environment in ~/.venvs/savior_glass
source ~/.venvs/savior_glass/bin/activate
python scripts/pi5_preflight.py           # checks every library and model
python main.py
```

Update later with `git pull`. Keep the project on the Pi's own disk: a FAT-formatted USB drive cannot hold the
links and pipes the Python environment and the GPIO library need.

## Run on a laptop (test window)

```bash
cd SARA/savior_glass
pip install -r requirements.txt
SHOW_PREVIEW=1 python main.py
```

Keys: **M** mode, **A** or **Space** action, **R** read again, **+ / -** volume, **L** live algorithm panel, **Q** quit.
The window draws a bounding box in every mode. `PREVIEW_LIVE_BOXES=0` turns off the live boxes on a slow machine.

## Online mode

Put the key in `savior_glass/.env` (never committed): `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`.

## Measured on the Raspberry Pi 5 (8 GB)

| | Median |
|---|---|
| Currency check, end to end | 327 ms |
| Taka detector (INT8 ONNX) | 254 ms |
| Watermark check | 50 ms |
| Object scan (YOLOv8n) | 207 ms |
| Text reading (one capture) | 18.7 s |

30 minutes of mixed use: peak 77.1 °C, no thermal throttling, peak memory 1.48 GB.

## Known limitations

- With the 640×480 fixed-focus glass camera, currency recognition and Bangla reading are weaker than in the
  laboratory; the cause is still being checked (`python scripts/diagnose_glass.py currency`).
- Reading a full page is slow on the Pi.
- The counterfeit check is cautious: on whole notes it usually asks for a check by hand.
- A study with visually impaired users has started; results are not yet published.

## Folders

- `savior_glass/` – the glass application, modes, preview window, tests and tools
- `realtime_bangla_taka_detection/` – the trained Taka models, the counterfeit and watermark models, and the
  few result files the application reads

The research code, datasets, experiments and thesis are kept in the project's research repository.

## Licences

The object detector uses Ultralytics YOLOv8 (AGPL-3.0). The models were trained on public datasets
(BanglaTaka, JaalTaka, RAF-DB, COCO); their own terms apply to any further use of the trained weights.
