# SARA (Smart Assistive Reading Aid)

**An Offline Smart Glass for Visually Impaired People that Reads Bangla and English Text, Detects Objects, and
Recognises and Checks Bangladeshi Taka Notes**

An assistive smart glass for blind and low-vision people in Bangladesh that speaks Bangla, built on a Raspberry Pi 5.
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

## Main results

- Taka detector trained on 22,333 automatically labelled images: mAP@0.5 0.995 on held-out images, 91.5% correct
  note value on 1,536 photographs taken by other people.
- Counterfeit check trained on random prefixes of the views: 98.2% with one view (against 71.2% without).
- On counterfeit prints never seen in training, the watermark check raises accuracy to 95.0%.
- New watermark method, the denomination-matched watermark residual (DMWR): it removes the lamp from the back-lit
  photo (quotient image Q = I / (G * I)), compares the watermark with a genuine portrait of the note value built
  from genuine notes only, and classifies the residual R = Q - T. With 47,818 parameters (0.21 MB) it reaches
  91.5% on unseen prints, statistically level with a 46 times larger MobileNetV2.
- UV fibre check (optional, off by default): the glass carries a small UV-A lamp (360-395 nm, the blue light on the
  frame), which covers the 365 nm used for banknote inspection. Genuine 200, 500 and 1,000 Taka notes show glowing
  security fibres on paper that stays dark; 103 of 111 genuine notes were accepted (92.8%) on phone photographs.
- Text reading: character error rate 9.3% to 2.4% (Bangla 11.9% to 3.2%) on a benchmark with correctly shaped Bangla.
- The glass never says "counterfeit": 0 of 88 test counterfeits were passed as genuine.

## Current work

- Tuning the glass camera for currency and Bangla reading (`python scripts/diagnose_glass.py currency`).
- Faster reading of whole pages on the Pi.
- Trial sessions with visually impaired users have started.

## Folders

- `savior_glass/` – the glass application, modes, preview window, tests and tools
- `realtime_bangla_taka_detection/` – the trained Taka models, the counterfeit and watermark models, and the
  few result files the application reads

The research code, datasets, experiments and thesis are kept in the project's research repository.

## Licences

The object detector uses Ultralytics YOLOv8 (AGPL-3.0). The models were trained on public datasets
(BanglaTaka, JaalTaka, RAF-DB, COCO); their own terms apply to any further use of the trained weights.
