# Text-reading demo

These files make a demo video of SARA's text mode. The glass's own OCR mode reads a photographed page, and the video shows what it read and plays what it said. All files here are new; no application file is changed.

- `input/page_vertical_farming.jpg`: the page photo.
- `input/page_vertical_farming.txt`: a hand transcription, used to score the reading (character error rate).
- `run_ocr_demo.py`: runs the OCR mode on the photo and writes `results/ocr_demo_*.json`.

## On the Raspberry Pi 5

```
cd ~/"SARA UPDATE"/SARA && git pull
cd savior_glass && . ~/.venvs/savior_glass/bin/activate
python demo_video/run_ocr_demo.py demo_video/input/page_vertical_farming.jpg
cd .. && git add savior_glass/demo_video/results && git commit -m "OCR demo result from the Pi" && git push
```

## What the result shows

The photo is read twice:
- **As the glass camera would deliver it:** fitted into a 640×480 frame.
- **At its own size:** like a phone picture.

This is a photo given to the program, not a page held in front of the glass camera. The video says so.
