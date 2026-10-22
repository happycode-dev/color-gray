# Color Gray

Colors a grayscale photo offline with [DDColor](https://github.com/piddnad/DDColor) artistic. The model predicts color from the gray image and keeps the original brightness, so fine detail stays in place.

Put the gray image at `gray.png`. The result is written to `colorized.png`.

## Run

From this folder, with the project virtual environment:

```bash
./venv/Scripts/python.exe run.py
```

The script uses the GPU when CUDA is available, and the CPU otherwise. Weights are read only from disk:

```text
models/ddcolor_artistic.pth
```

If that file is missing, download `ddcolor_artistic.pth` from [piddnad/DDColor-models](https://huggingface.co/piddnad/DDColor-models) into `models/`. The checkpoint is about 870 MB. After it is saved locally, colorization does not use the network.

## Setup

Python 3.10 is already set up in `venv`. Install the packages with:

```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

On this machine, PyTorch needs the CUDA build so it can use the RTX 4060:

```bash
./venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`requirements.txt` lists `opencv-python`, `numpy`, `torch`, and `torchvision`. Keep the existing NumPy 2 and OpenCV 5. Do not install the upstream DDColor pin of `numpy==1.24.3`.

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Load `gray.png`, run DDColor large at 512 px, save `colorized.png` |
| `models/ddcolor_artistic.pth` | Local artistic checkpoint (not committed) |
| `ddcolor/` | Inference pipeline vendored from DDColor |
| `basicsr/` | Model pieces the pipeline imports |
| `LICENSE.ddcolor` | Apache License 2.0 for the vendored code |

`reference.png` is not used. An earlier version copied colors from that photo, which only works when both pictures show the same scene.

## Model

DDColor artistic is the large DDColor model trained on ImageNet plus artistic images. It is intended for varied color with fewer odd color artifacts. Inference resizes the gray image to 512×512 to predict color, then applies that color to the original full-resolution brightness.
