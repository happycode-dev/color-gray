import cv2
import torch
from pathlib import Path

from ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model

BASE_DIR = Path(__file__).resolve().parent
gray_path = BASE_DIR / "gray.png"
output_path = BASE_DIR / "colorized.png"
model_path = BASE_DIR / "models" / "ddcolor_artistic.pth"

if not model_path.is_file():
    raise FileNotFoundError(
        f"Missing DDColor weights: {model_path}. "
        "Download ddcolor_artistic.pth into the models folder first."
    )

gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Cannot load grayscale image: {gray_path}")

gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_ddcolor_model(
    DDColor,
    model_path=str(model_path),
    input_size=512,
    model_size="large",
    device=device,
)
colorizer = ColorizationPipeline(model, input_size=512, device=device)
result = colorizer.process(gray_bgr)

if not cv2.imwrite(str(output_path), result):
    raise RuntimeError("Failed to save colorized image")

print("Colorization completed successfully!")
print(f"Device: {device}")
print(f"Grayscale image: {gray_path}")
print(f"Model: {model_path}")
print(f"Output image: {output_path}")
