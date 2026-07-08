import os
from ultralytics import YOLO

# ─── مسیرها بر اساس محل خود فایل (مستقل از pwd) ───────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# ────────────────────────────────────────────────────────

def train():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="african-wildlife.yaml",   # دیتاست رسمی Ultralytics - خودکار دانلود می‌شود
        epochs=30,
        imgsz=640,
        batch=16,
        project=RESULTS_DIR,
        name="wildlife_yolo",
        patience=10,
        seed=42,
    )

    best_path = os.path.join(RESULTS_DIR, "wildlife_yolo", "weights", "best.pt")
    print(f"\nTraining complete. Best weights saved at:\n{best_path}")

if __name__ == "__main__":
    train()
