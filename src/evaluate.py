import os
from ultralytics import YOLO

# ─── مسیرها بر اساس محل خود فایل (مستقل از pwd) ───────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BEST_WEIGHTS = os.path.join(RESULTS_DIR, "wildlife_yolo", "weights", "best.pt")
from ultralytics.utils import SETTINGS
TEST_IMAGES = os.path.join(SETTINGS["datasets_dir"], "african-wildlife", "images", "test")
# ────────────────────────────────────────────────────────

def evaluate():
    model = YOLO(BEST_WEIGHTS)

    metrics = model.val(data="african-wildlife.yaml")

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    model.predict(
        source=TEST_IMAGES,
        save=True,
        project=RESULTS_DIR,
        name="predictions",
        conf=0.4,
    )

if __name__ == "__main__":
    evaluate()
