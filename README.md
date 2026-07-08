# African Wildlife Object Detection (YOLOv8)

Fine-tuned YOLOv8n on the African Wildlife dataset to detect 4 species: 
buffalo, elephant, rhino, and zebra.

## Pipeline
1. Load pretrained YOLOv8n (COCO weights)
2. Fine-tune on African Wildlife dataset (auto-downloaded via Ultralytics)
3. Train for 30 epochs with early stopping (patience=10)
4. Evaluate: mAP50, mAP50-95, confusion matrix, PR curve
5. Run inference on held-out test images

## Dataset
[African Wildlife Dataset](https://github.com/ultralytics/assets) 
(auto-downloaded on first run via `african-wildlife.yaml`)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
cd src
python train.py
python evaluate.py
```

## Results
See `results/wildlife_yolo/results.png` for training curves and 
`results/predictions/` for sample detections.

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
