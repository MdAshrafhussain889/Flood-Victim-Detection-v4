# Flood Victim Detection v4

AI-powered flood-scene analysis app for detecting real flood images, segmenting floodwater, and locating people in flood scenes.

The current v4 system focuses on:

- 3-class flood classification: `non_flood`, `real_flood`, `maps_diagrams`
- Floodwater segmentation for real flood scenes
- YOLOv8 person detection and simple tracking
- Streamlit dashboard for images, videos, and webcam captures
- Plain person detection output without risk-level scoring

Risk analysis was removed from the active app flow for now. The old `risk_engine/` module remains in the repository only for possible future restoration.

---

## Current Status

| Area | Status |
| --- | --- |
| Streamlit app | Active |
| v4 classifier | Active |
| Flood segmentation | Active |
| YOLO person detection | Active |
| Person tracking | Active |
| Risk levels | Disabled / not shown |
| Maps, diagrams, posters | Treated as non-real-flood images |
| Streamlit Cloud readiness | Prepared |

---

## How It Works

```text
Input image / video / webcam frame
            |
            v
  EfficientNet-B0 v4 classifier
            |
    +-------+----------------+
    |                        |
Non-flood / diagram      Real flood
    |                        |
Skip flood analysis          v
                  Attention U-Net flood segmentation
                              |
                              v
                    YOLOv8 person detection
                              |
                              v
                    Streamlit dashboard output
```

For real flood scenes, the app displays:

- Uploaded image
- Detection result with person boxes
- Flood overlay
- Person count
- Detection table with track ID and person confidence

The detection overlay intentionally does not display `LOW`, `HIGH`, `CRITICAL`, or other risk labels.

---

## Models Used

| Component | Model / Method |
| --- | --- |
| Flood classification | EfficientNet-B0, 3 classes |
| Flood segmentation | Attention U-Net |
| Person detection | YOLOv8n |
| Tracking | Centroid tracker |
| UI | Streamlit |

Included model files:

- `checkpoints/best_model.pth` - flood segmentation checkpoint
- `checkpoints/flood_classifier_best.pth` - legacy classifier checkpoint
- `checkpoints_v4/flood_classifier_v4_best.pth` - active v4 classifier checkpoint
- `yolov8n.pt` - YOLOv8n detector weights

---

## Dataset Notes

The training dataset is not included in GitHub because it is large.

Ignored local data/artifacts include:

- `data_v4/`
- `processed_data/`
- `outputs/`
- `outputs_v4/`
- `tracking.zip`
- Python cache folders
- virtual environments
- last/intermediate checkpoints

The repository includes `splits_v4/*.csv` so the v4 split structure is documented, but the actual images must exist locally if you want to retrain or evaluate.

---

## Project Structure

```text
Flood-Victim-Detection-v4/
|
|-- classification/
|   |-- inference/
|   |-- models/
|   |-- training/
|   |-- dataset_v4.py
|   |-- metrics_v4.py
|
|-- checkpoints/
|-- checkpoints_v4/
|-- configs/
|-- detection/
|-- pipeline/
|-- segmentation/
|-- streamlit_app/
|-- tracking/
|-- video/
|-- visualization/
|
|-- requirements.txt
|-- packages.txt
|-- runtime.txt
|-- README.md
```

---

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/MdAshrafhussain889/Flood-Victim-Detection-v4.git
cd Flood-Victim-Detection-v4
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

---

## Streamlit Cloud Deployment

This repository is prepared for Streamlit Community Cloud.

Use these deployment settings:

| Field | Value |
| --- | --- |
| Repository | `MdAshrafhussain889/Flood-Victim-Detection-v4` |
| Branch | `main` |
| Main file path | `streamlit_app/app.py` |

`packages.txt` contains Linux system packages needed by OpenCV on Streamlit Cloud.

---

## App Behavior

### Real flood image

The app runs flood segmentation and person detection, then shows the flood overlay and detected persons.

### Non-flood image

The app skips flood analysis and shows a non-flood result.

### Poster, map, chart, or diagram

The v4 classifier may place these in the `maps_diagrams` class. The app treats them as non-real-flood images and skips segmentation/detection.

---

## Training / Evaluation

Create v4 splits:

```bash
python scripts/create_v4_splits.py
```

Train v4 classifier:

```bash
python classification/training/train_classifier_v4.py
```

Evaluate v4 classifier:

```bash
python classification/training/evaluate_classifier_v4.py
```

These commands require the local `data_v4/` image folders.

---

## Author

Mohammed Ashraf Hussain

---

## License

This project is developed for research, academic, and AI disaster-management purposes.
