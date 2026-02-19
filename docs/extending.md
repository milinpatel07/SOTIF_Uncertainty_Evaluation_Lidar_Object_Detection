# Extending the Pipeline

This guide covers how to add a new detector, a new dataset, or a new uncertainty indicator to the evaluation pipeline.

---

## Adding a New Detector

The pipeline is not tied to SECOND. Any 3D object detector that produces per-detection confidence scores can be used. The requirement is that you run K independently trained instances and collect their outputs.

### What the pipeline expects

Each ensemble member `k` produces, for each frame, a list of detections:
- Bounding box: `[x, y, z, w, l, h, yaw]` in LiDAR coordinates
- Confidence score: float in `[0, 1]`

After running all K members, the outputs are passed to `ensemble.cluster_detections()` which associates detections across members using DBSCAN on BEV IoU distance.

### Steps

1. **Train K models** with identical hyperparameters and different random seeds. The number K=6 was used in the paper; as few as K=3 works but with lower uncertainty quality.

2. **Run inference** for each member on the same set of frames. Store per-member detections in a list of dicts:

```python
# Per-frame, per-member format expected by cluster_detections()
member_detections = [
    {
        "boxes_lidar": np.ndarray,  # shape (D_k, 7)
        "score": np.ndarray,        # shape (D_k,)
    }
    for k in range(K)
]
```

3. **Cluster and evaluate**:

```python
from sotif_uncertainty.ensemble import cluster_detections, clustered_to_pipeline_format

# Associate detections across members
clustered = cluster_detections(
    member_detections,
    iou_threshold=0.5,
    voting="consensus",  # or "affirmative", "unanimous"
)

# Convert to pipeline format (scores matrix + boxes tensor)
scores, boxes, frame_ids, labels = clustered_to_pipeline_format(
    clustered_frames, gt_boxes_per_frame, iou_threshold=0.5
)

# From here, use the standard pipeline
from sotif_uncertainty import compute_all_indicators, compute_all_metrics
indicators = compute_all_indicators(scores, boxes)
```

4. **Alternatively**, use `scripts/run_inference.py` with `--ckpt_dirs` pointing to your K model checkpoints. This script handles the full inference-to-evaluation flow for OpenPCDet-compatible models.

---

## Adding a New Dataset

The pipeline supports any dataset that provides LiDAR point clouds and 3D bounding box annotations. Two integration paths exist.

### Path A: KITTI-format directory

The simplest approach. Arrange your data in KITTI directory layout:

```
your_dataset/
├── training/
│   ├── velodyne/       # {frame_id}.bin  (N x 4 float32: x, y, z, intensity)
│   ├── label_2/        # {frame_id}.txt  (KITTI label format)
│   └── calib/          # {frame_id}.txt  (KITTI calibration format)
├── ImageSets/
│   ├── train.txt       # frame IDs for training
│   └── val.txt         # frame IDs for evaluation
└── conditions.json     # optional: per-frame metadata for TC analysis
```

Then use the existing adapter:

```python
from sotif_uncertainty.dataset_adapter import DatasetAdapter

adapter = DatasetAdapter("your_dataset/", split="val", format="kitti")
frame_ids = adapter.get_frame_ids()
points = adapter.load_points(frame_ids[0])
gt_boxes = adapter.load_gt_boxes(frame_ids[0])
```

### Path B: Custom format

If your data is not in KITTI format, you have two options:

1. Write a conversion script that produces KITTI-format files (see `scripts/generate_carla_data.py` for an example).

2. Subclass `DatasetAdapter` and override `load_points()`, `load_gt_boxes()`, and `load_calibration()`.

### Triggering condition metadata

To use the TC ranking analysis (ISO 21448 Clause 7), provide a `conditions.json` file mapping frame IDs to condition categories:

```json
{
    "000001": {"tc_category": "heavy_rain"},
    "000002": {"tc_category": "clear"},
    "000003": {"tc_category": "fog"}
}
```

The category names are arbitrary strings. The pipeline computes FP share and mean uncertainty per category.

---

## Adding a New Uncertainty Indicator

The pipeline currently uses three indicators (Section 3.3 of the paper). To add a fourth:

### 1. Implement the computation

Add your function to `sotif_uncertainty/uncertainty.py`:

```python
def compute_your_indicator(scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute your indicator per proposal.

    Parameters
    ----------
    scores : np.ndarray, shape (M, K)
    boxes : np.ndarray, shape (M, K, 7)

    Returns
    -------
    np.ndarray, shape (M,)
    """
    # Your computation here
    return values
```

### 2. Register it in `compute_all_indicators()`

In `uncertainty.py`, add your indicator to the returned dict:

```python
def compute_all_indicators(scores, boxes=None):
    result = {
        "mean_confidence": compute_mean_confidence(scores),
        "confidence_variance": compute_confidence_variance(scores),
        "your_indicator": compute_your_indicator(scores, boxes),
    }
    if boxes is not None:
        result["geometric_disagreement"] = compute_geometric_disagreement(boxes)
    return result
```

### 3. Evaluate discrimination

Use `metrics.compute_auroc()` to measure how well your indicator separates TP from FP:

```python
from sotif_uncertainty.metrics import compute_auroc

auroc = compute_auroc(
    your_values,
    labels,
    higher_is_correct=False,  # True if higher values indicate correct detections
)
```

### 4. Add to acceptance gate (optional)

If your indicator should be used in the acceptance gate, extend `sotif_analysis.acceptance_gate()` with an additional threshold parameter.

### 5. Add tests

Add test cases to `tests/test_pipeline.py` following the existing pattern:

```python
def test_your_indicator(self):
    scores = np.array([[0.9, 0.85, 0.88], [0.3, 0.1, 0.0]])
    boxes = ...  # if needed
    values = compute_your_indicator(scores, boxes)
    self.assertEqual(values.shape, (2,))
    # Assert expected properties
```
