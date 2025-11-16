# Functional Ultrasound Localization Microscopy (fULM) Analysis

This folder contains scripts for analyzing functional brain activity using ultrasound localization microscopy with stimulus-based experimental protocols.

## Files

### 1. `explore_h5.py`
Interactive exploration of H5 files with stimulus labels.

**Features:**
- Displays top-level H5 structure
- Shows stimulus label blocks (rest vs watch)
- Maps acquisitions to their corresponding labels
- Visualizes experimental protocol timeline

**Usage:**
```bash
uv run python explore_h5.py
```

Edit the `h5_file_path` variable at the top to analyze different files.

### 2. `analyze_functional_activity.py`
Complete functional ULM analysis comparing rest vs stimulus conditions.

**Analysis Pipeline:**
1. Loads stimulus labels from H5 file
2. Processes all acquisitions (bubble detection & tracking)
3. Separates data by condition (rest vs watch)
4. Computes per-voxel detection rates
5. Performs statistical tests (z-test, p-values, effect sizes)
6. Generates activation maps

**Outputs:**
- `z_score_map.npy` - Statistical z-scores (watch vs rest)
- `p_value_map.npy` - Two-tailed p-values
- `cohens_d_map.npy` - Effect sizes (Cohen's d)
- `difference_map.npy` - Watch - Rest detection rate
- `rest_detection_rate.npy` - Bubble detection rate during rest
- `watch_detection_rate.npy` - Bubble detection rate during stimulus
- `functional_analysis.png` - 6-panel comprehensive visualization
- `activation_map_thresholded.png` - Only significant voxels
- `summary.json` - Analysis statistics

**Usage:**
```bash
uv run python analyze_functional_activity.py
```

**Configuration:**
Edit these variables at the top of the script:
- `h5_file_path` - Path to H5 file with labels
- `FILTER_METHOD` - 'highpass' or 'svd'
- `USE_GPU` - True/False
- `MIN_TRACK_LENGTH` - Minimum bubble track length (default: 5)
- `MIN_DETECTIONS_PER_VOXEL` - Minimum detections for statistics (default: 10)
- `SIGNIFICANCE_THRESHOLD` - P-value threshold (default: 0.05)

## Statistical Approach

For each voxel (z, x):
1. Count bubble detections during all "rest" acquisitions
2. Count bubble detections during all "watch" acquisitions
3. Normalize by number of acquisitions to get detection rates
4. Compute z-score: `z = (rate_watch - rate_rest) / SE`
   - Where SE = sqrt(var_watch/n_watch + var_rest/n_rest)
   - Assumes Poisson distribution for detection counts
5. Compute two-tailed p-value from z-score
6. Compute effect size (Cohen's d)

**Interpretation:**
- **Positive z-score**: More detections during "watch" (activation)
- **Negative z-score**: Fewer detections during "watch" (deactivation)
- **High |z-score|**: Strong statistical significance
- **p < 0.05**: Statistically significant difference
- **|Cohen's d| > 0.5**: Medium effect size
- **|Cohen's d| > 0.8**: Large effect size

## Data Format

The H5 file must contain:
- `/labels` - Array of label strings (b'rest', b'watch')
- `/label_timestamps` - Timestamps for each label block
- `/stimulus_metadata` - Per-acquisition stimulus metadata (475 entries)
- `/acquisitions/{i}/meta/...` - Standard ULM acquisition data

Example label structure:
```
Block 0: rest   at 2025-11-13T12:48:43.252111
Block 1: watch  at 2025-11-13T12:49:13.282651
Block 2: rest   at 2025-11-13T12:49:43.320996
...
```

## Example Workflow

1. **Explore the data:**
   ```bash
   uv run python explore_h5.py
   ```
   This shows you the label distribution and validates the experimental protocol.

2. **Run functional analysis:**
   ```bash
   uv run python analyze_functional_activity.py
   ```
   This processes all acquisitions and generates statistical maps.

3. **Load and visualize results:**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   z_scores = np.load('results/.../z_score_map.npy')
   p_values = np.load('results/.../p_value_map.npy')
   
   # Show significant activation
   sig_mask = p_values < 0.05
   plt.imshow(np.where(sig_mask, z_scores, np.nan), cmap='RdBu_r')
   plt.colorbar(label='Z-score')
   plt.title('Significant Functional Activation')
   plt.show()
   ```

## Performance

Processing ~475 acquisitions with GPU acceleration takes approximately:
- Bubble detection/tracking: ~15-30 minutes
- Statistical analysis: ~1-2 minutes
- Total: ~20-35 minutes

Without GPU: ~2-4 hours

## Notes

- The analysis uses a 2x upsampled grid for finer spatial resolution
- Voxels with insufficient data (< MIN_DETECTIONS_PER_VOXEL) are excluded
- Multiple comparison correction is NOT applied by default (consider FDR/Bonferroni for publication)
- The z-test assumes Poisson-distributed detection counts
