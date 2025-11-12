# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains MATLAB code demonstrating **Transcranial Adaptive Ultrasound Localization Microscopy (t-ULM)** for imaging human brain vasculature. The code accompanies the paper "Deep Transcranial Adaptive Ultrasound Localization Microscopy of the Human Brain Vascularization" by Demené et al.

**License**: Creative Commons CC-BY-NC (non-commercial use only)
**Contact**: charlie.demene(at)espci.fr

## Repository Structure

The codebase is organized into two main demo directories:

### 1. `1_aberration_correction_and_beamforming/`
Demonstrates aberration correction and RF beamforming pipeline.

**Main script**: `demo_aberration_correction_and_beamforming.m`

**Data files**:
- `ExampleDataSet.mat` - Single frame example (RF data, beamformed images, bubble positions)
- `Raw_ultrasonic_data_1s.mat` - 1 second of raw RF data with reconstruction parameters

**Key functions** (in `functions/`):
- `Beamformer_RF.m` - RF beamforming in polar coordinates with aberration correction
- `findAberrationLaw_PaperExample.m` - Iterative aberration law estimation from bubble positions
- `IDBulles.m` - Bubble detection and isolation
- `scanConversion.m` / `scanConversionCustom.m` - Polar to Cartesian coordinate transformation
- `filtreSVDClutterAndNoise.m` - Spatiotemporal SVD filtering
- `estimateDelayViaFTCorrelation.m` - Delay estimation via Fourier transform correlation
- `localize2D_isolated.m` - 2D bubble localization
- `filtre_secteur_angulaire.m` - Angular sector filtering

### 2. `2_localisation_and_display/`
Demonstrates bubble tracking, super-localization, and density map generation.

**Main script**: `demo_overlay_raw_data_with_tULMposition_and_showdensitymap.m`

**Data files**:
- `Raw_ultrasonic_data_1s.mat` - 1 second of beamformed data (800 frames at 800 Hz)
- `Bubbles_positions_and_speed_1s.mat` - Localization data for 1 second
- `Bubbles_positions_and_speed_45s.mat` - Complete 45-second dataset (as in paper)

**Key functions** (in `functions/`):
- `filterSVD.m` - Spatiotemporal filtering to reveal microbubbles
- `displayImageFromPositions.m` - Generate density ULM image from bubble positions
- `getPixelForDisplay.m` - Convert subpixel positions to pixel coordinates
- `scanConversion.m` - Coordinate transformation
- `subimagesc.m` - Custom image display utility

## Running the Demos

### Demo 1: Aberration Correction and Beamforming
```matlab
cd 1_aberration_correction_and_beamforming
demo_aberration_correction_and_beamforming
```

**What it does**:
1. Loads RF data and reconstruction parameters
2. Beamforms one frame without aberration correction
3. Detects isolated bubbles for use as point reflectors
4. Estimates aberration law iteratively (5 iterations)
5. Beamforms the same frame WITH aberration correction
6. Displays before/after comparison

**Note**: This simplified demo uses bubbles from a single frame. The full paper implementation uses all frames across isoplanatic patches for better results.

### Demo 2: Localization and Display
```matlab
cd 2_localisation_and_display
demo_overlay_raw_data_with_tULMposition_and_showdensitymap
```

**What it does**:
1. Loads 1s of aberration-corrected beamformed data (800 frames)
2. Applies SVD spatiotemporal filtering to reveal microbubbles
3. Displays movie overlaying:
   - Original pixel detection (red squares)
   - Subpixel super-localization (blue crosses)
   - Final bubble trajectories (red dots)
4. Generates partial density ULM image from 1s data
5. Generates complete density ULM image from 45s data (Figure 1 from paper)

**Frame offset**: Data corresponds to the 10th second of acquisition (frames 7200-8000)

## Key Technical Details

### Coordinate Systems
- **Reconstruction**: Performed in polar coordinates (R, Phi)
- **Display**: Converted to Cartesian (X, Z) via `scanConversion.m`
- **Origin**: `BFStruct.BFOrigin` defines coordinate system origin

### Data Structures
- **BFStruct**: Contains beamforming/reconstruction parameters
  - `R_extent`, `Phi_extent`: Image extent in polar coordinates
  - `Depth_BF`: Depth range for beamforming
  - `dR`, `dPhi`: Spatial steps
  - `RxFreq`: Receive frequency

- **CP**: Low-level ultrasound sequence parameters
  - `nbPiezos`, `piezoPitch`: Transducer element configuration
  - `nbSources`, `xApex`, `zApex`: Virtual source positions
  - `c`: Speed of sound (mm/µs)
  - `TxFreq`, `RxFreqReal`: Transmit/receive frequencies

- **Bubbles_positions_and_speed_table**: Localization results
  - Column 1-2: X, Z subpixel positions
  - Column 3: Frame number
  - Column 5: Track length
  - Column 6-7: Original pixel positions
  - Column 8: Trajectory frame number

### Important Parameters

**Aberration correction**:
- `nb_iter = 5` - Number of iterations for aberration law estimation
- `coh_threshold = 0.7` - Coherence threshold for bubble selection
- `Width_filter = 10` mm - Angular filter width around bubbles

**Bubble detection**:
- `nbBullesParFrame = 100` - Target number of bubbles per frame
- `ecartX = 10`, `ecartZ = 10` pixels - Isolation distance
- `margeX = 5`, `margeZ1 = 50`, `margeZ2 = 300` pixels - Margin exclusion zones

**Display/reconstruction**:
- `min_size_track = 5` - Minimum track length for density image
- SVD filter cutoff: 30 singular values removed

## Development Notes

- This is demonstration code; full paper implementation uses GPU acceleration
- Computation times are NOT representative of the paper's performance
- Complete dataset (83 GB) not included; simplified examples provided
- All measurements in mm and µs unless otherwise specified
