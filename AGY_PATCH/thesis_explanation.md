# 3D Vertex Reconstruction in the TMS Detector using Event-Level Machine Learning Regression

This document details the methodology, physical assumptions, coordinate systems, and machine learning regression models for identifying primary interaction vertices in the **The Muon Spectrometer (TMS)** from 2D scintillator projections. This documentation is structured as a methodology reference for a doctoral thesis.

---

## 1. Physics Context and Experimental Setup

The TMS detector is positioned downstream of a liquid argon (LAr) target. Its primary function is to range out and measure the momentum of muons produced in neutrino interactions. However, the steel planes within the TMS also serve as a target for neutrino-nucleus interactions. A neutrino interacting with a nucleon inside the iron planes produces a hadronic shower and one or more tracks (such as a muon or pion).

To study these interactions, it is essential to correctly reconstruct the 3D vertex position $(x_v, y_v, z_v)$ of the interaction. In experimental datasets, we do not have direct 3D track hits or pre-clustered tracks. We only have 2D line candidates (reconstructed via a Hough transform or similar algorithm) from the U-view (rotated $+1.5^\circ$) and V-view (rotated $-1.5^\circ$) planes.

---

## 2. Coordinate Systems & Geometry

The Cartesian reference coordinate system of the detector is defined as:
- **Z-axis**: Oriented downstream, parallel to the nominal neutrino beam line.
- **X-axis**: Horizontal, transverse to the beam line.
- **Y-axis**: Vertical, transverse to the beam line.

Scintillator strips in alternating planes are tilted at an angle $\theta = 3^\circ$ relative to the X-axis. 
- **U-plane coordinate**: $u = x \cos(\theta/2) + y \sin(\theta/2)$
- **V-plane coordinate**: $v = x \cos(\theta/2) - y \sin(\theta/2)$

For a U-view track line $u(z) = a_u z + b_u$ and a V-view track line $v(z) = a_v z + b_v$, the 3D line trajectory $(x(z), y(z), z)$ is calculated as:
$$x(z) = \frac{a_u + a_v}{2 \cos(1.5^\circ)} z + \frac{b_u + b_v}{2 \cos(1.5^\circ)}$$
$$y(z) = \frac{a_u - a_v}{2 \sin(1.5^\circ)} z + \frac{b_u - b_v}{2 \sin(1.5^\circ)}$$

### Physical Challenges & Constraints:
1. **Y-Coordinate Instability**: Because the stereo angle is very small ($\theta = 3^\circ$), the denominator for the vertical coordinate reconstruction ($2 \sin(1.5^\circ) \approx 0.052$) is extremely small. Consequently, any sub-millimeter shift or discretization error in $u$ or $v$ translates to a $\approx 19.1$-fold error in the vertical coordinate $y$.
2. **Hadronic Shower Visibility (Vertex Z-Shift)**: Neutrino interactions produce a hadronic shower at the true vertex. These short-range, heavily ionizing hadronic particles undergo multiple scattering and are not reconstructed as long Hough lines. Reconstructed tracks (such as muons) often only start showing clear Hough line structures several planes downstream of the true vertex, leading to a systematic downstream shift and a broad tail in Z.
3. **Discretization Limits**: The scintillator strips have a width of $\approx 35\text{ mm}$ and the active planes are separated by $\approx 80\text{ mm}$ of steel absorber. This sets a coarse mechanical limit on single-hit resolution.

---

## 3. Event-Level Direct ML Regression

To resolve the high error rate of baseline geometric solvers (which are heavily corrupted by combinatorial pairing mismatches and downstream hadronic shifts), we implement a direct **Event-Level Machine Learning Regression Model**.

Instead of performing candidate-level classification, we compile the entire event's Hough line information into a global, fixed-size event-level feature vector (23 features).

### Feature Engineering:
1. **Baseline Geometric Reconstructions (`x_reco`, `y_reco`, `z_reco`)**: Solved via track Z-boundary matching and a Point of Closest Approach (PCA) vertex solver.
2. **Line Counts (`nLinesU`, `nLinesV`)**: Number of reconstructed 2D track segments.
3. **Track Kinematics**: Mean, maximum, and sum of track lengths and energy depositions across both views.
4. **Hough Parameters**: Mean slope and intercept of U and V track lines.
5. **Detector Boundaries**: Minimum and maximum longitudinal Z-start and Z-end coordinates of the active hits across all tracks.

### Model Architecture:
We train three independent Gradient Boosting Regressors (`GradientBoostingRegressor`) to directly predict the true primary neutrino interaction vertex coordinates $(x_{true}, y_{true}, z_{true})$ from the event-level feature vector. The regressors automatically learn the non-linear corrections required to map baseline Hough line parameters to the true interaction vertex.

---

## 4. Evaluation and Performance Metrics

The model was trained and evaluated on 1,000 events inside the TMS detector region ($11124\text{ mm} \le Z \le 18544\text{ mm}$), using an 80/20 train/test split.

### A. Residuals (Loss) Comparison (Test set, $N = 188$):
| Metric | Baseline Geometric Solver | Event-Level Direct ML Regression | Error Reduction (%) |
| :--- | :---: | :---: | :---: |
| **Mean 3D Distance (dr)** | 4768.4 mm | **1213.0 mm** | **-74.6%** |
| **Median 3D Distance (dr)** | 2852.0 mm | **911.3 mm** | **-68.0%** |
| **Y-Residual Std Dev (dy)** | 6535.6 mm | **674.8 mm** | **-89.7%** |
| **Z-Residual Std Dev (dz)** | 3884.9 mm | **893.2 mm** | **-77.0%** |
| **X-Residual Std Dev (dx)** | 1036.7 mm | **1054.5 mm** | +1.7% (Noise scale) |

### B. Vertex Reconstruction Efficiency (Reconstruction Accuracy):
Accuracy is defined as the percentage of test set events reconstructed within a given distance threshold (tolerance) from the true primary neutrino vertex:
| Distance Cut (dr) | Baseline Geometric Solver | Event-Level Direct ML Regression | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **dr < 300 mm** (30 cm) | 1.5% | **12.6%** | **+11.1%** (8.4x increase) |
| **dr < 500 mm** (50 cm) | 3.0% | **27.3%** | **+24.2%** (9.1x increase) |
| **dr < 1000 mm** (1.0 m) | 11.6% | **51.5%** | **+39.9%** (4.4x increase) |
| **dr < 1500 mm** (1.5 m) | 20.2% | **63.6%** | **+43.4%** (3.1x increase) |

### C. Physical Interpretation:
1. **Collapsing the Combinatorial Tails**:
   Without ML, the raw geometric solver has massive tail errors (Y-axis standard deviation of 6.5 meters, median 3D error of 2.85 meters) due to mismatches of unrelated tracks. The event-level GBDT regressor successfully identifies these patterns, shrinking the Y-error standard dev by **$89.7\%$** and Z-error standard dev by **$77.0\%$**.
2. **Sub-Meter 3D Precision**:
   The model achieves a median 3D error of **$91.1\text{ cm}$** and reconstructs **$51.5\%$** of all interactions to within 1 meter of their true vertex location, offering a high-efficiency solution for neutrino event reconstruction.
