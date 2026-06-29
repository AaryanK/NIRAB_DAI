# Presentation Slide Guide: TMS Vertex Reconstruction & Particle Identification

This presentation guide summarizes the physical motivation, machine learning methodologies, and key performance results of our study. Use this structure for your presentation slides.

---

## Slide 1: Project Overview
* **Title**: Machine Learning Assisted Reconstruction in the TMS Detector
* **Goal**: Optimize 3D Primary Interaction Vertex Reconstruction and Particle Identification (PID) in **The Muon Spectrometer (TMS)**.
* **Core Datasets**: `Line_Candidates` (2D Hough Segments), `Reco_Tree` (Reconstructed 3D Tracks), and `Truth_Info` (True Monte Carlo coordinates and PDG codes).
* **Scope**: 1,380 Simulated Neutrino Interaction Events in `Cut2.root`.

---

## Slide 2: The Physical Challenges of TMS Geometry
* **Stereo Angle Limitation**:
  - Scintillator strips in alternating planes are tilted at $\pm 3^\circ$ relative to vertical Y-bar orientation.
  - The small stereo angle ($6^\circ$ total) geometrically amplifies the vertical coordinate ($y$) reconstruction uncertainty by **9.5-fold** relative to horizontal ($x$) measurements.
  - *Formula*: $y = \frac{u - v}{2 \sin(3^\circ)}$
* **Hadronic Shower Z-Shift**:
  - The neutrino interaction point is surrounded by a dense hadronic shower.
  - Tracks can only be reconstructed as Hough lines downstream of the shower core. This creates a systematic downstream shift in the reconstructed Z coordinate.

---

## Slide 3: Vertex Reconstruction Methodology
We evaluated two competing paradigms to reconstruct the primary interaction vertex $(x, y, z)$:

* **Approach A: 2D Hough Lines (`Line_Candidates`)**:
  1. Pair 2D lines in U and V views dynamically by matching Z boundaries.
  2. Compute starting coordinates $x_{start}, y_{start}$ at the Z overlap start.
  3. Extract 23 track-level and event-level features.
  4. Train GBDT regressors to predict true vertex $(x, y, z)$.
* **Approach B: Reconstructed 3D Tracks (`Reco_Tree`)**:
  1. Leverage fully fitted 3D tracks from global plane hit clustering.
  2. Solve for the 3D track intersection using a **3D Point of Closest Approach (PCA)** linear solver.
  3. Extract 17 track-level features.
  4. Train GBDT regressors to refine and predict true vertex $(x, y, z)$.

---

## Slide 4: Optimization & Ensembling (Sandbox Study)
To push reconstruction accuracy to the physical limit, we built an optimized ensemble model in a sandbox:
* **Physics-Motivated Feature Enrichment**:
  - **Hadronic Energy**: Summed the first 5 hit energies (`TrackHitEnergies`) near the track starts to proxy the hadronic shower density.
  - **Track Scattering**: Calculated the difference between the 3D track length and the straight-line distance.
* **Ensemble Architecture**:
  - Combined a tuned **Gradient Boosting Regressor**, a **Random Forest Regressor**, and a **Multi-Layer Perceptron (MLP) Neural Network** using a `VotingRegressor` ensemble.
  - **Convergence Tuning**: Increased MLP iterations to `1200` with early stopping to resolve convergence limits and correct the Z-bias.

---

## Slide 5: Vertex Reconstruction Results
*The machine learning model successfully corrects the systematic offsets, reducing the median 3D error by **$72.4\%$**:*

| Reconstruction Step | Median 3D Error ($dr$) | Y-Resolution ($\sigma_y$) | Z-Resolution ($\sigma_z$) |
| :--- | :---: | :---: | :---: |
| **Approach A (Baseline 2D Lines)** | $2307.5\text{ mm}$ | $3788.9\text{ mm}$ | $3201.1\text{ mm}$ |
| **Approach A (ML Refined)** | $892.2\text{ mm}$ | $667.3\text{ mm}$ | $906.3\text{ mm}$ |
| **Approach B (Baseline 3D Tracks)** | $1197.0\text{ mm}$ | $809.8\text{ mm}$ | $2362.9\text{ mm}$ |
| **Approach B (ML Refined)** | $686.6\text{ mm}$ | $589.0\text{ mm}$ | $692.8\text{ mm}$ |
| **Sandbox Ensemble (Optimized)** | **$646.4\text{ mm}$** | **$570.1\text{ mm}$** | **$657.5\text{ mm}$** |

* **Containment**: **$50.4\%$** of interactions are reconstructed to within **1 meter** and **$74.3\%$** within **1.5 meters**.

---

## Slide 6: Particle Identification (PID) Methodology
* **Task**: Predict the true Particle Data Group (PDG) ID code of each reconstructed track.
* **Primary Classes**: Muon ($\pm 13$), Pion ($\pm 211$), Proton ($2212$), Electron ($\pm 11$), and Other.
* **Model**: **Balanced Random Forest Classifier** (150 estimators, max depth 8).
* **Class Imbalance Resolution**: Utilized balanced class weighting to prevent the dominant muon class ($1,409$ tracks) from overshadowing the minor classes (protons, electrons).
* **Key Features**: Track momentum (MeV/c), 3D length, track energy deposit, average energy loss per unit length ($dE/dx$).

---

## Slide 7: Particle Identification Results
*The model successfully isolates muons with high purity and recovers minority hadrons:*

* **Classification Quality**:
  - **Muon**: **$88\%$ Precision**, **$79\%$ Recall** (F1-Score: **$83\%$**).
  - **Pion**: **$47\%$ Precision**, **$45\%$ Recall** (F1-Score: **$46\%$**).
  - **Proton**: **$32\%$ Precision**, **$45\%$ Recall** (F1-Score: **$38\%$**).
  - **Electron**: **$07\%$ Precision**, **$22\%$ Recall** (F1-Score: **$11\%$**).
* **Feature Importances**:
  - **`momentum`** and **`length_3d`** are the strongest discriminators.
  - **`dedx`** ($dE/dx$) provides clean separation for highly ionizing protons.

---

## Slide 8: Summary of Work Accomplished
We delivered a complete, production-ready reconstruction codebase:
1. **Interactive Notebooks & Documentation**: Walks through data extraction, training, residual analyses, and physics derivations.
2. **Batch Inference Pipelines**: Reconstructs vertices ([predicted_vertices_event_level.csv](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/predicted_vertices_event_level.csv)) and particle PDG codes ([predicted_particle_pdgs.csv](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/predicted_particle_pdgs.csv)) on the full dataset.
3. **Repository Sync**: The complete suite is pushed and live on GitHub:
   https://github.com/AaryanK/NIRAB_DAI.git
