# Particle Identification (PID) and PDG Prediction in the TMS Steel Spectrometer

This document details the design, physics motivation, and performance of the Machine Learning Particle Identification (PID) classifier developed for **The Muon Spectrometer (TMS)**.

---

## 1. Physics Motivation for Particle Identification

The Muon Spectrometer (TMS) is primarily designed to measure muon momentum via range and curvature. However, deep-inelastic scattering (DIS) and resonant neutrino interactions generate other final-state particles (hadrons and electrons) that enter the spectrometer alongside the primary muon. Distinguishing these species is essential for:
- Reconstructing the neutrino energy (which requires summing hadron and lepton energy correctly).
- Eliminating backgrounds from cosmic rays and neutral current interactions.

Different particles interact with the steel absorber plates in characteristic ways:
1. **Muons ($\mu^\pm$, PDG $\pm 13$)**: Minimum ionizing particles (MIPs) that do not undergo strong interactions. They penetrate deeply, forming very long, clean, and straight tracks.
2. **Charged Pions ($\pi^\pm$, PDG $\pm 211$)**: Undergo hadronic interactions in the steel plates, resulting in nuclear scattering, larger track curvature residuals (non-straightness), and shorter track lengths.
3. **Protons ($p$, PDG $2212$)**: Highly ionizing at typical neutrino-interaction energies. They stop rapidly in the steel, leaving short tracks with extremely high energy deposition per unit length ($dE/dx$).
4. **Electrons ($e^\pm$, PDG $\pm 11$)**: Undergo bremsstrahlung, immediately initiating electromagnetic showers that stop in the first few steel planes, yielding short, high-density hit clusters.

---

## 2. Engineered Track-Level Features

For each reconstructed track, we extract 16 physics-motivated features:
- **`momentum`**: The track momentum (in MeV/c). This is the single strongest classifier because muons typically carry much higher momentum than final-state hadrons.
- **`length` & `length_3d`**: The Euclidean and 3D track lengths.
- **`nhits`**: The number of scintillator hits in the track.
- **`z_depth`**: The longitudinal depth of the track (maximum Z - minimum Z), measuring penetration power.
- **`straightness`**: The mean square residuals of the hits relative to a 3D PCA line fit, proxying nuclear scattering.
- **`total_energy` & `dedx`**: The total energy deposit and average energy deposit per unit length ($dE/dx$).
- **`dca` & `angle`**: Distance of closest approach and track angle relative to the reconstructed vertex.
- **`length_rank` & `z_depth_rank`**: Topological ranks within the event (muons are typically the longest and deepest tracks).

---

## 3. Classifier Performance (Balanced Random Forest)

We trained a **Balanced Random Forest Classifier** to resolve the class imbalance (muons dominate the dataset). 

### A. Classification Report:
The model achieves an overall classification accuracy of **$59\%$** across the 5 primary classes on the test set ($N=583$ tracks):

| Particle Class | PDG Codes | Test Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Muon** | $\pm 13$ | 282 | **$88\%$** | $79\%$ | **$83\%$** |
| **Pion** | $\pm 211$ | 140 | $47\%$ | $45\%$ | $46\%$ |
| **Proton** | $2212$ | 60 | $32\%$ | **$45\%$** | $38\%$ |
| **Electron** | $\pm 11$ | 9 | $07\%$ | **$22\%$** | $11$ |
| **Other** | - | 92 | $39\%$ | $35\%$ | $37\%$ |

### B. Confusion Matrix Plot:
The heatmap below shows the predicted vs. actual classification matrix:

![PID Confusion Matrix](/C:/Users/Moktan/.gemini/antigravity-cli/brain/45e71d8c-f6eb-4ea0-a644-9dd3c5e4ec20/pid_confusion_matrix.png)

---

## 4. Feature Importances

The figure below shows the relative importance score of each feature in the Random Forest model:

![PID Feature Importances](/C:/Users/Moktan/.gemini/antigravity-cli/brain/45e71d8c-f6eb-4ea0-a644-9dd3c5e4ec20/pid_feature_importances.png)

### Key Interpretations:
1. **`momentum` & `length_3d`**: Are confirmed as the most critical features for distinguishing muons from other particles.
2. **`start_z` & `end_z`**: Reflect whether the track started inside the active vertex region or entered from outside, which is highly predictive of primary vs. secondary particles.
3. **`dedx`**: Provides clean separation for protons (which have much higher average energy deposition per mm).

---

## 5. Saved Assets & Pipeline
- **Training Script**: [train_particle_pid_classifier.py](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/train_particle_pid_classifier.py)
- **Model**: [particle_pid_classifier.joblib](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/particle_pid_classifier.joblib)
- **Inference Pipeline**: [particle_pid_pipeline.py](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/particle_pid_pipeline.py)
- **Output Predictions**: [predicted_particle_pdgs.csv](file:///C:/MY_CODES/NIRAB_DAI/AGY_PATCH/predicted_particle_pdgs.csv)
