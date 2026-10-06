## Code Summary
< Periodicity-Driven Adaptive Localization of Respiratory Regions for Non-Contact Respiration Monitoring in Sleeping Infants >
- Published in Bioengineering (MDPI): [Paper](https://www.mdpi.com/2306-5354/13/10/1158)
- We propose a method for estimating respiratory rate by adaptively identifying respiration-related regions, without requiring deep learning pre-training or predefined anatomical ROIs such as the chest or abdomen.

- Model Arch
<img width="1449" height="298" alt="Figure 1  Architecture of the spatial respiratory region estimation model" src="https://github.com/user-attachments/assets/23e3d32f-e724-4cc7-b2ed-3cbcd61d443b" />

#### Method Overview
1. Divide each video frame into a 32 × 18 spatial grid.
2. Extract chrominance/intensity and vertical optical-flow signals from each grid cell.
3. Compute respiration-related periodicity within the 0.2–0.8 Hz frequency band.
4. Select the grid cell with the strongest fused periodicity as the Anchor Grid Cell.
5. Estimate respiratory rate from the dominant frequency of the selected respiratory signal.

#### Code
- `src/main.py` : Main respiration-region localization and respiratory-rate estimation pipeline
- `src/validation.py` : Subject-wise cross-validation and performance evaluation
- `src/6fold_explanation.md` : Description of the 6-fold validation protocol

#### Dataset
The dataset used in this study is based on the data available on the GitHub repository linked below.
- link : https://github.com/michaelwwan/air-400
