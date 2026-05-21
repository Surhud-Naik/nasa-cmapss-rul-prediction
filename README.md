# Aircraft Engine Remaining Useful Life (RUL) Prediction

This repository implements a complete machine learning pipeline to estimate the **Remaining Useful Life (RUL)** of turbofan aircraft engines using multi-channel sensor data. Utilizing the classic **NASA CMAPSS (FD001) dataset**, this project develops, optimizes, and evaluates a variety of modeling architectures, ranging from baseline statistical regressions to deep sequential architectures (LSTMs and 1D CNNs).

## Project Objectives & Accomplishments

* **End-to-End Pipeline Creation:** Built a modular Python-based workflow implementing data ingestion, target label generation, exploratory analysis, preprocessing, and model evaluation.
* **Piece-Wise Linear Target Construction:** Formulated the continuous inductive target labels ($RUL$) by calculating maximum running cycles per engine unit and applied a domain-standard $125$-cycle ceiling threshold to limit early-life noise.
* **Leakage-Free Engine-Level Splits:** Implemented a rigorous train/test division stratified strictly at the engine level ($80/20$) to prevent temporal data leakage across time-series histories.
* **Advanced Feature Engineering:** Extracted temporal trends per engine unit group by creating rolling statistics ($15$-cycle window), short-term temporal dynamics via lag states, and calculated overall cumulative asset degradation.
* **Robust Preprocessing Flow:** Sandboxed models using outlier clipping ($\pm3\sigma$), standard scaling, and automated low-variance feature thresholding fit exclusively on training splits.
* **Multi-Architecture Benchmark:** Implemented, tuned, and evaluated 5 distinct algorithmic approaches including Classical Regressions, Ensemble Trees, Fully-Connected Neural Networks, LSTMs, and Temporal 1D Convolutional Networks.

---

## 📊 Model Performance Summary

The metrics below represent predictive accuracy ($R^2$ Score) evaluated on held-out test engines. Models are strictly evaluated under physical constraints where negative predictions are post-processed to $0$.

| Model Type | Architecture Details | Baseline $R^2$ | Tuned / Final $R^2$ | Final RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Ordinary Least Squares $\rightarrow$ 2nd Degree Polynomial Expansion | $0.803480$ | $0.831801$ | $17.1085$ |
| **Random Forest** | 200 trees $\rightarrow$ 400 trees, Max Depth: 15, Leaf Samples: 5 | $0.842333$ | **0.868517** | **15.1263** |
| **Neural Network (FCNN)** | 3-Layer Dense Net $\rightarrow$ LeakyReLU Upgrade, Epoch Extensions | $0.836965$ | $0.844105$ | $16.4708$ |
| **1D CNN** | Temporal Convolutions (64 & 128 channels, Sliding Window = 20) | — | $0.839684$ | $16.7499$ |
| **LSTM** | 128 Hidden Units Sequential RNN (Sliding Window = 20) | — | $0.838396$ | $16.8170$ |

> **Key Takeaway:** The **Tuned Random Forest Regressor** achieved the highest overall accuracy ($R^2 = 0.8685$). Because the FD001 subset contains a single operating condition and single fault mode, its asset degradation trend is smooth and near-linear post-clipping. Therefore, well-regularized tabular architectures pairing explicit temporal features (lag, rolling mean) can match or out-perform complex deep sequence networks on simpler prognostic settings.

---

## Tech Stack & Dependencies

* **Core Data Wrangling:** `numpy`, `pandas`
* **Modeling Frameworks:** `scikit-learn`, `torch` (PyTorch)
* **Statistical Analysis:** `scipy`
* **Visualization Engine:** `matplotlib`, `seaborn`

---

## Final Conclusion

The project demonstrates that ensemble learning and sequence-based deep learning architectures are highly effective for Remaining Useful Life prediction tasks.

Among all evaluated approaches:

- **Random Forest** achieved the highest classical ML performance.
- **1D CNN** achieved the best sequence-model performance.
- Temporal models such as LSTM and CNN were particularly effective in learning degradation behavior from sequential sensor measurements.
