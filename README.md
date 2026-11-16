# Beyond Confidence: Investigating OOD Detection Failures Under Image Corruptions, Training-Time Augmentation and Threshold Generalization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Framework](https://img.shields.io/badge/Framework-Keras-red)
![Research](https://img.shields.io/badge/Project-Experimental%20Research-purple)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

## Overview

Machine learning models are generally trained on a specific dataset and are expected to make predictions on similar inputs. However, real-world applications frequently encounter data that differs from the training distribution.

A model may receive unfamiliar images, noisy inputs, blurred images or images with significantly different visual characteristics. Even when these inputs are unfamiliar, neural networks can produce highly confident predictions.

**Out-of-Distribution (OOD) Detection** aims to identify such inputs and determine whether they differ from the distribution on which a model was trained.

This project investigates the reliability of OOD detection methods under different distribution shifts, with particular emphasis on the effect of training-time data augmentation, image corruption severity and threshold calibration.

Rather than simply comparing detection algorithms, we conduct controlled experiments to understand **when OOD detection methods succeed, when they fail, and how their behaviour changes under different training and testing conditions.**

---

## Research Problem

A major challenge in OOD detection is that a method performing well on one type of distribution shift may perform poorly on another.

For example:

- A detector may identify completely unfamiliar images but fail to identify corrupted versions of familiar images.
- A model may become more robust to image corruption without becoming better at detecting that corruption.
- A threshold calibrated on clean validation data may not behave consistently when the input distribution changes.
- Confidence-based and feature-based detection methods may respond differently to the same input transformation.

This project investigates these behaviours through a comparative experimental framework.

## Research Questions

The study is guided by the following research questions:

| ID | Research Question |
|---|---|
| RQ1 | How do different image corruption types and severity levels affect OOD detection performance? |
| RQ2 | Do confidence-based detection methods behave differently from feature-based detection methods under distribution shifts? |
| RQ3 | Does training-time data augmentation improve OOD detection and classification robustness? |
| RQ4 | Does higher prediction confidence necessarily indicate that an input belongs to the training distribution? |
| RQ5 | Can thresholds calibrated on clean validation data generalize to corrupted and semantically unfamiliar inputs? |

## Research Objectives

1. Implement and evaluate multiple OOD detection methods using TensorFlow/Keras.
2. Investigate the effect of different image corruptions and severity levels on OOD detection.
3. Compare confidence-based and feature-based detection approaches.
4. Evaluate the influence of training-time data augmentation on model robustness and OOD detection.
5. Investigate threshold calibration and generalization under distribution shifts.
6. Analyse the differences between classification accuracy and OOD detection performance.
7. Identify limitations and failure cases of the evaluated detection methods.

---

## Experimental Framework

### 1. Datasets

The experiments use the following datasets:

| Dataset | Role | Description |
|---|---|---|
| CIFAR-10 | In-Distribution (ID) | Primary dataset used for model training, validation and testing. |
| SVHN | Semantic OOD | Street View House Numbers dataset, used to evaluate detection of unfamiliar image content. |
| Synthetic CIFAR-10 Corruptions | Covariate Shift | Controlled corruptions applied to familiar CIFAR-10 test images. |

**Dataset references:**

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [SVHN](https://ufldl.stanford.edu/housenumbers/)

**Important:** The synthetic corruptions generated in this project are not the official CIFAR-10-C benchmark.

### 2. Types of Distribution Shifts

The project evaluates two primary categories of distribution shifts.

**A. Semantic OOD**

Semantic OOD inputs contain unfamiliar semantic content compared with the training dataset.

Example: A model trained on CIFAR-10 object categories encounters SVHN street-number images.

**B. Covariate Shift**

Covariate-shifted inputs retain their original semantic classes but have altered visual characteristics.

The following synthetic corruptions are implemented:

| Corruption | Description |
|---|---|
| Gaussian Noise | Adds random Gaussian noise to image pixels. |
| Gaussian Blur | Applies a Gaussian blur transformation. |
| Brightness | Changes image brightness. |
| Contrast | Modifies image contrast. |

Each corruption is evaluated at five severity levels (1–5) in the main corruption experiment.

---

## OOD Detection Methods

Four detection methods are implemented and compared.

| Method | Approach | Principle |
|---|---|---|
| MSP | Confidence-based | Uses the maximum predicted softmax probability. |
| Energy | Logit-based | Uses the log-sum-exp of model logits to calculate an energy score. |
| Entropy | Uncertainty-based | Measures uncertainty across the predicted class probabilities. |
| Mahalanobis Distance | Feature-based | Measures the distance between extracted feature representations and learned class distributions. |

### Methodological distinction

MSP, Energy and Entropy use the model's prediction outputs or logits.

Mahalanobis distance uses intermediate feature representations extracted from the CNN.

This comparison allows us to investigate whether prediction-based and feature-based approaches respond differently to semantic and covariate distribution shifts.

All detection scores are oriented so that higher values indicate more OOD-like inputs.

---

## Experimental Design

The project is organised into five main experiments.

### Experiment 1: Effect of Corruption Type and Severity

**Objective:** Investigate how increasing image corruption severity influences OOD detection.

The model is evaluated using clean CIFAR-10 test images and their synthetically corrupted versions.

- Corruptions: Gaussian noise, blur, brightness and contrast.
- Severity levels: 1, 2, 3, 4 and 5.
- Detection methods: MSP, Energy, Entropy and Mahalanobis.

**Measurements:**
- AUROC
- AUPR
- FPR95 (as implemented in the notebook)
- Classification accuracy on corrupted images

**Expected analysis:** Determine whether detection performance changes consistently with increasing corruption severity and whether different methods respond differently to each corruption.

### Experiment 2: Confidence and Uncertainty Analysis

**Objective:** Investigate whether model confidence and prediction uncertainty can reliably distinguish familiar from unfamiliar inputs.

The following quantities are evaluated:

- Maximum Softmax Probability (MSP)
- Predictive Entropy
- Energy scores
- Classification correctness

The experiment investigates whether incorrect or unfamiliar predictions can still receive high confidence scores.

### Experiment 3: Comparison of Detection Methods

**Objective:** Compare confidence-based and feature-based OOD detection methods.

The four methods are evaluated on:

- Clean CIFAR-10 versus SVHN.
- Clean CIFAR-10 versus corrupted CIFAR-10.
- Different corruption severities.

The comparison helps identify which methods perform better under particular distribution shifts.

### Experiment 4: Effect of Training-Time Data Augmentation

**Objective:** Investigate whether data augmentation during training improves OOD detection reliability.

Two CNN models are trained:

| Model | Training Strategy |
|---|---|
| Model A | Clean CIFAR-10 training images |
| Model B | CIFAR-10 training images with random data augmentation |

The augmentation strategy includes:

- Random horizontal flipping.
- Random translation.
- Random rotation.
- Random contrast adjustment.

Both models are evaluated using the same datasets and detection methods.

**Primary comparison:** Classification robustness and OOD detection performance under semantic and covariate shifts.

### Experiment 5: Threshold Calibration and Generalization

**Objective:** Investigate whether a threshold calibrated on clean data remains reliable under shifted input distributions.

Two calibration strategies are compared:

**A. Clean-only calibration**

The threshold is selected using clean CIFAR-10 validation images.

**B. Corruption-aware calibration**

The threshold is selected using clean validation images combined with selected low-severity corrupted validation images.

The fixed thresholds are then evaluated on held-out clean test images, SVHN and shifted CIFAR-10 images.

For corruption-aware calibration, severities 1 and 2 are used, while stronger severities are evaluated separately.

This experiment investigates whether exposure to known distribution shifts during calibration improves generalization to stronger or different shifts.

---

## Model Architecture

A custom Convolutional Neural Network (CNN) is implemented using TensorFlow/Keras.

The architecture consists of:

1. Input layer: 32 × 32 × 3.
2. Convolutional layer with 32 filters.
3. Max-pooling layer.
4. Convolutional layer with 64 filters.
5. Max-pooling layer.
6. Convolutional layer with 128 filters.
7. Global Average Pooling.
8. Dense feature layer with 128 units.
9. Output layer with 10 logits.

The model exposes two outputs:

- **Logits:** Used for MSP, Energy and Entropy calculations.
- **Intermediate features:** Used for Mahalanobis distance calculations.

Both clean-trained and augmentation-trained models use the same basic CNN architecture.

---

## Evaluation Metrics

The following metrics are used to evaluate the detection methods:

| Metric | Purpose |
|---|---|
| AUROC | Measures the ability to distinguish ID and OOD samples across thresholds. |
| AUPR | Evaluates precision-recall performance for OOD detection. |
| FPR95 | Evaluates false-positive behaviour at a specified ROC operating point. |
| ID Rejection Rate | Measures the percentage of familiar images incorrectly rejected. |
| Classification Accuracy | Measures the classifier's ability to correctly classify familiar images. |
| OOD Rejection Rate | Measures the percentage of unfamiliar inputs rejected at a selected threshold. |

**Note:** AUROC measures ranking performance rather than the performance of one fixed threshold. Threshold-based rejection rates are therefore analysed separately.

The exact FPR95 convention should be checked against the implementation before comparing the value with published benchmarks.

---

## Preliminary Experimental Findings

The following observations are from the current experimental run. They should be interpreted as preliminary results rather than statistically established conclusions.

### 1. Semantic OOD Detection: CIFAR-10 vs SVHN

| Detection Method | Clean Training | Augmented Training |
|---|---:|---:|
| Energy | 0.7148 | **0.8211** |
| Entropy | 0.7279 | **0.7884** |
| MSP | 0.7463 | **0.7774** |
| Mahalanobis | 0.5977 | **0.6407** |

**Observation:** Augmentation improved AUROC across all four methods in the current SVHN experiment. Energy achieved the highest AUROC of 0.8211 with augmentation.

### 2. Covariate-Shift Detection

The results were not consistent across all corruption types.

| Corruption | Main Observation |
|---|---|
| Blur | Clean training performed better at medium severity, while augmentation improved selected methods at high severity. |
| Brightness | Confidence-based methods showed small improvements with augmentation at lower and medium severity, with mixed results at high severity. |
| Gaussian Noise | Mahalanobis performed particularly well with clean training at higher severity, while augmentation improved MSP and Entropy at some higher severities. |

**Observation:** The effectiveness of data augmentation depended on corruption type, severity and detection method.

### 3. Classification Robustness

Some improvements in classification accuracy were observed with augmentation:

| Condition | Clean Training | Augmented Training |
|---|---:|---:|
| Blur — Severity 3 | 33.82% | **52.78%** |
| Blur — Severity 5 | 27.05% | **39.94%** |
| Brightness — Severity 5 | 46.59% | **53.48%** |
| Gaussian Noise — Severity 5 | **21.66%** | 21.31% |

**Observation:** Augmentation improved classification accuracy for selected corruptions, but it did not consistently improve performance under Gaussian noise.

### 4. Threshold Generalization

Threshold behaviour varied across distribution shifts.

For example, under Gaussian noise, the Mahalanobis detector's clean-calibrated rejection rate increased from approximately 4.29% at severity 1 to 96.04% at severity 5.

However, Mahalanobis behaved differently under blur and brightness changes.

**Observation:** A threshold calibrated on clean images did not maintain consistent rejection behaviour across all shifted conditions.

### Overall Preliminary Finding

The current results suggest that:

- Augmentation can improve semantic OOD detection.
- Improvements in classification robustness do not necessarily translate into improvements in OOD detection.
- Different detection methods respond differently to different distribution shifts.
- Threshold calibration is sensitive to the characteristics of the calibration and evaluation distributions.

There is no evidence from these experiments that one detection method or training strategy is universally superior.

---

## Technology Stack

| Technology | Usage |
|---|---|
| Python | Primary programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN construction and training |
| TensorFlow Datasets | Loading SVHN |
| NumPy | Numerical operations |
| Pandas | Experimental result analysis |
| Scikit-learn | Evaluation metrics, covariance estimation and dimensionality reduction |
| Matplotlib | Visualisation |
| Seaborn | Statistical plots |
| Google Colab | Development and experimentation |

---

## Getting Started

### Requirements

- Google account.
- Google Colab.
- Internet connection for downloading datasets and dependencies.
- GPU runtime recommended for faster training.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git

cd <your-repository>
```

### 2. Open the Notebook in Google Colab

Open the `.ipynb` notebook from your cloned repository in Google Colab.

Alternatively, upload the notebook directly to Colab.

### 3. Install Dependencies

The notebook installs the required dependencies.

For a local Python environment, install:

```bash
pip install tensorflow tensorflow-datasets scikit-learn pandas numpy matplotlib seaborn
```

### 4. Configure the Experiment

The notebook contains configurable parameters:

```python
SEED = 42

FAST_RUN = False

EPOCHS = 15

BATCH_SIZE = 128

MAX_ID_EVAL = 10000

MAX_OOD_EVAL = 10000
```

For an initial pipeline check, enable `FAST_RUN`.

For more complete experiments, use the full training data, increase the number of epochs if necessary, and repeat training with multiple random seeds.

### 5. Execute the Notebook

Run all cells sequentially.

The notebook performs:

1. Dataset loading and preprocessing.
2. CNN model construction.
3. Model training.
4. Synthetic corruption generation.
5. OOD score calculation.
6. Detection metric evaluation.
7. Threshold calibration.
8. Clean versus augmented training comparison.
9. Feature-space visualisation.
10. Result export.

### 6. Access the Results

The notebook saves experiment outputs to:

```text
/content/ood_research_results/
```

The output directory includes:

```text
ood_research_results/
├── semantic_ood_results.csv
├── corruption_results.csv
├── threshold_results.csv
├── training_strategy_results.csv
├── model_clean.keras
├── model_augmented.keras
└── experiment_metadata.json
```

The notebook also packages the generated results into a ZIP archive for download.

---

## Repository Structure

The recommended repository organisation is:

```text
ood-detection-research/
│
├── README.md
│
├── notebooks/
│   └── ood_detection_experiments.ipynb
│
├── src/
│   ├── model.py
│   ├── corruptions.py
│   ├── detection_methods.py
│   ├── evaluation.py
│   └── calibration.py
│
├── results/
│   ├── semantic_ood_results.csv
│   ├── corruption_results.csv
│   ├── threshold_results.csv
│   └── training_strategy_results.csv
│
├── figures/
│   ├── corruption_severity.png
│   ├── semantic_ood_comparison.png
│   └── feature_visualization.png
│
├── references/
│   └── literature_review.md
│
└── requirements.txt
```

This is the proposed organisation for the GitHub project. The current implementation is primarily contained in the Google Colab-exported Python notebook source; the modular files can be introduced as the project is organised for reuse.

---

## Limitations

The current experimental framework has several limitations:

1. **Limited datasets:** CIFAR-10 and SVHN are used for the current evaluation. More datasets are needed to establish broader generalization.
2. **Synthetic corruptions:** The corruptions are controlled transformations and are not equivalent to the official CIFAR-10-C benchmark.
3. **Limited architecture:** The experiments use a relatively small custom CNN.
4. **Training randomness:** The current comparison uses different random seeds for clean and augmented training. Multiple matched-seed experiments are needed for stronger conclusions.
5. **Statistical significance:** Confidence intervals and statistical significance tests have not yet been established.
6. **Threshold dependence:** Rejection rates depend on the selected calibration strategy and operating threshold.
7. **Feature-space assumptions:** Mahalanobis detection depends on the quality of learned features and covariance estimation.
8. **Dataset-level differences:** CIFAR-10 and SVHN differ in image characteristics and acquisition domains in addition to semantic content.

The results should therefore be interpreted as findings from a controlled student research experiment rather than universal claims about OOD detection.

---

## Future Work

Potential extensions include:

- Evaluating the methods on the official CIFAR-10-C benchmark.
- Introducing additional semantic OOD datasets.
- Repeating training with multiple matched random seeds.
- Reporting confidence intervals for AUROC.
- Investigating temperature scaling and improved uncertainty calibration.
- Exploring Monte Carlo Dropout and ensemble-based uncertainty.
- Investigating adversarial examples and their effect on OOD detectors.
- Comparing additional feature-space detection methods.
- Packaging corruption generation and evaluation functions into a reusable OOD evaluation toolkit.
- Exploring opportunities to contribute useful improvements to existing open-source OOD detection frameworks.

---

## Research References

The following papers provide the theoretical foundation for the implemented detection methods and benchmarking approach.

1. **Hendrycks, D. and Gimpel, K. (2017).** A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.  
   [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)

2. **Liu, W. et al. (2020).** Energy-based Out-of-distribution Detection.  
   [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)

3. **OpenOOD: Benchmarking Generalized Out-of-Distribution Detection.**  
   [arXiv:2210.07242](https://arxiv.org/abs/2210.07242)  
   [GitHub Repository](https://github.com/Jingkang50/OpenOOD)

4. **CIFAR-10 Dataset.**  
   [Dataset Website](https://www.cs.toronto.edu/~kriz/cifar.html)

5. **Street View House Numbers Dataset.**  
   [Dataset Website](https://ufldl.stanford.edu/housenumbers/)

---

## Authors

**Developed as an academic Machine Learning research mini-project.**

- Member 1: [Your Name]
- Member 2: [Your Teammate's Name]

**Institution:** Vivekanand Education Society's Institute of Technology (VESIT), Mumbai  
**Department:** Artificial Intelligence and Data Science

---

## Acknowledgements

We acknowledge the researchers and open-source communities whose work on Out-of-Distribution Detection, uncertainty estimation and benchmarking provides the foundation for this investigation.

This project is intended as an educational and experimental research contribution. It is not an official implementation of the OpenOOD benchmark.

---


