# GREEN-IDS: Build Plan & Implementation Roadmap

**Project**: Green Intrusion Detection System  
**Framework**: Green Learning (GL) - Sustainable ML Paradigm  
**Target Dataset**: UNSW-NB15 Network Intrusion Detection Dataset  
**Status**: Planning Phase

---

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Green Learning Overview](#green-learning-overview)
3. [Project Architecture](#project-architecture)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Technical Specifications](#technical-specifications)
6. [Success Criteria](#success-criteria)

---

## Executive Summary

**Green-IDS** is an intrusion detection system built on the **Green Learning (GL)** paradigm—a sustainable alternative to Deep Learning that achieves comparable performance with:
- **Low carbon footprint** (minimal training/inference energy)
- **Small model sizes** (edge-deployable)
- **High interpretability** (logical, transparent decisions)
- **Fast inference** (suitable for real-time detection)

### Key Objectives
1. Implement GL framework for network intrusion detection
2. Achieve F1-score comparable to state-of-the-art IDS
3. Reduce computational complexity and carbon footprint vs Deep Learning
4. Demonstrate edge-device deployment readiness
5. Provide transparent, auditable decision-making

---

## Green Learning Overview

### What is Green Learning?

Green Learning is an alternative ML paradigm that addresses two critical challenges with Deep Learning:

**Challenge 1: Environmental Sustainability**
- DL requires massive compute resources (GPUs/TPUs) and energy
- Training large models on huge datasets has high carbon footprint
- Not suitable for resource-constrained environments

**Challenge 2: Trustworthiness**
- DL models are "black boxes" - decisions cannot be easily explained
- Unsuitable for high-stakes applications (medical, security, finance)
- Adversarial vulnerability and lack of interpretability

### GL Approach: Three-Module Architecture

Green Learning uses a **modularized, cascading design** with three distinct modules:

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: Raw Network Traffic / Network Data                    │
└────────────────────────────┬─────────────────────────────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │ MODULE 1: REPRESENTATION LEARNING      │
        │ • Saab/Saak Transforms                 │
        │ • Unsupervised dimensionality reduction│
        │ • Remove redundancy, keep signal       │
        └────────────────────┬──────────────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │ MODULE 2: FEATURE LEARNING             │
        │ • Discriminant Feature Test (DFT)      │
        │ • Supervised feature selection         │
        │ • Rank by relevance to attack/normal   │
        └────────────────────┬──────────────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │ MODULE 3: DECISION LEARNING            │
        │ • Subspace Learning Machine (SLM)      │
        │ • Space partitioning classifier        │
        │ • Ensemble-based decision making       │
        └────────────────────┬──────────────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │ OUTPUT: Detection (Attack/Normal)      │
        │ + Confidence Scores & Explanations     │
        └────────────────────────────────────────┘
```

### Why Three Modules?

**Separation of Concerns:**
- **Module 1** focuses purely on representation quality
- **Module 2** incorporates label information for discrimination
- **Module 3** optimizes for final classification performance

**Advantages:**
- Each module uses appropriate techniques (unsupervised, supervised, ensemble)
- Individual optimization for efficiency
- Explicit, observable intermediate results (not latent)
- Easy to debug and interpret each stage

---

## Project Architecture

### Technical Stack
- **Language**: Python 3.8+
- **Dependencies**: NumPy, Pandas, Scikit-learn, Matplotlib
- **Dataset**: UNSW-NB15 (176,000 training + 82,000 test records)
- **Hardware Target**: CPU-based, edge-device compatible

### Directory Structure
```
green-ids/
├── data/
│   ├── unsw_nb15/
│   │   ├── training_set.csv
│   │   ├── test_set.csv
│   │   └── features.md
│   └── preprocessing.py
├── src/
│   ├── module1_representation/
│   │   ├── saab_transform.py
│   │   ├── pixelhop_units.py
│   │   └── representation_pipeline.py
│   ├── module2_feature_learning/
│   │   ├── dft_selector.py
│   │   └── feature_analysis.py
│   ├── module3_decision_learning/
│   │   ├── slm_classifier.py
│   │   ├── slm_forest.py
│   │   └── slm_boost.py
│   ├── gl_pipeline.py
│   └── utils.py
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   └── test_integration.py
├── evaluation/
│   ├── benchmark.py
│   ├── latency_profiler.py
│   └── carbon_analyzer.py
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── METHODOLOGY.md
└── README.md
```

---

## Implementation Roadmap

### Phase 1: Setup & Data Preparation (Week 1)
**Goals**: Initialize development environment, prepare dataset

- [ ] **Task 1.1**: Setup Python environment
  - Create virtual environment
  - Install dependencies (numpy, pandas, scikit-learn)
  - Setup testing framework (pytest)
  
- [ ] **Task 1.2**: UNSW-NB15 Dataset Preparation
  - Download and explore dataset
  - Analyze feature types and distributions
  - Implement data loading pipeline
  - Create train/validation/test splits

**Deliverables**: 
- Working development environment
- Dataset loading module
- Data exploration report

---

### Phase 2: Module 1 - Unsupervised Representation Learning (Weeks 2-3)
**Goal**: Extract rich, low-dimensional representations from raw network data

#### 2.1 Implement Saab Transform
- [ ] **Task 2.1.1**: Understand Saab theory
  - Study PCA-based dimension reduction
  - Understand DC/AC kernels
  - Learn bias term handling

- [ ] **Task 2.1.2**: Implement Saab Transform
  ```python
  class SaabTransform:
      def __init__(self, num_components=25, bias=True):
          # DC kernel + PCA-based AC kernels
          self.dc_kernel = None
          self.ac_kernels = None
          
      def fit(self, X):
          # Compute DC kernel (mean)
          # Compute AC kernels (PCA)
          # Handle energy thresholding
          
      def transform(self, X):
          # Apply DC kernel
          # Apply AC kernels
          # Return compact representation
  ```

#### 2.2 Implement PixelHop Units
- [ ] **Task 2.2.1**: Multi-stage representation pipeline
  - Design cascading PixelHop units
  - Implement spatial pooling
  - Handle tensor operations (spectral-spatial)

- [ ] **Task 2.2.2**: Channel-wise Saab optimization
  - Decompose 3D tensors efficiently
  - Apply Saab per channel
  - Reduce model size

#### 2.3 Validation
- [ ] **Task 2.3.1**: Representation quality assessment
  - Measure dimensionality reduction
  - Verify information preservation
  - Visualize learned representations

**Deliverables**:
- Saab Transform module
- PixelHop pipeline
- Representation validation tests
- Quality report

**Success Metrics**:
- Dimension reduction ratio > 10x
- Information preservation (correlation with labels) > 0.8

---

### Phase 3: Module 2 - Supervised Feature Learning (Week 4)
**Goal**: Select most discriminative features using label information

#### 3.1 Implement Discriminant Feature Test (DFT)
- [ ] **Task 3.1.1**: DFT algorithm implementation
  ```python
  class DiscriminantFeatureTest:
      def fit(self, representations, labels):
          # Rank features by discriminant power
          # Auto-determine threshold
          # Return feature importance scores
          
      def select_features(self, threshold=None):
          # Return mask of discriminant features
          # Can use auto-threshold or manual
  ```

#### 3.2 Feature Analysis
- [ ] **Task 3.2.1**: Apply DFT to UNSW-NB15
  - Rank features by importance
  - Identify attack signatures
  - Generate feature statistics

- [ ] **Task 3.2.2**: Feature set validation
  - Verify discriminant features separate classes
  - Measure class separability improvements
  - Document top features

**Deliverables**:
- DFT implementation
- Feature selection results
- Feature importance ranking
- Class separability analysis

**Success Metrics**:
- Feature reduction > 90% (keep top 10%)
- Class separability metric > 0.9

---

### Phase 4: Module 3 - Supervised Decision Learning (Week 5)
**Goal**: Train ensemble classifier on selected features

#### 4.1 Implement Subspace Learning Machine
- [ ] **Task 4.1.1**: SLM classifier (tree-based)
  ```python
  class SubspaceLearningMachine:
      def __init__(self, max_depth=10):
          self.tree = None
          
      def fit(self, X, y):
          # Build decision tree with space partitioning
          # Optimize splits for class separation
          
      def predict(self, X):
          # Return class predictions
          
      def predict_proba(self, X):
          # Return confidence scores
  ```

#### 4.2 Implement Ensemble Methods
- [ ] **Task 4.2.1**: SLM Forest (bagging)
  - Multiple SLM trees
  - Majority voting
  - Confidence aggregation

- [ ] **Task 4.2.2**: SLM Boost (boosting)
  - Adaptive reweighting
  - Iterative tree building
  - Error reduction

#### 4.3 Training & Validation
- [ ] **Task 4.3.1**: Train on UNSW-NB15
  - Hyperparameter tuning
  - Cross-validation
  - Performance evaluation

**Deliverables**:
- SLM classifier implementation
- SLM Forest ensemble
- SLM Boost variant
- Training pipeline
- Validation results

**Success Metrics**:
- F1-score > 0.95
- Precision > 0.95
- Recall > 0.90

---

### Phase 5: End-to-End Integration (Week 6)
**Goal**: Connect all modules into unified pipeline

- [ ] **Task 5.1**: Pipeline integration
  ```python
  class GreenIDSPipeline:
      def __init__(self):
          self.module1 = SaabPixelHopPipeline()
          self.module2 = DFTFeatureSelector()
          self.module3 = SLMForestClassifier()
          
      def fit(self, X_train, y_train):
          # Train Module 1
          repr_train = self.module1.fit_transform(X_train)
          # Train Module 2
          self.module2.fit(repr_train, y_train)
          X_selected = self.module2.transform(repr_train)
          # Train Module 3
          self.module3.fit(X_selected, y_train)
          
      def predict(self, X_test):
          repr_test = self.module1.transform(X_test)
          X_selected = self.module2.transform(repr_test)
          return self.module3.predict(X_selected)
  ```

- [ ] **Task 5.2**: End-to-end validation
  - Full pipeline test on UNSW-NB15
  - Performance verification
  - Latency measurement

**Deliverables**:
- Complete GL-IDS pipeline
- Integration tests
- Performance report

---

### Phase 6: Evaluation & Benchmarking (Week 7)
**Goal**: Comprehensive evaluation against baselines

- [ ] **Task 6.1**: Benchmark against baselines
  - SVM classifier
  - Random Forest
  - Neural Network (DL baseline)
  - Compare F1-scores
  
- [ ] **Task 6.2**: Latency & throughput testing
  - Inference latency measurement
  - Throughput under load
  - Memory profiling
  
- [ ] **Task 6.3**: Carbon footprint analysis
  - Training energy consumption
  - Inference energy measurement
  - Comparison with DL

- [ ] **Task 6.4**: Robustness testing
  - Adversarial attack resistance
  - Model stability
  - Explain predictions

**Deliverables**:
- Benchmark report (F1-scores, latency, energy)
- Performance comparison table
- Sustainability metrics

**Success Criteria**:
- F1-score within 2% of best DL model
- Inference latency < 10ms per sample
- Training time < 1 hour
- Energy consumption < 10% of DL

---

### Phase 7: Optimization & Edge Deployment (Week 8)
**Goal**: Prepare for production and edge deployment

- [ ] **Task 7.1**: Model optimization
  - Quantization (8-bit precision)
  - Pruning unnecessary features
  - Model serialization

- [ ] **Task 7.2**: Edge deployment
  - Create lightweight inference engine
  - Test on ARM devices
  - Document deployment

- [ ] **Task 7.3**: Production readiness
  - Error handling
  - Logging & monitoring
  - Configuration management

**Deliverables**:
- Optimized model artifacts
- Deployment package
- Edge deployment guide

---

### Phase 8: Documentation & Release (Week 9)
**Goal**: Complete documentation and prepare for sharing

- [ ] **Task 8.1**: API documentation
  - Module docstrings
  - Usage examples
  - Parameter documentation

- [ ] **Task 8.2**: User guides
  - Installation guide
  - Quick start tutorial
  - Configuration guide

- [ ] **Task 8.3**: Technical documentation
  - Methodology paper
  - Architecture overview
  - Performance analysis

- [ ] **Task 8.4**: Code release
  - GitHub repository setup
  - License selection
  - Release notes

**Deliverables**:
- Complete API documentation
- User guides
- Technical paper
- GitHub repository

---

## Technical Specifications

### Module 1: Representation Learning (Saab/PixelHop)

**Input**: Raw network traffic records or network features (e.g., 41 UNSW-NB15 features)

**Process**:
1. Organize features into local neighborhoods (e.g., 5×5 spatial patches for image-like data)
2. Apply Saab Transform:
   - Compute DC kernel (local mean)
   - Compute AC kernels via PCA
   - Generate compact representation
3. Multi-stage cascade:
   - Apply pooling between stages
   - Expand receptive field
   - Enrich representations

**Output**: Compact representations (e.g., 512-1024 dimensions from 41 input features)

**Key Parameters**:
- `num_saab_components`: Number of AC components (default: 25)
- `kernel_size`: Local neighborhood size
- `pooling_stride`: Spatial pooling stride
- `num_stages`: Number of PixelHop stages (default: 2-3)

---

### Module 2: Feature Learning (DFT)

**Input**: Representations + training labels

**Process**:
1. For each feature dimension:
   - Measure discriminant power (e.g., class separability)
   - Rank by importance
2. Auto-threshold: Select top K% features that discriminate attacks from normal traffic
3. Dimensionality reduction: From 512+ → 50-100 dimensions

**Output**: Selected feature indices and importance scores

**Key Parameters**:
- `threshold_percentile`: Keep top X% features (default: 5-10%)
- `min_features`: Minimum features to keep (default: 30)
- `max_features`: Maximum features to keep (default: 200)

---

### Module 3: Decision Learning (SLM)

**Input**: Selected features + training labels

**Process**:
1. Build Subspace Learning Machine tree:
   - Find discriminant hyperplanes
   - Partition feature space into pure regions
2. Ensemble:
   - SLM Forest: Multiple trees + voting
   - SLM Boost: Iterative tree building with reweighting
3. Output predictions with confidence

**Output**: Attack/Normal classification + confidence scores

**Key Parameters**:
- `max_depth`: Max tree depth (default: 10)
- `num_trees`: Ensemble size (default: 50)
- `min_samples_leaf`: Minimum samples per leaf (default: 10)

---

## Success Criteria

### Performance Targets
| Metric | Target | Acceptable |
|--------|--------|-----------|
| F1-Score | > 0.96 | > 0.94 |
| Precision | > 0.97 | > 0.95 |
| Recall | > 0.95 | > 0.93 |
| Accuracy | > 0.97 | > 0.95 |

### Efficiency Targets
| Metric | Target | Acceptable |
|--------|--------|-----------|
| Inference Latency | < 10 ms | < 20 ms |
| Training Time | < 1 hour | < 2 hours |
| Model Size | < 5 MB | < 10 MB |
| Memory (Inference) | < 100 MB | < 200 MB |

### Sustainability Targets
| Metric | Target | Acceptable |
|--------|--------|-----------|
| Training Energy | < 100 kWh | < 200 kWh |
| Energy vs DL | < 10% | < 20% |
| Carbon Footprint | < 50 kg CO2e | < 100 kg CO2e |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Dimension too high after Module 1 | Use lossy Saab, aggressive pooling |
| Feature selection removes discriminative power | Use ensemble of selectors |
| SLM overfitting | Pruning, ensemble methods, regularization |
| Edge deployment latency | Model quantization, feature reduction |
| Class imbalance in UNSW-NB15 | Stratified sampling, weighted loss |

---

## References

**Green Learning Papers**:
- Kuo, C.-C. J. "Green Learning: Introduction, Examples and Outlook" (2022)
- Kuo, C.-C. J. & Chen, Y. "Understanding Convolutional Neural Networks with A Mathematical Model" (2016-2019)
- Chen, Y. & Kuo, C.-C. J. "PixelHop: A Successive Subspace Learning Framework" (2020)

**Dataset**:
- UNSW-NB15: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

---

**Status**: Ready for Phase 1 Implementation  
**Last Updated**: 2026-07-20
