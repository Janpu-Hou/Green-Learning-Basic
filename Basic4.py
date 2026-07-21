# To implement SLM Boost as described on Page 14 of the paper, 
# we apply the Gradient Boosting framework using our custom Subspace Learning Machine (SLM) 
# Tree as the weak learner. Instead of training trees independently (like a Forest), 
# SLM Boost trains a sequence of SLM Trees where each successive tree fits the pseudo-residuals 
# (gradients of the loss function) of the previous trees' predictions.For multi-class classification, 
# the standard approach is to use the Multinomial Log-Loss function. This means for a dataset with \(C\) 
# classes, each boosting round builds \(C\) separate SLM trees—one for each class's residual vector.
# Here is the complete, high-performance pipeline extending our architecture to SLM Boost:

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from scipy.special import softmax

# =====================================================================
# MODULE 1 & 2: SAAB TRANSFORM & DFT (PRE-PROCESSING BLOCKS)
# =====================================================================
class TabularSaabTransform(BaseEstimator, TransformerMixin):
    def __init__(self, k_prime=0.95):
        self.k_prime = k_prime
        self.scaler = StandardScaler()
        self.pca = None
        self.bias_vector_ = None

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.k_prime, random_state=42)
        self.pca.fit(X_scaled)
        raw_responses = np.dot(X_scaled, self.pca.components_.T)
        self.bias_vector_ = np.abs(np.min(raw_responses, axis=0)) + 1e-5
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        ac_responses = np.dot(X_scaled, self.pca.components_.T)
        return ac_responses + self.bias_vector_

class MultiClassDiscriminantFeatureTest(BaseEstimator, TransformerMixin):
    def __init__(self, percentile_threshold=70):
        self.percentile_threshold = percentile_threshold
        self.selected_indices_ = []

    def _calculate_entropy(self, y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def fit(self, X, y):
        num_features = X.shape[1]
        feature_costs = []
        y = np.array(y)

        for i in range(num_features):
            f_col = X[:, i]
            thresholds = np.percentile(f_col, [25, 50, 75])
            best_entropy = float('inf')

            for t in thresholds:
                left_mask = f_col < t
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue
                entropy_split = (
                    (np.sum(left_mask) / len(y)) * self._calculate_entropy(y[left_mask]) + 
                    (np.sum(~left_mask) / len(y)) * self._calculate_entropy(y[~left_mask])
                )
                if entropy_split < best_entropy:
                    best_entropy = entropy_split
            feature_costs.append(best_entropy if best_entropy != float('inf') else 1.0)

        cost_cutoff = np.percentile(feature_costs, self.percentile_threshold)
        self.selected_indices_ = [idx for idx, cost in enumerate(feature_costs) if cost <= cost_cutoff]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

# =====================================================================
# MODIFIED REGRESSION SLM TREE NODE FOR RESIDUAL FITTING (SLR)
# =====================================================================
class SLMRegTreeNode:
    def __init__(self, depth=0, max_depth=4):
        self.depth = depth
        self.max_depth = max_depth
        self.w = None          # Hyperplane projection weights
        self.threshold = None  # Scalar split point
        self.left = None       # Left child
        self.right = None      # Right child
        self.is_leaf = False
        self.value = None      # Leaf regression prediction output value

class SLMRegTree:
    """Subspace Learning Regressor (SLR) Tree to fit raw floating-point gradients."""
    def __init__(self, max_depth=4, num_proj_candidates=20, random_state=42):
        self.max_depth = max_depth
        self.num_proj_candidates = num_proj_candidates
        self.random_state = random_state
        self.root = None

    def _find_best_oblique_split(self, X, residuals, rng):
        num_samples, num_features = X.shape
        best_mse_reduction = -1.0
        best_w, best_t = None, None
        
        initial_mse = np.var(residuals)

        candidates = []
        for i in range(min(num_features, self.num_proj_candidates // 2)):
            w = np.zeros(num_features)
            w[i] = 1.0
            candidates.append(w)
        
        while len(candidates) < self.num_proj_candidates:
            w = rng.normal(0, 1, size=num_features)
            norm = np.linalg.norm(w)
            if norm > 0:
                candidates.append(w / norm)

        for w in candidates:
            X_projected = np.dot(X, w)
            split_targets = np.percentile(X_projected, [20, 40, 60, 80])
            
            for t in split_targets:
                left_mask = X_projected < t
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                mse_split = (np.sum(left_mask) * np.var(residuals[left_mask]) + 
                             np.sum(right_mask) * np.var(residuals[right_mask])) / num_samples
                
                mse_reduction = initial_mse - mse_split
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_w = w
                    best_t = t
                    
        return best_w, best_t

    def _build_tree(self, X, residuals, node, rng):
        num_samples = X.shape[0]
        node.value = np.mean(residuals) if num_samples > 0 else 0.0

        if node.depth >= node.max_depth or num_samples < 5:
            node.is_leaf = True
            return

        w, t = self._find_best_oblique_split(X, residuals, rng)
        
        if w desertion or t is None:
            node.is_leaf = True
            return

        node.w = w
        node.threshold = t

        X_projected = np.dot(X, w)
        left_mask = X_projected < t
        
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            node.is_leaf = True
            return

        node.left = SLMRegTreeNode(depth=node.depth + 1, max_depth=node.max_depth)
        node.right = SLMRegTreeNode(depth=node.depth + 1, max_depth=node.max_depth)

        self._build_tree(X[left_mask], residuals[left_mask], node.left, rng)
        self._build_tree(X[~left_mask], residuals[~left_mask], node.right, rng)

    def fit(self, X, residuals):
        rng = np.random.default_rng(self.random_state)
        self.root = SLMRegTreeNode(depth=0, max_depth=self.max_depth)
        self._build_tree(X, residuals, self.root, rng)
        return self

    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.value
        if np.dot(x, node.w) < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])


# =====================================================================
# SYSTEM ARCHITECTURE: SLM BOOST CLASS FIRE
# =====================================================================
class SLMBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    SLM Boost Multi-Class Classifier.
    Sequentially builds ensembles of Subspace Learning Regressor (SLR) 
    Trees to minimize Multi-Class Log-Loss.
    """
    def __init__(self, n_estimators=15, learning_rate=0.1, max_depth=4, num_proj_candidates=20, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_proj_candidates = num_proj_candidates
        self.random_state = random_state
        self.estimators_ = []  # Matrix structured storage layout: [round][class_idx]
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        num_samples = X.shape[0]
        
        # 1. Initialize Log-odds margins to 0.0
        F = np.zeros((num_samples, num_classes))
        
        # Convert target vectors to multi-class one-hot matrices
        y_onehot = np.zeros((num_samples, num_classes))
        for i, c in enumerate(self.classes_):
            y_onehot[:, i] = (y == c).astype(int)

        print(f" -> Beginning Boosting Execution Loop iterations...")
        for r in range(self.n_estimators):
            # Compute probability metrics mapping across the samples via softmax
            probs = softmax(F, axis=1)
            
            # Step 2: Compute pseudo-residuals (Gradients = True Labels - Softmax Probabilities)
            residuals = y_onehot - probs
            
            round_estimators = []
            # Fit one oblique SLR tree for each discrete category gradient
            for c_idx in range(num_classes):
                tree = SLMRegTree(
                    max_depth=self.max_depth, 
                    num_proj_candidates=self.num_proj_candidates, 
                    random_state=self.random_state + r + c_idx
                )
                tree.fit(X, residuals[:, c_idx])
                
                # Update current raw margin predictions using calculated step updates
                F[:, c_idx] += self.learning_rate * tree.predict(X)
                round_estimators.append(tree)
                
            self.estimators_.append(round_estimators)
            if (r + 1) % 5 == 0 or r == 0:
                print(f"    Round {r+1}/{self.n_estimators} Complete.")
        return self

    def predict_proba(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classes_)
        F = np.zeros((num_samples, num_classes))
        
        # Score mapping loops across ensemble matrices sequential rounds
        for round_estimators in self.estimators_:
            for c_idx, tree in enumerate(round_estimators):
                F[:, c_idx] += self.learning_rate * tree.predict(X)
                
        return softmax(F, axis=1)





def predict(self, X):
  probas = self.predict_proba(X)
  return self.classes_[np.argmax(probas, axis=1)]

# =======================================================
# INTEGRATED UNSW-NB15 SLM BOOST PIPELINE PIPELINE
# =======================================================

def run_slm_boost_pipeline(train_path, test_path):
  print("Step 0: Processing Tabular Datasets & Multi-Class Categories...")
  train_df = pd.read_csv(train_path)
  test_df = pd.read_csv(test_path)


for df in [train_df, test_df]:
  if 'attack_cat' in df.columns:
    df['attack_cat'] = df['attack_cat'].str.strip().fillna('Normal')
    df.loc[df['attack_cat'] == '', 'attack_cat'] = 'Normal'

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['attack_cat'])
y_test = label_encoder.transform(test_df['attack_cat'])

cols_to_drop = ['id', 'label', 'attack_cat']
train_raw = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
X_test_raw = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
X_combined = pd.concat([X_train_raw, X_test_raw], axis=0)
X_combined_encoded = pd.get_dummies(X_combined, columns=categorical_cols, drop_first=True)

X_train_numeric = X_combined_encoded.iloc[:lenX_train_raw)].values.astype(np.float64)X_
test_numeric = X_combined_encoded.iloc[len(X_train_raw):].values.astype(np.float64)

# --- Module 1: Unsupervised Representation learning ---
print("\nStep 1: Commencing Tabular Saab Subspace Approximation...")
saab = TabularSaabTransform(k_prime=0.95)
X_train_rep = saab.fit_transform(X_train_numeric)
X_test_rep = saab.transform(X_test_numeric)

# --- Module 2: Supervised Feature selection ---
print("\nStep 2: Executing Multi-Class Discriminant Feature Test (DFT)...")
dft = MultiClassDiscriminantFeatureTest(percentile_threshold=70)
X_train_features = dft.fit_transform(X_train_rep, y_train)
X_test_features = dft.transform(X_test_rep)

# --- Module 3: Custom SLM Boost Classifier Execution ---
print("\nStep 3: Training Custom Subspace Learning Machine (SLM) Boost Pipeline Architecture...")
slm_boost = SLMBoostClassifier(n_estimators=15, learning_rate=0.1, max_depth=4, num_proj_candidates=30, random_state=42)
slm_boost.fit(X_train_features, y_train)

predictions = slm_boost.predict(X_test_features)
print("\n================ SLM BOOST EVALUATION REPORT ================")
print(f"Oblique SLM Boost Testing Pipeline Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))

if name == "main":
# run_slm_boost_pipeline("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
    pass


### SLM Boost Operational Highlights:
# * **Subspace Learning Regressor (SLR)**: Nodes calculate continuous target deviations rather than categorical counts. It fits residual values directly via hyperplane minimization to gradually lower loss variables across sequential rounds.
# * **Additive Multi-Projection Layers**: Because every weak tree candidate explores a unique set of randomized multi-feature projection combinations (\(w\)), the final boosted combination naturally achieves strong ensemble generalization over localized or unseen cyber anomalies.

