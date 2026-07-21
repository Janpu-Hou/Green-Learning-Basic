# Critical Updates For Multi-Class Cyber Analysis:
# Weighted Class Balances: Certain attack groups in UNSW-NB15 (such as Analysis or Backdoors) 
# contain very few samples compared to Normal traffic or Exploits. Setting class_weight='balanced' 
# ensures the decision engine does not neglect minority malicious vectors.
# True Multi-Class DFT: The entropy function computes distributions across all \(C\) 
# discrete attack groups instead of treating them as binary variables, optimizing 
# the partition boundaries specifically to isolate distinct patterns of different attack techniques.


# Replacing the standard Random Forest classifier with the custom Subspace Learning Machine (SLM)
# Tree configuration changes the split mechanism. Unlike conventional decision trees that perform 
# axis-aligned splits on a single feature at a time, an SLM Tree performs oblique (hyperplane) 
# splits.As detailed on Page 13 of the paper, the optimization of an SLM Tree node 
# follows a two-step framework:Find a discriminant 1D direction vector (\(w_{opt}\)) 
# using a probabilistic search or optimization method.Project the multi-dimensional 
# feature space onto that 1D vector and locate the optimal split threshold point (\(w_{0}\) or \(\phi \))
# to minimize the weighted multi-class entropy.The production-ready pipeline below 
# builds this custom SLM Tree Classifier from scratch using a probabilistic search 
# mechanism to optimize the hyperplanes, completely avoiding global backpropagation:



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# MODULE 1: UNSUPERVISED REPRESENTATION LEARNING (SAAB TRANSFORM)
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

# =====================================================================
# MODULE 2: SUPERVISED FEATURE LEARNING (MULTI-CLASS DFT)
# =====================================================================
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
                entropy_split = (
                    (np.sum(left_mask) / len(y)) * self._calculate_entropy(y[left_mask]) + 
                    (np.sum(~left_mask) / len(y)) * self._calculate_entropy(y[~left_mask])
                )
                if entropy_split < best_entropy:
                    best_entropy = entropy_split
            feature_costs.append(best_entropy)

        cost_cutoff = np.percentile(feature_costs, self.percentile_threshold)
        self.selected_indices_ = [idx for idx, cost in enumerate(feature_costs) if cost <= cost_cutoff]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

# =====================================================================
# MODULE 3: SUBSPACE LEARNING MACHINE (SLM) TREE CLASSIFIER
# =====================================================================
class SLMTreeNode:
    """Internal structural node for the Subspace Learning Machine Tree."""
    def __init__(self, depth=0, max_depth=10, min_samples_split=5):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.w = None          # Projection hyperplane vector (w_opt)
        self.threshold = None  # Split boundary point (phi)
        self.left = None       # Left child node
        self.right = None      # Right child node
        self.is_leaf = False
        self.probs = None      # Class probability distribution mapping

class SLMTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Subspace Learning Machine (SLM) Tree Classifier.
    Implements multi-class oblique splits using probabilistic direction search.
    """
    def __init__(self, max_depth=10, min_samples_split=5, num_proj_candidates=50, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_proj_candidates = num_proj_candidates
        self.random_state = random_state
        self.root = None
        self.classes_ = None

    def _calculate_entropy(self, y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _find_best_oblique_split(self, X, y, rng):
        num_samples, num_features = X.shape
        best_entropy = float('inf')
        best_w = None
        best_t = None

        # Step 1: Generate projection vector candidates (Probabilistic Search Strategy)
        candidates = []
        # Include baseline axis-aligned vectors
        for i in range(min(num_features, self.num_proj_candidates // 2)):
            w = np.zeros(num_features)
            w[i] = 1.0
            candidates.append(w)
        
        # Sample random oblique projections (combinations of multiple feature vectors)
        while len(candidates) < self.num_proj_candidates:
            w = rng.normal(0, 1, size=num_features)
            norm = np.linalg.norm(w)
            if norm > 0:
                candidates.append(w / norm)

        # Step 2: Project data onto 1D subspace and locate the optimal split point
        for w in candidates:
            X_projected = np.dot(X, w)
            # Evaluate splitting boundaries on data percentiles
            split_targets = np.percentile(X_projected, [10, 25, 50, 75, 90])
            
            for t in split_targets:
                left_mask = X_projected < t
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                w_left = np.sum(left_mask) / num_samples
                w_right = np.sum(right_mask) / num_samples
                
                entropy_split = (w_left * self._calculate_entropy(y[left_mask]) + 
                                 w_right * self._calculate_entropy(y[right_mask]))
                
                if entropy_split < best_entropy:
                    best_entropy = entropy_split
                    best_w = w
                    best_t = t
                    
        return best_w, best_t, best_entropy

    def _build_tree(self, X, y, node, rng):
        num_samples = X.shape[0]
        unique_classes, counts = np.unique(y, return_counts=True)

        # Calculate class distribution maps for this boundary
        node.probs = {c: 0.0 for c in self.classes_}
        for c, count in zip(unique_classes, counts):
            node.probs[c] = count / num_samples

        # Evaluate terminal conditions
        if (node.depth >= node.max_depth or 
            num_samples < node.min_samples_split or 
            len(unique_classes) == 1):
            node.is_leaf = True
            return

        # Execute Oblique Subspace Hyperplane Split
        w, t, split_ent = self._find_best_oblique_split(X, y, rng)
        
        if w desertion or t is None:
            node.is_leaf = True
            return

        node.w = w
        node.threshold = t

        # Partition feature mappings downstream
        X_projected = np.dot(X, w)
        left_mask = X_projected < t
        
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            node.is_leaf = True
            return

        node.left = SLMTreeNode(depth=node.depth + 1, max_depth=node.max_depth, min_samples_split=node.min_samples_split)
        node.right = SLMTreeNode(depth=node.depth + 1, max_depth=node.max_depth, min_samples_split=node.min_samples_split)

        self._build_tree(X[left_mask], y[left_mask], node.left, rng)
        self._build_tree(X[~left_mask], y[~left_mask], node.right, rng)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.root = SLMTreeNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        self._build_tree(X, y, self.root, rng)
        return self

    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.probs
        # Use equation: w^T * f - phi to check structural partition direction
        if np.dot(x, node.w) < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict_proba(self, X):
        proba_list = [self._predict_sample(x, self.root) for x in X]
        return np.array([[p[c] for c in self.classes_] for p in proba_list])

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

# =====================================================================
# INTEGRATED PIPELINE EXECUTION
# =====================================================================
def run_slm_multi_class_pipeline(train_path, test_path):
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
    X_train_raw = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    X_test_raw = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
    X_combined = pd.concat([X_train_raw, X_test_raw], axis=0)
