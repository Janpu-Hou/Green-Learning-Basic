# To implement multi-class classification for predicting specific threat families 
# (e.g., Exploits, Fuzzers, DoS, Reconnaissance, Normal), 
# we need to shift our target label from the binary label column to the attack_cat column.
# Because the UNSW-NB15 dataset represents benign traffic with blank spaces, NaN values, 
# or the string 'Normal' depending on the file version, 
# we must safely standardize the column, encode the labels numerically, 
# and adapt our Discriminant Feature Test (DFT) module 
# to evaluate multi-class weighted entropy.
# Here is the updated complete pipeline designed to 
# process, extract, select, and categorize structural threat classes:

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# MODULE 1: UNSUPERVISED REPRESENTATION LEARNING (SAAB TRANSFORM)
# =====================================================================
class TabularSaabTransform(BaseEstimator, TransformerMixin):
    """
    Subspace Approximation via Adjusted Bias (Saab) for Tabular Flows.
    Extracts DC component, computes PCA on mean-removed residuals, 
    and applies a positive bias vector to eliminate sign confusion.
    """
    def __init__(self, k_prime=0.95):
        self.k_prime = k_prime  # Keeps principal components explaining % variance
        self.scaler = StandardScaler()
        self.pca = None
        self.bias_vector_ = None

    def fit(self, X, y=None):
        # 1. Standardize the data across features
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Extract AC Components using PCA
        self.pca = PCA(n_components=self.k_prime, random_state=42)
        self.pca.fit(X_scaled)
        
        # 3. Calculate Adjusted Bias vector to handle sign confusion
        raw_responses = np.dot(X_scaled, self.pca.components_.T)
        self.bias_vector_ = np.abs(np.min(raw_responses, axis=0)) + 1e-5
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        ac_responses = np.dot(X_scaled, self.pca.components_.T)
        # Shift responses to keep them strictly positive without non-linear activations
        positive_representations = ac_responses + self.bias_vector_
        return positive_representations

# =====================================================================
# MODULE 2: SUPERVISED FEATURE LEARNING (MULTI-CLASS DFT)
# =====================================================================
class MultiClassDiscriminantFeatureTest(BaseEstimator, TransformerMixin):
    """
    Supervised Multi-Class Feature Selection via 1D partitioning cost curves.
    Measures weighted entropy of target threat class distributions.
    """
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
            # Use percentiles as evaluation split points for quick local processing
            thresholds = np.percentile(f_col, [20, 40, 60, 80])
            best_entropy = float('inf')

            for t in thresholds:
                left_mask = f_col < t
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                w_left = len(y_left) / len(y)
                w_right = len(y_right) / len(y)
                
                # Compute multi-class split entropy cost
                entropy_split = (w_left * self._calculate_entropy(y_left) + 
                                 w_right * self._calculate_entropy(y_right))
                
                if entropy_split < best_entropy:
                    best_entropy = entropy_split
            
            feature_costs.append(best_entropy)

        # Retain features with the lowest entropy (highest purity)
        cost_cutoff = np.percentile(feature_costs, self.percentile_threshold)
        self.selected_indices_ = [idx for idx, cost in enumerate(feature_costs) if cost <= cost_cutoff]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

# =====================================================================
# COMPREHENSIVE PIPELINE PIPELINE WRAPPER
# =====================================================================
def run_multi_class_green_learning(train_path, test_path):
    print("Step 0: Loading Data and Standardizing Multi-Class Targets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Standardize 'attack_cat' string categories (filling NaN/spaces with 'Normal')
    for df in [train_df, test_df]:
        if 'attack_cat' in df.columns:
            df['attack_cat'] = df['attack_cat'].str.strip()
            df['attack_cat'] = df['attack_cat'].fillna('Normal')
            df.loc[df['attack_cat'] == '', 'attack_cat'] = 'Normal'
            
    # Encode textual threat categories into flat index arrays
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['attack_cat'])
    y_test = label_encoder.transform(test_df['attack_cat'])
    
    # Drop identifying/target keys safely from feature sets
    cols_to_drop = ['id', 'label', 'attack_cat']
    X_train_raw = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    X_test_raw = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # Low-overhead categorical conversion for tabular variables (proto, service, state)
    categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
    X_combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    X_combined_encoded = pd.get_dummies(X_combined, columns=categorical_cols, drop_first=True)
    
    X_train_numeric = X_combined_encoded.iloc[:len(X_train_raw)].values.astype(np.float64)
    X_test_numeric = X_combined_encoded.iloc[len(X_train_raw):].values.astype(np.float64)
    
    print(f" -> Raw Feature Matrix Extracted: {X_train_numeric.shape}")
    print(f" -> Identified Threat Class Encoded Pairs: {dict(enumerate(label_encoder.classes_))}")

    # --- Module 1: Subspace Compression ---
    print("\nStep 1: Commencing Tabular Saab Subspace Approximation...")
    saab = TabularSaabTransform(k_prime=0.98) # Target 98% explained variance thresholds
    X_train_rep = saab.fit_transform(X_train_numeric)
    X_test_rep = saab.transform(X_test_numeric)
    print(f" -> Condensed Representation Dimension: {X_train_rep.shape}")

    # --- Module 2: Multi-Class Filtering ---
    print("\nStep 2: Executing Label-Assisted Multi-Class Feature Selection (DFT)...")
    dft = MultiClassDiscriminantFeatureTest(percentile_threshold=65) # Top 65% informative variables
    X_train_features = dft.fit_transform(X_train_rep, y_train)
    X_test_features = dft.transform(X_test_rep)
    print(f" -> Filtered Threat Features Vector Dimension: {X_train_features.shape}")

    # --- Module 3: Decision Engine ---
    print("\nStep 3: Training Statistical Decision Ensemble Architecture...")
    # Balanced weights help offset the heavy class imbalance in UNSW-NB15 attack categories
    classifier = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    classifier.fit(X_train_features, y_train)
    
    # Final Metrics Generation
    predictions = classifier.predict(X_test_features)
    print("\n================ MULTI-CLASS GREEN LEARNING EVALUATION ================")
    print(f"Overall Testing Pipeline Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nDetailed Per-Class Category Threat Analysis:")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))

# =====================================================================
# INITIALIZATION EXAMPLE
# =====================================================================
if __name__ == "__main__":
    # Ensure correct filenames for 'UNSW_NB15_training-set.csv' and 'UNSW_NB15_testing-set.csv'
    # run_multi_class_green_learning("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
    pass
