import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        self.k_prime = k_prime  # Can be integer (components) or float (variance %)
        self.scaler = StandardScaler()
        self.pca = None
        self.bias_vector_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        # 1. Compute DC Component: Tabular baseline mean per row/feature group
        X_scaled = self.scaler.fit_transform(X_df)
        
        # 2. Extract AC Components: Run PCA on mean-removed data
        self.pca = PCA(n_components=self.k_prime, random_state=42)
        self.pca.fit(X_scaled)
        
        # 3. Calculate Bias: Find min response to offset sign-confusion safely
        raw_responses = np.dot(X_scaled, self.pca.components_.T)
        self.bias_vector_ = np.abs(np.min(raw_responses, axis=0)) + 1e-5
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        # Apply projection kernels
        ac_responses = np.dot(X_scaled, self.pca.components_.T)
        # Apply adjusted bias vector (Ensures structural logical transparency)
        positive_representations = ac_responses + self.bias_vector_
        return positive_representations

# =====================================================================
# MODULE 2: SUPERVISED FEATURE LEARNING (DISCRIMINANT FEATURE TEST)
# =====================================================================
class DiscriminantFeatureTest(BaseEstimator, TransformerMixin):
    """
    Supervised Feature Selection filtering via optimal 1D partitioning cost curves.
    Calculates weighted entropy across subset splits to discard noisy features.
    """
    def __init__(self, percentile_threshold=75):
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
            # Test split candidates (quantiles for efficiency)
            thresholds = np.percentile(f_col, [25, 50, 75])
            best_entropy = float('inf')

            for t in thresholds:
                left_mask = f_col < t
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                w_left = len(y_left) / len(y)
                w_right = len(y_right) / len(y)
                
                # Weighted split entropy cost calculation
                entropy_split = (w_left * self._calculate_entropy(y_left) + 
                                 w_right * self._calculate_entropy(y_right))
                
                if entropy_split < best_entropy:
                    best_entropy = entropy_split
            
            feature_costs.append(best_entropy)

        # Feature Ranking: Select top variables below cost threshold
        cost_cutoff = np.percentile(feature_costs, self.percentile_threshold)
        self.selected_indices_ = [idx for idx, cost in enumerate(feature_costs) if cost <= cost_cutoff]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

# =====================================================================
# COMPREHENSIVE PIPELINE PIPELINE WRAPPER
# =====================================================================
def run_green_learning_pipeline(train_path, test_path):
    print("Step 0: Loading and Encoding UNSW-NB15 Subsets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Isolate labels ('label' column: 0 for normal, 1 for attack)
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Drop identifying/target keys safely
    cols_to_drop = ['id', 'label', 'attack_cat']
    X_train_raw = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    X_test_raw = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # Handle Tabular Categoricals via quick low-overhead dummy mapping
    categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
    X_combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    X_combined_encoded = pd.get_dummies(X_combined, columns=categorical_cols, drop_first=True)
    
    X_train_numeric = X_combined_encoded.iloc[:len(X_train_raw)].values.astype(np.float64)
    X_test_numeric = X_combined_encoded.iloc[len(X_train_raw):].values.astype(np.float64)
    
    print(f" -> Base Encoded Tabular Matrix Shape: {X_train_numeric.shape}")

    # --- Module 1 Execution ---
    print("\nStep 1: Commencing Tabular Saab Subspace Approximation...")
    saab = TabularSaabTransform(k_prime=0.99)  # Keep 99% explained variance kernels
    X_train_rep = saab.fit_transform(X_train_numeric)
    X_test_rep = saab.transform(X_test_numeric)
    print(f" -> Output Representations Dimension: {X_train_rep.shape}")

    # --- Module 2 Execution ---
    print("\nStep 2: Executing Label-Assisted Discriminant Feature Test (DFT)...")
    dft = DiscriminantFeatureTest(percentile_threshold=60) # Preserve top 60% purest splits
    X_train_features = dft.fit_transform(X_train_rep, y_train)
    X_test_features = dft.transform(X_test_rep)
    print(f" -> High-Value Target Features Selected Dimension: {X_train_features.shape}")

    # --- Module 3 Execution ---
    print("\nStep 3: Training Statistical Decision Architecture...")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    classifier.fit(X_train_features, y_train)
    
    # Final Metrics Generation
    predictions = classifier.predict(X_test_features)
    print("\n================ GREEN LEARNING EVALUATION REPORT ================")
    print(f"Pipeline Test Accuracy Score: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=["Normal Traffic", "Attack / Intrusion"]))

# =====================================================================
# INITIALIZATION EXAMPLE
# =====================================================================
if __name__ == "__main__":
    # Replace strings with exact paths to 'UNSW_NB15_training-set.csv' and 'UNSW_NB15_testing-set.csv'
    # run_green_learning_pipeline("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
    pass
