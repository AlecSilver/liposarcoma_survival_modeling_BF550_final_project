import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SurvivalModelPipeline:
    """
    Pipeline for testing multiple survival analysis models.
    """
    
    def __init__(self, models=None, random_state=42):
        """
        Initialize the pipeline with models to test.
        
        Parameters:
        -----------
        models : dict, optional
            Dictionary of model names and model instances
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
        
        # Default models if none provided
        if models is None:
            self.models = {
                'CoxLasso': CoxnetSurvivalAnalysis(l1_ratio=1.0,
                                                    alpha_min_ratio=0.01,
                                                    fit_baseline_model =True),
                'CoxElasticNet' : CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01),
                'RSF': RandomSurvivalForest(n_estimators=100, 
                                           min_samples_split=10,
                                           min_samples_leaf=15,
                                           max_features="sqrt",
                                           random_state=random_state)
            }
        else:
            self.models = models
    
    def load_and_prepare_data(self, X=None, y=None, test_size=0.3):
        """
        Load and prepare survival data.
        
        Parameters:
        -----------
        X : array-like, optional
            Feature matrix
        y : structured array, optional
            Survival target with 'event' and 'time' fields
        test_size : float
            Proportion of data for testing
        """
        # Load example dataset if no data provided
        if X is None or y is None:
            print("Loading example GBSG2 dataset...")
            X, y = load_gbsg2()
            
            # Convert categorical variables to numeric
            X_encoded = pd.get_dummies(X, drop_first=True)
        else:
            X_encoded = X
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=self.random_state
        )
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Features: {self.X_train.shape[1]}")
    
    def train_and_evaluate(self):
        """
        Train all models and compute evaluation metrics.
        """
        print("\n" + "="*60)
        print("Training and Evaluating Models")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            try:
                # Train model
                print("  Fitting model...", name)
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predict risk scores
                print("  Predicting risk scores...", name)
                risk_scores = model.predict(self.X_test_scaled)
                
                # need to split y into event and time for metrics
                state, time = zip(*self.y_test)
                state = np.array(state)
                time = np.array(time)
                # Calculate C-index

                print("  Calculating C-index...", name)
                c_index = concordance_index_censored(
                    state,
                    time,
                    risk_scores
                )[0]
                
                # Calculate time-dependent AUC
                times = np.percentile(self.y_train['time'], 
                                     np.linspace(10, 90, 9))
                
                try:
                    # Get survival functions for AUC calculation
                    if hasattr(model, 'predict_survival_function'):
                        surv_funcs = model.predict_survival_function(self.X_test_scaled)
                        auc_scores, mean_auc = cumulative_dynamic_auc(
                            self.y_train, self.y_test, risk_scores, times
                        )
                    else:
                        mean_auc = None
                        auc_scores = None
                except Exception as e:
                    mean_auc = None
                    auc_scores = None
                    print(f"  Could not calculate AUC: {str(e)}")
                
                # Calculate IBS if possible
                try:
                    if hasattr(model, 'predict_survival_function'):
                        surv_funcs = model.predict_survival_function(self.X_test_scaled)
                        
                        # Create prediction array
                        times_ibs = np.percentile(self.y_train['time'], 
                                                 np.linspace(5, 95, 20))
                        preds = np.array([[fn(t) for t in times_ibs] 
                                         for fn in surv_funcs])
                        
                        ibs = integrated_brier_score(
                            self.y_train, self.y_test, preds, times_ibs
                        )
                    else:
                        ibs = None
                except Exception as e:
                    ibs = None
                    print(f"  Could not calculate IBS: {str(e)}")
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'c_index': c_index,
                    'mean_auc': mean_auc,
                    'auc_scores': auc_scores,
                    'auc_times': times if mean_auc is not None else None,
                    'ibs': ibs,
                    'risk_scores': risk_scores
                }
                
                print(f"  C-index: {c_index:.4f}")
                if mean_auc is not None:
                    print(f"  Mean AUC: {mean_auc:.4f}")
                if ibs is not None:
                    print(f"  IBS: {ibs:.4f}")
                    
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.results[name] = {'error': str(e)}
    
    def plot_results(self):
        """
        Visualize model comparison results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: C-index comparison
        models = []
        c_indices = []
        for name, result in self.results.items():
            if 'c_index' in result:
                models.append(name)
                c_indices.append(result['c_index'])
        
        axes[0].barh(models, c_indices, color='steelblue')
        axes[0].set_xlabel('C-index', fontsize=12)
        axes[0].set_title('Model Comparison: C-index', fontsize=14, fontweight='bold')
        axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        axes[0].legend()
        axes[0].set_xlim([0, 1])
        
        # Plot 2: Time-dependent AUC
        for name, result in self.results.items():
            if result.get('mean_auc') is not None:
                axes[1].plot(result['auc_times'], result['auc_scores'], 
                           marker='o', label=name)
        
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title('Time-Dependent AUC', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """
        Return a summary DataFrame of all results.
        """
        summary_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                summary_data.append({
                    'Model': name,
                    'C-index': result.get('c_index', np.nan),
                    'Mean AUC': result.get('mean_auc', np.nan),
                    'IBS': result.get('ibs', np.nan)
                })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('C-index', ascending=False)
        return df


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SurvivalModelPipeline(random_state=42)
    
    # Load and prepare data
    pipeline.load_and_prepare_data(test_size=0.3)
    
    # Train and evaluate all models
    pipeline.train_and_evaluate()
    
    # Plot results
    pipeline.plot_results()
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Results")
    print("="*60)
    summary = pipeline.get_summary()
    print(summary.to_string(index=False))
    
    # Get best model
    best_model_name = summary.iloc[0]['Model']
    print(f"\nBest model by C-index: {best_model_name}")