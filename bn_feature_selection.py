import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC
import networkx as nx

def run_bayesian_feature_selection(csv_path, target_node='target_runoff'):
    print(f"Loading engineered features from {csv_path}...")
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # 1. Pre-Filtering (Filter Method)
    # BNs scale exponentially. We first find the top 15 most correlated features 
    # to prevent computation timeout.
    print("Step 1: Pre-filtering features using Pearson correlation...")
    correlations = df.corr()[target_node].abs().sort_values(ascending=False)
    
    # Drop the target itself from the predictors list
    top_15_features = correlations.drop(target_node).head(15).index.tolist()
    
    # Create a smaller dataframe with just the top 15 and the target
    df_reduced = df[top_15_features + [target_node]]
    
    # 2. Bayesian Network Causal Discovery (Wrapper Method)
    print(f"Step 2: Running Bayesian Network on the top 15 causal candidates...")
    
    # Discretize the continuous data into bins for the BN probability mapping
    df_discrete = df_reduced.apply(lambda x: pd.qcut(x, q=3, labels=False, duplicates='drop'))
    
    # Initialize the Hill Climb Search algorithm
    hc = HillClimbSearch(df_discrete)
    
    # Estimate the best network structure using the BIC Score
    best_model = hc.estimate(scoring_method=BIC(df_discrete))
    
    print("\nCausal Graph Edges Discovered:")
    for edge in best_model.edges():
        print(f" -> {edge[0]} influences {edge[1]}")
        
    # Select only the features that have a direct path to our target variable
    selected_features = []
    for edge in best_model.edges():
        if edge[1] == target_node:
            selected_features.append(edge[0])
            
    # Fallback in case the BN structure prunes everything (rare, but good for stability)
    if not selected_features:
        print(f"\nNo direct causal links found by BN. Defaulting to top 3 correlated features.")
        selected_features = top_15_features[:3]
        
    return df_reduced, selected_features

if __name__ == "__main__":
    try:
        df_reduced, optimal_features = run_bayesian_feature_selection('engineered_features_data.csv')
        
        print("\n=========================================")
        print("FINAL CAUSAL FEATURES SELECTED BY BAYESIAN NETWORK:")
        for feat in optimal_features:
            print(f" * {feat}")
        print("=========================================")
        
        # Save the final optimized dataset for the DL model
        final_df = df_reduced[optimal_features + ['target_runoff']]
        final_df.to_csv('model_ready_data.csv')
        print("\nOptimized dataset saved to model_ready_data.csv")
        
    except FileNotFoundError:
        print("Error: Could not find engineered_features_data.csv.")
        