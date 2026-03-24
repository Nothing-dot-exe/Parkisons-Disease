import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_graphs():
    # Create directory for graphs
    os.makedirs('graphs', exist_ok=True)
    
    # 1. Target Distribution Graph
    print("Generating Target Distribution Graph...")
    df = pd.read_csv('data.csv')
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='status', data=df, palette=['#1f77b4', '#d62728'])
    plt.title("Distribution of Target Variable (Status)", fontsize=14)
    plt.xlabel("Status (0: Healthy, 1: Parkinson's)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
    plt.savefig('graphs/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation Heatmap (Top 10 features correlated with status)
    print("Generating Correlation Heatmap...")
    # Drop name column 
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    
    corr = df.corr()
    top_corr_features = corr.index[abs(corr["status"]) > 0.3]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Heatmap (Highly Correlated Features)", fontsize=14)
    plt.savefig('graphs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model Performance Comparison
    print("Generating Model Performance Comparison...")
    if os.path.exists('model_metrics_comparison.csv'):
        metrics_df = pd.read_csv('model_metrics_comparison.csv')
        metrics_df = metrics_df.sort_values(by='F1-Score', ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plotting F1-Score as it's the primary metric for imbalanced classes
        ax2 = plt.barh(metrics_df['Model'], metrics_df['F1-Score'], color='skyblue')
        plt.xlabel('F1-Score', fontsize=12)
        plt.title('Model Performance Comparison (F1-Score)', fontsize=14)
        plt.xlim(0, 1.0)
        
        # Add values to the bars
        for patch in ax2.patches:
            plt.text(patch.get_width() - 0.05, patch.get_y() + patch.get_height()/2, 
                     f'{patch.get_width()*.100:.2f}%', va='center', color='black', fontsize=10)

        plt.savefig('graphs/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Warning: model_metrics_comparison.csv not found. Run main.py first to generate model metrics.")

    print("All graphs successfully generated and saved in 'graphs/' directory.")

if __name__ == "__main__":
    create_graphs()
