import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)

def get_metrics_report(y_test, y_pred, model_name):
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Classification Report
    print(f"\nPerformance Report for: {model_name}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    
    return results

def plot_confusion_matrix(y_test, y_pred, model_name, classes=None):

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes if classes is not None else 'auto',
                yticklabels=classes if classes is not None else 'auto')
    
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_model_comparison(final_results):

    df_results = pd.DataFrame(list(final_results.items()), columns=['Model', 'Accuracy'])
    df_results = df_results.sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Accuracy', y='Model', data=df_results, palette='viridis')
    
    for index, value in enumerate(df_results['Accuracy']):
        plt.text(value, index, f'{value*100:.2f}%')
        
    plt.title('Comparison of Model Accuracies')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Model Name')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def compare_top_models_detailed(y_test, predictions_dict):
    comparison_data = []
    for name, y_pred in predictions_dict.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        comparison_data.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
    
    df_comp = pd.DataFrame(comparison_data).melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_comp)
    plt.xticks(rotation=45)
    plt.title('Detailed Performance Comparison (Accuracy vs F1-Score)')
    plt.show()