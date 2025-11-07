#TODO: Compare parameter importance with domain knowledge
#TODO: Add preprocessing steps from the previous pipeline
#TODO: Add better parameter searching, maybe grid search
#TODO: Implement voting system
#TODO: What does 'C': 0.1, 'penalty': 'l1' mean, remove if bad


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Baseline model function
def baseline_model(input_data):
    smoking_data = input_data['smoking']
    majority_class = smoking_data.mode()[0]
    no_of_rows = input_data.shape[0]
    no_of_rows_maj_class = sum(smoking_data == majority_class)
    accuracy = no_of_rows_maj_class / no_of_rows
    return majority_class, accuracy

#read csv
data = pd.read_csv('train_dataset.csv')
#run baseline model
baseline_prediction = baseline_model(data)
print("Baseline model prediction (majority class):", baseline_prediction[0])
print("Baseline model accuracy:", baseline_prediction[1])
# Baseline model function ends here

X = data.drop(columns=['smoking'])
y = data['smoking']


def nested_cross_validation_with_metrics(model, params_list, X, y, outer_folds=5, inner_folds=3, model_name="Model"):
    """
    Manual parameter selection with comprehensive metrics and detailed progress tracking
    """
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    outer_scores = []
    all_best_params = []
    
    # Store comprehensive metrics
    metrics_data = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': [],
        'roc_data': {
            'fpr': [],
            'tpr': [],
            'roc_auc': [],
            'fold_numbers': []
        }
    }

    print(f"\n{'='*60}")
    print(f"STARTING NESTED CV FOR: {model_name}")
    print(f"{'='*60}")
    
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        print(f"\n📁 OUTER FOLD {fold_num}/{outer_folds}")
        print(f"   Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_score = -1
        best_params = None
        param_scores = []
        
        print(f"   🔍 Inner CV - Testing {len(params_list)} parameter sets...")
        
        # Test each parameter set in inner CV
        for param_idx, params in enumerate(params_list, 1):
            print(f"      Testing parameter set {param_idx}/{len(params_list)}: {params}")
            
            inner_scores = []
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
            
            for inner_fold_num, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), 1):
                X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
                
                # Create new model instance
                model_instance = model.__class__(**model.get_params())
                model_instance.set_params(**params)
                
                # Train on inner training fold
                model_instance.fit(X_inner_train, y_inner_train)
                
                # Score on validation
                score = accuracy_score(y_inner_val, model_instance.predict(X_inner_val))
                inner_scores.append(score)
                
                print(f"         Inner Fold {inner_fold_num}/{inner_folds}: Accuracy = {score:.4f}")
            
            mean_inner_score = np.mean(inner_scores)
            param_scores.append((params, mean_inner_score))
            print(f"      ✅ Parameter set {param_idx} - Mean Inner CV Score: {mean_inner_score:.4f}")
            
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = params
                print(f"      🎯 NEW BEST: {best_params} (Score: {best_score:.4f})")
        
        # Store best parameters for this outer fold
        all_best_params.append(best_params)
        print(f"\n   🏆 Selected best parameters for Outer Fold {fold_num}:")
        print(f"      {best_params}")
        print(f"      Best Inner CV Score: {best_score:.4f}")
        
        # Train best model on full inner training set
        print(f"   🚀 Training final model with best parameters on full training set...")
        best_model = model.__class__(**model.get_params())
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        
        # Predict on outer test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        outer_scores.append(accuracy)
        metrics_data['accuracy'].append(accuracy)
        metrics_data['precision'].append(precision)
        metrics_data['recall'].append(recall)
        metrics_data['f1'].append(f1)
        metrics_data['confusion_matrices'].append(cm)
        
        print(f"\n   📊 Outer Fold {fold_num} Test Results:")
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
        
        # Print detailed classification report
        print(f"\n      Detailed Classification Report:")
        report = classification_report(y_test, y_pred, zero_division=0)
        for line in report.split('\n'):
            print(f"        {line}")
        
        # Print confusion matrix
        print(f"      Confusion Matrix:")
        print(f"        {cm}")
        
        # Calculate ROC curve for this fold (for binary classification)
        if len(np.unique(y)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            metrics_data['roc_data']['fpr'].append(fpr)
            metrics_data['roc_data']['tpr'].append(tpr)
            metrics_data['roc_data']['roc_auc'].append(roc_auc)
            metrics_data['roc_data']['fold_numbers'].append(fold_num)
            
            print(f"      AUC:       {roc_auc:.4f}")
        
        print(f"   ✅ Outer Fold {fold_num} completed")
        print(f"   {'─'*50}")

    mean_score = np.mean(outer_scores)
    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Mean Accuracy:  {mean_score:.4f}")
    print(f"   Mean Precision: {np.mean(metrics_data['precision']):.4f}")
    print(f"   Mean Recall:    {np.mean(metrics_data['recall']):.4f}")
    print(f"   Mean F1-Score:  {np.mean(metrics_data['f1']):.4f}")
    
    if metrics_data['roc_data']['fpr']:
        mean_auc = np.mean(metrics_data['roc_data']['roc_auc'])
        print(f"   Mean AUC:      {mean_auc:.4f}")
    
    print(f"\n   Individual Fold Metrics:")
    print(f"   {'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}" + 
          (f" {'AUC':<10}" if metrics_data['roc_data']['fpr'] else ""))
    print(f"   {'─'*60}")
    
    for i in range(outer_folds):
        row = f"   {i+1:<6} {metrics_data['accuracy'][i]:<10.4f} {metrics_data['precision'][i]:<10.4f} {metrics_data['recall'][i]:<10.4f} {metrics_data['f1'][i]:<10.4f}"
        if metrics_data['roc_data']['fpr']:
            row += f" {metrics_data['roc_data']['roc_auc'][i]:<10.4f}"
        print(row)
    
    return mean_score, outer_scores, all_best_params, metrics_data

def plot_roc_curves(roc_data, model_name):
    """Plot ROC curves for all folds"""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each fold
    for i, (fpr, tpr, roc_auc, fold_num) in enumerate(zip(
        roc_data['fpr'], roc_data['tpr'], roc_data['roc_auc'], roc_data['fold_numbers']
    )):
        plt.plot(fpr, tpr, lw=2, alpha=0.7,
                label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    # Calculate mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for fpr, tpr in zip(roc_data['fpr'], roc_data['tpr']):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= len(roc_data['fpr'])
    mean_auc = auc(mean_fpr, mean_tpr)
    
    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=3,
             label=f'Mean ROC (AUC = {mean_auc:.3f})', linestyle='-')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}\n(Mean AUC = {mean_auc:.3f})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    return mean_auc

def plot_metrics_comparison(rf_metrics, linear_metrics):
    """Plot comparison of metrics between models"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_means = [
        np.mean(rf_metrics['accuracy']),
        np.mean(rf_metrics['precision']),
        np.mean(rf_metrics['recall']),
        np.mean(rf_metrics['f1'])
    ]
    linear_means = [
        np.mean(linear_metrics['accuracy']),
        np.mean(linear_metrics['precision']),
        np.mean(linear_metrics['recall']),
        np.mean(linear_metrics['f1'])
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, rf_means, width, label='Random Forest', alpha=0.8)
    rects2 = ax.bar(x + width/2, linear_means, width, label='Logistic Regression', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Define parameter sets
rf_params_list = [
    {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
    {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3}
]

linear_params_list = [
    {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'},
    {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
]

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
linear_model = LogisticRegression(random_state=42, max_iter=1000)

print("🚀 STARTING NESTED CROSS-VALIDATION FOR BOTH MODELS")
print("="*70)

# Train Random Forest
rf_mean_score, rf_fold_scores, rf_best_params, rf_metrics = nested_cross_validation_with_metrics(
    rf_model, rf_params_list, X, y, outer_folds=5, inner_folds=3, model_name="RANDOM FOREST"
)

print("\n" + "="*70)

# Train Logistic Regression
linear_mean_score, linear_fold_scores, linear_best_params, linear_metrics = nested_cross_validation_with_metrics(
    linear_model, linear_params_list, X, y, outer_folds=5, inner_folds=3, model_name="LOGISTIC REGRESSION"
)

# Final summary
print("\n" + "="*70)
print("🎯 FINAL COMPARISON SUMMARY")
print("="*70)

# Plot metrics comparison
plot_metrics_comparison(rf_metrics, linear_metrics)

# Plot ROC curves if binary classification
if rf_metrics['roc_data']['fpr']:
    print(f"\n📈 GENERATING ROC CURVES...")
    rf_mean_auc = plot_roc_curves(rf_metrics['roc_data'], "Random Forest")
    linear_mean_auc = plot_roc_curves(linear_metrics['roc_data'], "Logistic Regression")

# Final detailed comparison
print(f"\n🏆 FINAL MODEL COMPARISON:")
print(f"{'Metric':<15} {'Random Forest':<15} {'Logistic Regression':<15}")
print(f"{'-'*50}")
print(f"{'Accuracy':<15} {np.mean(rf_metrics['accuracy']):<15.4f} {np.mean(linear_metrics['accuracy']):<15.4f}")
print(f"{'Precision':<15} {np.mean(rf_metrics['precision']):<15.4f} {np.mean(linear_metrics['precision']):<15.4f}")
print(f"{'Recall':<15} {np.mean(rf_metrics['recall']):<15.4f} {np.mean(linear_metrics['recall']):<15.4f}")
print(f"{'F1-Score':<15} {np.mean(rf_metrics['f1']):<15.4f} {np.mean(linear_metrics['f1']):<15.4f}")

if rf_metrics['roc_data']['fpr']:
    print(f"{'AUC':<15} {rf_mean_auc:<15.4f} {linear_mean_auc:<15.4f}")

print(f"\n✅ Nested Cross-Validation completed successfully!")