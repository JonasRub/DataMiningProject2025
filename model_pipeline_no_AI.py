
#TODO: Compare parameter importance with domain knowledge
#TODO: Add better parameter searching, maybe grid search
#Implement voting system: Done
# What does 'C': 0.1, 'penalty': 'l1' mean, remove if bad: Ok not good or bad, just different regularization, can fix using grid search
#TODO: Add another simple baseline model
#Add feature importance for logistic regression and catboost: Done, hopefully there is no mixup of coefficients?
#TODO: Increase folds when all other tasks are done (2,2) at the moment for testing speed

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib # For saving models

from sklearn.preprocessing import StandardScaler

from helper_KMv2 import get_preprocessor, dataCleaning #MMO

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
baseline_accuracy = baseline_prediction[1]
X = data.drop(columns=['smoking'])
y = data['smoking']



# Create results directory
results_dir = "nested_cross_validation_noAI_results"
os.makedirs(results_dir, exist_ok=True)
print(f" Created results directory: {results_dir}")

def nested_cross_validation_with_metrics(model, params_list, X, y, outer_folds=5, inner_folds=3, model_name="Model"):
    """
    Manual parameter selection with comprehensive metrics and detailed progress tracking
    """
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    outer_scores = []
    all_best_params = []  # Store best params for each fold
    best_overall_params = None  # Store the single best parameter set
    best_overall_score = -1  # Track the best score across all folds
    all_feature_importances = []
    
    # Store comprehensive metrics
    metrics_data = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': [],
        'feature_importances': [],
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
        print(f"\n OUTER FOLD {fold_num}/{outer_folds}")
        print(f"   Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_score = -1
        best_params = None
        param_scores = []
        
        cleaner = dataCleaning() #MMO
        preprocessor = get_preprocessor() #MMO

        print(f"    Inner CV - Testing {len(params_list)} parameter sets...")
        
        # Test each parameter set in inner CV
        for param_idx, params in enumerate(params_list, 1):
            print(f"      Testing parameter set {param_idx}/{len(params_list)}: {params}")
            
            inner_scores = []
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
            
            for inner_fold_num, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), 1):
                X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

                ##########################################################
                # ONLY IMPORTANT FOR VOTİNG CLASSIFIER
                # Create new model instance (use sklearn.clone to correctly handle complex estimators
                # like VotingClassifier / pipelines without passing unexpected kwargs to __init__)
                model_instance = clone(model)

                # If the parameter set is a nested dict for a meta-estimator (e.g. VotingClassifier),
                # flatten it to the 'estimator__param' format that set_params expects.
                if isinstance(params, dict) and any(isinstance(v, dict) for v in params.values()):
                    flat_params = {}
                    for est_name, est_params in params.items():
                        if isinstance(est_params, dict):
                            for k, v in est_params.items():
                                flat_params[f"{est_name}__{k}"] = v
                        else:
                            flat_params[est_name] = est_params
                    model_instance.set_params(**flat_params)
                else:
                    model_instance.set_params(**params)
                # ONLY IMPORTANT FOR VOTİNG CLASSIFIER
                ##########################################################
                
                X_inner_train_clean = cleaner.fit_transform(X_inner_train)
                y_inner_train = y_inner_train.loc[X_inner_train_clean.index]
                X_inner_val_clean = cleaner.transform(X_inner_val)
                y_inner_val = y_inner_val.loc[X_inner_val_clean.index]

                X_inner_train_prep = preprocessor.fit_transform(X_inner_train_clean)
                X_inner_val_prep = preprocessor.transform(X_inner_val_clean)
                
                # Train on inner training fold
                model_instance.fit(X_inner_train_prep, y_inner_train)
                
                # Score on validation
                score = accuracy_score(y_inner_val, model_instance.predict(X_inner_val_prep))
                inner_scores.append(score)
                
                print(f"         Inner Fold {inner_fold_num}/{inner_folds}: Accuracy = {score:.4f}")
            
            mean_inner_score = np.mean(inner_scores)
            param_scores.append((params, mean_inner_score))
            print(f"       Parameter set {param_idx} - Mean Inner CV Score: {mean_inner_score:.4f}")
            
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = params
                print(f"       NEW BEST: {best_params} (Score: {best_score:.4f})")
        
        # Store best parameters for this outer fold and update overall best if needed
        all_best_params.append(best_params)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_params = best_params.copy()  # Make a copy to be safe
        
        print(f"\n    Selected best parameters for Outer Fold {fold_num}:")
        print(f"      {best_params}")
        print(f"      Best Inner CV Score: {best_score:.4f}")
        
        # Train best model on full inner training set
        print(f"    Training final model with best parameters on full training set...")
        best_model = clone(model)

        #######################################################
        # ONLY IMPORTANT FOR VOTİNG CLASSIFIER
        # Handle nested dict parameter sets for meta-estimators (VotingClassifier etc.)
        if isinstance(best_params, dict) and any(isinstance(v, dict) for v in best_params.values()):
            flat_best = {}
            for est_name, est_params in best_params.items():
                if isinstance(est_params, dict):
                    for k, v in est_params.items():
                        flat_best[f"{est_name}__{k}"] = v
                else:
                    flat_best[est_name] = est_params
            best_model.set_params(**flat_best)
        else:
            best_model.set_params(**best_params)
        
        #MMO
        X_train_clean = cleaner.fit_transform(X_train)
        X_test_clean = cleaner.transform(X_test)

        y_train = y_train.loc[X_train_clean.index]
        y_test = y_test.loc[X_test_clean.index]

        X_train_prep =  preprocessor.fit_transform(X_train_clean)
        X_test_prep = preprocessor.transform(X_test_clean)
        #MMO
        
        if model_name == 'LOGISTIC REGRESSION':
            scaler = StandardScaler()
            X_train_prep= scaler.fit_transform(X_train_prep)
            X_test_prep= scaler.transform(X_test_prep)

        best_model.fit(X_train_prep, y_train)
        # ONLY IMPORTANT FOR VOTİNG CLASSIFIER
        #######################################################
        
        
        # Store feature importances if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            metrics_data['feature_importances'].append(feature_importances)
            all_feature_importances.append(feature_importances)
        elif hasattr(best_model, 'coef_'):
            # coef_ can be 2D (n_classes, n_features). Flatten to 1D so downstream code
            # (mean across folds and DataFrame creation) receives a 1-D array per fold.
            feature_importances = best_model.coef_.ravel()
            metrics_data['feature_importances'].append(feature_importances)
            all_feature_importances.append(feature_importances)

        
        # Predict on outer test set
        y_pred = best_model.predict(X_test_prep)
        y_pred_proba = best_model.predict_proba(X_test_prep)
        
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
        
        print(f"\n    Outer Fold {fold_num} Test Results:")
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
        
        print(f"    Outer Fold {fold_num} completed")
        print(f"   {'─'*50}")

    mean_score = np.mean(outer_scores)
    print(f"\n {model_name} - FINAL RESULTS:")
    print(f"   Mean Accuracy:  {mean_score:.4f}")
    print(f"   Mean Precision: {np.mean(metrics_data['precision']):.4f}")
    print(f"   Mean Recall:    {np.mean(metrics_data['recall']):.4f}")
    print(f"   Mean F1-Score:  {np.mean(metrics_data['f1']):.4f}")
    print(f"   Best Parameters: {best_overall_params}")
    
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
    
    return mean_score, outer_scores, best_overall_params, metrics_data, all_feature_importances

def plot_roc_curves(roc_data, model_name, save_dir):
    """Plot ROC curves for all folds and save to file"""
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
    
    # Save the plot
    filename = f"{model_name.replace(' ', '_')}_ROC_Curves.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved ROC curve: {filepath}")
    return mean_auc

def plot_feature_importance(feature_importances, feature_names, model_name, save_dir):
    """Plot feature importance and save to file"""
    if len(feature_importances) == 0:
        print(f"     No feature importances available for {model_name}")
        return
    
    # Calculate mean feature importance across folds
    mean_importance = np.mean(feature_importances, axis=0)

    # Ensure we have a 1-D array for pandas (coef_ can be shape (1, n_features) or similar)
    mean_importance = np.squeeze(mean_importance)
    if getattr(mean_importance, 'ndim', 1) != 1:
        mean_importance = np.asarray(mean_importance).flatten()

    # If lengths mismatch, try to adjust by taking the first matching slice
    if len(mean_importance) != len(feature_names):
        try:
            mean_importance = mean_importance.reshape(-1)[:len(feature_names)]
        except Exception:
            raise ValueError(f"Feature names length ({len(feature_names)}) does not match importance length ({len(mean_importance)})")

    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importance
    }).sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name.replace(' ', '_')}_Feature_Importance.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved feature importance: {filepath}")
    
    # Also save feature importance data to CSV
    csv_filename = f"{model_name.replace(' ', '_')}_Feature_Importance.csv"
    csv_filepath = os.path.join(save_dir, csv_filename)
    importance_df.to_csv(csv_filepath, index=False)
    print(f"    Saved feature importance data: {csv_filepath}")
    
    return importance_df

def plot_metrics_comparison(rf_metrics, linear_metrics, catboost_metrics, baseline_accuracy, save_dir):
    """Plot comparison of metrics between models and save to file"""
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
    catboost_means = [
        np.mean(catboost_metrics['accuracy']),
        np.mean(catboost_metrics['precision']),
        np.mean(catboost_metrics['recall']),
        np.mean(catboost_metrics['f1'])
    ]
    baseline_means = [baseline_accuracy, 0, 0, 0]

    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    rects1 = ax.bar(x - 1.5*width, rf_means, width, label='Random Forest', alpha=0.8, color='skyblue')
    rects2 = ax.bar(x - 0.5*width, linear_means, width, label='Logistic Regression', alpha=0.8, color='lightcoral')
    rects3 = ax.bar(x + 0.5*width, catboost_means, width, label='CatBoost', alpha=0.8, color='lightgreen')
    rects4 = ax.bar(x + 1.5*width, baseline_means, width, label='Baseline', alpha=0.8, color='grey')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for rect in rects1 + rects2 + rects3 + rects4:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = "Model_Performance_Comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved metrics comparison: {filepath}")

def plot_combined_roc(rf_roc_data, linear_roc_data, catboost_roc_data, save_dir):
    """Plot both models' ROC curves on the same plot and save"""
    plt.figure(figsize=(12, 8))
    
    # Calculate mean ROC for Random Forest
    mean_fpr_rf = np.linspace(0, 1, 100)
    mean_tpr_rf = np.zeros_like(mean_fpr_rf)
    for fpr, tpr in zip(rf_roc_data['fpr'], rf_roc_data['tpr']):
        mean_tpr_rf += np.interp(mean_fpr_rf, fpr, tpr)
    mean_tpr_rf /= len(rf_roc_data['fpr'])
    mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)
    
    # Calculate mean ROC for Logistic Regression
    mean_fpr_lr = np.linspace(0, 1, 100)
    mean_tpr_lr = np.zeros_like(mean_fpr_lr)
    for fpr, tpr in zip(linear_roc_data['fpr'], linear_roc_data['tpr']):
        mean_tpr_lr += np.interp(mean_fpr_lr, fpr, tpr)
    mean_tpr_lr /= len(linear_roc_data['fpr'])
    mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)

    # Calculate mean ROC for CatBoost
    mean_fpr_cb = np.linspace(0, 1, 100)
    mean_tpr_cb = np.zeros_like(mean_fpr_cb)
    for fpr, tpr in zip(catboost_roc_data['fpr'], catboost_roc_data['tpr']):
        mean_tpr_cb += np.interp(mean_fpr_cb, fpr, tpr)
    mean_tpr_cb /= len(catboost_roc_data['fpr'])
    mean_auc_cb = auc(mean_fpr_cb, mean_tpr_cb)

    # Plot mean ROC curves
    plt.plot(mean_fpr_rf, mean_tpr_rf, color='blue', lw=3,
             label=f'Random Forest (AUC = {mean_auc_rf:.3f})')
    plt.plot(mean_fpr_lr, mean_tpr_lr, color='red', lw=3,
             label=f'Logistic Regression (AUC = {mean_auc_lr:.3f})')
    plt.plot(mean_fpr_cb, mean_tpr_cb, color='green', lw=3,
             label=f'CatBoost (AUC = {mean_auc_cb:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison - Random Forest vs Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save the plot
    filename = "ROC_Curve_Comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved combined ROC curve: {filepath}")

    return mean_auc_rf, mean_auc_lr, mean_auc_cb

# Define parameter sets
rf_params_list = [
    {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
]


linear_params_list = [
    {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'},
    {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}
]

catboost_params_list = [
    {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3},
    {'iterations': 300, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 5}
]

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
linear_model = LogisticRegression(random_state=42, max_iter=1000)
catboost_model = CatBoostClassifier(random_state=42, verbose=0)

print(" STARTING NESTED CROSS-VALIDATION FOR BOTH MODELS")
print("="*70)


# Train Catboost
catboost_model = CatBoostClassifier(random_state=42, verbose=0)
catboost_mean_score, catboost_fold_scores, catboost_best_params, catboost_metrics, catboost_feature_importances = nested_cross_validation_with_metrics(
    catboost_model, catboost_params_list, X, y, outer_folds=2, inner_folds=2, model_name="CATBOOST"
)
plot_feature_importance(catboost_feature_importances, X.columns, "CatBoost", results_dir)



# Train Logistic Regression
linear_mean_score, linear_fold_scores, linear_best_params, linear_metrics, linear_feature_importances = nested_cross_validation_with_metrics(
    linear_model, linear_params_list, X, y, outer_folds=2, inner_folds=2, model_name="LOGISTIC REGRESSION"
)

print(f"    Note: Logistic Regression feature importance based on coefficients")
coef_importances = np.mean(np.abs(linear_feature_importances), axis=0)
plot_feature_importance([coef_importances], X.columns, "Logistic Regression", results_dir)






# Train Random Forest
rf_mean_score, rf_fold_scores, rf_best_params, rf_metrics, rf_feature_importances = nested_cross_validation_with_metrics(
    rf_model, rf_params_list, X, y, outer_folds=2, inner_folds=2, model_name="RANDOM FOREST"
)

print("\n" + "="*70)





# Generate and save all plots
print(f"\n GENERATING AND SAVING PLOTS...")
print(f"    Saving to: {results_dir}")

# Plot metrics comparison
plot_metrics_comparison(rf_metrics, linear_metrics, catboost_metrics, baseline_accuracy, results_dir)

# Plot ROC curves if binary classification
if rf_metrics['roc_data']['fpr']:
    rf_mean_auc = plot_roc_curves(rf_metrics['roc_data'], "Random Forest", results_dir)
    linear_mean_auc = plot_roc_curves(linear_metrics['roc_data'], "Logistic Regression", results_dir)
    catboost_mean_auc = plot_roc_curves(catboost_metrics['roc_data'], "CatBoost", results_dir)
    plot_combined_roc(rf_metrics['roc_data'], linear_metrics['roc_data'], catboost_metrics['roc_data'], results_dir)

# Plot feature importance
print(f"\n GENERATING FEATURE IMPORTANCE PLOTS...")
plot_feature_importance(rf_feature_importances, X.columns, "Random Forest", results_dir)


# Final summary
print("\n" + "="*70)
print(" FINAL COMPARISON SUMMARY")
print("="*70)

print(f"\n FINAL MODEL COMPARISON:")
print(f"{'Metric':<15} {'Random Forest':<15} {'Logistic Regression':<15}")
print(f"{'-'*50}")
print(f"{'Accuracy':<15} {np.mean(rf_metrics['accuracy']):<15.4f} {np.mean(linear_metrics['accuracy']):<15.4f}")
print(f"{'Precision':<15} {np.mean(rf_metrics['precision']):<15.4f} {np.mean(linear_metrics['precision']):<15.4f}")
print(f"{'Recall':<15} {np.mean(rf_metrics['recall']):<15.4f} {np.mean(linear_metrics['recall']):<15.4f}")
print(f"{'F1-Score':<15} {np.mean(rf_metrics['f1']):<15.4f} {np.mean(linear_metrics['f1']):<15.4f}")

if rf_metrics['roc_data']['fpr']:
    print(f"{'AUC':<15} {rf_mean_auc:<15.4f} {linear_mean_auc:<15.4f}")

print(f"\n Nested Cross-Validation completed successfully!")
print(f" All results saved in: {results_dir}/")

#Using the best parameters, add a final training on the whole dataset
# and add a voter system that uses all three models to make a final prediction.
print("\n FINAL TRAINING ON FULL DATASET WITH BEST PARAMETERS")

# Train CatBoost with best params
catboost_model = CatBoostClassifier(**catboost_best_params, random_state=42, verbose=0)
catboost_model.fit(X, y)

# Train Random Forest with best params
rf_model = RandomForestClassifier(**rf_best_params, random_state=42)
rf_model.fit(X, y)

# Train Logistic Regression with best params
linear_model = LogisticRegression(**linear_best_params, random_state=42, max_iter=1000)
linear_model.fit(X, y)

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('catboost', catboost_model),
    ('random_forest', rf_model),
    ('logistic_regression', linear_model)
], voting='soft')

# Train the voting classifier
voting_clf.fit(X, y)
print(" Final models trained and voting classifier created.")
# Save the voting classifier

voting_model_path = os.path.join(results_dir, "voting_classifier_model.pkl")
joblib.dump(voting_clf, voting_model_path)

# Evaluate voting classifier with nested CV, but I need parameters for catboost, rf, and linear
catboost_params = catboost_best_params
rf_params = rf_best_params
linear_params = linear_best_params

voting_params_list = [
    {
        'catboost': catboost_params,
        'random_forest': rf_params,
        'logistic_regression': linear_params
    }
]

voting_mean_score, voting_fold_scores, voting_best_params, voting_metrics, voting_feature_importances = nested_cross_validation_with_metrics(
    voting_clf, voting_params_list, X, y, outer_folds=2, inner_folds=2, model_name="VOTING CLASSIFIER"
)
plot_roc_curves(voting_metrics['roc_data'], "Voting Classifier", results_dir)

#Use joblist to save the other models as well
catboost_model_path = os.path.join(results_dir, "catboost_model.pkl")
joblib.dump(catboost_model, catboost_model_path)
rf_model_path = os.path.join(results_dir, "random_forest_model.pkl")
joblib.dump(rf_model, rf_model_path)
linear_model_path = os.path.join(results_dir, "logistic_regression_model.pkl")
joblib.dump(linear_model, linear_model_path)