from install import check_and_install
check_and_install("scikit-learn")
check_and_install("xgboost")

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def impute_scale_train(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                          model, ax_roc, ax_pr, imputation_strategy: str = "median", scaler_type: str = "StandardScaler"):

    # Set up pipeline impute -> scaler -> model train
    imputer = SimpleImputer(strategy = imputation_strategy)
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    pipe = Pipeline(steps = [("impute", imputer), 
                             ("scale", scaler), 
                             ("classifier", model)])
    pipe.fit(X_train, y_train)
    model_scores = pipe.predict_proba(X_val)[:,1]

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, model_scores)
    auroc = roc_auc_score(y_val, model_scores)
    ax_roc.plot(fpr, tpr, label = f"{model.__class__.__name__} (AUC = {auroc:.4f})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    # Plot PR curve
    fpr, tpr, thresholds = precision_recall_curve(y_val, model_scores)
    auprc = average_precision_score(y_val, model_scores)
    ax_pr.plot(fpr, tpr, label = f"{model.__class__.__name__} (AUC = {auprc:.4f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()

def impute_scale_tune_train(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                            model, ax_roc, ax_pr, params,
                            imputation_strategy: str = "median", scaler_type: str = "StandardScaler", 
                            num_folds: int = 5, cv_scoring: str = "average_precision"):

    # Set up pipeline impute -> scaler -> model train
    imputer = SimpleImputer(strategy = imputation_strategy)
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    pipe = Pipeline(steps = [("impute", imputer), 
                             ("scale", scaler), 
                             ("classifier", model)])
    grid = GridSearchCV(pipe, params, cv = num_folds, n_jobs = -1, scoring = cv_scoring, refit = True)   
    grid.fit(X_train, y_train)
    model_scores = grid.predict_proba(X_val)[:,1]
    # Print the best parameters
    print("Best parameters:", grid.best_params_)
    # Print the best score
    print("Best score:", grid.best_score_)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, model_scores)
    auroc = roc_auc_score(y_val, model_scores)
    ax_roc.plot(fpr, tpr, label = f"{model.__class__.__name__} (AUC = {auroc:.4f})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    # Plot PR curve
    fpr, tpr, thresholds = precision_recall_curve(y_val, model_scores)
    auprc = average_precision_score(y_val, model_scores)
    ax_pr.plot(fpr, tpr, label = f"{model.__class__.__name__} (AUC = {auprc:.4f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()

    return grid.best_estimator_