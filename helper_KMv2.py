# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1xr5Z40HWgBmgMOTe_gdgmRx4K4IX7GqW
[Smoker Status Prediction using Bio-Signals](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)
"""

# Standard libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics & math
from scipy.stats import chi2
from scipy.stats.mstats import winsorize

# Machine learning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class dataCleaning(BaseEstimator, TransformerMixin):
    """
    Perform basic dataset cleaning before preprocessing.

    Steps:
    1. Normalize column names: lowercase, replace spaces and parentheses with underscores.
    2. Remove duplicate rows.
    3. Convert numeric columns to float.
    4. Encode hearing-related columns: {1→0, 2→1}.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 1. Normalize column names
        df.columns = (
            df.columns.str.lower()
            .str.replace(r"[ (]", "_", regex=True)
            .str.replace(")", "", regex=False)
        )

        # 2. Remove duplicate rows
        df = df.drop_duplicates()

        # 3. Convert selected numeric columns to float (if present)
        numeric_cols = [
            'height_cm', 'weight_kg', 'systolic', 'relaxation',
            'fasting_blood_sugar', 'cholesterol', 'triglyceride',
            'hdl', 'ldl', 'ast', 'alt', 'gtp'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Encode hearing-related columns (1→0, 2→1)
        if 'hearing_left' in df.columns:
            df['hearing_left'] = df['hearing_left'].replace({1: 0, 2: 1})
        if 'hearing_right' in df.columns:
            df['hearing_right'] = df['hearing_right'].replace({1: 0, 2: 1})

        def get_feature_names_out(self, input_features=None):
            return np.array(input_features)

        return df


groups = {
    'Physical': ['height_cm', 'weight_kg', 'waist_cm'],
    'Vision': ['eyesight_left', 'eyesight_right'],
    'Blood_Pressure': ['systolic', 'relaxation'],
    'Blood_Sugar': ['fasting_blood_sugar'],
    'Lipid': ['cholesterol', 'triglyceride', 'hdl', 'ldl'],
    'Hematologic_Renal': ['hemoglobin', 'serum_creatinine'],
    'Liver': ['ast', 'alt', 'gtp']
}


"""transformers"""
class MahalanobisOutlierFlagger(BaseEstimator, TransformerMixin):
    """
    Identify and flag outliers using the Mahalanobis distance,
    calculated from the physical feature group ['height_cm', 'weight_kg', 'waist_cm'].
    """
    def __init__(self, alpha=0.99):
        self.alpha = alpha

    def fit(self, X, y=None):
        X_np = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X_np, axis=0)
        self.cov_ = np.cov(X_np, rowvar=False)
        self.inv_cov_ = np.linalg.inv(self.cov_)
        self.threshold_ = np.sqrt(chi2.ppf(self.alpha, df=X_np.shape[1]))
        return self

    def transform(self, X):
        X_np = np.asarray(X, dtype=float)
        diff = X_np - self.mean_
        m_dist = np.sqrt(np.sum(diff @ self.inv_cov_ * diff, axis=1))
        X_out = X.copy()
        X_out["physical_outlier_flag"] = pd.Series((m_dist > self.threshold_).astype("int32"), index=X_out.index)
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        return np.append(input_features, 'physical_outlier_flag')


class VisionReplacer(BaseEstimator, TransformerMixin):
    """
    Replace specific invalid eyesight values (e.g., 9.9) with NaN,
    then impute those NaN values with the column median.
    Compatible with sklearn Pipeline / ColumnTransformer.
    """
    def __init__(self, value=9.9):
        self.value = value

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        # 중앙값 계산 (NaN 무시)
        self.median_ = np.nanmedian(np.where(X_arr == self.value, np.nan, X_arr), axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_arr = np.where(X_arr == self.value, np.nan, X_arr)
        # NaN을 중앙값으로 대체
        idx = np.isnan(X_arr)
        X_arr[idx] = np.take(self.median_, np.where(idx)[1])
        return X_arr
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)


# BP clipper
class BloodPressureClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers in blood pressure features to predefined bounds.
    Out-of-range values are replaced with the corresponding boundary values.
    Columns expected: ['systolic', 'relaxation']
    """

    def __init__(self,
                 systolic_lower=70, systolic_upper=250,
                 relaxation_lower=40, relaxation_upper=150):
        self.systolic_lower = systolic_lower
        self.systolic_upper = systolic_upper
        self.relaxation_lower = relaxation_lower
        self.relaxation_upper = relaxation_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_out = X.copy()

        # systolic bounds
        X_out["systolic"] = np.clip(
            X_out["systolic"].astype(float), self.systolic_lower, self.systolic_upper
            )
        
        # relaxation bounds
        X_out["relaxation"] = np.clip(
            X_out["relaxation"].astype(float), self.relaxation_lower, self.relaxation_upper
            )
        
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)


class BloodSugarOutlier(BaseEstimator, TransformerMixin):
    """
    Detect and handle outliers for 'fasting_blood_sugar':
    - Clip physically impossible values
    - Detect outliers via IQR + IsolationForest (multivariate context)
    - Winsorize only samples detected by both
    """

    def __init__(self,
                 lower_bound=10, upper_bound=600,
                 ref_features=None, random_state=42):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.ref_features = ref_features or ['fasting_blood_sugar', 'triglyceride', 'hdl', 'ldl', 'cholesterol', 'waist_cm']
        self.random_state = random_state
        self.target_col = 'fasting_blood_sugar'
        self.iso_ = None

    def fit(self, X, y=None):
        df = X.copy()
        col = self.target_col

        valid_features = [f for f in self.ref_features if f in df.columns]

        # 1. Clip physically impossible values
        df[col] = np.clip(df[col], self.lower_bound, self.upper_bound)

        # 2. Compute IQR bounds
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        self.lower_iqr_ = Q1 - 1.5 * IQR
        self.upper_iqr_ = Q3 + 1.5 * IQR

        # 3. Determine contamination from IQR rate
        rate_iqr = ((df[col] < self.lower_iqr_) | (df[col] > self.upper_iqr_)).mean()
        self.contamination_ = max(rate_iqr, 1e-4)

        self.iso_ = IsolationForest(contamination=self.contamination_, random_state=self.random_state)
        self.iso_.fit(df[valid_features])

        return self

    def transform(self, X):

        if self.iso_ is None:
            raise RuntimeError(
                "BloodSugarOutlier instance is not fitted yet. "
                "Call fit() before transform()."
            )

        df = X.copy()
        col = self.target_col

        # 1. Clip physically impossible values
        df[col] = np.clip(df[col], self.lower_bound, self.upper_bound)

        # 2. Recalculate IQR bounds for current fold
        # Q1, Q3 = df[col].quantile([0.25, 0.75])
        # IQR = Q3 - Q1
        # lower_iqr = Q1 - 1.5 * IQR
        # upper_iqr = Q3 + 1.5 * IQR

        lower_iqr = self.lower_iqr_ 
        upper_iqr = self.upper_iqr_ 

        # 3. Detect outliers again for current fold
        mask_iqr = (df[col] < lower_iqr) | (df[col] > upper_iqr)
        valid_features = [f for f in self.ref_features if f in df.columns]
        # iso = IsolationForest(contamination=self.contamination_, random_state=self.random_state)
        # pred_iforest = iso.fit_predict(df[valid_features])
        pred_iforest = self.iso_.predict(df[valid_features])
        mask_iforest = (pred_iforest == -1)

        # 4. Combine both methods
        both_mask = mask_iqr & mask_iforest

        # 5. Winsorize agreed outliers
        df[col] = df[col].astype(float)
        df.loc[both_mask & (df[col] < lower_iqr), col] = float(lower_iqr)
        df.loc[both_mask & (df[col] > upper_iqr), col] = float(upper_iqr)

        # Debug info — now uses same mask, no double call
        # print("transform shape df:", df.shape)
        # print("fit_predict shape:", df[valid_features].shape)
        # print("mask_iqr vs mask_iforest:", mask_iqr.shape, mask_iforest.shape)

        return df[[col]]

    def get_feature_names_out(self, input_features=None):
        return np.array([self.target_col])


# Detect outliers using measurable (physical) ranges, IQR, and Isolation Forest,
# then handle detected outliers by clipping them to their IQR-based limits.

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Generalized outlier handler for numeric feature groups.

    Steps:
    1. Clip values to measurable physical bounds.
    2. Compute IQR-based lower/upper bounds per feature.
    3. Compute average IQR outlier rate → contamination ratio.
    4. In transform(), apply IsolationForest and winsorize values
       where both IQR and IsolationForest detect outliers.
    """

    def __init__(self, features, clip_bounds, random_state=42):
        self.features = features
        self.clip_bounds = clip_bounds
        self.random_state = random_state
        self.iso_ = None

    
    # compute IQR bounds and contamination rate only
    def fit(self, X, y=None):
        df = X.copy()

        # store bounds
        self.bounds_ = {}

        # 1. clip to measurable bounds
        for col, (low, high) in self.clip_bounds.items():
            df[col] = df[col].clip(lower=low, upper=high)

        # 2. compute IQR bounds and individual outlier flags
        iqr_rates = []
        for col in self.features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_iqr = Q1 - 1.5 * IQR
            upper_iqr = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower_iqr, upper_iqr)
            outlier_rate = ((df[col] < lower_iqr) | (df[col] > upper_iqr)).mean()
            iqr_rates.append(outlier_rate)

        # 3. determine contamination (average IQR rate)
        avg_rate = np.mean(iqr_rates)
        self.contamination_ = max(avg_rate, 1e-4)

        self.iso_ = IsolationForest(contamination=self.contamination_, random_state=self.random_state)
        self.iso_.fit(df[self.features])

        return self

    # re-apply IsolationForest and winsorize per fold
    def transform(self, X):

        if self.iso_ is None:
            raise RuntimeError(
                "OulierHandler instance is not fitted yet. "
                "Call fit() before transform()."
            )



        df = X.copy()

        # 1. clip again to measurable bounds
        for col, (low, high) in self.clip_bounds.items():
            df[col] = df[col].clip(lower=low, upper=high)

        # 2. IsolationForest for current fold
        # iso = IsolationForest(contamination=self.contamination_, random_state=self.random_state)
        # mask_iforest = (iso.fit_predict(df[self.features]) == -1)
        mask_iforest = (self.iso_.predict(df[self.features]) == -1)
        # # df["outlier_iforest"] = pd.Series(mask_iforest.astype("int32"), index=df.index)

        # 3. per-column IQR mask and winsorization
        for col in self.features:
            # lower_iqr, upper_iqr = self.bounds_[col]
            lower_iqr, upper_iqr = self.bounds_[col]
            mask_iqr = (df[col] < lower_iqr) | (df[col] > upper_iqr)
            both_mask = mask_iforest & mask_iqr

            df[col] = df[col].astype(float)
            df.loc[both_mask & (df[col] < lower_iqr), col] = float(lower_iqr)
            df.loc[both_mask & (df[col] > upper_iqr), col] = float(upper_iqr)

        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)

class KidneyOutlier(BaseEstimator, TransformerMixin):
    """
    Flag outliers in ['hemoglobin', 'serum_creatinine'] based on:
    (1) Mahalanobis distance (covariance-based deviation)
    (2) Robust regression (RANSAC) residuals
    If either method flags a sample as an outlier, flag = 1.
    """

    def __init__(self, alpha=0.999, residual_sigma=3.0, residual_threshold=1.5, random_state=42):
        self.alpha = alpha
        self.residual_sigma = residual_sigma
        self.residual_threshold = residual_threshold
        self.random_state = random_state
        self.features = ['hemoglobin', 'serum_creatinine']

    def fit(self, X, y=None):
        df = X[self.features].copy().dropna()
        a = df.values

        # --- (1) Mahalanobis distance setup ---
        self.mean_vec_ = np.mean(a, axis=0)
        self.cov_mat_ = np.cov(a, rowvar=False)
        self.inv_covmat_ = np.linalg.inv(self.cov_mat_)
        self.threshold_mahal_ = np.sqrt(chi2.ppf(self.alpha, df=len(self.features)))

        # --- (2) RANSAC regression setup ---
        X_r = df['serum_creatinine'].values.reshape(-1, 1)
        y_r = df['hemoglobin'].values
        self.ransac_ = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self.residual_threshold,
            random_state=self.random_state
        )
        self.ransac_.fit(X_r, y_r)

        # Calculate residual threshold from training data
        y_pred = self.ransac_.predict(X_r)
        residuals = np.abs(y_r - y_pred)
        self.resid_threshold_ = np.mean(residuals) + self.residual_sigma * np.std(residuals)

        return self

    def transform(self, X):
        df = X.copy()

        # Handle possible NaNs
        valid_mask = df[self.features].notna().all(axis=1)
        outlier_flag = np.zeros(len(df), dtype=int)

        if valid_mask.sum() > 0:
            df[self.features] = df[self.features].astype(float)
            a = df.loc[valid_mask, self.features].values

            # (1) Mahalanobis distance
            diff = a - self.mean_vec_
            m_dist = np.sqrt(np.sum(diff @ self.inv_covmat_ * diff, axis=1))
            outlier_mahal = (m_dist > self.threshold_mahal_).astype(int)

            # (2) RANSAC residuals
            X_r = df.loc[valid_mask, 'serum_creatinine'].values.reshape(-1, 1)
            y_r = df.loc[valid_mask, 'hemoglobin'].values
            y_pred = self.ransac_.predict(X_r)
            residuals = np.abs(y_r - y_pred)
            outlier_resid = (residuals > self.resid_threshold_).astype(int)

            # Combine: Mahalanobis OR Residual outlier
            outlier_flag[valid_mask.values] = np.maximum(outlier_mahal, outlier_resid)

        df["kidney_outlier_flag"] = pd.Series(outlier_flag.astype("int32"), index=df.index)

        return df
    
    def get_feature_names_out(self, input_features=None):
        return np.append(input_features, 'kidney_outlier_flag')


def get_preprocessor():
    """
    Build and return the full preprocessing pipeline (ColumnTransformer)
    combining all feature-group specific transformers.
    """
    # Vision
    vision_pipe = Pipeline([
        ('replace_and_impute', VisionReplacer(value=9.9))
    ])

    # Physical
    physical_pipe = Pipeline([
        ('outlier_flag', MahalanobisOutlierFlagger(alpha=0.99))
    ])

    # Blood Pressure
    bp_pipe = Pipeline([
        ('clipper', BloodPressureClipper())
    ])

    # Blood Sugar
    sugar_pipe = Pipeline([
        ('outlier_handler', BloodSugarOutlier())
    ])

    # Lipid
    lipid_pipe = Pipeline([
        ('lipid_outlier', OutlierHandler(
            features=['cholesterol', 'triglyceride', 'hdl', 'ldl'],
            clip_bounds={
                'cholesterol': (50, 400),
                'triglyceride': (30, 1000),
                'hdl': (10, 150),
                'ldl': (0, 400)
            }
        ))
    ])

    # Liver
    liver_pipe = Pipeline([
        ('liver_outlier', OutlierHandler(
            features=['ast', 'alt', 'gtp'],
            clip_bounds={
                'ast': (5, 500),
                'alt': (5, 500),
                'gtp': (5, 1000)
            }
        ))
    ])

    # Kidney
    kidney_pipe = Pipeline([
        ('kidney_outlier', KidneyOutlier())
    ])

    # ColumnTransformer 통합
    preprocessor = ColumnTransformer([
        ('vision', vision_pipe, ['eyesight_left', 'eyesight_right']),
        ('physical', physical_pipe, ['height_cm', 'weight_kg', 'waist_cm']),
        ('blood_pressure', bp_pipe, ['systolic', 'relaxation']),
        ('blood_sugar', sugar_pipe, ['fasting_blood_sugar']),
        ('lipid', lipid_pipe, ['cholesterol', 'triglyceride', 'hdl', 'ldl']),
        ('liver', liver_pipe, ['ast', 'alt', 'gtp']),
        ('kidney', kidney_pipe, ['hemoglobin', 'serum_creatinine'])
    ], remainder='passthrough',
       verbose_feature_names_out=False,
       force_int_remainder_cols=False
       )

    return preprocessor