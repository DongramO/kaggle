import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
 
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
 
# 상대 경로 사용 (같은 폴더에 있는 파일 참조)
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sub = pd.read_csv('sample_submission.csv')

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(df_train.head())
 
print('df_train.shape', df_train.shape)
print('df_test.shape', df_test.shape)
 
print(df_train.info())
print(df_test.info())
print(df_train.describe())
 
 
print('df_train.shape:', df_train.shape)
print('df_test.shape:', df_test.shape)
 
 
print(df_train.info())
print(df_test.info())
print(df_train.describe())
 
cols = ['id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade', 'loan_paid_back']
 
for col in cols:
    print(col, df_train[col].nunique())
 
 
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 
            'grade_subgrade']
 
# Target Distribution Visualization
counts = df_train['loan_paid_back'].value_counts()
labels = counts.index
values = counts.values
 
plt.figure(figsize=(15,5.5))
 
bars = plt.barh(labels, values, color = 'crimson')
plt.ylabel("loan_paid_back")
plt.xlabel("Frequency")
plt.title("The Distribution of the Target Column 'loan_paid_back'")
plt.yticks([1, 0])
 
total = values.sum()
for bar, count in zip(bars, values):
    width = bar.get_width()
    pct = count / total * 100
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f"{count}\n({pct:.1f}%)",
             ha='left', va='center')
# # plt.show()
 
 
# Data dirtribution visualization
n_vars = len(num_cols)
fig, axes = plt.subplots(n_vars, 2, figsize=(12, n_vars*3))
for i, col in enumerate(num_cols):
    axes[i,0].hist(df_train[col], bins=50, edgecolor='black', color='crimson')
    axes[i,0].set_title(f'{col} histogram')
    axes[i,1].boxplot(df_train[col], vert=False)
    axes[i,1].set_title(f'{col} boxplot')
 
plt.tight_layout()
# # plt.show()
 
 
# outliers
 
n_vars = len(num_cols)
n_cols = 2
n_rows = (n_vars + 1) // n_cols
 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_vars * 3))
 
for i, col in enumerate(num_cols):
    row = i // 2
    col_idx = i % 2
    sns.boxplot(x='loan_paid_back', y=col, data=df_train, ax=axes[row, col_idx], palette='pastel')
    axes[row, col_idx].set_title(f'{col} by loan_paid_back')
 
if n_vars % 2 !=0:
    fig.delaxes(axes[n_rows-1, 1])
 
plt.tight_layout()
# # plt.show()
 
 
 
n_vars = len(cat_cols)
fig, axes = plt.subplots(n_vars, 2, figsize=(14, n_vars*8))
 
for i, col in enumerate(cat_cols):
    sns.countplot(x=df_train[col], ax=axes[i,0], order=df_train[col].value_counts().index, palette='pastel')
    axes[i,0].set_title(f'{col} countplot')
    axes[i,0].set_xlabel('')
    axes[i,0].set_ylabel('Count')
    axes[i,0].tick_params(axis='x', rotation=45)
 
    df_train[col].value_counts().plot.pie(
        ax=axes[i, 1],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel'),
        legend=False,
        ylabel=''
    )
    axes[i,1].set_title(f'{col} Pie chart')
 
plt.tight_layout()
# # plt.show()
 
n_vars = len(cat_cols)
n_cols = 2
n_rows = (n_vars + 1) // 2
 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
 
for i, col in enumerate(cat_cols):
    row  = i // 2
    col_idx = i % 2
    sns.countplot(x=col, hue='loan_paid_back', data=df_train, ax=axes[row, col_idx], palette='pastel')
    axes[row, col_idx].set_title(f"{col} by loan_paid_back")
    axes[row, col_idx].tick_params(axis='x', rotation=45)
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Count')
 
if n_vars % 2 != 0:
    fig.delaxes(axes[n_rows - 1, 1])
 
plt.tight_layout()
# # plt.show()
 
 
n_vars = len(cat_cols)
n_cols = 2
n_rows = (n_vars + 1) // 2
 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
 
for i, col in enumerate(cat_cols):
    row = i // 2
    col_idx = i % 2
 
    ratio = (
        df_train.groupby(col)['loan_paid_back']
        .value_counts(normalize=True)
        .rename('ratio')
        .mul(100)
        .reset_index()
    )
 
    ratio = ratio[ratio['loan_paid_back'] == 0]
 
    sns.barplot(
        data=ratio,
        x=col,
        y='ratio',
        ax=axes[row, col_idx],
        palette='pastel'
    )
 
    axes[row, col_idx].set_title(f"{col}: % Not Paid Back")
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('% Not Paid Back')
    axes[row, col_idx].tick_params(axis='x', rotation=45)
    axes[row, col_idx].bar_label(axes[row, col_idx].containers[0], fmt='%.1f%%', label_type='edge', fontsize=9)
 
if n_vars % 2 != 0:
    fig.delaxes(axes[n_rows - 1, 1])
 
plt.tight_layout()
# # plt.show()


def remove_outliers(df, cols):

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3- Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # lower와 upper를 구하고 사이의 값을 사용함 정상적인 값들만 사용하겠다는 의지
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df



def feature_engineering(df):
    # df['employment_status_grade_subgrade'] = df['employment_status'].astype(str) + '_' + df['grade_subgrade'].astype(str)
    
    #  # 2. interest_rate / debt_to_income_ratio
    df["interest_rate_to_dti"] = df["interest_rate"] / (df["debt_to_income_ratio"] + 1e-6)
    
    # # 3. education_level & loan_purpose
    df["loan_purpose_interest_rate"] = df["loan_purpose"].astype(str) + "_" + np.log1p(df["interest_rate"].round(1)).astype(str)
    
    # # 4. employment_status & loan_purpose
    df["employment_loan_purpose"] = df["employment_status"].astype(str) + "_" + df["loan_purpose"].astype(str)

    df["monthly_income"] = df["annual_income"] / 12
    df["debt_to_monthly_income"] = df["debt_to_income_ratio"] / (df["monthly_income"] + 1e-6)
    df["monthly_income_interest_amount"] = df["monthly_income"] / ( df["interest_rate"] * df["loan_amount"] / 12)
    # # 5. education_level & grade_subgrade
    # df["education_grade_subgrade"] = df["education_level"].astype(str) + "_" + df["grade_subgrade"].astype(str)
    df["head_grade"] = df["grade_subgrade"].astype(str).str.split('_').str[0]
    # df["sub_grade"] = df["grade_subgrade"].astype(str).str.split('_').str[1]
    df["loan_amount_credit"] = df["loan_amount"].astype(float) / (df["credit_score"].astype(float)+ 1e-6)
    df["loan_amount_div_income"] = df["loan_amount"].astype(int) / (df["annual_income"].astype(float)+ 1e-6)
    df["loan_amount_div_ratio"] = df["loan_amount"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ 1e-6)
    df["credit_div_ratio"] = df["credit_score"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ 1e-6)

    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
# df_train = remove_outliers(df_train, num_cols)
df_train.head()
df_test.head()

rs = 42

def prepare_data(df_train, target_col, num_cols, cat_cols):
    
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    

    df_train["loan_amount"] = np.log1p(df_train["loan_amount"])
    df_train["interest_rate"] = np.log1p(df_train["interest_rate"])
    df_train["annual_income"] = np.log1p(df_train["annual_income"])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    # 순서가 없는 범주형 변수들 onehot encoding
    onehot_cols = ['gender', 'marital_status', 'loan_purpose']
    
    # 순서가 있는 범주형 변수들 ordinal encoding
    ordinal_cols = ['education_level', 'employment_status', 'grade_subgrade', 'head_grade']
    
    num_transformer = StandardScaler()
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ordinal_transformer = OrdinalEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'  
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    return X_train_processed, X_val_processed, y_train, y_val, preprocessor

def optimize_models(X_train, y_train, model_type):
    def objective(trial):
        # model_type = trial.suggest_categorical("model", ["logistic", "lgbm", "xgb"])
        
        if model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
                "depth": trial.suggest_int("depth", 4, 10),
                "random_seed": rs,
                "loss_function": "Logloss",      # 2진 분류 기본
                "eval_metric": "AUC",            # 모니터링할 지표
                "verbose": False,   
            }
            model = CatBoostClassifier(**params)
        
        elif model_type == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": rs,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)
        
        else:  # XGBoost
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": rs,
                # "use_label_encoder": False,
                "eval_metric": "auc"
            }
            model = xgb.XGBClassifier(**params)
        
        # k-fold를 통해 학습할 수 있도록 객체 생성
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
        
        # 교차 검증을 통해 성능 점수 계산
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return np.mean(scores)

    # optuna를 통해 최적의 파라메터를 찾아가는 과정
    study = optuna.create_study(direction="maximize")

    # 실제로 모델을 돌려보면서 최적의 파라메터를 찾는 과정
    study.optimize(objective, n_trials=50)
    
    print("\nBest trial:" , model_type)
    print(study.best_trial.params)
    return study.best_trial.params


def create_models_with_optuna(X_train, y_train, model_type, use_fixed_params):
    """Optuna 최적화 또는 고정 파라미터로 모델 생성"""
    
    if use_fixed_params:
        # 고정 파라미터 사용
        if model_type == "catboost":
            fixed_params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "iterations": 478,
                "learning_rate": 0.09918246016024668,
                'l2_leaf_reg': 0.10336505202563943,
                'random_strength': 0.0933390543005998,
                "depth": 7,
                "random_seed": rs,
                "verbose": False,
            }
            model = CatBoostClassifier(**fixed_params)
            
        elif model_type == "lgbm":
            fixed_params = {
                "n_estimators": 774,
                "learning_rate": 0.047051628921977035,
                "num_leaves": 34,
                "max_depth": 12,
                "subsample": 0.7046955802104758,
                "colsample_bytree": 0.854461065661112,
                "reg_alpha": 8.876014045236223,
                "reg_lambda": 1.41657000413366,
                "random_state": rs,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**fixed_params)
            
        elif model_type == "xgb":
            fixed_params = {
                "n_estimators": 767,
                "learning_rate": 0.04941706054533553,
                "max_depth": 5,
                "subsample": 0.9032294063778931,
                "colsample_bytree": 0.5574373428339369,
                "min_child_weight": 10,
                "gamma": 0.14827696333170587,
                "reg_alpha": 0.0034542264442943855,
                "reg_lambda": 0.0017437934010550243,
                "random_state": rs,
                "eval_metric": "auc"
            }
            model = xgb.XGBClassifier(**fixed_params)
        
        print(f"✅ {model_type.upper()} 모델 생성 완료 (고정 파라미터 사용)")
        return model
    
    else:
        try:
            print(f"\n🚀 {model_type.upper()} 최적화 시작...")
            best_params = optimize_models(X_train, y_train, model_type)
            
            # 최적 파라미터 재확인 출력
            print(f"✅ {model_type.upper()} 최종 사용 파라미터:")
            print("-" * 80)
            for key, value in sorted(best_params.items()):
                print(f"  {key:25s}: {value}")
            print("-" * 80)
            
            # 모델 생성
            if model_type == "catboost":
                model = CatBoostClassifier(**best_params)
            elif model_type == "lgbm":
                model = lgb.LGBMClassifier(**best_params)
            elif model_type == "xgb":
                model = xgb.XGBClassifier(**best_params)
            
            print(f"✅ {model_type.upper()} 모델 생성 완료\n")
            return model
            
        except Exception as e:
            print(f"\n❌ {model_type.upper()} 모델 생성 실패: {e}")
            print(f"에러 타입: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

def ensemble_predict(models, X):
    cb_model, lgb_model, xgb_model = models
    
    cb_pred = cb_model.predict_proba(X)[:, 1]
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    
    ensemble_pred_proba = np.mean([cb_pred, lgb_pred, xgb_pred], axis=0)
    ensemble_pred = (ensemble_pred_proba >= 0.25).astype(float)
    
    return ensemble_pred, ensemble_pred_proba
def main(df_train, target_col, num_cols, cat_cols):
    X_train_processed, X_val_processed, y_train, y_val, preprocessor = prepare_data(df_train, target_col, num_cols, cat_cols)
    
    # Feature 이름 추출을 위한 정보
    onehot_cols = ['gender', 'marital_status', 'loan_purpose']
    ordinal_cols = ['education_level', 'employment_status', 'grade_subgrade', 'head_grade']
    feature_names = get_feature_names(preprocessor, num_cols, onehot_cols, ordinal_cols)
    
    model_types = ["catboost", "lgbm", "xgb"]

    print("🔍 Optimizing models with Optuna...")
    cb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[0], use_fixed_params=True)
    lgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[1], use_fixed_params=True)
    xgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[2], use_fixed_params=True)

    print("Training optimized models...")
    cb_model.fit(X_train_processed, y_train)
    lgb_model.fit(X_train_processed, y_train)
    xgb_model.fit(X_train_processed, y_train)
    
    models = {'catboost': cb_model, 
              'LightGBM': lgb_model, 
              'XGBoost': xgb_model}
    
    for name, model in models.items():
        try:
            pred = model.predict(X_val_processed)
            proba = model.predict_proba(X_val_processed)[:, 1]
            acc = accuracy_score(y_val, pred)
            auc = roc_auc_score(y_val, proba)
            print(f"{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        except Exception as e:
            print(f"{name} - Error: {e}")

    try:
        ensemble_pred, ensemble_pred_proba = ensemble_predict([cb_model, lgb_model, xgb_model], X_val_processed)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred_proba)
        print(f"Ensemble - Accuracy: {ensemble_acc:.4f}, AUC: {ensemble_auc:.4f}")
    except Exception as e:
        print(f"Ensemble Error: {e}")
    
    return preprocessor, cb_model, lgb_model, xgb_model, feature_names, onehot_cols, ordinal_cols


def get_feature_names(preprocessor, num_cols, onehot_cols, ordinal_cols):
    """전처리된 feature 이름을 추출하는 함수"""
    # ColumnTransformer의 get_feature_names_out 사용 (가장 안전한 방법)
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            return list(preprocessor.get_feature_names_out())
        except:
            pass
    
    # 대체 방법: 수동으로 구성
    feature_names = []
    
    # 수치형 컬럼
    feature_names.extend(num_cols)
    
    # OneHot 인코딩된 컬럼 이름
    onehot_transformer = preprocessor.named_transformers_['onehot']
    if hasattr(onehot_transformer, 'get_feature_names_out'):
        try:
            onehot_features = onehot_transformer.get_feature_names_out(onehot_cols)
            feature_names.extend(onehot_features)
        except:
            # 대체 방법
            for i, col in enumerate(onehot_cols):
                if hasattr(onehot_transformer, 'categories_') and i < len(onehot_transformer.categories_):
                    categories = onehot_transformer.categories_[i]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
    else:
        # 구버전 호환성
        for i, col in enumerate(onehot_cols):
            if hasattr(onehot_transformer, 'categories_') and i < len(onehot_transformer.categories_):
                categories = onehot_transformer.categories_[i]
                for cat in categories:
                    feature_names.append(f"{col}_{cat}")
    
    # Ordinal 인코딩된 컬럼
    feature_names.extend(ordinal_cols)
    
    return feature_names


def analyze_model_performance(models, X_val, y_val, model_names=None):
    """모델별 상세 성능 지표를 출력하는 함수"""
    if model_names is None:
        model_names = list(models.keys())
    
    print("\n" + "="*80)
    print("📊 모델별 상세 성능 지표")
    print("="*80)
    
    for name in model_names:
        if name not in models:
            continue
            
        model = models[name]
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n🔹 {name} 모델")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   ROC-AUC:  {auc:.4f}")
        print("\n   Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['Not Paid', 'Paid']))
        print("\n   Confusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(f"   [[{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"    [{cm[1,0]:5d}  {cm[1,1]:5d}]]")


def get_feature_importance(model, feature_names, model_name):
    """모델별 feature importance를 추출하는 함수"""
    importance_dict = {}
    
    if model_name == 'catboost' or isinstance(model, CatBoostClassifier):
        # CatBoost
        importance = model.get_feature_importance()
        importance_dict = dict(zip(feature_names, importance))
        
    elif model_name == 'LightGBM' or isinstance(model, lgb.LGBMClassifier):
        # LightGBM
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        
    elif model_name == 'XGBoost' or isinstance(model, xgb.XGBClassifier):
        # XGBoost
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
    
    return importance_dict


def visualize_feature_importance(models, feature_names, top_n=20, figsize=(15, 10)):
    """모델별 feature importance를 시각화하는 함수"""
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        importance_dict = get_feature_importance(model, feature_names, name)
        
        # 중요도 순으로 정렬
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:top_n]
        
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # 시각화
        axes[idx].barh(range(len(features)), importances, color='crimson')
        axes[idx].set_yticks(range(len(features)))
        axes[idx].set_yticklabels(features)
        axes[idx].set_xlabel('Feature Importance')
        axes[idx].set_title(f'{name} - Top {top_n} Feature Importance')
        axes[idx].invert_yaxis()
        
        # 값 표시
        for i, v in enumerate(importances):
            axes[idx].text(v, i, f' {v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature importance 시각화가 'feature_importance.png'로 저장되었습니다.")
    # plt.show()


def compare_feature_importance(models, feature_names, top_n=15):
    """모델별 feature importance를 비교하는 함수"""
    print("\n" + "="*80)
    print("📈 모델별 Feature Importance 비교 (Top {} features)".format(top_n))
    print("="*80)
    
    all_importances = {}
    
    for name, model in models.items():
        importance_dict = get_feature_importance(model, feature_names, name)
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🔹 {name} - Top {top_n} Features:")
        print("-" * 80)
        for i, (feature, importance) in enumerate(sorted_importance[:top_n], 1):
            print(f"  {i:2d}. {feature:40s} : {importance:10.4f}")
            if feature not in all_importances:
                all_importances[feature] = {}
            all_importances[feature][name] = importance
    
    # 통합 비교 시각화
    common_features = set()
    for name in models.keys():
        importance_dict = get_feature_importance(models[name], feature_names, name)
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_importance[:top_n]]
        if not common_features:
            common_features = set(top_features)
        else:
            common_features = common_features.intersection(set(top_features))
    
    if common_features:
        print(f"\n🔹 공통 중요 Feature (모든 모델에서 Top {top_n}에 포함):")
        print("-" * 80)
        for feature in sorted(common_features):
            print(f"  - {feature}")


def save_model_analysis(models, X_val, y_val, feature_names, preprocessor, 
                        num_cols, onehot_cols, ordinal_cols, filename='model_analysis.txt'):
    """모델 분석 결과를 텍스트 파일로 저장하는 함수"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("모델 상세 분석 결과\n")
        f.write("="*80 + "\n\n")
        
        # Feature 이름 정보
        f.write("📋 사용된 Feature 목록:\n")
        f.write("-" * 80 + "\n")
        f.write(f"수치형 Feature ({len(num_cols)}개): {', '.join(num_cols)}\n")
        f.write(f"OneHot Feature ({len(onehot_cols)}개): {', '.join(onehot_cols)}\n")
        f.write(f"Ordinal Feature ({len(ordinal_cols)}개): {', '.join(ordinal_cols)}\n")
        f.write(f"전체 Feature 수: {len(feature_names)}\n\n")
        
        # 모델별 성능
        f.write("="*80 + "\n")
        f.write("📊 모델별 성능 지표\n")
        f.write("="*80 + "\n\n")
        
        for name, model in models.items():
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            f.write(f"🔹 {name} 모델\n")
            f.write(f"   Accuracy: {acc:.4f}\n")
            f.write(f"   ROC-AUC:  {auc:.4f}\n")
            f.write("\n   Classification Report:\n")
            f.write(classification_report(y_val, y_pred, target_names=['Not Paid', 'Paid']))
            f.write("\n")
        
        # Feature Importance
        f.write("="*80 + "\n")
        f.write("📈 모델별 Feature Importance\n")
        f.write("="*80 + "\n\n")
        
        for name, model in models.items():
            importance_dict = get_feature_importance(model, feature_names, name)
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"🔹 {name} - Top 20 Features:\n")
            f.write("-" * 80 + "\n")
            for i, (feature, importance) in enumerate(sorted_importance[:20], 1):
                f.write(f"  {i:2d}. {feature:40s} : {importance:10.4f}\n")
            f.write("\n")
    
    print(f"\n✅ 모델 분석 결과가 '{filename}'로 저장되었습니다.")


if __name__ == "__main__":

    df_train = df_train
    df_test = df_test
    df_sub = df_sub

    num_cols = [
        'debt_to_income_ratio', 'credit_score', 'loan_amount_div_income',
        'loan_amount', 'interest_rate', 'annual_income',
        'interest_rate_to_dti', 'loan_amount_div_ratio',
        'credit_div_ratio', "monthly_income", "debt_to_monthly_income"
    ]

    cat_cols = [
        'employment_status',
        'loan_purpose',
        'grade_subgrade',              # head_grade 제거
        'employment_loan_purpose',
        'loan_purpose_interest_rate'
        # gender, marital_status, education_level 제거
    ]
    target_col = 'loan_paid_back'
    
    preprocessor, cb_model, lgb_model, xgb_model, feature_names, onehot_cols, ordinal_cols = main(
        df_train, target_col, num_cols, cat_cols
    )
    
    # 모델 분석 수행
    models = {'catboost': cb_model, 'LightGBM': lgb_model, 'XGBoost': xgb_model}
    X_train_processed, X_val_processed, y_train, y_val, _ = prepare_data(df_train, target_col, num_cols, cat_cols)
    
    print("\n" + "="*80)
    print("🔍 모델 상세 분석 시작")
    print("="*80)
    
    # 1. 모델별 상세 성능 지표 출력
    analyze_model_performance(models, X_val_processed, y_val)
    
    # 2. Feature Importance 비교
    compare_feature_importance(models, feature_names, top_n=20)
    
    # 3. Feature Importance 시각화
    visualize_feature_importance(models, feature_names, top_n=20)
    
    # 4. 분석 결과를 파일로 저장
    save_model_analysis(models, X_val_processed, y_val, feature_names, preprocessor,
                       num_cols, onehot_cols, ordinal_cols, filename='model_analysis.txt')
    
    # 테스트 데이터 예측 및 제출 파일 생성
    X_test_final_processed = preprocessor.transform(df_test)
    
    _, y_pred_ensemble = ensemble_predict(
        [cb_model, lgb_model, xgb_model], 
        X_test_final_processed
    )
    
    submission = pd.DataFrame({
        'id': df_sub['id'],
        'loan_paid_back': y_pred_ensemble
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    df_confirm = pd.read_csv('submission_ensemble.csv')

    print(df_confirm.head())