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
 
from sklearn.metrics import accuracy_score, roc_auc_score
 
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
 
# 상대 경로 사용 (같은 폴더에 있는 파일 참조)
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sub = pd.read_csv('sample_submission.csv')
 
 
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
# plt.show()
 
 
# Data dirtribution visualization
n_vars = len(num_cols)
fig, axes = plt.subplots(n_vars, 2, figsize=(12, n_vars*3))
for i, col in enumerate(num_cols):
    axes[i,0].hist(df_train[col], bins=50, edgecolor='black', color='crimson')
    axes[i,0].set_title(f'{col} histogram')
    axes[i,1].boxplot(df_train[col], vert=False)
    axes[i,1].set_title(f'{col} boxplot')
 
plt.tight_layout()
# plt.show()
 
 
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
# plt.show()
 
 
 
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
# plt.show()
 
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
# plt.show()
 
 
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
# plt.show()


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
    # df["interest_rate_to_dti"] = df["interest_rate"] / (df["debt_to_income_ratio"] + 1e-6)
    
    # # 3. education_level & loan_purpose
    # df["education_loan_purpose"] = df["education_level"].astype(str) + "_" + df["loan_purpose"].astype(str)
    
    # # 4. employment_status & loan_purpose
    # df["employment_loan_purpose"] = df["employment_status"].astype(str) + "_" + df["loan_purpose"].astype(str)
    
    # # 5. education_level & grade_subgrade
    # df["education_grade_subgrade"] = df["education_level"].astype(str) + "_" + df["grade_subgrade"].astype(str)

    df["head_grade"] = df["grade_subgrade"].astype(str).str.split('_').str[0]
    df["sub_grade"] = df["grade_subgrade"].astype(str).str.split('_').str[1]
    df["loan_amount_div_income"] = df["loan_amount"].astype(float) / (df["annual_income"].astype(float)+ 1e-6)
    df["loan_amount_log"] = np.log1p(df["loan_amount"])
    df["interest_rate_log"] = np.log1p(df["interest_rate"])
    df["annual_income_log"] = np.log1p(df["annual_income"])

    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
df_train = remove_outliers(df_train, num_cols)
df_train.head()
df_test.head()

rs = 42

def prepare_data(df_train, target_col, num_cols, cat_cols):
    
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    # 순서가 없는 범주형 변수들 onehot encoding
    onehot_cols = ['gender', 'marital_status', 'loan_purpose']
    
    # 순서가 있는 범주형 변수들 ordinal encoding
    ordinal_cols = ['education_level', 'employment_status', 'grade_subgrade', 'head_grade', 'sub_grade']
    
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
                "loss_function": "Logloss",      # 2진 분류 기본
                "eval_metric": "AUC",            # 모니터링할 지표
                "iterations": trial.suggest_int("iterations", 100, 500),               # 트리 개수
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "random_state": rs,
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
    
    print("\nBest trial:")
    print(study.best_trial.params)
    return study.best_trial.params


def create_models_with_optuna(X_train, y_train, model_type):
    best_params = optimize_models(X_train, y_train, model_type)
    
    # 최적의 파라메터를 갖는 모델을 반환
    # model_type = best_params.pop("model")

    # 모델별로 하나씩 돌려서 파라메터를 확보하는건가???
    if model_type == "catboost":
        model = CatBoostClassifier(**best_params)
    elif model_type == "lgbm":
        model = lgb.LGBMClassifier(**best_params)
    elif model_type == "xgb":
        model = xgb.XGBClassifier(**best_params)
    
    print(f"\n✅ Selected model: {model_type}")
    return model

def ensemble_predict(models, X):
    cb_model, lgb_model, xgb_model = models
    
    lr_pred = cb_model.predict_proba(X)[:, 1]
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    
    ensemble_pred_proba = np.mean([lr_pred, lgb_pred, xgb_pred], axis=0)
    ensemble_pred = (ensemble_pred_proba >= 0.5).astype(float)
    
    return ensemble_pred, ensemble_pred_proba
def main(df_train, target_col, num_cols, cat_cols):
    X_train_processed, X_val_processed, y_train, y_val, preprocessor = prepare_data(df_train, target_col, num_cols, cat_cols)
    
    model_types = ["catboost", "lgbm", "xgb"]

    print("🔍 Optimizing models with Optuna...")
    cb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[0])
    lgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[1])
    xgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[2])

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
    
    return preprocessor, cb_model, lgb_model, xgb_model


if __name__ == "__main__":

    df_train = df_train
    df_test = df_test
    df_sub = df_sub

    num_cols = ['debt_to_income_ratio', 'credit_score', 'loan_amount_div_income', 'loan_amount_log', 'interest_rate_log', 'annual_income_log']
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade', 'head_grade', 'sub_grade']
    target_col = 'loan_paid_back'
    
    preprocessor, cb_model, lgb_model, xgb_model = main(df_train, target_col, num_cols, cat_cols)
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