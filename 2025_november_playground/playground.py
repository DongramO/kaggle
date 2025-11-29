import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import log_loss
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

def eda(df_train, df_test, df_sub):
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

    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    
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
    eps = 1e-6
    # 1. interest_rate / debt_to_income_ratio
    df["interest_rate_to_dti"] = df["interest_rate"] / (df["debt_to_income_ratio"] + eps)
    
    # 2. education_level & loan_purpose
    df["loan_purpose_interest_rate"] = df["loan_purpose"].astype(str) + "_" + np.log1p(df["interest_rate"].round(1)).astype(str)
    
    # 3. employment_status & loan_purpose
    df['employment_status_grade_subgrade'] = df['employment_status'].astype(str) + '_' + df['grade_subgrade'].astype(str)
    df["employment_loan_purpose"] = df["employment_status"].astype(str) + "_" + df["loan_purpose"].astype(str)
    df["education_loan_purpose"] = df["education_level"].astype(str) + "_" + df["loan_purpose"].astype(str)

    # 4. monthly_income 
    df["monthly_income"] = df["annual_income"] / 12
    df["debt_to_monthly_income"] = df["debt_to_income_ratio"] / (df["monthly_income"] + eps)
    # df["monthly_income_interest_amount"] = df["monthly_income"] / ( df["interest_rate"] * df["loan_amount"] / 12)
    df["estimated_monthly_payment"] = (df["loan_amount"] * df["interest_rate"]) / 12
    df["pti_ratio"] = df["estimated_monthly_payment"] / (df["monthly_income"] + eps)
    
    # 5. education_level & grade_subgrade
    # df["education_grade_subgrade"] = df["education_level"].astype(str) + "_" + df["grade_subgrade"].astype(str)
    df["head_grade"] = df["grade_subgrade"].astype(str).str.split('_').str[0]
    
    # 6. loan_amount_div
    df["loan_amount_credit"] = df["loan_amount"].astype(float) / (df["credit_score"].astype(float)+ eps)
    df["loan_amount_div_income"] = df["loan_amount"].astype(int) / (df["annual_income"].astype(float)+ eps)
    df["loan_amount_div_ratio"] = df["loan_amount"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ eps)
    
    # 7. creadit
    df["credit_div_ratio"] = df["credit_score"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ eps)
    
    # 범주형 특성 변수
    
    
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
    onehot_cols = ['gender', 'marital_status', 'loan_purpose',
                #    'employment_loan_purpose',
                #    'education_loan_purpose',
                   ]
    
    # 순서가 있는 범주형 변수들 ordinal encoding
    ordinal_cols = ['education_level', 'employment_status', 'grade_subgrade', 'head_grade',
                    'employment_status_grade_subgrade',
                    ]
    
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
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100.0, log=True),
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
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
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
                "eval_metric": "auc",
            }
            model = xgb.XGBClassifier(**params)
        
        # k-fold를 통해 학습할 수 있도록 객체 생성
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
        
        # 교차 검증을 통해 성능 점수 계산
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return np.mean(scores)

    # optuna를 통해 최적의 파라메터를 찾아가는 과정
    sampler = optuna.samplers.TPESampler(seed=rs)
    study = optuna.create_study(direction="maximize", sampler=sampler)
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
                "iterations": 475,
                "learning_rate": 0.09983620901362981,
                'l2_leaf_reg': 15.332725842146589,
                'random_strength': 0.7485180520264623,
                "depth": 7,
                "random_seed": rs,
                "verbose": False,
            }
            model = CatBoostClassifier(**fixed_params)
            
        elif model_type == "lgbm":
            fixed_params = {
                "colsample_bytree": 0.5295722198581925,
                "learning_rate": 0.04919972117020984,
                "max_depth": 8,
                "n_estimators": 576,
                "num_leaves": 93,
                "reg_alpha": 9.320749906598678,
                "reg_lambda": 0.045393254289234894,
                "subsample": 0.8368214696612971,
                "random_state": rs,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**fixed_params)
            
        elif model_type == "xgb":
            fixed_params = {
                "colsample_bytree": 0.9738845151688945,
                "gamma": 0.6097633807345567,
                "learning_rate": 0.03398329815843991,
                "max_depth": 7,
                "min_child_weight": 5,
                "n_estimators": 690,
                "reg_alpha": 4.931603458294555,
                "reg_lambda": 0.14289737491259996,
                "subsample": 0.7574655551921472,
                "random_state": rs,
                "eval_metric": "auc",
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

def find_optimal_weights(models, X_valid, y_valid):
    cb_model, lgb_model, xgb_model = models
    
    # 각 모델 확률 예측 가져오기
    cb_pred = cb_model.predict_proba(X_valid)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_valid)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_valid)[:, 1]
    
    preds = np.vstack([cb_pred, lgb_pred, xgb_pred]).T  # shape: (N, 3)

    # 초기값 (균등분배)
    init_w = np.array([1/3, 1/3, 1/3])

    # 제약조건: w >= 0, sum(w)=1
    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    min_w = 0.1
    bounds = [(min_w, 1)] * 3

    # 목적 함수: logloss 최소화
    def loss_fn(w):
        blended = np.dot(preds, w)
        return log_loss(y_valid, blended)

    result = minimize(loss_fn, init_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    optimal_w = result.x
    # print(f"Optimal weights: {optimal_w}")
    return optimal_w

def ensemble_predict(models, X, weights):
    cb_model, lgb_model, xgb_model = models
    
    cb_pred = cb_model.predict_proba(X)[:, 1]
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    
    preds = np.vstack([cb_pred, lgb_pred, xgb_pred]).T  # (N, 3)
    
    ensemble_pred_proba = np.dot(preds, weights)
    ensemble_pred = (ensemble_pred_proba >= 0.25).astype(float)
    
    return ensemble_pred, ensemble_pred_proba

def main(df_train, target_col, num_cols, cat_cols):
    X_train_processed, X_val_processed, y_train, y_val, preprocessor = prepare_data(df_train, target_col, num_cols, cat_cols)
    
    # Feature 이름 추출을 위한 정보
    onehot_cols = ['gender', 'marital_status', 'loan_purpose']
    ordinal_cols = ['education_level', 'employment_status', 'grade_subgrade', 'head_grade']
    
    model_types = ["catboost", "lgbm", "xgb"]

    print("🔍 Optimizing models with Optuna...")

    cb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[0], use_fixed_params=True)
    lgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[1], use_fixed_params=True)
    xgb_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[2], use_fixed_params=True)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    
    print("Training optimized models...")
    cb_model.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        use_best_model=True,
        verbose=False
    )
    
    lgb_model.fit(
        X_train_df, y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(0)
        ]
    )
    
    xgb_model.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=False
    )
    
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
        weights = find_optimal_weights([cb_model, lgb_model, xgb_model], X_val_processed, y_val)
        ensemble_pred, ensemble_pred_proba = ensemble_predict([cb_model, lgb_model, xgb_model], X_val_processed, weights)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred_proba)
        print(f"Ensemble - Accuracy: {ensemble_acc:.4f}, AUC: {ensemble_auc:.4f}")
    except Exception as e:
        print(f"Ensemble Error: {e}")
    
    return preprocessor, cb_model, lgb_model, xgb_model, onehot_cols, ordinal_cols, X_train_processed, X_val_processed, y_train, y_val


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


def visualize_feature_importance(models, feature_names, top_n=30, figsize=(15, 10)):
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
        
        # 전체 Feature 목록 출력 (신규 피쳐 포함)
        f.write("📋 전체 Feature 목록 (신규 조합 특성 포함):\n")
        f.write("-" * 80 + "\n")
        
        # 원본 특성과 신규 특성 구분
        original_features = get_feature_names(preprocessor, num_cols, onehot_cols, ordinal_cols)
        new_features = [f for f in feature_names if f not in original_features]
        
        f.write(f"원본 특성 수: {len(original_features)}\n")
        f.write(f"신규 조합 특성 수: {len(new_features)}\n\n")
        
        if new_features:
            f.write("🆕 신규 조합 특성 목록:\n")
            for i, feat in enumerate(new_features, 1):
                f.write(f"  {i:3d}. {feat}\n")
            f.write("\n")
        
        f.write("전체 특성 목록:\n")
        for i, feat in enumerate(feature_names, 1):
            is_new = "🆕" if feat in new_features else "  "
            f.write(f"  {i:3d}. {is_new} {feat}\n")
        f.write("\n")
        
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
        
        # Feature Importance - 모든 특성 저장
        f.write("="*80 + "\n")
        f.write("📈 모델별 Feature Importance (전체 특성)\n")
        f.write("="*80 + "\n\n")
        
        for name, model in models.items():
            importance_dict = get_feature_importance(model, feature_names, name)
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"🔹 {name} - 전체 Features ({len(sorted_importance)}개):\n")
            f.write("-" * 80 + "\n")
            for i, (feature, importance) in enumerate(sorted_importance, 1):
                is_new = "🆕" if feature in new_features else "  "
                f.write(f"  {i:3d}. {is_new} {feature:50s} : {importance:10.4f}\n")
            f.write("\n")
    
    print(f"\n✅ 모델 분석 결과가 '{filename}'로 저장되었습니다.")
    print(f"   - 전체 특성 수: {len(feature_names)}개")
    if new_features:
        print(f"   - 신규 조합 특성 수: {len(new_features)}개")

if __name__ == "__main__":

    df_train = df_train
    df_test = df_test
    df_sub = df_sub
    # eda(df_train, df_test, df_sub)

    num_cols = [
        'debt_to_income_ratio', 'credit_score', 'loan_amount_div_income',
        'loan_amount', 'interest_rate', 'annual_income',
        'interest_rate_to_dti', 'loan_amount_div_ratio',
        'credit_div_ratio', 
        'loan_amount_credit',
        # 'estimated_monthly_payment', 'pti_ratio'
    ]

    cat_cols = [
        'employment_status',
        'loan_purpose',
        'grade_subgrade',              # head_grade 제거
        'loan_purpose_interest_rate'
        # 'employment_loan_purpose',
        # 'employment_status_grade_subgrade',
        # 'education_loan_purpose',
        # gender, marital_status, education_level 제거
    ]
    target_col = 'loan_paid_back'
    
    preprocessor, cb_model, lgb_model, xgb_model, onehot_cols, ordinal_cols, X_train_processed, X_val_processed, y_train, y_val = main(
        df_train, target_col, num_cols, cat_cols
    )
    # 모델 분석 수행
    models = {'catboost': cb_model, 'LightGBM': lgb_model, 'XGBoost': xgb_model}
    
    print("\n" + "="*80)
    print("🔍 모델 상세 분석 시작")
    print("="*80)
    
    # # 1. 모델별 상세 성능 지표 출력
    # analyze_model_performance(models, X_val_processed, y_val)
    
    # # 2. Feature Importance 비교
    # compare_feature_importance(models, feature_names, top_n=20)
    
    # # 3. Feature Importance 시각화
    # visualize_feature_importance(models, feature_names, top_n=20)
    
    # 4. 분석 결과를 파일로 저장
    # save_model_analysis(models, X_val_processed, y_val, feature_names, preprocessor,
    #                    num_cols, onehot_cols, ordinal_cols, filename='model_analysis.txt')
    
    # 테스트 데이터 예측 및 제출 파일 생성
    X_test_final_processed = preprocessor.transform(df_test)
    
    weights = find_optimal_weights([cb_model, lgb_model, xgb_model], X_val_processed, y_val)
    _, y_pred_ensemble = ensemble_predict(
        [cb_model, lgb_model, xgb_model], 
        X_test_final_processed,
        weights
    )
    
    submission = pd.DataFrame({
        'id': df_sub['id'],
        'loan_paid_back': y_pred_ensemble
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    df_confirm = pd.read_csv('submission_ensemble.csv')

    print(df_confirm.head())