import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
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

rs = 42
model_types = ["catboost", "lgbm", "xgb"]

def data_preprocessing(df_array, target_col, df_num_cols, df_cat_cols):
    
    def log_regularization(df_array):
        
        for df in df_array:

            df["loan_amount"] = np.log1p(df["loan_amount"])
            # df["interest_rate"] = np.log1p(df["interest_rate"])
            df["annual_income"] = np.log1p(df["annual_income"])    
        
        return df_array
    
    def remove_outliers(df_array, cols):

        for df in df_array:

            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3- Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # lower와 upper를 구하고 사이의 값을 사용함 정상적인 값들만 사용하겠다는 의지
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        return df_array

    def remove_data_identify(df_array):
        
        for df in df_array:
            df.drop(columns=['id'], inplace=True)
        
        return df_array

    def feature_engineering(df_array):
        
        eps = 1e-6
        clip_rules = {
            "annual_income": 0.995,
            "loan_amount": 0.995,
            "debt_to_income_ratio": 0.99,
            "interest_rate": 0.995,
            "credit_score": 0.99
        }

        for df in df_array:
            for col, q in clip_rules.items():
                upper = df[col].quantile(q)

                if col == "credit_score":
                    lower = df[col].quantile(1 - q)
                    df[col] = df[col].clip(lower=lower, upper=upper)
                else:
                    df[col] = df[col].clip(upper=upper)

            df["annual_income"] = df["annual_income"].clip(lower=eps)
            df["loan_amount"]   = df["loan_amount"].clip(lower=eps)
            df["interest_rate"] = df["interest_rate"].clip(lower=eps)
            df["credit_score"]  = df["credit_score"].clip(lower=300)
            # 1. interest_rate / debt_to_income_ratio
            df["dti_to_interest_rate"] = (df["debt_to_income_ratio"] / (df["interest_rate"]+ eps))
            
            # 2. education_level & loan_purpose
            # df["loan_purpose_interest_rate"] = df["loan_purpose"].astype(str) + "_" + np.log1p(df["interest_rate"].round(1)).astype(str)
            
            # 3. employment_status & loan_purpose
            # df['employment_status_grade_subgrade'] = df['employment_status'].astype(str) + '_' + df['grade_subgrade'].astype(str)
            df["employment_loan_purpose"] = df["employment_status"].astype(str) + "_" + df["loan_purpose"].astype(str)
            # df["education_loan_purpose"] = df["education_level"].astype(str) + "_" + df["loan_purpose"].astype(str)

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
            # df["loan_amount_credit"] = df["loan_amount"].astype(float) / (df["credit_score"].astype(float)+ eps)
            # df["loan_amount_div_income"] = df["loan_amount"].astype(int) / (df["annual_income"].astype(float)+ eps)
            # df["loan_amount_div_ratio"] = df["loan_amount"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ eps)
            
            # 7. creadit
            df["credit_div_ratio"] = df["credit_score"].astype(float) / (df["debt_to_income_ratio"].astype(float)+ eps)

            ratio_cols = [
                "dti_to_interest_rate",
                "debt_to_monthly_income",
                "pti_ratio",
                "credit_div_ratio",
            ]

            for col in ratio_cols:
                upper = df[col].quantile(0.995)
                df[col] = df[col].clip(lower=0, upper=upper)

        return df_array

    df_array = remove_data_identify(df_array)
    df_array = log_regularization(df_array)
    df_array = feature_engineering(df_array)
    # df_array = remove_outliers(df_array, df_num_cols)
    return df_array

def encoding_data(df_train, target_col, model_type, X, y,X_train, X_val, y_train, y_val):

    df_num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    education_order = [
        "High School",
        "Other",
        "Bachelor's",
        "Master's",
        "PhD",
    ]

    grade_order = [
        'G5','G4','G3','G2','G1',
        'F5','F4','F3','F2','F1',
        'E5','E4','E3','E2','E1',
        'D5','D4','D3','D2','D1',
        'C5','C4','C3','C2','C1',
        'B5','B4','B3','B2','B1',
        'A5','A4','A3','A2','A1',
    ]

    head_grade_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']

  
    # encoder.fit(X_train, y_train)

    
    # num_transformer = 'passthrough'
    num_transformer = StandardScaler()
    
    if model_type == "catboost":
        cat_cat_transformer = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1)
        cat_cat_cols = [
            'education_level',
            'grade_subgrade',
            'head_grade',
            'gender',
            'marital_status',
            'loan_purpose',
            'employment_status',    
            'employment_loan_purpose',
            # 'employment_status_grade_subgrade',
        ]
    elif model_type == "lgbm":
        lgbm_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        lgbm_one_cols = [
            'gender',
            'marital_status',
            'loan_purpose',
            # 'employment_status',    
            'employment_loan_purpose',
        ]
        lgbm_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', 
          categories=[
                education_order,
                grade_order,
                head_grade_order,
            ],
            unknown_value=-1)
        lgbm_ordinal_cols = [
            'education_level',
            'grade_subgrade',
            'head_grade',
        ]
        lgbm_cb_cols = [
            'employment_status',
            'employment_loan_purpose',
            # 'employment_status_grade_subgrade',
        ]
        lgbm_cb_encoder = CatBoostEncoder(
            return_df=False,
            random_state=rs,
            cols=lgbm_cb_cols,
        )
       
    elif model_type == "xgb":
        xgb_cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        xgb_cat_cols = [
            'gender',
            'marital_status',
            'loan_purpose',
            # 'employment_status',    
            'employment_loan_purpose',
        ]

        xgb_cb_cols = [
            'employment_status',
            'employment_loan_purpose',
            # 'employment_status_grade_subgrade',
        ]
        xgb_cb_encoder = CatBoostEncoder(
            return_df=False,
            random_state=rs,
            cols=xgb_cb_cols,
        )
      

    if model_type == "catboost":
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, df_num_cols),
                ('cat', cat_cat_transformer, cat_cat_cols),
                
            ],
            remainder='drop'
        )
    elif model_type == "lgbm":
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, df_num_cols),
                ('onehot', lgbm_onehot_transformer, lgbm_one_cols),
                ('ordinal', lgbm_ordinal_transformer, lgbm_ordinal_cols),
                ('catboost', lgbm_cb_encoder, lgbm_cb_cols),
            ],
            remainder='drop'  
        )
    elif model_type == "xgb":
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, df_num_cols),
                ('cat', xgb_cat_transformer, xgb_cat_cols),
                ('catboost', xgb_cb_encoder, xgb_cb_cols),
            ],
            remainder='drop'  
        )
   

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)
 
    return [X_train_processed, X_val_processed, y_train, y_val, preprocessor]


def optimize_models(X_train, y_train, model_type, cat_features=None):

    pos = (df_train['loan_paid_back'] == 1).sum()
    neg = (df_train['loan_paid_back'] == 0).sum()
    spw = neg / pos

    def objective(trial):
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
                "objective": "binary",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 300, 900),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": rs,
                "n_jobs": -1,
                "metric": "auc",
            }
            model = lgb.LGBMClassifier(**params)
        
        else:  
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
                'scale_pos_weight': spw,
                # "use_label_encoder": False,
                "eval_metric": "auc",
            }
            model = xgb.XGBClassifier(**params)
        
        # k-fold를 통해 학습할 수 있도록 객체 생성
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)

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

def create_models_with_optuna(X_train, y_train, model_type, use_fixed_params, cat_features=None):
    """Optuna 최적화 또는 고정 파라미터로 모델 생성"""
    pos = (df_train['loan_paid_back'] == 1).sum()
    neg = (df_train['loan_paid_back'] == 0).sum()
    spw = neg / pos

    if use_fixed_params:
        # 고정 파라미터 사용
        if model_type == "catboost":
            fixed_params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "depth": 8,
                "iterations": 477,
                "l2_leaf_reg": 77.30156271338825,
                "learning_rate": 0.08217342500676761,
                "random_strength": 0.6335140795029303,
                "random_seed": rs,
                "verbose": False,
            }
            model = CatBoostClassifier(**fixed_params)
            
        elif model_type == "lgbm":
            fixed_params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "colsample_bytree": 0.7689967650165342,
                "learning_rate": 0.0903227448273408,
                "max_depth": 7,
                "min_child_samples": 105,
                "min_split_gain": 0.8904577433341123,
                "n_estimators": 841,
                "num_leaves": 84,
                "reg_alpha": 0.01946164724781516,
                "reg_lambda": 3.665810648369544,
                "subsample": 0.9127034245997915,
                "subsample_freq": 1,
                "random_state": rs,
                "n_jobs": -1,
                "metric": "auc",
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**fixed_params)
            
        elif model_type == "xgb":
            fixed_params = {
                "colsample_bytree": 0.6059606379994678,
                "gamma": 0.861757820514575,
                "learning_rate": 0.04114729921642164,
                "max_depth": 7,
                "min_child_weight": 8,
                "n_estimators": 639,
                "reg_alpha": 3.568274751351765,
                "reg_lambda": 0.014158740787660969,
                "subsample": 0.8489898860887624,
                "random_state": rs,
                "scale_pos_weight": spw,
                "eval_metric": "auc",
            }
            model = xgb.XGBClassifier(**fixed_params)
            
        print(f"✅ {model_type.upper()} 모델 생성 완료 (고정 파라미터 사용)")
        return model
    
    else:
        try:
            print(f"\n🚀 {model_type.upper()} 최적화 시작...")
            best_params = optimize_models(X_train, y_train, model_type, cat_features=cat_features)
            
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

def find_optimal_weights(models, preprocessor_list):
    """각 모델별 전처리된 validation set을 사용하여 optimal weights 찾기"""
    
    cb_model = models['catboost']
    lgb_model = models['lgbm']
    xgb_model = models['xgb']
    
    # 각 모델의 전처리된 validation set 가져오기
    # preprocessor_list 구조: [X_train, X_val, y_train, y_val, preprocessor]
    X_val_lgbm = preprocessor_list[0][1]      # lgbm용 전처리된 validation set
    X_val_xgb = preprocessor_list[1][1]       # xgb용 전처리된 validation set
    X_val_catboost = preprocessor_list[2][1]  # catboost용 전처리된 validation set
    y_val = preprocessor_list[0][3]           # 모든 모델의 y_val은 동일
    
    # 각 모델에 맞는 전처리된 validation set으로 예측
    cb_pred = cb_model.predict_proba(X_val_catboost)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_val_lgbm)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_val_xgb)[:, 1]
    
    # 각 모델의 개별 성능 확인
    cb_loss = log_loss(y_val, cb_pred)
    lgb_loss = log_loss(y_val, lgb_pred)
    xgb_loss = log_loss(y_val, xgb_pred)
    
    cb_auc = roc_auc_score(y_val, cb_pred)
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    
    print("\n" + "="*80)
    print("📊 각 모델의 개별 성능 (Validation Set)")
    print("="*80)
    print(f"CatBoost  - LogLoss: {cb_loss:.6f}, AUC: {cb_auc:.6f}")
    print(f"LightGBM  - LogLoss: {lgb_loss:.6f}, AUC: {lgb_auc:.6f}")
    print(f"XGBoost   - LogLoss: {xgb_loss:.6f}, AUC: {xgb_auc:.6f}")
    
    # 모델들 간의 예측 상관관계 확인
    preds_dict = {
        'CatBoost': cb_pred,
        'LightGBM': lgb_pred,
        'XGBoost': xgb_pred
    }
    preds_df = pd.DataFrame(preds_dict)
    corr_matrix = preds_df.corr()
    
    print("\n" + "="*80)
    print("📈 모델들 간의 예측 상관관계")
    print("="*80)
    print(corr_matrix.round(4))
    
    preds = np.vstack([cb_pred, lgb_pred, xgb_pred]).T  # shape: (N, 3)

    # 성능 기반 초기 weight 설정 (LogLoss가 낮을수록 높은 weight)
    losses = np.array([cb_loss, lgb_loss, xgb_loss])
    # LogLoss를 역수로 변환하여 weight로 사용 (정규화)
    inv_losses = 1.0 / (losses + 1e-10)  # 0으로 나누기 방지
    init_w = inv_losses / inv_losses.sum()
    
    print(f"\n초기 weight (성능 기반) - CatBoost: {init_w[0]:.4f}, LightGBM: {init_w[1]:.4f}, XGBoost: {init_w[2]:.4f}")

    # 제약조건: w >= 0, sum(w)=1
    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    # 최소 weight 제약 추가 (각 모델이 최소 5% 이상 기여하도록)
    min_w = 0.33
    bounds = [(min_w, 1)] * 3

    # 목적 함수: logloss 최소화
    def loss_fn(w):
        blended = np.dot(preds, w)
        return log_loss(y_val, blended)

    result = minimize(loss_fn, init_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    optimal_w = result.x
    print(f"\n✅ 최적 weight - CatBoost: {optimal_w[0]:.4f}, LightGBM: {optimal_w[1]:.4f}, XGBoost: {optimal_w[2]:.4f}")
    
    # 최적 weight로 앙상블 성능 확인
    ensemble_pred = np.dot(preds, optimal_w)
    ensemble_loss = log_loss(y_val, ensemble_pred)
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    print(f"앙상블 성능 - LogLoss: {ensemble_loss:.6f}, AUC: {ensemble_auc:.6f}")
    print("="*80 + "\n")
    
    return optimal_w

def ensemble_predict(models, preprocessed_X_list, weights):
    """각 모델별 전처리된 데이터를 사용하여 ensemble 예측"""
    cb_model = models['catboost']
    lgb_model = models['lgbm']
    xgb_model = models['xgb']
    
    X_lgbm = preprocessed_X_list[0]      # lgbm용 전처리된 데이터
    X_xgb = preprocessed_X_list[1]       # xgb용 전처리된 데이터
    X_catboost = preprocessed_X_list[2]  # catboost용 전처리된 데이터

    # 각 모델에 맞는 전처리된 데이터로 예측
    # predict_proba가 1차원 배열을 반환할 수 있으므로 안전하게 처리
    def safe_predict_proba(model, X):
        """predict_proba 결과를 안전하게 처리 (1차원 또는 2차원 배열 모두 지원)"""
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            # 1차원 배열인 경우 (CatBoost 등에서 발생 가능)
            return proba
        elif proba.ndim == 2 and proba.shape[1] > 1:
            # 2차원 배열이고 클래스가 여러 개인 경우
            return proba[:, 1]  # 양성 클래스 확률
        else:
            # 2차원이지만 클래스가 1개인 경우
            return proba.flatten()
    
    cb_pred = safe_predict_proba(cb_model, X_catboost)
    lgb_pred = safe_predict_proba(lgb_model, X_lgbm)
    xgb_pred = safe_predict_proba(xgb_model, X_xgb)
    
    preds = np.vstack([cb_pred, lgb_pred, xgb_pred]).T
    ensemble_pred_proba = np.dot(preds, weights)
    ensemble_pred = (ensemble_pred_proba >=0.5).astype(float)
    
    return ensemble_pred, ensemble_pred_proba

def optimize_weights_random_search(models, X_val_list, y_val, n_trials=3000):
    """
    무작위 가중치 조합을 3000번 시도하여 AUC가 가장 높은 조합을 찾습니다.
    X_val_list: 각 모델별로 전처리된 X_val 데이터의 리스트 [X_cat, X_lgbm, X_xgb]
    """
    print(f"\n⚖️ Optimizing Ensemble Weights (Random Search {n_trials} trials)...")
    model_order = ['lgbm', 'xgb', 'catboost']

    # 각 모델의 예측값 미리 계산 (속도 향상)
    preds_list = []
    for model_name in model_order:
        model = models[model_name]
        preds_list.append(model.predict_proba(X_val_list[model_order.index(model_name)])[:, 1])
    
    best_auc = 0
    best_weights = [0.33, 0.33, 0.33]
    
    for _ in range(n_trials):
        # 랜덤 가중치 생성 (Dirichlet 분포 사용 시 합이 1이 됨)
        weights = np.random.dirichlet(np.ones(len(models)), size=1)[0]
        
        # 가중 평균 계산
        final_pred = np.zeros_like(preds_list[0])
        for i, pred in enumerate(preds_list):
            final_pred += pred * weights[i]
            
        score = roc_auc_score(y_val, final_pred)
        if score > best_auc:
            best_auc = score
            best_weights = weights
            
    print(f"✅ Optimal Weights Found: {best_weights}")
    print(f"✅ Best Validation AUC: {best_auc:.5f}")
    return best_weights

# def ensemble_predict(models, X_list, weights):
#     model_order = ['lgbm', 'xgb', 'catboost']
#     final_pred_proba = np.zeros(X_list[0].shape[0])

#     for i, model_name in enumerate(model_order):
#         model = models[model_name]
#         pred = model.predict_proba(X_list[i])[:, 1]
#         final_pred_proba += pred * weights[i]
        
#     final_pred = (final_pred_proba >= 0.3).astype(float)
#     return final_pred, final_pred_proba

def analyze_feature_importance(models, preprocessor_list, top_n=20):
    """
    각 모델별로 feature importance를 분석하고 시각화하는 함수
    
    Parameters:
    -----------
    models : dict
        학습된 모델 딕셔너리 {'lgbm': model, 'xgb': model, 'catboost': model}
    preprocessor_list : list
        각 모델별 전처리기 리스트
    top_n : int
        상위 N개 feature만 표시 (기본값: 20)
    """
    print("\n" + "="*80)
    print("📊 모델별 Feature Importance 분석")
    print("="*80)
    
    model_types = ["lgbm", "xgb", "catboost"]
    all_importances = {}
    
    for idx, model_name in enumerate(model_types):
        model = models[model_name]
        preprocessor = preprocessor_list[idx][4]  # preprocessor 객체
        feature_names = preprocessor.get_feature_names_out()
        
        # 모델별로 feature importance 추출
        if model_name == "catboost":
            importances = model.get_feature_importance()
        elif model_name == "lgbm":
            importances = model.feature_importances_
        elif model_name == "xgb":
            importances = model.feature_importances_
        else:
            importances = None
        
        if importances is not None:
            # DataFrame으로 변환
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            all_importances[model_name] = importance_df
            
            # 상위 N개 출력
            print(f"\n🔹 {model_name.upper()} - Top {top_n} Features:")
            print("-" * 80)
            for i, row in importance_df.head(top_n).iterrows():
                print(f"  {row['feature']:50s} : {row['importance']:10.4f}")
    
    # 시각화
    visualize_feature_importance(all_importances, top_n=top_n)
    
    # 공통 중요 feature 찾기
    find_common_important_features(all_importances, top_n=top_n)
    
    return all_importances

def visualize_feature_importance(all_importances, top_n=20, figsize=(15, 12)):
    """
    모델별 feature importance를 시각화
    """
    n_models = len(all_importances)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, importance_df) in enumerate(all_importances.items()):
        top_features = importance_df.head(top_n)
        
        # 수평 막대 그래프
        axes[idx].barh(range(len(top_features)), top_features['importance'].values, color='crimson')
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['feature'].values)
        axes[idx].set_xlabel('Feature Importance', fontsize=12)
        axes[idx].set_title(f'{model_name.upper()} - Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, v in enumerate(top_features['importance'].values):
            axes[idx].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Feature importance 시각화가 'feature_importance.png'로 저장되었습니다.")

def find_common_important_features(all_importances, top_n=20):
    """
    모든 모델에서 공통으로 중요하게 판단하는 feature 찾기
    """
    print("\n" + "="*80)
    print(f"🔍 공통 중요 Feature (모든 모델에서 Top {top_n}에 포함)")
    print("="*80)
    
    # 각 모델의 top N feature 집합
    top_features_sets = {}
    for model_name, importance_df in all_importances.items():
        top_features = set(importance_df.head(top_n)['feature'].values)
        top_features_sets[model_name] = top_features
    
    # 교집합 찾기
    common_features = set.intersection(*top_features_sets.values())
    
    if common_features:
        print(f"\n총 {len(common_features)}개의 공통 중요 feature 발견:")
        print("-" * 80)
        
        # 중요도 평균 계산
        common_importance = {}
        for feature in common_features:
            avg_importance = np.mean([
                all_importances[model_name][
                    all_importances[model_name]['feature'] == feature
                ]['importance'].values[0]
                for model_name in all_importances.keys()
            ])
            common_importance[feature] = avg_importance
        
        # 중요도 순으로 정렬
        sorted_common = sorted(common_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, avg_imp) in enumerate(sorted_common, 1):
            print(f"  {i:2d}. {feature:50s} : 평균 중요도 {avg_imp:10.4f}")
    else:
        print("\n공통 중요 feature가 없습니다.")
    
    # 모델별 비교 테이블
    print("\n" + "="*80)
    print("📈 모델별 Feature Importance 비교 (공통 feature)")
    print("="*80)
    
    if common_features:
        comparison_data = []
        for feature in sorted_common[:10]:  # 상위 10개만
            row = {'feature': feature[0]}
            for model_name in all_importances.keys():
                imp = all_importances[model_name][
                    all_importances[model_name]['feature'] == feature[0]
                ]['importance'].values[0]
                row[model_name] = imp
            row['average'] = feature[1]
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))


def main(df_array, target_col, num_cols, cat_cols):
    
    df_train = df_array[0]
    df_test = df_array[1]
    
    model_types = ["lgbm", "xgb", "catboost"]

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    preprocessor_list = []
    for idx, model_type in enumerate(model_types):
        preprocessor_list.append(encoding_data(df_train, target_col, model_types[idx], X, y, X_train, X_val, y_train, y_val))

    model_result = []

    for idx, preprocessor in enumerate(preprocessor_list):
        
        X_train_processed = preprocessor[0]
        X_val_processed = preprocessor[1]
        y_train = preprocessor[2]
        y_val = preprocessor[3]
        preprocessor = preprocessor[4]
        
        print("🔍 Optimizing models with Optuna...")
        
        # if idx == 0:
        #     processed_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[idx], use_fixed_params=False)
        # else:
        processed_model = create_models_with_optuna(X_train_processed, y_train, model_type=model_types[idx], use_fixed_params=True)

        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        
        processed_model.fit(X_train_df, y_train)

        model_result.append(processed_model)

    # model_result.append(cb_model_trained)

    models = {
        'lgbm': model_result[0],
        'xgb': model_result[1],
        'catboost': model_result[2],
    }
    # preprocessor_list.append([X_train, X_val, y_train, y_val, cb_model])

    
    for idx, (name, model) in enumerate(models.items()):
        X_val_data = preprocessor_list[idx][1]  # validation set
        y_val_data = preprocessor_list[idx][3]  # validation labels
        
        try:
            pred = model.predict(X_val_data)
            proba = model.predict_proba(X_val_data)[:, 1]
            
            acc = accuracy_score(y_val_data, pred)
            auc = roc_auc_score(y_val_data, proba)

            print(f"{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        except Exception as e:
            print(f"{name} - Error: {e}")
            import traceback
            traceback.print_exc()
    best_weights = [1/3, 1/3, 1/3]
    try:
        # 각 모델의 전처리된 validation set 가져오기
        X_val_list = [preprocessor_list[i][1] for i in range(3)] 
        y_val = preprocessor_list[0][3]  # 모든 모델의 y_val은 동일
        
        # model_result 대신 models 딕셔너리 사용
        best_weights = find_optimal_weights(models, preprocessor_list)
        # best_weights = optimize_weights_random_search(models, X_val_list, y_val, n_trials=5000)
        ensemble_pred, ensemble_pred_proba = ensemble_predict(models, X_val_list, best_weights)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred_proba)
        print(f"Ensemble - Accuracy: {ensemble_acc:.4f}, AUC: {ensemble_auc:.4f}")
    except Exception as e:
        print(f"Ensemble Error: {e}")
    
    # Feature Importance 분석 추가
    try:
        feature_importances = analyze_feature_importance(models, preprocessor_list, top_n=20)
    except Exception as e:
        print(f"Feature Importance 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    return preprocessor_list, models, df_array, best_weights

if __name__ == "__main__":

    df_train = df_train
    df_test = df_test
    df_sub = df_sub
    
    # eda(df_train, df_test, df_sub)

    df_num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    target_col = 'loan_paid_back'
    df_array = [df_train, df_test]
    df_array = data_preprocessing(df_array, target_col, df_num_cols, df_cat_cols )
    

    df_train.head()
    df_test.head()
    
    preprocessor_list, models, df_array, best_weights = main(df_array, target_col, df_num_cols, df_cat_cols)
    
    print("\n" + "="*80)
    print("🔍 모델 상세 분석 시작")
    print("="*80)
     
    # 테스트 데이터 예측 및 제출 파일 생성
    X_test_processed_list = []
    
    for i in range(len(preprocessor_list)):
        preprocessor = preprocessor_list[i][4]  # preprocessor 객체
        X_test_processed = preprocessor.transform(df_test)
        X_test_processed_list.append(X_test_processed)
    
    # X_test_processed_list.append(df_array[1])
    
    # 2. Validation set 가져오기 (weights 계산용)
    X_val_list = [preprocessor_list[i][1] for i in range(3)]  # [catboost, lgbm, xgb]
    y_val = preprocessor_list[0][3]  # 모든 모델의 y_val은 동일
    # best_weights = optimize_weights_random_search(models, X_val_list, n_trials=5000)
    # 4. Ensemble 예측
    _, y_pred_ensemble = ensemble_predict(
        models, 
        X_test_processed_list,  # 각 모델별 전처리된 테스트 데이터 리스트
        best_weights
    )

    # 5. 제출 파일 생성
    submission = pd.DataFrame({
        'id': df_sub['id'],
        'loan_paid_back': y_pred_ensemble
    })

    submission.to_csv('submission_ensemble.csv', index=False)
    df_confirm = pd.read_csv('submission_ensemble.csv')

    print(df_confirm.head())