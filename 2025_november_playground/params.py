[catboost]
depth                    : 7
iterations               : 475
l2_leaf_reg              : 15.332725842146589
learning_rate            : 0.09983620901362981
random_strength          : 0.7485180520264623    
    
[LGBM]
colsample_bytree         : 0.5295722198581925
learning_rate            : 0.04919972117020984
max_depth                : 8
n_estimators             : 576
num_leaves               : 93
reg_alpha                : 9.320749906598678
reg_lambda               : 0.045393254289234894
subsample                : 0.8368214696612971

[XGB]
colsample_bytree         : 0.9738845151688945
  gamma                    : 0.6097633807345567
  learning_rate            : 0.03398329815843991
  max_depth                : 7
  min_child_weight         : 5
  n_estimators             : 690
  reg_alpha                : 4.931603458294555
  reg_lambda               : 0.14289737491259996
  subsample                : 0.7574655551921472
  
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