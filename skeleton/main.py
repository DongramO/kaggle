"""
skeleton 메인: data_loader → prepare_data → eda → modeling 순서로 각 모듈 참조.
"""

from pathlib import Path

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns
from eda import run_eda, summary, missing_report
from modeling import train_model, predict, evaluate


def main(data_dir: str | Path = "data", target_col: str | None = None, id_col: str | None = "id"):
    """전체 파이프라인: 로드 → 전처리 → EDA → 모델링."""
    data_dir = Path(data_dir)

    # 1. data_loader
    train_df, test_df, submission_df = load_all(data_dir)

    # 2. prepare_data
    train_prepared = prepare_data(train_df, target_col=target_col)
    test_prepared = prepare_data(test_df, target_col=target_col)
    feature_cols = get_feature_columns(train_prepared, target_col=target_col, id_col=id_col)

    # 3. eda (선택)
    eda_result = run_eda(train_prepared, target_col=target_col)

    # 4. modeling (모델 주입 시 학습/예측/평가)
    X = train_prepared[feature_cols]
    y = train_prepared[target_col]

    model = train_model(X, y, model=None)

    y_pred = predict(model, X)
    score = evaluate(y, y_pred)
    print(f"Train score: {score}")

    X_test = test_prepared[feature_cols]
    test_pred = predict(model, X_test)
    submission_df["target"] = test_pred
    submission_df.to_csv("submission.csv", index=False)

    return train_prepared, test_prepared, eda_result


if __name__ == "__main__":
    main(data_dir="data", target_col="target", id_col="id")
