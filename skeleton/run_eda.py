"""
EDA 독립 실행 진입점.
- main.py와 분리하여 EDA만 별도 실행 가능.
- python run_eda.py 로 실행.
"""

from pathlib import Path

from data_loader import load_all
from eda import run_eda, summary, missing_report


def main(
    data_dir: str | Path = "data",
    target_col: str | None = "target",
):
    """EDA만 실행: 로드 → EDA."""
    data_dir = Path(data_dir)

    train_df, _, _ = load_all(data_dir)

    eda_result = run_eda(train_df, target_col=target_col)

    print("=== EDA 요약 ===")
    print(f"Shape: {eda_result['shape']}")
    print("\n--- 결측 리포트 ---")
    print(eda_result["missing"])
    if "target_distribution" in eda_result:
        print("\n--- 타겟 분포 ---")
        print(eda_result["target_distribution"])

    return eda_result


if __name__ == "__main__":
    # ========== 문제별 수정 영역 ==========
    TARGET_COL = "target"
    # ======================================
    main(data_dir="data", target_col=TARGET_COL)
