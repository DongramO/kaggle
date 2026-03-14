"""
실행 이력 관리: AUC 기록, Feature Importance 누적 저장 및 시각화.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).parent
FI_HISTORY_PATH = BASE_DIR / "fi_history.json"
AUC_LOG_PATH = BASE_DIR / "auc_history.json"


def save_auc_log(auc_dict: dict, run_type: str) -> None:
    """모델별 OOF AUC를 타임스탬프와 함께 누적 저장.
    auc_dict: {"catboost": 0.914, "ensemble": 0.916, ...}
    run_type: "ensemble:catboost+lightgbm" | "single:xgboost" 등
    """
    log = []
    if AUC_LOG_PATH.exists():
        with open(AUC_LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_type": run_type,
        "auc": {k: round(v, 6) for k, v in auc_dict.items()},
        "lb_score": None,
    }
    log.append(entry)

    with open(AUC_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n[AUC 기록]")
    for k, v in auc_dict.items():
        print(f"  {k:15s}: {v:.5f}")
    print(f"  → {AUC_LOG_PATH.name} 저장 (lb_score는 제출 후 직접 입력)")


def save_fi_history(fi_accumulator: dict) -> None:
    """fi_accumulator를 타임스탬프와 함께 히스토리 파일에 누적 저장."""
    history = []
    if FI_HISTORY_PATH.exists():
        with open(FI_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    for model_type, fi_df in fi_accumulator.items():
        entry[model_type] = dict(zip(fi_df["feature"], fi_df["importance"].round(6)))

    history.append(entry)
    with open(FI_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"      fi_history saved (total {len(history)} runs): {FI_HISTORY_PATH.name}")


def plot_fi_history(top_n: int = 15, save_path: str = "fi_history.png") -> None:
    """히스토리 파일을 읽어 모델별 top feature의 importance 변화를 시각화."""
    if not FI_HISTORY_PATH.exists():
        print(f"히스토리 파일 없음: {FI_HISTORY_PATH.name}")
        return

    with open(FI_HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    if len(history) < 2:
        print("시각화를 위해 최소 2회 이상 실행이 필요합니다.")
        return

    timestamps = [e["timestamp"] for e in history]
    model_types = [k for k in history[0] if k != "timestamp"]

    fig, axes = plt.subplots(1, len(model_types), figsize=(10 * len(model_types), 8))
    if len(model_types) == 1:
        axes = [axes]

    for ax, model_type in zip(axes, model_types):
        last_fi = history[-1].get(model_type, {})
        top_features = sorted(last_fi, key=last_fi.get, reverse=True)[:top_n]

        for feat in top_features:
            values = [e.get(model_type, {}).get(feat, 0) for e in history]
            ax.plot(range(len(history)), values, marker="o", label=feat)

        ax.set_xticks(range(len(history)))
        ax.set_xticklabels([t[5:16] for t in timestamps], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{model_type} — Top {top_n} Feature Importance 변화")
        ax.set_xlabel("Run")
        ax.set_ylabel("Importance")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      fi_history plot saved: {save_path}")
