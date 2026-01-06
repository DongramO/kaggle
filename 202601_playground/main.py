"""
메인 실행 파일
모델 학습 및 앙상블을 실행합니다.
"""
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from modeling.train import main

if __name__ == "__main__":
    print("="*60)
    print("🎯 Kaggle Competition - 모델 학습 및 앙상블")
    print("="*60)
    
    # GPU 사용 여부 설정 (True로 설정하면 GPU 사용)
    USE_GPU = True  # GPU를 사용하려면 True로 변경
    
    # 하이퍼파라미터 저장 경로 설정
    # 기본값: 프로젝트 루트의 modeling 디렉토리
    PARAMS_FILEPATH = os.path.join(
        project_root,
        'modeling',
        'best_hyperparameters.json'
    )
    
    results, ensemble_pred, submission = main(
        use_gpu=USE_GPU,
        params_filepath=PARAMS_FILEPATH
    )
    
    print("\n" + "="*60)
    print("✅ 모든 작업이 완료되었습니다!")
    print("="*60)

