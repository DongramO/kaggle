"""
유틸리티 헬퍼 함수들
"""
import subprocess
from typing import Optional


def check_gpu_availability() -> bool:
    """GPU 사용 가능 여부 확인"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def get_gpu_device() -> Optional[str]:
    """사용 가능한 GPU 디바이스 번호 반환 ('0', '1', ... 또는 None)"""
    if check_gpu_availability():
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return '0'
        except:
            pass
    return None


def setup_gpu_params(use_gpu: bool, model_type: str) -> dict:
    """
    모델별 GPU 파라미터 설정
    
    Parameters:
    -----------
    use_gpu : bool
        GPU 사용 여부
    model_type : str
        모델 타입 ('catboost', 'lightgbm', 'xgboost')
    
    Returns:
    --------
    dict: GPU 파라미터 딕셔너리
    """
    if not use_gpu:
        if model_type == 'xgboost':
            return {'tree_method': 'hist'}
        return {}
    
    gpu_device = get_gpu_device()
    if gpu_device is None:
        if model_type == 'xgboost':
            return {'tree_method': 'hist'}
        return {}
    
    if model_type == 'catboost':
        return {'task_type': 'GPU', 'devices': gpu_device}
    elif model_type == 'lightgbm':
        return {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': int(gpu_device)}
    elif model_type == 'xgboost':
        return {'tree_method': 'hist', 'device': f'cuda:{gpu_device}'}
    return {}
