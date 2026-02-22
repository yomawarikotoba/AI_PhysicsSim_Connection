import numpy as np

# ==========================================
# Data Generation & Preprocessing
# ==========================================

def get_rastrigin_data(num_points=300):
    """1D Rastrigin関数データを生成する"""
    x = np.linspace(-4, 4, num_points).reshape(-1, 1)
    y_true = 10 + x**2 - 10 * np.cos(2 * np.pi * x)
    return x, y_true

def normalize_data(x, y_true):
    """データを正規化する"""
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min) * 2 - 1 
    
    y_min, y_max = y_true.min(), y_true.max()
    y_norm = (y_true - y_min) / (y_max - y_min) * 0.8 + 0.1
    
    # 逆正規化に使うための値を返す
    return x_norm, y_norm, (x_min, x_max, y_min, y_max)

def denormalize_data(y_pred_norm, y_min, y_max):
    """予測結果を元のスケールに戻す"""
    return (y_pred_norm - 0.1) / 0.8 * (y_max - y_min) + y_min
