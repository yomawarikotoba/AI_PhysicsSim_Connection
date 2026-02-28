import numpy as np
from data_utils import get_rastrigin_data, normalize_data
from training import train_experiment
from visualization import plot_results

# ==========================================
# Main Experiment Execution
# ==========================================
if __name__ == "__main__":
    # 乱数シード固定
    np.random.seed(1)

    # 1. データの準備
    x_raw, y_raw = get_rastrigin_data(num_points=300)
    x_norm, y_norm, norm_params = normalize_data(x_raw, y_raw)
    
    # 2. 実験の実行
    # Hybridを追加
    modes = ["None", "Horizontal", "Vertical", "Hybrid"]
    results = {}
    
    for m in modes:
        net, losses = train_experiment(m, x_norm, y_norm, epochs=15000, lr=0.5)
        results[m] = {"net": net, "loss": losses}

    # 3. 結果の可視化
    for layer in net.layers:
        # layer.biases.size でバイアスの合計数（＝ニューロン数）を取得する
        print(f"Layer_Name：{layer.name}, Num_Newrons：{layer.biases.size}")
    plot_results(x_raw, y_raw, results, norm_params)
