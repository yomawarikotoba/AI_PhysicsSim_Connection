import matplotlib.pyplot as plt

# ==========================================
# Result Visualization
# ==========================================

# 出力する画像ファイルの名前はここで編集
outputfile = "result_v4_rastrigin_bimodal_simaltanius_学習率ノイズ小粒子小さく.png"

def plot_results(x, y_true, results, normalization_params):
    """フィッティング結果とLossの推移をプロットする"""
    
    _, _, y_min, y_max = normalization_params
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 色設定: Hybridは緑色(green)
    colors = {"None": "black", "Horizontal": "red", "Vertical": "blue", "Hybrid": "green"}

    # 1. フィッティング結果
    ax1 = axes[0]
    ax1.scatter(x, y_true, color="gray", alpha=0.3, label="True Data (Rastrigin)")
    
    # 正規化されたxを再計算
    x_norm = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    
    for m, data in results.items():
        net = data["net"]
        y_pred_norm = net.forward(x_norm)
        
        # スケール戻し
        y_pred = (y_pred_norm - 0.1) / 0.8 * (y_max - y_min) + y_min
        
        final_loss = data["loss"][-1]
        ax1.plot(x, y_pred, label=f"{m} (Loss={final_loss:.5f})", 
                 color=colors.get(m, "purple"), linewidth=2, alpha=0.8)
        
    ax1.set_title("Approximating Rastrigin Function")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Loss推移
    ax2 = axes[1]
    for m, data in results.items():
        ax2.plot(data["loss"], label=m, color=colors.get(m, "purple"), alpha=0.6, linewidth=1)
    
    ax2.set_yscale("log")
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (Log Scale)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(outputfile)
    print(outputfile)
