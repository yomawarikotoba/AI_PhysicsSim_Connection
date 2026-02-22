import numpy as np

# ==========================================
# Physics-based Optimizer
# ==========================================
class PhysicsOptimizer:
    def __init__(self, mode="None", noise_scale=0.01, shock_interval=100, shock_scale=0.5):
        self.mode = mode
        self.noise_scale = noise_scale
        self.shock_interval = shock_interval
        self.shock_scale = shock_scale
    
    def apply_horizontal_vibration(self, network):
        """水平振動: 勾配にノイズを乗せる
           Hybridモードでも有効
        """
        if self.mode not in ["Horizontal", "Hybrid"]: return

        for layer in network.layers:
            # 勾配の大きさに関わらず一定のノイズ
            noise_w = np.random.normal(0, self.noise_scale, layer.weights_gradient.shape)
            noise_b = np.random.normal(0, self.noise_scale, layer.biases_gradient.shape)
            layer.weights_gradient += noise_w
            layer.biases_gradient += noise_b

    def apply_vertical_vibration(self, network, epoch):
        """鉛直振動: 重みに衝撃を与える
           Hybridモードでも有効
        """
        if self.mode not in ["Vertical", "Hybrid"]: return

        if (epoch + 1) % self.shock_interval == 0:
            # print(f"  >>> BOOM! Shock at epoch {epoch+1} (Mode: {self.mode})")
            for layer in network.layers:
                shock_w = np.random.normal(0, self.shock_scale, layer.weights.shape)
                layer.weights += shock_w 
                shock_b = np.random.normal(0, self.shock_scale, layer.biases.shape)
                layer.biases += shock_b
