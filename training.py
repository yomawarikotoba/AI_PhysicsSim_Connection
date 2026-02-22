import numpy as np
import copy

from ml_core.network import Network
from ml_core.layer import DenseLayer
from ml_core.loss import MSELoss
from ml_core.activations import swish, swish_deriv, identity, identity_deriv
from physics_sim.vibration import PhysicsOptimizer

# ==========================================
# Training Function
# ==========================================
def train_experiment(mode_name, x, y, epochs=10000, batch_size=32, lr=0.005):
    print(f"Training Mode: {mode_name}...")
    
    # ネットワーク構築 (1->64->64->1)
    net = Network()
    net.add(DenseLayer(1, 64, swish, swish_deriv, name="Hidden1")) 
    net.add(DenseLayer(64, 64, swish, swish_deriv, name="Hidden2")) 
    net.add(DenseLayer(64, 1, identity, identity_deriv, name="Output"))
    
    loss_func = MSELoss()
    
    # 物理オプティマイザの設定
    optimizer = PhysicsOptimizer(
        mode=mode_name, 
        noise_scale=0.03,    # Horizontal/Hybrid用
        shock_interval=2000, # Vertical/Hybrid用
        shock_scale=0.3      # Vertical/Hybrid用
    )
    
    loss_history = []
    num_samples = x.shape[0]
    
    # ベストな状態を保存するための変数
    best_loss = float('inf')
    best_net_state = None

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            x_batch = x[indices[i : i + batch_size]]
            y_batch = y[indices[i : i + batch_size]]
            
            y_pred = net.forward(x_batch)
            loss = loss_func.forward(y_pred, y_batch)
            epoch_loss += loss
            
            grad = loss_func.backward(y_pred, y_batch)
            net.backward(grad)
            
            # 水平振動の適用
            optimizer.apply_horizontal_vibration(net)
            net.update(lr)
        
        avg_loss = epoch_loss / (num_samples / batch_size)
        loss_history.append(avg_loss)

        # VerticalまたはHybridモードの場合、ベストなモデルを保存
        if mode_name in ["Vertical", "Hybrid"]:
            is_shock_time = ((epoch + 1) % optimizer.shock_interval == 0)
            is_final = (epoch == epochs - 1)
            
            if is_shock_time or is_final:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_net_state = copy.deepcopy(net.layers)
        
        # 鉛直振動の適用
        optimizer.apply_vertical_vibration(net, epoch)

    # VerticalまたはHybridモードの場合、最終的にベストなモデルを復元
    if mode_name in ["Vertical", "Hybrid"] and best_net_state is not None:
        print(f"  >>> Restoring Best Model (Loss: {best_loss:.5f})")
        net.layers = best_net_state
        loss_history[-1] = best_loss

    return net, loss_history
