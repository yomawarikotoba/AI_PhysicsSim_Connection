from ml_core.activations import swish, swish_deriv, identity, identity_deriv
from ml_core.layer import DenseLayer
class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        # ネットワークに層を追加
        self.layers.append(layer)

    def forward(self, inputs):
        # ネットワーク全体の順伝播
        # ここではlayersに追加されたlayerを順にそのforward()を回して全体の順伝播を構成している
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, upstream_gradient):
        # ネットワーク全体の逆伝播
        grad = upstream_gradient
        # reversed()は組み込み関数の一つでリストやarrayを受け取り、その要素を逆順で一つずつ取り出すメソッド
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        # 勾配降下法によるパラメータの更新
        for layer in self.layers:
            # マクロ層：基準の学習率でどっしりと動かす
            if layer.name in ["Macro_Hidden1", "Macro_Hidden2"]:
                layer.weights -= learning_rate * layer.weights_gradient
                layer.biases -= learning_rate * layer.biases_gradient

            # ミクロ層：隙間を素早く埋めるために学習率を少し高める（ここは自分で調整）
            elif layer.name in ["Micro_Hidden1", "Micro_Hidden2"]:
                layer.weights -= (learning_rate*0.2) * layer.weights_gradient
                layer.biases -= (learning_rate*0.2) * layer.biases_gradient

            # 出力層：暴走を抑えるためにマクロと同じか少し小さくする（要調整）
            elif layer.name in ["Macro_Output", "Micro_Output"]:
                layer.weights -= (learning_rate*0.1) * layer.weights_gradient
                layer.biases -= (learning_rate*0.1) * layer.biases_gradient

class BimodalNetwork:
    def __init__(self):
        # 既存のNetworkクラスを使って2つのルートを構築する
        self.macro_net = Network()
        self.macro_net.add(DenseLayer(1, 8, swish, swish_deriv, name="Macro_Hidden1"))
        self.macro_net.add(DenseLayer(8, 8, swish, swish_deriv, name="Macro_Hidden2"))
        self.macro_net.add(DenseLayer(8, 1, swish, swish_deriv, name="Macro_Output"))

        self.micro_net = Network()
        self.micro_net.add(DenseLayer(1, 64, swish, swish_deriv, name="Micro_Hidden1"))
        self.micro_net.add(DenseLayer(64, 64, swish, swish_deriv, name="Micro_Hidden2"))
        self.micro_net.add(DenseLayer(64, 1, swish, swish_deriv, name="Micro_Output"))

    @property
    def layers(self):
        # PhysicsOptimizerから要求されたら、両方の層を足し合わせたリストを返す
        return self.macro_net.layers + self.micro_net.layers
    
    def forward(self, inputs):
        # アウトプットの合算
        return self.macro_net.forward(inputs) + self.micro_net.forward(inputs)
    
    def backward(self, upstream_gradient):
        # 勾配の並列伝播
        self.macro_net.backward(upstream_gradient)
        self.micro_net.backward(upstream_gradient)

    def update(self, learning_rate):
        self.macro_net.update(learning_rate)
        self.micro_net.update(learning_rate)