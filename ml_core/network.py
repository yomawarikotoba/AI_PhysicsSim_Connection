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
            # 入力層に近い層を大粒子として専用の学習率を適用する
            if layer.name == "Hidden1":
                layer.weights -= learning_rate * layer.weights_gradient
                layer.biases -= learning_rate * layer.biases_gradient

            # 出力層に近い層を小粒子として専用の学習率を適用する
            if layer.name in ["Hidden2", "Output"]:
                layer.weights -= (learning_rate*1.5) * layer.weights_gradient
                layer.biases -= (learning_rate*1.5) * layer.biases_gradient          
