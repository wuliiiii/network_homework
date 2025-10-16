import numpy as np

class module:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def update(self, lr):
        pass


class tensor(module):
    def __init__(self, data, requires_grad=False):
        super().__init__()
        self.data = np.array(data)  # 转换为numpy数组
        self.grad = None  # 存储梯度
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self._prev = set()  # 记录计算图中的前驱节点
        self._backward_func = None  # 反向传播函数

    def __add__(self, other):
        """逐元素相加，支持广播机制"""
        if not isinstance(other, tensor):
            other = tensor(other, requires_grad=False)
            
        # 检查形状是否兼容（支持广播）
        try:
            np.broadcast_shapes(self.data.shape, other.data.shape)
        except ValueError as e:
            raise ValueError(f"相加操作形状不兼容: {self.data.shape} 和 {other.data.shape}") from e
            
        # 前向计算：逐元素相加
        out_data = self.data + other.data
        out = tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # 记录计算图关系
        out._prev.add(self)
        out._prev.add(other)
        
        # 保存原始形状用于反向传播
        self_shape = self.data.shape
        other_shape = other.data.shape
        
        # 反向传播函数：加法的梯度逐个传递
        def _backward():
            # 计算self的梯度（逐元素传递，需要广播调整形状）
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # 对输出梯度求和以匹配原始形状（解决维度不匹配问题）
                grad = self._reduce_grad(out.grad, self_shape)
                self.grad += grad
            
            # 计算other的梯度（逐元素传递）
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # 对输出梯度求和以匹配原始形状
                grad = self._reduce_grad(out.grad, other_shape)
                other.grad += grad
        
        out._backward_func = _backward
        return out

    def __mul__(self, other):
        """逐元素相乘（哈达玛积），支持广播机制"""
        if not isinstance(other, tensor):
            other = tensor(other, requires_grad=False)
            
        # 检查形状是否兼容
        try:
            np.broadcast_shapes(self.data.shape, other.data.shape)
        except ValueError as e:
            raise ValueError(f"相乘操作形状不兼容: {self.data.shape} 和 {other.data.shape}") from e
            
        # 前向计算：逐元素相乘
        out_data = self.data * other.data
        out = tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # 记录计算图关系
        out._prev.add(self)
        out._prev.add(other)
        
        # 保存原始形状用于反向传播
        self_shape = self.data.shape
        other_shape = other.data.shape
        
        # 反向传播函数：乘法的梯度是逐元素相乘
        def _backward():
            # 计算self的梯度：other.data逐元素乘以out.grad
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # 先逐元素相乘，再调整形状
                grad = other.data * out.grad
                grad = self._reduce_grad(grad, self_shape)
                self.grad += grad
            
            # 计算other的梯度：self.data逐元素乘以out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # 先逐元素相乘，再调整形状
                grad = self.data * out.grad
                grad = self._reduce_grad(grad, other_shape)
                other.grad += grad
        
        out._backward_func = _backward
        return out

    def _reduce_grad(self, grad, target_shape):
        """
        将梯度调整为目标形状，通过对额外维度求和实现
        这是处理广播反向传播的关键逻辑
        """
        # 计算需要缩减的维度
        grad_shape = grad.shape
        reduce_dims = []
        
        # 从后往前比较维度
        for i in range(1, max(len(grad_shape), len(target_shape)) + 1):
            grad_dim = grad_shape[-i] if i <= len(grad_shape) else 1
            target_dim = target_shape[-i] if i <= len(target_shape) else 1
            
            if grad_dim != target_dim and target_dim == 1:
                reduce_dims.append(-i)
        
        # 对需要缩减的维度求和
        if reduce_dims:
            grad = np.sum(grad, axis=tuple(reduce_dims), keepdims=True)
        
        # 去除大小为1的维度以匹配目标形状
        grad = np.squeeze(grad)
        
        # 如果目标形状是标量，确保梯度也是标量
        if target_shape == ():
            return np.array(grad.item())
        
        return grad

    def backward(self, grad_output=None):
        """反向传播，计算梯度"""
        if grad_output is None:
            # 默认梯度为1（适用于标量输出）
            grad_output = np.ones_like(self.data)
        else:
            # 确保传入的梯度形状与数据形状兼容
            grad_output = np.array(grad_output)
            try:
                # 先尝试广播
                grad_output = np.broadcast_to(grad_output, self.data.shape)
            except ValueError as e:
                raise ValueError(f"梯度形状不兼容: {grad_output.shape} 无法广播到 {self.data.shape}") from e
        
        self.grad = grad_output
        
        # 拓扑排序构建计算图
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)
        
        build_topo(self)
        
        # 反向传播计算梯度
        for node in reversed(topo):
            if node._backward_func:
                node._backward_func()

    def __repr__(self):
        return f"tensor(data=\n{self.data},\nrequires_grad={self.requires_grad},\ngrad={self.grad})"




class Linear(module):
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化全连接层，支持批量处理
        :param in_features: 输入特征数量
        :param out_features: 输出特征数量
        :param bias: 是否使用偏置项
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重 (Xavier初始化)
        scale = np.sqrt(1.0 / in_features)
        self.weight = np.random.randn(in_features, out_features) * scale  # 形状: (in_features, out_features)
        
        # 初始化偏置
        self.bias = np.random.randn(1, out_features) * scale if bias else None  # 形状: (1, out_features)
        
        # 存储梯度
        self.grad_weight = None  # 形状: (in_features, out_features)
        self.grad_bias = None    # 形状: (1, out_features)
        
        # 存储前向传播的输入，用于反向传播
        self.x = None  # 形状: (batch_size, in_features)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, in_features)
        :return: 输出数据，形状为 (batch_size, out_features)
        """
        self.x = x  # 保存输入用于反向传播
        
        # 矩阵乘法: (batch_size, in_features) × (in_features, out_features) = (batch_size, out_features)
        output = np.dot(x, self.weight)
        
        # 如果使用偏置，添加偏置 (利用广播机制)
        if self.bias is not None:
            output += self.bias
        
        return output

    def backward(self, grad_output):
        """
        反向传播
        :param grad_output: 来自下一层的梯度，形状为 (batch_size, out_features)
        :return: 传递给上一层的梯度，形状为 (batch_size, in_features)
        """
        batch_size = self.x.shape[0]
        
        # 计算权重的梯度: (in_features, batch_size) × (batch_size, out_features) = (in_features, out_features)
        # 除以批量大小取平均
        self.grad_weight = np.dot(self.x.T, grad_output) / batch_size
        
        # 计算偏置的梯度: 对批量维度求平均
        if self.bias is not None:
            self.grad_bias = np.mean(grad_output, axis=0, keepdims=True)
        
        # 计算传递给上一层的梯度: (batch_size, out_features) × (out_features, in_features) = (batch_size, in_features)
        grad_input = np.dot(grad_output, self.weight.T)
        
        return grad_input

    def update(self, lr):
        """
        使用梯度下降更新参数
        :param lr: 学习率
        """
        # 更新权重
        self.weight -= lr * self.grad_weight
        
        # 更新偏置
        if self.bias is not None:
            self.bias -= lr * self.grad_bias
    


