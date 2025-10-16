import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import seaborn as sns
import time

# 首先定义tensor类
class tensor:
    def __init__(self, data, requires_grad=False):
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

    def __matmul__(self, other):
        """矩阵乘法，使用@运算符，支持二维张量乘法"""
        if not isinstance(other, tensor):
            other = tensor(other, requires_grad=False)
            
        # 检查矩阵乘法的形状兼容性
        if len(self.data.shape) != 2 or len(other.data.shape) != 2:
            raise ValueError("矩阵乘法仅支持二维张量")
            
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(f"矩阵乘法形状不兼容: {self.data.shape} 和 {other.data.shape}")
            
        # 前向计算：矩阵乘法
        out_data = self.data @ other.data  # 使用numpy的矩阵乘法
        out = tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # 记录计算图关系
        out._prev.add(self)
        out._prev.add(other)
        
        # 保存原始张量用于反向传播
        self_data = self.data.copy()
        other_data = other.data.copy()
        
        # 反向传播函数：矩阵乘法的梯度计算
        def _backward():
            # 计算self的梯度：out.grad @ other.T
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # 矩阵乘法梯度公式：dL/dA = (dL/dC) @ B^T
                grad = out.grad @ other_data.T
                self.grad += grad
            
            # 计算other的梯度：self.T @ out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # 矩阵乘法梯度公式：dL/dB = A^T @ (dL/dC)
                grad = self_data.T @ out.grad
                other.grad += grad
        
        out._backward_func = _backward
        return out

    def __sub__(self, other):
        """减法操作"""
        return self + (-other)

    def __neg__(self):
        """取负操作"""
        return self * (-1)

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


# 定义激活函数
class Sigmoid:
    def forward(self, x):
        """前向传播：sigmoid(x) = 1 / (1 + exp(-x))"""
        if not isinstance(x, tensor):
            x = tensor(x)
            
        self.out = tensor(1.0 / (1.0 + np.exp(-x.data)), requires_grad=x.requires_grad)
        self.out._prev.add(x)
        
        # 反向传播函数
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # sigmoid导数: sigmoid(x) * (1 - sigmoid(x))
                grad = self.out.data * (1 - self.out.data) * self.out.grad
                x.grad += grad
        
        self.out._backward_func = _backward
        return self.out


class ReLU:
    def forward(self, x):
        """前向传播：ReLU(x) = max(0, x)"""
        if not isinstance(x, tensor):
            x = tensor(x)
            
        self.out = tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        self.out._prev.add(x)
        self.x_data = x.data  # 保存输入数据用于反向传播
        
        # 反向传播函数
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # ReLU导数: 1 if x > 0 else 0
                grad = (self.x_data > 0) * self.out.grad
                x.grad += grad
        
        self.out._backward_func = _backward
        return self.out


class Tanh:
    def forward(self, x):
        """前向传播：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        if not isinstance(x, tensor):
            x = tensor(x)
            
        self.out = tensor(np.tanh(x.data), requires_grad=x.requires_grad)
        self.out._prev.add(x)
        
        # 反向传播函数
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # tanh导数: 1 - tanh(x)^2
                grad = (1 - self.out.data**2) * self.out.grad
                x.grad += grad
        
        self.out._backward_func = _backward
        return self.out


class Softmax:
    def forward(self, x):
        """前向传播：softmax(x) = exp(x) / sum(exp(x))"""
        if not isinstance(x, tensor):
            x = tensor(x)
            
        # 数值稳定版softmax
        exp_vals = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        self.out = tensor(exp_vals / np.sum(exp_vals, axis=-1, keepdims=True), requires_grad=x.requires_grad)
        self.out._prev.add(x)
        self.x_data = x.data  # 保存输入数据用于反向传播
        
        # 反向传播函数
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                
                batch_size, num_classes = self.out.data.shape
                # softmax导数: softmax(x) * (I - softmax(x)^T)
                for i in range(batch_size):
                    s = self.out.data[i].reshape(-1, 1)
                    jacobian = np.diagflat(s) - np.dot(s, s.T)
                    x.grad[i] += np.dot(self.out.grad[i], jacobian)
        
        self.out._backward_func = _backward
        return self.out


# 定义线性层
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        """初始化全连接层"""
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重 (Xavier初始化)
        scale = np.sqrt(1.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = tensor(weight_data, requires_grad=True)  # 形状: (in_features, out_features)
        
        # 初始化偏置
        if bias:
            bias_data = np.random.randn(1, out_features) * scale
            self.bias = tensor(bias_data, requires_grad=True)  # 形状: (1, out_features)
        else:
            self.bias = None

    def forward(self, x):
        """前向传播"""
        # 确保输入是tensor类型
        if not isinstance(x, tensor):
            x = tensor(x, requires_grad=False)
        
        # 矩阵乘法: (batch_size, in_features) @ (in_features, out_features) = (batch_size, out_features)
        output = x @ self.weight
        
        # 如果使用偏置，添加偏置 (利用广播机制)
        if self.bias is not None:
            output = output + self.bias
        
        return output

    def parameters(self):
        """返回层的参数列表"""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


# 定义损失函数
class MSELoss:
    def forward(self, pred, target):
        """前向传播：计算均方误差损失"""
        if not isinstance(pred, tensor):
            pred = tensor(pred)
        if not isinstance(target, tensor):
            target = tensor(target)
            
        self.pred = pred
        self.target = target
        
        # 计算MSE损失: 1/N * sum((pred - target)^2)
        diff = pred - target
        loss_data = np.mean(diff.data ** 2)
        loss = tensor(loss_data, requires_grad=pred.requires_grad)
        loss._prev.add(pred)
        
        # 反向传播函数
        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = np.zeros_like(pred.data)
                # MSE导数: 2*(pred - target)/N
                grad = 2 * diff.data / pred.data.size
                pred.grad += grad * loss.grad  # loss.grad通常为1.0
        
        loss._backward_func = _backward
        return loss


class CrossEntropyLoss:
    def forward(self, pred, target):
        """前向传播：计算交叉熵损失"""
        if not isinstance(pred, tensor):
            pred = tensor(pred)
        if not isinstance(target, tensor):
            target = tensor(target)
            
        self.pred = pred
        self.target = target
        
        # 防止log(0)的数值稳定性处理
        eps = 1e-10
        pred_data = np.clip(pred.data, eps, 1 - eps)
        
        # 计算交叉熵损失: -sum(target * log(pred)) / N
        loss_data = -np.mean(np.sum(target.data * np.log(pred_data), axis=1))
        loss = tensor(loss_data, requires_grad=pred.requires_grad)
        loss._prev.add(pred)
        
        # 反向传播函数
        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = np.zeros_like(pred.data)
                # 交叉熵导数: -target / pred / N
                grad = -target.data / pred_data / pred.data.shape[0]
                pred.grad += grad * loss.grad  # loss.grad通常为1.0
        
        loss._backward_func = _backward
        return loss


# 定义优化器
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        """随机梯度下降优化器"""
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]
        
    def step(self):
        """更新参数"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            # 计算动量
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
            # 更新参数
            param.data += self.velocities[i]
            
            # 重置梯度
            param.grad = None


class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """Adam优化器"""
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # 初始化一阶矩和二阶矩
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        
    def step(self):
        """更新参数"""
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            # 更新一阶矩和二阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # 重置梯度
            param.grad = None

