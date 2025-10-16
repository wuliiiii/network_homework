import numpy as np
from util.model import tensor, Linear


def tensor_test():
    # 测试广播操作
    print("=== 广播操作测试 ===")
    m = tensor([[1, 2], [3, 4]], requires_grad=True)
    v = tensor([5, 6], requires_grad=True)
    e = m + v  # 广播相加: [[6,8], [8,10]]
    f = m * v  # 广播相乘: [[5,12], [15,24]]
    print("e = m + v =\n", e.data)
    print("f = m * v =\n", f.data)
    
    # 反向传播
    e.backward([[1, 1], [1, 1]])
    print("e对m的梯度:\n", m.grad)  # 应为[[1,1],[1,1]]
    print("e对v的梯度:", v.grad)    # 应为[2,2]（每列梯度之和）
    
    # 重置梯度
    m.grad = None
    v.grad = None
    
    f.backward([[1, 1], [1, 1]])
    print("f对m的梯度:\n", m.grad)  # 应为[[5,6],[5,6]]
    print("f对v的梯度:", v.grad)    # 应为[4,6]（1+3和2+4）

def main():
    tensor_test()


if __name__ == "__main__":
    main()
