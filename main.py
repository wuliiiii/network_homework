import numpy as np
from util.model import tensor, Linear, ReLU, Sigmoid, Tanh, Softmax,CrossEntropyLoss,Adam
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import seaborn as sns
import time

# 定义神经网络模型
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        初始化神经网络
        :param input_size: 输入特征数量
        :param hidden_sizes: 隐藏层大小列表，如[64, 32]表示两个隐藏层
        :param output_size: 输出特征数量
        :param activation: 激活函数，可选'relu', 'sigmoid', 'tanh'
        """
        self.layers = []
        self.activations = []
        
        # 选择激活函数
        if activation == 'relu':
            activation_func = ReLU
        elif activation == 'sigmoid':
            activation_func = Sigmoid
        elif activation == 'tanh':
            activation_func = Tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输入层到第一个隐藏层
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            self.activations.append(activation_func())
            prev_size = hidden_size
        
        # 最后一个隐藏层到输出层
        self.layers.append(Linear(prev_size, output_size))
        
        # 输出层使用softmax激活函数（用于分类）
        self.output_activation = Softmax()
        
    def forward(self, x):
        """前向传播"""
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            x = self.activations[i].forward(x)
        
        # 输出层
        x = self.layers[-1].forward(x)
        x = self.output_activation.forward(x)
        return x
    
    def parameters(self):
        """返回所有参数列表"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# 训练和测试函数
def train(model, train_loader, criterion, optimizer, epochs=10):
    """训练模型"""
    train_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # 前向传播
            outputs = model.forward(batch_x)
            
            # 计算损失
            loss = criterion.forward(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.data
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 打印训练进度
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {end_time-start_time:.2f}s")
    
    return train_losses


def predict(model, x):
    """预测函数"""
    outputs = model.forward(x)
    # 返回预测的类别（概率最大的类别）
    return np.argmax(outputs.data, axis=1)


def evaluate(model, test_loader):
    """评估模型在测试集上的性能"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_x, batch_y in test_loader:
        outputs = model.forward(batch_x)
        preds = np.argmax(outputs.data, axis=1)
        labels = np.argmax(batch_y, axis=1)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(outputs.data)
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# 数据加载和预处理
def load_mnist(subset_size=None):
    """加载MNIST数据集并进行预处理"""
    # 加载MNIST数据集
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # 转换为float32并归一化到[0, 1]
    X = X.astype(np.float32) / 255.0
    
    # 如果需要使用子集
    if subset_size is not None and subset_size < len(X):
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    # 将标签转换为整数
    y = y.astype(np.int32)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 对标签进行独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    return (X_train, y_train_onehot), (X_test, y_test_onehot), y_test


def create_data_loader(X, y, batch_size=32):
    """创建数据加载器"""
    dataset = list(zip(X, y))
    loader = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_x = np.array([item[0] for item in batch])
        batch_y = np.array([item[1] for item in batch])
        loader.append((batch_x, batch_y))
    return loader


# 可视化函数
def plot_loss_curve(losses):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('./output/loss_curve.png')


def plot_confusion_matrix(labels, preds, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./output/confusion_matrix.png')


def plot_roc_curve(labels, probs, n_classes):
    """绘制多类别的ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 为每个类别计算ROC曲线和AUC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    # 绘制随机猜测的基准线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.savefig('./output/roc_curve.png')


def visualize_samples(X, y_true, y_pred, num_samples=10):
    """可视化样本及其预测结果"""
    plt.figure(figsize=(15, 3))
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {np.argmax(y_true[idx])}\nPred: {y_pred[idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./output/sample_visualization.png')


# 主函数
def main():
    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    
    # 加载数据（使用10000个样本加速训练）
    print("加载MNIST数据集...")
    (X_train, y_train), (X_test, y_test), y_test_labels = load_mnist(subset_size=10000)
    
    # 创建数据加载器
    batch_size = 64
    train_loader = create_data_loader(X_train, y_train, batch_size)
    test_loader = create_data_loader(X_test, y_test, batch_size)
    
    # 定义模型参数
    input_size = 784  # MNIST图像是28x28=784像素
    hidden_sizes = [128, 64]  # 两个隐藏层，分别有128和64个神经元
    output_size = 10  # 10个类别（0-9）
    
    # 创建模型
    print("创建神经网络模型...")
    model = NeuralNetwork(input_size, hidden_sizes, output_size, activation='relu')
    
    # 定义损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("开始训练...")
    epochs = 20
    train_losses = train(model, train_loader, criterion, optimizer, epochs)
    
    # 绘制损失曲线
    plot_loss_curve(train_losses)
    
    # 在测试集上评估
    print("在测试集上评估...")
    accuracy, true_labels, pred_labels, probs = evaluate(model, test_loader)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 可视化部分测试样本
    visualize_samples(X_test, y_test, pred_labels)
    
    # 绘制混淆矩阵
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, pred_labels, class_names)
    
    # 绘制ROC曲线
    plot_roc_curve(true_labels, probs, output_size)


if __name__ == "__main__":
    main()
    