import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import random

def plot_loss(train_losses, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='red', linestyle='-', linewidth=2)
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.savefig("train_loss_curve.png")
    plt.close()
    pass

def plot_embeddings(model, vocab_size):
    # 可视化输入嵌入向量的分布 (使用 PCA 降维至 2 维)
    # 从 embedding 层中抽取嵌入矩阵，shape: (vocab_size, d_model)
    # visualize the distribution of input embedding vectors (using PCA to reduce to 2D)
    # Extract the embedding matrix from the embedding layer, shape: (vocab_size, d_model)
    embeddings = model.embedding.weight.cpu().detach().numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.subplot(2,2,2)
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=np.arange(vocab_size), cmap="viridis", s=100)
    plt.title("Embedding Distribution (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.colorbar(scatter)
    pass

def plot_sample_prediction(seq_len, sample_input, sample_target, model):
    with torch.no_grad():
        sample_output = model(sample_input)
    sample_pred = torch.argmax(sample_output, dim=-1).cpu().numpy()[0]
    plt.figure()
    plt.plot(range(seq_len), sample_target, marker='o', color='blue', linestyle='-', label="Ground Truth")
    plt.plot(range(seq_len), sample_pred, marker='x', color='green', linestyle='--', label="Prediction")
    plt.title("Prediction vs Ground Truth", fontsize=14)
    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Token ID", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("sample_prediction.png")
    pass

def plot_attention_weights(model, sample_src, sample_input):
    # 触发 forward，计算注意力权重 
    # Trigger forward pass to compute attention weights
    _ = model(sample_src, sample_input)  
    # 获取最后一层的注意力权重
    # Obtain the last layer's attention weights
    attn_weights = model.last_attn 
    # shape of attn_weights: (1, heads, seq_len, seq_len)
    for i in range(attn_weights.size(1)):
        attn_heatmap = attn_weights[0, i].cpu().detach().numpy().T # shape: (seq_len, seq_len)
        plt.figure()
        im = plt.imshow(attn_heatmap, cmap="plasma", aspect='auto')
        # 让 y=0 出现在图底部，符合坐标系习惯
        # invert the y-axis so that y=0 appears at the bottom of the plot, following the coordinate system convention
        plt.gca().invert_yaxis()  
        plt.title("Attention Weight Heatmap", fontsize=14)
        plt.xlabel("Query Position(i)", fontsize=12)
        plt.ylabel("Key Position(j)", fontsize=12)
        plt.colorbar(im)
        plt.savefig(f"attention_heatmap_head_{i}.png")
        plt.close()  
    pass

def visualize(model, train_losses, num_epochs, vocab_size, seq_len, test_input, test_target, device):
    # 可视化预测结果
    # 随机选择一个样本进行可视化
    # Visualize the prediction results
    # Randomly select a sample for visualization
    sample_idx = random.randint(0, len(test_input)-1)
    sample_input = test_input[sample_idx].unsqueeze(0).to(device)  # shape: (1, seq_len)
    sample_true = test_target[sample_idx].numpy()
    
    plt.figure(figsize=(18, 12), constrained_layout=True)
    plt.subplot(2,2,1)
    # 绘制训练损失曲线
    # Plot the training loss curve
    plot_loss(train_losses, num_epochs)
    plt.subplot(2,2,2)
    # 可视化嵌入向量分布
    # Visualize the distribution of embedding vectors
    plot_embeddings(model, vocab_size)
    plt.subplot(2,2,3)
    # 绘制样本预测结果
    # Plot the sample prediction results
    plot_sample_prediction(seq_len, sample_input, sample_true, model)
    plt.subplot(2,2,4)
    # 绘制注意力权重热图
    # Plot the attention weight heatmaps
    plot_attention_weights(model, sample_input)
    
    plt.suptitle("Transformer Input/Output Embedding and Linear Transformation Analysis", fontsize=16)
    plt.show()