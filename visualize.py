import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import random

def plot_loss(train_losses, val_losses, num_epochs, save_path):
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='red', linestyle='-', linewidth=2, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='x', color='blue', linestyle='--', linewidth=2, label='Validation Loss')
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/loss_curve.png")
    plt.close()
    pass

def plot_embeddings(model, vocab_size, save_path):
    # 可视化输入嵌入向量的分布 (使用 PCA 降维至 2 维)
    # 从 embedding 层中抽取嵌入矩阵，shape: (vocab_size, d_model)
    # visualize the distribution of input embedding vectors (using PCA to reduce to 2D)
    # Extract the embedding matrix from the embedding layer, shape: (vocab_size, d_model)
    embeddings = model.src_embedding.weight.cpu().detach().numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure()
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=np.arange(vocab_size), cmap="viridis", s=100)
    plt.title("Embedding Distribution (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.colorbar(scatter)
    plt.savefig(f"{save_path}/embedding_distribution.png")
    plt.close()
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

def plot_attention_weights(model, sample_src, sample_input, save_path, sample_src_text):
    # 触发 forward，计算注意力权重 
    # Trigger forward pass to compute attention weights
    _ = model(sample_src, sample_input)  
    # 获取最后一层的注意力权重
    # Obtain the last layer's attention weights
    e_attn_weights = model.last_e_attn
    dc_attn_weights = model.last_d_cross_attn
    d_attn_weights = model.last_d_attn
    # shape of attn_weights: (1, heads, seq_len, seq_len)
    for i in range(e_attn_weights.size(1)):
        # encoder
        attn_heatmap = e_attn_weights[0, i].cpu().detach().numpy().T # shape: (seq_len, seq_len)
        plt.figure()
        im = plt.imshow(attn_heatmap, cmap="plasma", aspect='auto')
        # 让 y=0 出现在图底部，符合坐标系习惯
        # invert the y-axis so that y=0 appears at the bottom of the plot, following the coordinate system convention
        plt.gca().invert_yaxis()  
        plt.title(f"Encoder Attention Weight Heatmap_head{i+1}", fontsize=14)
        plt.xlabel(f"Query Position(i):{sample_src_text}", fontsize=12)
        plt.ylabel("Key Position(j)", fontsize=12)
        plt.colorbar(im)
        plt.savefig(f"{save_path}/{sample_src_text}_encoder_attn_head_{i+1}.png")
        plt.close()  
        # decoder self-attention
        attn_heatmap = d_attn_weights[0, i].cpu().detach().numpy().T # shape: (seq_len, seq_len)
        plt.figure()
        im = plt.imshow(attn_heatmap, cmap="plasma", aspect='auto')
        # 让 y=0 出现在图底部，符合坐标系习惯
        # invert the y-axis so that y=0 appears at the bottom of the plot, following the coordinate system convention
        plt.gca().invert_yaxis()  
        plt.title(f"Decoder Self-Attention Weight Heatmap_head{i+1}", fontsize=14)
        plt.xlabel(f"Query Position(i):{sample_src_text}", fontsize=12)
        plt.ylabel("Key Position(j)", fontsize=12)
        plt.colorbar(im)
        plt.savefig(f"{save_path}/{sample_src_text}_decoder_self-attn_head_{i+1}.png")
        plt.close()  
        # decoder cross-attention
        attn_heatmap = dc_attn_weights[0, i].cpu().detach().numpy().T # shape: (seq_len, seq_len)
        plt.figure()
        im = plt.imshow(attn_heatmap, cmap="plasma", aspect='auto')
        # 让 y=0 出现在图底部，符合坐标系习惯
        # invert the y-axis so that y=0 appears at the bottom of the plot, following the coordinate system convention
        plt.gca().invert_yaxis()  
        plt.title(f"Decoder Cross-Attention Weight Heatmap_head{i+1}", fontsize=14)
        plt.xlabel(f"Query Position(i):{sample_src_text}", fontsize=12)
        plt.ylabel("Key Position(j)", fontsize=12)
        plt.colorbar(im)
        plt.savefig(f"{save_path}/{sample_src_text}_decoder_cross-attn_head_{i+1}.png")
        plt.close()  
    pass

# def visualize(model, train_losses, num_epochs, vocab_size, seq_len, test_input, test_target, device):
#     # 可视化预测结果
#     # 随机选择一个样本进行可视化
#     # Visualize the prediction results
#     # Randomly select a sample for visualization
#     sample_idx = random.randint(0, len(test_input)-1)
#     sample_input = test_input[sample_idx].unsqueeze(0).to(device)  # shape: (1, seq_len)
#     sample_true = test_target[sample_idx].numpy()
    
#     plt.figure(figsize=(18, 12), constrained_layout=True)
#     plt.subplot(2,2,1)
#     # 绘制训练损失曲线
#     # Plot the training loss curve
#     plot_loss(train_losses, num_epochs)
#     plt.subplot(2,2,2)
#     # 可视化嵌入向量分布
#     # Visualize the distribution of embedding vectors
#     plot_embeddings(model, vocab_size)
#     plt.subplot(2,2,3)
#     # 绘制样本预测结果
#     # Plot the sample prediction results
#     plot_sample_prediction(seq_len, sample_input, sample_true, model)
#     plt.subplot(2,2,4)
#     # 绘制注意力权重热图
#     # Plot the attention weight heatmaps
#     plot_attention_weights(model, sample_input)
    
#     plt.suptitle("Transformer Input/Output Embedding and Linear Transformation Analysis", fontsize=16)
#     plt.show()