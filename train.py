import torch
import torch.nn as nn
import numpy as np
from transformer import TransformerSeq2SeqModel
from pig_latin_dataset import dataset_generator
from visualize import plot_loss, plot_attention_weights
import torch.optim as optim
from tqdm import tqdm
import os


def train(batch_size, num_epochs, device, lr = 1e-4, pad_idx = 0):

    # ===== 数据集 dataset =====
    train_loader, val_loader, test_loader = dataset_generator(batch_size=batch_size)

    # 从任一数据集中取出 vocab_size（都是一个 PigLatinDataset 对象）
    # from any dataset, we can get the vocab_size (all are PigLatinDataset objects)
    vocab_size = train_loader.dataset.dataset.vocab_size  # nested：Subset -> Dataset

    # ===== 模型初始化 initalization =====
    model = TransformerSeq2SeqModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=20
    ).to(device)

    # ===== 优化器与损失 optimizer and loss function =====
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ===== 训练 train =====
    def train_one_epoch(model, dataloader):
        model.train()
        total_loss = 0
        for src, tgt_input, tgt_output in tqdm(dataloader, desc="Training"):
        # for src, tgt_input, tgt_output in tqdm(itertools.islice(train_loader, 5), desc="Training (debug)"):
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            logits = model(src, tgt_input)

            logits = logits.view(-1, logits.size(-1))
            tgt_output = tgt_output.view(-1)

            loss = criterion(logits, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # ===== 验证 evaluate =====
    @torch.no_grad()
    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0
        for src, tgt_input, tgt_output in tqdm(dataloader, desc="Evaluating"):
        # for src, tgt_input, tgt_output in tqdm(itertools.islice(val_loader, 5), desc="Evaluating (debug)"):
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            logits = model(src, tgt_input)

            logits = logits.view(-1, logits.size(-1))
            tgt_output = tgt_output.view(-1)

            loss = criterion(logits, tgt_output)
            total_loss += loss.item()

        return total_loss / len(dataloader)


    # ===== 主训练循环 main training loop =====
    os.makedirs("checkpoints", exist_ok=True)

    train_loss_history = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader)
        val_loss = evaluate(model, val_loader)
        train_loss_history.append(train_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/transformer_epoch{epoch+1}.pt")
        
    np.save("train_loss_history.npy", train_loss_history)


    plot_loss(train_loss_history, num_epochs)
    # 取出一个batch的样本进行可视化
    # Get a batch of samples for visualization
    sample_src, sample_input, sample_target = next(iter(test_loader))
    sample_src, sample_input, sample_target = sample_src.to(device), sample_input.to(device), sample_target.to(device)
    plot_attention_weights(model, sample_src, sample_input)

if __name__ == "__main__":
    # ===== 超参数 Hyperparameters =====
    batch_size = 64
    num_epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_idx = 0  
    train(batch_size=batch_size, num_epochs=num_epochs, device=device, lr=lr, pad_idx=pad_idx)