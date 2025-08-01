import torch
import torch.nn as nn
import numpy as np
from transformer import TransformerSeq2SeqModel
from pig_latin_dataset import dataset_generator
from visualize import plot_loss, plot_attention_weights, plot_embeddings
import torch.optim as optim
from tqdm import tqdm
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import date
import itertools
from coderlayer_with_attn import TransformerDecoderLayerWithAttn

# ===== 训练 train =====
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt_input, tgt_output in tqdm(dataloader, desc="Training"):
    # for src, tgt_input, tgt_output in tqdm(itertools.islice(dataloader, 5), desc="Training (debug)"):
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
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt_input, tgt_output in tqdm(dataloader, desc="Evaluating"):
    # for src, tgt_input, tgt_output in tqdm(itertools.islice(dataloader, 5), desc="Evaluating (debug)"):
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

        logits = model(src, tgt_input)

        logits = logits.view(-1, logits.size(-1))
        tgt_output = tgt_output.view(-1)

        loss = criterion(logits, tgt_output)
        total_loss += loss.item()

    return total_loss / len(dataloader)

# ===== 测试 test =====
@torch.no_grad()
def test(model, dataloader, device, id2char, pad_idx, bos_idx, eos_idx, max_len=20, num_print=5):
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0
    total_bleu = 0.0

    smooth_fn = SmoothingFunction().method1  # BLEU 平滑器 BLEU Smoothing Function

    for i, (src, tgt_input, tgt_output) in enumerate(dataloader):
        src = src.to(device)
        batch_size, src_len = src.size()

        # 初始化 tgt 为 bos_token
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len):
            out = model(src, tgt)
            next_token_logits = out[:, -1, :]  # 每个序列最后一个位置的输出 the output of the last position in each sequence
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.eq(eos_idx).all():
                break

        pred = tgt[:, 1:]  # 去掉起始的 <BOS> remove the initial <BOS> token

        # ===== 评估 Evaluation =====
        for j in range(batch_size):
            pred_seq = pred[j].tolist()
            target_seq = tgt_output[j].tolist()

            # 去掉 padding 和 eos
            # Remove padding and eos tokens
            pred_seq = [tok for tok in pred_seq if tok != pad_idx and tok != eos_idx]
            target_seq = [tok for tok in target_seq if tok != pad_idx and tok != eos_idx]

            # Token-level Accuracy
            for p_tok, t_tok in zip(pred_seq, target_seq):
                total_tokens += 1
                if p_tok == t_tok:
                    correct_tokens += 1

            # Sequence-level Accuracy
            total_sequences += 1
            if pred_seq == target_seq:
                correct_sequences += 1

            # BLEU
            bleu = sentence_bleu([target_seq], pred_seq, smoothing_function=smooth_fn)
            total_bleu += bleu

            # 打印前几个例子看看效果
            # Print a few examples to see the results
            if i == 0 and j < num_print:
                pred_str = ''.join([id2char[c] for c in pred_seq])
                target_str = ''.join([id2char[c] for c in target_seq])
                print("Examples:")
                print(f"[{j}] Source: {''.join([id2char[c] for c in src[j].tolist() if c != pad_idx])}")
                print(f"[{j}] Target: {target_str}")
                print(f"[{j}] Output: {pred_str}")
                print()

    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_acc = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    bleu_score = total_bleu / total_sequences if total_sequences > 0 else 0.0

    print(f"Token-level Accuracy: {token_acc:.4f}")
    print(f"Sequence-level Accuracy: {seq_acc:.4f}")
    print(f"BLEU Score: {bleu_score:.4f}")
    

def main(batch_size, num_epochs, device, my_seed, lr = 1e-4, pad_idx = 0):

    # ===== 数据集 dataset =====
    train_loader, val_loader, test_loader = dataset_generator(batch_size=batch_size, seed=my_seed)

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

    # ===== 主训练循环 main training loop =====
    datestamp = date.today().strftime("%Y%m%d")
    save_dir = f"{datestamp}"
    os.makedirs(save_dir, exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), f"{save_dir}/model_epoch{epoch+1}.pth")
        # early stopping condition
        if epoch > 0 and abs(val_loss - val_loss_history[-2]) < 1e-4:
            final_epoch = epoch + 1
            print("Early stopping condition met, stopping training.")
            break
    
    np.save(f"{save_dir}/train_losses.npy", train_loss_history)
    np.save(f"{save_dir}/val_losses.npy", val_loss_history)
    print(f"Training complete. Final epoch: {final_epoch}")
    
    test(model, test_loader, device, train_loader.dataset.dataset.id2char, pad_idx, 
         train_loader.dataset.dataset.char2id[train_loader.dataset.dataset.bos_token],
         train_loader.dataset.dataset.char2id[train_loader.dataset.dataset.eos_token])

    model.eval()
    plot_loss(train_loss_history, val_loss_history, final_epoch, save_path=save_dir)
    # 取出单条样本进行可视化
    # get a single sample for visualization
    for i in range(6):
        sample_src, sample_input, sample_target = test_loader.dataset[i]  # 第i个样本
        # 获取sample_src的原始文本
        sample_src_text = ''.join([train_loader.dataset.dataset.id2char[c] for c in sample_src.tolist() if c != pad_idx])
        sample_src = sample_src.unsqueeze(0).to(device)
        sample_input = sample_input.unsqueeze(0).to(device)
        plot_attention_weights(model, sample_src, sample_input, save_dir, sample_src_text)
        print(f"Visualized attention weights for sample {i+1}: {sample_src_text}")

if __name__ == "__main__":
    # ===== 超参数 Hyperparameters =====
    batch_size = 64
    num_epochs = 100
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_seed = 86
    torch.manual_seed(my_seed)
    pad_idx = 0  
    # main(batch_size=batch_size, num_epochs=num_epochs, device=device, lr=lr, pad_idx=pad_idx, my_seed=my_seed)
    
    
    # ===== 仅测试 test only =====
    # train_loss_history = np.load("train_loss_history.npy")
    # val_loss_history = np.load("train_loss_history.npy")
    # final_epoch = len(train_loss_history)
    # plot_loss(train_loss_history, val_loss_history, final_epoch, save_path="checkpoints")
    train_loader, val_loader, test_loader = dataset_generator(batch_size=batch_size)
    vocab_size = train_loader.dataset.dataset.vocab_size
    model = TransformerSeq2SeqModel(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1, 
        max_seq_len=20,
    ).to(device)
    model.load_state_dict(torch.load("20250731/model_epoch8.pth", map_location=device))
    model.eval()
    save_dir = "test_visualization"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(6):
        sample_src, sample_input, sample_target = test_loader.dataset[i]  # 第i个样本
        # sample_src, sample_input, sample_target = next(iter(test_loader))
        # 获取sample_src的原始文本
        sample_src_text = ''.join([train_loader.dataset.dataset.id2char[c] for c in sample_src.tolist() if c != pad_idx])
        sample_src = sample_src.unsqueeze(0).to(device)
        sample_input = sample_input.unsqueeze(0).to(device)
        plot_attention_weights(model, sample_src, sample_input, save_dir, sample_src_text)
        print(f"Visualized attention weights for sample {i+1}: {sample_src_text}")

    # plot_embeddings(model, vocab_size, save_path="checkpoints")
    # bos_idx = train_loader.dataset.dataset.char2id[train_loader.dataset.dataset.bos_token]
    # eos_idx = train_loader.dataset.dataset.char2id[train_loader.dataset.dataset.eos_token]
    # test(model, test_loader, device, train_loader.dataset.dataset.id2char, pad_idx, bos_idx, eos_idx)