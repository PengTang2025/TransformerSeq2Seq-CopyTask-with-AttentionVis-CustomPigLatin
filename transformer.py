import torch
import torch.nn as nn
import math
from coderlayer_with_attn import TransformerEncoderLayerWithAttn, TransformerDecoderLayerWithAttn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerSeq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_seq_len=20):
        super().__init__()

        # # 为了方便后续提取注意力权重，我们保存最后一层 encoder 的注意力
        self.last_attn = None 
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        # 需要手动绑定fc_out.weight = embedding.weight，且nn.linear的biase==false,才能实现所谓输出映射矩阵与输入的嵌入矩阵相同
        # linear层内部存储的是表面参数转置的矩阵，即（tgt_vocab_size，d_model），和底层运算有关
        # if we want to use the same embedding matrix for output, we need to set bias=False, and fc_out.weight = embedding.weight
        # linear layer stores the transposed matrix of the surface parameters, i.e., (tgt_vocab_size, d_model), which is related to the underlying operations

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        """
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # 保存最后一层的注意力权重 
        # Save the last layer's attention weights
        self.last_attn = self.encoder.layers[-1].attn_weights
        
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

        output = self.fc_out(output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        return output

    def generate_square_subsequent_mask(self, sz):
        # 生成 tgt 序列的 mask，防止 decoder 看到未来 token。
        # 返回形状为 (sz, sz)，对角线以上为 -inf，其余为 0 的 float tensor。
        # generate a mask for the tgt sequence to prevent the decoder from seeing future tokens.
        # Returns a float tensor of shape (sz, sz) with -inf above the diagonal
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

