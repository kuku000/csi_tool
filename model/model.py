import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Token 與 Position Embedding
# ----------------------
class TokenAndPositionEmbedding(nn.Module):
    """
    maxlen: 序列的最大長度
    embed_dim: 嵌入維度
    """
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = nn.Linear(1, embed_dim)
        self.token_cnn = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1)
        print("Using Conv1d as token embedding.")
        # 保持不變
        self.position_embedding = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        # 確保 x 是 [batch_size, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embeddings = self.position_embedding(positions)  # [1, seq_len, embed_dim]
        token_embeddings = self.token_embedding(x)  # [batch_size, seq_len, embed_dim]
        return token_embeddings + pos_embeddings


# ----------------------
# Transformer Block
# ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        embed_dim: 嵌入維度
        num_heads: 多頭注意力的頭數
        ff_dim: 前饋神經網路隱藏層的維度
        dropout_rate: Dropout 機率
        """
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 自注意力機制
        attn_output, _ = self.attention(x, x, x)  # [batch_size, seq_len, embed_dim]
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 殘差連接 + LayerNorm

        # 前饋神經網路
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)  # 殘差連接 + LayerNorm


# ----------------------
# 整個模型結構 (支持多層 Transformer Block)
# ----------------------
class TransformerClassifier(nn.Module):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, num_classes, num_layers=1, dropout_rate=0.1):
        """
        maxlen: 序列的最大長度
        embed_dim: 嵌入維度
        num_heads: 多頭注意力的頭數
        ff_dim: 前饋神經網路隱藏層的維度
        num_classes: 最終分類的類別數
        num_layers: Transformer Block 的層數
        dropout_rate: Dropout 機率
        """
        super(TransformerClassifier, self).__init__()
        self.embedding = TokenAndPositionEmbedding(maxlen, embed_dim)

        # 使用 nn.ModuleList 來建立多層 Transformer Block
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, 1]
        """
        # 嵌入與位置編碼
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # 多層 Transformer Block
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # [batch_size, seq_len, embed_dim]

        # 全局平均池化
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        x = self.global_avg_pool(x).squeeze(-1)  # [batch_size, embed_dim]

        # Dense 層
        x = self.dropout(F.relu(self.fc1(x)))  # [batch_size, 64]
        logits = self.fc_out(x)  # [batch_size, num_classes]
        return F.log_softmax(logits, dim=-1)


# ----------------------
# 測試模型
# ----------------------
# if __name__ == '__main__':
#     # 模型參數
#     maxlen = 48       # 序列長度
#     embed_dim = 64    # token embedding 維度
#     num_heads = 4     # 多頭注意力數量
#     ff_dim = 128      # 前饋神經網路隱藏層維度
#     num_classes = 49  # 最終分類類別數
#     num_layers = 3    # Transformer Block 的層數
#     dropout_rate = 0.1

#     # 初始化模型
#     model = TransformerClassifier(maxlen, embed_dim, num_heads, ff_dim, num_classes, num_layers, dropout_rate)
#     print(model)

#     # 假設的輸入數據: [batch_size, seq_len, 1]
#     dummy_input = torch.randn(32, maxlen, 1)  # [32, 48, 1]
#     output = model(dummy_input)

#     print(f"Output shape: {output.shape}")  # 預期 [32, 49]
