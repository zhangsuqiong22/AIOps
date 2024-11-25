import torch.nn as nn
import torch, math
from icecream import ic
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""

class Transformer(nn.Module):
    # d_model : number of features
    #def __init__(self,feature_size=10,num_layers=3,dropout=0):
    def __init__(self,feature_size=8,num_layers=3,dropout=0):

        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,
                                                        nhead=4, dim_feedforward=512,
                                                        batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device=None):
        if device:
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            #self._forward_with_attention(src, mask)  # 不需要返回值
            output = self.transformer_encoder(src, mask)
        else:
            output = self.transformer_encoder(src)

        output = self.decoder(output)
        return output


    def _forward_with_attention(self, src, mask):
        # 创建一个列表来存储每层的注意力权重
        src = src.double()
        mask = mask.double() if mask is not None else None

        attn_weights_list = []
        output = src

        for layer in self.transformer_encoder.layers:
            # 调用 Transformer 层，并接收输出和注意力权重
            output, attn_weights = layer.self_attn(output, output, output, attn_mask=mask)

            # 将注意力权重添加到列表中
            attn_weights_list.append(attn_weights)

        print("Attention weights shape:", attn_weights.shape)  # 预期为 (num_layers, num_heads, 60, 60)

        # 将所有层的注意力权重堆叠成一个张量
        attn_weights = torch.stack(attn_weights_list)
        print("Attention weights shape after stack:", attn_weights.shape)  # 预期为 (num_layers, num_heads, 60, 60)

        # 绘制或其他处理
        self.plot_attention_weights(attn_weights)



    def plot_attention_weights(self, attn_weights):
        # 将注意力权重转换为numpy数组
        attn_weights = attn_weights.detach().cpu().numpy()  # (num_layers, num_heads, seq_len, seq_len)

        # 只取最后一层的权重
        #last_layer_attn_weights = attn_weights[-1]  # 形状为 (num_heads, seq_len, seq_len)

        # 计算所有头的平均注意力权重
        #avg_attn_weights = np.mean(last_layer_attn_weights, axis=0)  # (seq_len, seq_len)

        # 只取最后一层的权重
        last_layer_attn_weights = attn_weights[-1]  # 形状为 (num_heads, seq_len, seq_len)

        # 取第一个头的注意力权重
        head0_attn_weights = last_layer_attn_weights[0]  # 形状为 (seq_len, seq_len)

        plt.figure(figsize=(8, 6))
        sns.heatmap(head0_attn_weights, cmap="plasma", cbar=True)  # 使用 "plasma" 颜色映射绘制第一个头的权重
        plt.title("Attention Weights Heatmap (Head 0)")

        #plt.figure(figsize=(8, 6))
        #sns.heatmap(avg_attn_weights, cmap="coolwarm", cbar=True)  # 绘制平均权重
        #sns.heatmap(avg_attn_weights, cmap="plasma", cbar=True)  # 绘制平均权重
        #sns.heatmap(avg_attn_weights, cmap="Oranges", cbar=True)  # 绘制平均权重
        #sns.heatmap(avg_attn_weights, cmap="Spectral", cbar=True)
        #plt.title("Average Attention Weights Heatmap (All Heads)")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")

        output_dir = "attention_weights"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{output_dir}/average_attention_weights_{timestamp}.png"
        plt.savefig(filename)
        plt.close()

