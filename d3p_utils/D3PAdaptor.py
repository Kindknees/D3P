import torch
import torch.nn as nn
from torch.distributions import Normal
# 假設你的 MLP 類別放在這個路徑，請根據實際情況修改 import
from model.common.mlp import MLP 

class D3PAdaptor(nn.Module):
    def __init__(self, obs_dim, action_dim, output_mean, seq_len=1, chunk_size=4, mlp_dims=[256, 512, 1024, 512, 256]):
        super().__init__()
        
        # 1. 計算輸入維度
        input_dim = (obs_dim * seq_len) + (action_dim * chunk_size)
        
        # 2. 建構完整的 dim_list (包含輸入層與所有隱藏層)
        dim_list = [input_dim] + mlp_dims
        
        # 3. 實例化你提供的 MLP 類別
        # DPPO 官方通常愛用 Mish 或 SiLU，這裡我們跟隨主流設定
        self.net = MLP(
            dim_list=dim_list,
            activation_type="Mish",      # 隱藏層的激勵函數
            out_activation_type="Mish",  # 特徵輸出層的激勵函數
            use_layernorm=False          # 如果覺得難收斂，可以改成 True
        )
        
        # 4. 預測步長 k 的均值 (接在 MLP 的最後一層輸出之後)
        self.mean_layer = nn.Linear(mlp_dims[-1], 1)
        
        # 預測步長 k 的對數標準差 (獨立的可學習參數)
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # 🚨 關鍵初始化：讓 Adaptor 初始傾向於每步走 1 步
        nn.init.constant_(self.mean_layer.bias, output_mean)
        nn.init.constant_(self.mean_layer.weight, 0.0)

    def forward(self, obs_dict, noisy_action):
        B = noisy_action.shape[0]
        
        # 攤平輸入
        obs_flat = obs_dict["state"].view(B, -1)
        action_flat = noisy_action.view(B, -1)
        
        # 拼接後丟入你的 MLP
        x = torch.cat([obs_flat, action_flat], dim=-1)
        feat = self.net(x)
        
        # 計算高斯分佈參數
        mean = self.mean_layer(feat).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return Normal(mean, std)