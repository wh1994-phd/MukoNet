import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.linear_branch import PFDO_Block, StructuredKoopman
from layers.nonlinear_branch import NonlinearBranch
from models.revin import RevIN

def calculate_group_sizes(K, num_groups):
    if K < num_groups:
        raise ValueError("K must be greater than or equal to the number of groups.")
    base_size = K // num_groups
    remainder = K % num_groups
    group_sizes = [base_size] * num_groups
    for i in range(remainder):
        group_sizes[i] += 1
    return group_sizes


class MuKo_Block(nn.Module):


    def __init__(self, T_L, T_H, C, K, group_sizes, rank,
                 num_nl_layers, num_hyperedge_prototypes, d_node_feature,
                 mlp_hidden_dim=256, mlp_hidden_layers=4):
        super(MuKo_Block, self).__init__()
        self.T_L = T_L
        self.T_H = T_H
        self.C = C
        self.K = K

        self.input_norm = nn.LayerNorm(C) 

        self.pfdo = PFDO_Block(
            T_L=T_L, K=K,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_hidden_layers=mlp_hidden_layers
        )
        self.koopman_operator = StructuredKoopman(
            K=K, T_L=T_L,
            group_sizes=group_sizes,
            rank=rank,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_hidden_layers=mlp_hidden_layers
        )

        self.nonlinear_branch = NonlinearBranch(
            T_L=T_L, T_H=T_H, C=C,
            num_layers=num_nl_layers,
            num_hyperedge_prototypes=num_hyperedge_prototypes,
            d_node_feature=d_node_feature
        )

        self.reconstruction_layer = nn.Linear(K, 1, bias=False)

    def forward(self, x_in):
        """
        Args:
            x_in (torch.Tensor): 块的输入，形状 (B, T_L, C)。

        Returns:
            y_b (torch.Tensor): 当前块的预测增量, 形状 (B, T_H, C)。
            r_b (torch.Tensor): 用于下一块的输入残差, 形状 (B, T_L, C)。
            l_linear (torch.Tensor): 当前块的线性动力学损失 。
        """
        B = x_in.size(0)

        # 将 B 和 C 维度合并，以独立处理每个通道
        # (B, T_L, C) -> (B, C, T_L) -> (B*C, T_L)
        x_in_flat = x_in.permute(0, 2, 1).reshape(B * self.C, self.T_L)

        # --- 线性分支计算 ---

        # 1. PFDO 分解
        c_x, B_x = self.pfdo(x_in_flat)  # c_x: (B*C, K), B_x: (B*C, K, L)

        # 2. 构造 Koopman 的输入：用系数加权自适应基
        # c_x: (B*C, K) -> (B*C, K, 1)
        # B_x: (B*C, K, L)
        # koopman_input: (B*C, K, L) -> permute -> (B*C, L, K)
        koopman_input = (c_x.unsqueeze(-1) * B_x).permute(0, 2, 1)

        # 3. 学习动力学并计算损失 (Koopman的训练模式)
        koopman_one_step_pred = self.koopman_operator(koopman_input)  # (B*C, L-1, K)
        l_linear = F.mse_loss(koopman_one_step_pred, koopman_input[:, 1:, :])

        # 4. 进行多步预测 (Koopman的推理模式)
        koopman_input_last_step = koopman_input[:, -1, :]  # (B*C, K)
        koopman_future_pred = self.koopman_operator.predict(koopman_input_last_step, self.T_H)  # (B*C, H, K)

        # 5. 重构时域预测 y_linear
        # (B*C, H, K) -> (B*C, H, 1) -> (B*C, H)
        y_linear_flat = self.reconstruction_layer(koopman_future_pred).squeeze(-1)
        # (B*C, H) -> (B, C, H) -> (B, H, C)
        y_linear = y_linear_flat.view(B, self.C, self.T_H).permute(0, 2, 1)

        # 6. 重构过去信号 x_recon 用于计算残差
        # (B*C, L, K) -> (B*C, L, 1) -> (B*C, L)
        x_recon_flat = self.reconstruction_layer(koopman_input).squeeze(-1)
        # (B*C, L) -> (B, C, L) -> (B, L, C)
        x_recon = x_recon_flat.view(B, self.C, self.T_L).permute(0, 2, 1)

        # --- 非线性分支计算 ---
        y_nonlinear = self.nonlinear_branch(x_in)

        # --- 组合输出 ---
        y_b = y_linear + y_nonlinear
        r_b = x_in - x_recon

        return y_b, r_b, l_linear


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        T_L = configs.seq_len
        T_H = configs.pred_len
        C = configs.enc_in
        c_out = configs.c_out

        K = getattr(configs, 'K', 64)
        rank = getattr(configs, 'rank', 10)
        num_groups = getattr(configs, 'num_groups', 4)
        num_blocks = getattr(configs, 'num_blocks', 3)
        num_nl_layers = getattr(configs, 'num_nl_layers', 2)
        num_hyperedge_prototypes = getattr(configs, 'num_hyperedge_prototypes', 5)
        d_node_feature = getattr(configs, 'd_node_feature', 16)
        revin_affine = getattr(configs, 'revin_affine', True)

        group_sizes = calculate_group_sizes(K, num_groups)

        self.revin_layer = RevIN(num_features=C, affine=revin_affine)

        shared_block = MuKo_Block(
            T_L, T_H, C, K, group_sizes, rank,
            num_nl_layers, num_hyperedge_prototypes, d_node_feature
        )
        self.blocks = nn.ModuleList([shared_block] * num_blocks)
        self.final_projection = nn.Linear(C, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
      
        x_norm = self.revin_layer(x_enc, 'norm')
        current_residual = x_norm
        y_total = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(2), device=x_enc.device)
        total_l_linear = 0.0

        for block in self.blocks:
            y_b, r_b, l_linear = block(current_residual)
            y_total += y_b
            total_l_linear += l_linear
            current_residual = r_b

        y_intermediate_norm = y_total / len(self.blocks)

        y_denorm = self.revin_layer(y_intermediate_norm, 'denorm')

        y_final = self.final_projection(y_denorm)

        if self.training:
            B_global = self.blocks[0].pfdo.B_global
            K = B_global.size(0)
            identity_matrix = torch.eye(K, device=B_global.device)
            l_ortho = torch.norm(B_global @ B_global.T - identity_matrix, p='fro') ** 2
            koopman_op = self.blocks[0].koopman_operator
            K_diag = koopman_op.get_K_diag()
            K_inter = koopman_op.get_K_inter()
            K_base = K_diag + K_inter
            l_k_norm = torch.norm(K_base, p='fro')
            try:
                _, s, _ = torch.linalg.svd(K_base)
                l_k_spectral = torch.max(s)
            except:
                l_k_spectral = torch.tensor(0.0, device=K_base.device)
            return y_final, total_l_linear, l_ortho, l_k_norm, l_k_spectral
        else:
            return y_final
