import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=2, dropout=0.1, activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(f_in, hidden_dim), activation_func, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_func, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PFDO_Block(nn.Module):


    def __init__(self, T_L, K, mlp_hidden_dim=128, mlp_hidden_layers=2):

        super(PFDO_Block, self).__init__()
        self.T_L = T_L
        self.K = K


        self.B_global = nn.Parameter(torch.randn(K, T_L))
        nn.init.xavier_uniform_(self.B_global)
        self.f_phi = MLP(f_in=T_L, f_out=K * K, hidden_dim=mlp_hidden_dim, hidden_layers=mlp_hidden_layers)
        self.f_psi = MLP(f_in=T_L, f_out=K, hidden_dim=mlp_hidden_dim, hidden_layers=mlp_hidden_layers)

    def forward(self, x_in):
        
        B = x_in.size(0)

        M_x = self.f_phi(x_in).view(B, self.K, self.K)

        B_x = torch.bmm(M_x, self.B_global.unsqueeze(0).expand(B, -1, -1))

        c_x = self.f_psi(x_in) 
        return c_x, B_x


class StructuredKoopman(nn.Module):

    def __init__(self, K, T_L, group_sizes, rank, mlp_hidden_dim=64, mlp_hidden_layers=2):

        super(StructuredKoopman, self).__init__()
        assert sum(group_sizes) == K, "Sum of group_sizes must be equal to K."
        self.K = K
        self.T_L = T_L
        self.group_sizes = group_sizes
        self.rank = rank
        self.K_diag_blocks = nn.ModuleList()
        for size in group_sizes:
            block = nn.Linear(size, size, bias=False)
            nn.init.xavier_uniform_(block.weight)
            self.K_diag_blocks.append(block)
        self.U = nn.Parameter(torch.randn(K, rank))
        self.V = nn.Parameter(torch.randn(K, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.positional_encoding = nn.Parameter(torch.randn(T_L, K), requires_grad=True)
        self.f_mod = MLP(f_in=K * 2, f_out=K * K, hidden_dim=mlp_hidden_dim, hidden_layers=mlp_hidden_layers)

    def get_K_diag(self):
        K_diag = torch.zeros(self.K, self.K, device=self.U.device)
        current_idx = 0
        for i, size in enumerate(self.group_sizes):
            K_diag[current_idx:current_idx + size, current_idx:current_idx + size] = self.K_diag_blocks[i].weight
            current_idx += size
        return K_diag

    def get_K_inter(self):
        return self.U @ self.V.T

    def forward(self, C_x):
        B = C_x.size(0)
        K_diag = self.get_K_diag()
        K_inter = self.get_K_inter()
        K_base = K_diag + K_inter  # (K, K)

        C_x_pred_list = []
        for t in range(self.T_L - 1):
            c_t = C_x[:, t, :]  # (B, K)
            pos_t = self.positional_encoding[t, :].unsqueeze(0).expand(B, -1)  # (B, K)
            mod_input = torch.cat([c_t, pos_t], dim=1)  # (B, 2*K)
            M_mod_t = self.f_mod(mod_input).view(B, self.K, self.K)
            M_mod_t = torch.sigmoid(M_mod_t) 
            # (1, K, K) * (B, K, K) -> (B, K, K)
            K_structured_t = K_base.unsqueeze(0) * M_mod_t
            # (B, 1, K) @ (B, K, K) -> (B, 1, K)
            c_t_plus_1_pred = torch.bmm(c_t.unsqueeze(1), K_structured_t.transpose(1, 2)).squeeze(1)
            C_x_pred_list.append(c_t_plus_1_pred)
        return torch.stack(C_x_pred_list, dim=1)

    @torch.no_grad()
    def predict(self, c_x_last, H):
        B = c_x_last.size(0)
        K_diag = self.get_K_diag()
        K_inter = self.get_K_inter()
        K_base = K_diag + K_inter

        pos_last = self.positional_encoding[-1, :].unsqueeze(0).expand(B, -1)
        mod_input = torch.cat([c_x_last, pos_last], dim=1)
        M_mod_last = torch.sigmoid(self.f_mod(mod_input).view(B, self.K, self.K))
        K_tilde = K_base.unsqueeze(0) * M_mod_last  # (B, K, K)
        try:
            eigvals, eigvecs = torch.linalg.eig(K_tilde)
            eigvecs_inv = torch.linalg.inv(eigvecs)
            eigvals_abs = torch.abs(eigvals)
            clip_scales = torch.ones_like(eigvals_abs, dtype=eigvals_abs.dtype)
            unstable_mask = eigvals_abs > 0.99
            clip_scales[unstable_mask] = 0.99 / eigvals_abs[unstable_mask]
            eigvals_clipped = eigvals * clip_scales.to(eigvals.dtype)
            
        except torch.linalg.LinAlgError:
            print("Warning: Singular matrix encountered in eigenvalue decomposition. Using identity for prediction.")
            return c_x_last.unsqueeze(1).expand(-1, H, -1)

        c_x_last_complex = c_x_last.to(torch.cfloat)

        C_x_future_list = []
        for h in range(1, H + 1):
            Lambda_h = torch.diag_embed(torch.pow(eigvals_clipped, h))
            K_h = eigvecs @ Lambda_h @ eigvecs_inv

            # (B, 1, K) @ (B, K, K) -> (B, 1, K)
            c_future_h = torch.bmm(c_x_last_complex.unsqueeze(1), K_h.transpose(1, 2)).squeeze(1)
            C_x_future_list.append(c_future_h.real) 

        return torch.stack(C_x_future_list, dim=1)
