import torch
import torch.nn as nn
from layers.RevIN import RevIN

# TRAnsient Chaotic Evolution Network
class Model(nn.Module):
     def __init__(self, args):
          super(Model, self).__init__()
          self.pred_len = args.pred_len
          self.enc_in = args.enc_in
          self.seq_len = args.seq_len
          self.invert = getattr(args, 'invert', 0)

          self.d_model = args.d_model
          self.dec_in = args.dec_in

          self.chaos_type = getattr(args, 'chaos_type', 'lorenz').lower()
          self.d_chaos = 2 if self.chaos_type in ['vdp', 'damped_oscillator'] else 3
          self.num_step = getattr(args, 'num_step', 20)
          self.derivation = getattr(args, 'derivation', 'euler').lower()

          self.num_nodes = self.enc_in if self.invert else self.seq_len

          if args.chaos_para_nn:
               self.t_evo = nn.Parameter(torch.tensor(args.t_evo))
          else:
               self.t_evo = torch.tensor(args.t_evo)

          if self.chaos_type == 'linear_ode':
            # 线性 ODE 转移矩阵 (d_chaos x d_chaos)
            # dS/dt = S * A^T
            self.linear_A = nn.Parameter(torch.randn(self.d_chaos, self.d_chaos) * 0.1)
            
          elif self.chaos_type == 'gru':
               # GRU 作为连续向量场 dS/dt = GRU(0, S) - S
               self.gru_cell = nn.GRUCell(input_size=self.d_chaos, hidden_size=self.d_chaos)

          self.decay_rate = args.decay_rate
          self.coupling_strength = args.coupling_strength

          self.revin = RevIN(self.enc_in, affine=True, subtract_last=False)

          decoder_modules = []
          encoder_modules = []
          # nn.Linear(self.d_model, self.d_model),
          #           nn.Tanh()

          if self.invert:
               encoder_modules.append(nn.Linear(self.seq_len, self.d_model))
               encoder_modules.append(nn.GELU())

               decoder_modules.append(nn.Linear(self.d_chaos, self.dec_in))
               decoder_modules.append(nn.GELU())
          else:
               encoder_modules.append(nn.Linear(self.enc_in, self.d_model))
               encoder_modules.append(nn.GELU())

               decoder_modules.append(nn.Linear(self.d_chaos * self.num_nodes, self.dec_in))
               decoder_modules.append(nn.GELU())

          for _ in range(getattr(args, 'enc_layers', 1) - 1):
               encoder_modules.append(nn.Linear(self.d_model, self.d_model))
               encoder_modules.append(nn.GELU())
          encoder_modules.append(nn.Linear(self.d_model, self.d_chaos))
          encoder_modules.append(nn.Tanh())

          for _ in range(getattr(args, 'dec_layers', 1) - 1):
               decoder_modules.append(nn.Linear(self.dec_in, self.dec_in))
               decoder_modules.append(nn.GELU())

          if self.invert: decoder_modules.append(nn.Linear(self.dec_in, self.pred_len))
          else: decoder_modules.append(nn.Linear(self.dec_in, self.pred_len * self.enc_in))

          self.decoder = nn.Sequential(*decoder_modules)
          self.encoder = nn.Sequential(*encoder_modules)

          self.dropout = nn.Dropout(args.dropout)

     def get_coupling_matrix(self, device):
          indices = torch.arange(self.num_nodes, device=device).float()
          dist_matrix = indices.unsqueeze(0) - indices.unsqueeze(1)

          mask = (dist_matrix > 0).float() if not self.invert else torch.ones_like(dist_matrix)
          tmp = torch.abs(dist_matrix) * mask
          coupling_matrix = torch.exp(-self.decay_rate * tmp)

          return coupling_matrix
     
     def physics_derivative(self, state, coupling_matrix):
          x, y = state[..., 0], state[..., 1]

          if self.chaos_type == 'lorenz':
               z = state[..., 2]
               d_internal = torch.stack([10.*(y-x), x*(28.-z)-y, x*y-(8./3.)*z], dim=-1)
          elif self.chaos_type == 'rossler':
               z = state[..., 2]
               d_internal = torch.stack([-y-z, x+0.2*y, 0.2+z*(x-5.7)], dim=-1)
          elif self.chaos_type == 'vdp':
               d_internal = torch.stack([y, 1.*(1-x**2)*y-x], dim=-1)
          elif self.chaos_type == 'damped_oscillator':
               # 强阻尼周期系统 (2D ODE): x'' + gamma*x' + omega^2*x = 0
               # 转化为 d_x = y, d_y = -omega^2*x - gamma*y
               # 强阻尼条件: gamma^2 - 4*omega^2 > 0 (例如: omega=1.0, gamma=3.0)
               omega_sq = 1.0
               gamma = 3.0
               d_internal = torch.stack([y, -omega_sq * x - gamma * y], dim=-1)
            
          elif self.chaos_type == 'linear_ode':
               # 线性 ODE (ND ODE): dS/dt = S * A^T
               # 这里 state 的形状是 [B, N, d_chaos]
               d_internal = torch.matmul(state, self.linear_A.t())
               
          elif self.chaos_type == 'gru':
               # 标准网络模块 GRU 作为连续向量场: dS/dt = GRU(0, S) - S
               B, N, D = state.shape
               # GRUCell 要求输入是 2D 张量 [batch_size, input_size]，因此需要展平
               state_flat = state.view(B * N, D)
               # 自治系统不需要外部输入，输入全置为 0
               dummy_input = torch.zeros_like(state_flat)
               # 计算隐状态更新
               h_next = self.gru_cell(dummy_input, state_flat)
               # 欧拉框架下的导数 = \Delta S
               d_internal = (h_next - state_flat).view(B, N, D)
               
          else:
               raise ValueError(f"Unknown chaos type: {self.chaos_type}")
          # elif self.chaos_type == 'chen' / 'lorenz-96':
          # more

          influence_sum = torch.matmul(coupling_matrix, state)
          degree = coupling_matrix.sum(dim=-1, keepdim=True)
          coupling_effect = influence_sum - degree * state

          # print(f'd_internal.shape {d_internal.shape}, coupling_effect.shape {coupling_effect.shape}, state.shape {state.shape}, degree.shape {degree.shape}')
          # exit(0)

          return d_internal + self.coupling_strength * coupling_effect

     def _evolve(self, state, coupling_matrix):
          dt = torch.abs(self.t_evo) / self.num_step if isinstance(self.t_evo, nn.Parameter) else self.t_evo / self.num_step
          for _ in range(self.num_step):
               if self.derivation == 'rk4':
                    k1 = self.physics_derivative(state, coupling_matrix)
                    k2 = self.physics_derivative(state + dt*0.5*k1, coupling_matrix)
                    k3 = self.physics_derivative(state + dt*0.5*k2, coupling_matrix)
                    k4 = self.physics_derivative(state + dt*k3, coupling_matrix)
                    state = state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
               else:
                    state = state + dt * self.physics_derivative(state, coupling_matrix)
          return state

     def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
          B, T, D = x.shape
          x_enc = self.revin(x, 'norm')

          # print(f'x_enc head {x_enc[:, :5, :]}')

          if self.invert:
               # [B, T, D] -> [B, D, T] -> [B, D, d_chaos]
               state = self.encoder(x_enc.permute(0, 2, 1))
          else:
               # [B, T, D] -> [B, T, d_chaos]
               state = self.encoder(x_enc)

          # print(f'state head {state[:, :5, :]}')

          coupling_matrix = self.get_coupling_matrix(x_enc.device)
          final_state = self._evolve(state, coupling_matrix)

          if self.invert:
               # final_state: [B, D, d_chaos]  -> [B, D, pred_len]
               out = self.decoder(final_state)
               out = out.permute(0, 2, 1) # [B, pred_len, D]
          else:
               # final_state: [B, T, d_chaos] -> [B, pred_len * D]
               state_flat = final_state.reshape(B, -1)
               out = self.decoder(state_flat).reshape(B, self.pred_len, D)

          # print(f'out head {out[:, :5, :]}',)

          out_tmp = self.dropout(out)

          final_out = self.revin(out_tmp, mode='denorm')
          # print(f'final_out head {final_out[:, :5, :]}')
          return final_out
