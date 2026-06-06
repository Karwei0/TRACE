import os
import torch

from model import Chaos_demo, MLP_wo_chaos, ESN, Chaos_ODE, TRACE, NOT_TRACE, TRACE_CNN, TRACE_damped, TRACE_more, TRACE96, TRACE_flexible, RNN

class Exp_Basic(object):
     def __init__(self, args):
          self.args = args
          self.model_dict = {
               'Chaos_demo': Chaos_demo,
               'MLP_wo_chaos': MLP_wo_chaos,
               'ESN': ESN,
               'Chaos_ODE': Chaos_ODE,
               'TRACE': TRACE,
               'NOT_TRACE': NOT_TRACE,
               'TRACE_CNN': TRACE_CNN,
               'TRACE_damped': TRACE_damped,
               'TRACE_more': TRACE_more,
               'TRACE96': TRACE96,
               'TRACE_flexible': TRACE_flexible,
               'RNN': RNN
          }
          self.device = self._acquire_device()
          self.model = self._build_model().to(self.device)

     def _build_model(self):
          raise NotImplementedError
     
     def _acquire_device(self):
          if self.args.use_gpu:
               os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
               device = torch.device('cuda:{}'.format(self.args.gpu))
               print('Use GPU: cuda:{}'.format(self.args.gpu))
          else:
               device = torch.device('cpu')
               print('Use CPU')
          return device
     
     def _get_data(self):
          pass

     def vali(self):
          pass

     def train(self):
          pass

     def test(self):
          pass