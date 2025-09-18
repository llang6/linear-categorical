import torch
import torch.nn as nn
from math import sqrt
import time
import random
import numpy as np
from utilities.utils import fit_shape_templates, classify_fits, parse_label


class MyRNN(nn.Module):
    """
    modified from 'FullRankRNN' in https://github.com/adrian-valente/lowrank_inference/blob/main/low_rank_rnns/modules.py
    """
    def __init__(self, 
                 input_size,
                 observed_size,
                 hidden_size,
                 input_expansion_size=None,
                 noise_std=0.05,
                 alpha=0.2, 
                 rho=0.1,
                 train_wi=False, 
                 train_wrec=True, 
                 train_h0=False, 
                 train_si=False, 
                 train_wz=False,
                 train_b=True,
                 wi_init=None,   
                 wz_init=None,
                 wrec_init=None, 
                 h0_init=None, 
                 si_init=None,  
                 b_init=None, 
                 non_linearity=torch.nn.functional.relu):
        """
        :param input_size: int
        :param observed_size: int
        :param hidden_size: int
        :param input_expansion_size: int or None
        :param noise_std: float
        :param alpha: float, value of dt / tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_wz: bool
        :param train_b: bool
        :param wi_init: 
            tuple of torch tensors of shapes (input_size, input_expansion_size) and (input_expansion_size, observed_size + hidden_size) 
            - or -
            torch tensor of shape (input_size, observed_size + hidden_size)
            - or -
            None
        :param wz_init: torch tensor of shape (observed_size + hidden_size, 1) or None
        :param wrec_init: torch tensor of shape (observed_size + hidden_size, observed_size + hidden_size) or None
        :param h0_init: torch tensor of shape (observed_size + hidden_size) or None
        :param si_init: 
            input scaling
            tuple of torch tensors of shapes (input_size) and (input_expansion_size) 
            - or -
            torch tensor of shape (input_size)
            - or -
            None
        :param b_init: torch tensor of shape (observed_size + hidden_size) or None
        :param non_linearity: torch module with a forward method
        """
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.observed_size = observed_size
        self.hidden_size = hidden_size
        self.network_size = observed_size + hidden_size
        self.input_expansion_size = input_expansion_size
        self.has_expansion_layer = input_expansion_size is not None
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_wz = train_wz
        self.train_b = train_b
        self.non_linearity = non_linearity

        # Define parameters
        network_size = observed_size + hidden_size
        if not self.has_expansion_layer:
            self.wi = nn.Parameter(torch.Tensor(input_size, network_size))
            self.si = nn.Parameter(torch.Tensor(input_size))
            if train_wi:
                self.si.requires_grad = False
            else:
                self.wi.requires_grad = False
            if not train_si:
                self.si.requires_grad = False
        else:
            self.wi_1 = nn.Parameter(torch.Tensor(input_size, input_expansion_size))
            self.si_1 = nn.Parameter(torch.Tensor(input_size))
            self.wi_2 = nn.Parameter(torch.Tensor(input_expansion_size, network_size))
            self.si_2 = nn.Parameter(torch.Tensor(input_expansion_size))
            if train_wi:
                self.si_1.requires_grad = False
                self.si_2.requires_grad = False
            else:
                self.wi_1.requires_grad = False
                self.wi_2.requires_grad = False
            if not train_si:
                self.si_1.requires_grad = False
                self.si_2.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(network_size, network_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(network_size))
        if not train_b:
            self.b.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(network_size))
        if not train_h0:
            self.h0.requires_grad = False
        self.wz = nn.Parameter(torch.Tensor(network_size, 1))
        if not train_wz:
            self.wz.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                if not self.has_expansion_layer:
                    self.wi.normal_()
                else:
                    self.wi_1.normal_()
                    self.wi_2.normal_()
            else:
                if not self.has_expansion_layer:
                    self.wi.copy_(wi_init)
                else:
                    self.wi_1.copy_(wi_init[0])
                    self.wi_2.copy_(wi_init[1])
            if si_init is None:
                if not self.has_expansion_layer:
                    self.si.set_(torch.ones_like(self.si))
                else:
                    self.si_1.set_(torch.ones_like(self.si_1))
                    self.si_2.set_(torch.ones_like(self.si_2))
            else:
                if not self.has_expansion_layer:
                    self.si.copy_(si_init)
                else:
                    self.si_1.copy_(si_init[0])
                    self.si_2.copy_(si_init[1])
            if wrec_init is None:
                # self-connections are not allowed
                W_init = (rho / sqrt(network_size)) * torch.randn(network_size, network_size)
                W_init = W_init * (torch.ones(network_size, network_size) - torch.eye(network_size));
            else:
                W_init = wrec_init
                W_init = W_init * (torch.ones(network_size, network_size) - torch.eye(network_size))
            self.wrec.copy_(W_init)
            if b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(b_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
            if wz_init is None:
                self.wz.normal_(std=1 / network_size)
            else:
                self.wz.copy_(wz_init)
        self.wi_full = None
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        if not self.has_expansion_layer:
            self.wi_full = (self.wi.t() * self.si).t()
        else:
            self.wi_1_full = (self.wi_1.t() * self.si_1).t()
            self.wi_2_full = (self.wi_2.t() * self.si_2).t()

    def forward(self, input_, return_dynamics=False, initial_states=None, silenced_neurons=None):
        """
        :param input_: tensor of shape (batch_size, #timesteps, input_size)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial if desired
        :param silenced_neurons: None or list/array of integer indices of neurons to ablate if desired
        :return: if return_dynamics=False:
                     output tensor of shape (batch_size, #timesteps, observed_size + 1)
                 if return_dynamics=True:
                     (output tensor, trajectories tensor of shape (batch_size, #timesteps, observed_size + hidden_size))
        """
        batch_size = input_.shape[0]
        seq_len = input_.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(h + self.b)
        if silenced_neurons is not None:
            r[silenced_neurons] = 0
        z = torch.zeros(1)
        c = torch.zeros(1)
        self._define_proxy_parameters()
        W_out = torch.zeros(self.network_size, self.observed_size)
        W_out[:self.observed_size, :] = torch.eye(self.observed_size)
        noise = torch.randn(batch_size, seq_len, self.network_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.observed_size + 1, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.network_size, device=self.wrec.device)
            trajectories[:, 0, :] = h.clone()

        # simulation loop
        for i in range(seq_len):
            
            # stimulus
            if self.has_expansion_layer:
                stim_expand = torch.tanh( input_[:, i, :].matmul(self.wi_1_full) )
                stim = stim_expand.matmul(self.wi_2_full)
            else:
                stim = input_[:, i, :].matmul(self.wi_full)
                
            # current
            W = self.wrec * (torch.ones_like(self.wrec) - torch.eye(self.wrec.shape[0])) # ensure self-connections are not trained
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(W.t()) + stim)
           
            # firing rate
            r = self.non_linearity(h + self.b)
            if silenced_neurons is not None:
                r[:, silenced_neurons] = 0
            
            # behavior
            z = z + self.alpha * (-z + r @ self.wz)
            c = torch.tanh(z)

            # output
            output[:, i, :self.observed_size] = r @ W_out
            output[:, i, -1] = c.flatten()

            if return_dynamics:
                trajectories[:, i + 1, :] = h.clone()
        
        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        if self.has_expansion_layer:
            wi = (self.wi_1, self.wi_2)
            si = (self.si_1, self.si_2)
        else:
            wi = self.wi
            si = self.si
        new_net = MyRNN(
            input_size=self.input_size,
            observed_size=self.observed_size,
            hidden_size=self.hidden_size,
            input_expansion_size=self.input_expansion_size,
            noise_std=self.noise_std,
            alpha=self.alpha, 
            rho=self.rho,
            train_wi=self.train_wi,  
            train_wrec=self.train_wrec, 
            train_h0=self.train_h0, 
            train_si=self.train_si, 
            train_wz=self.train_wz,
            train_b=self.train_b,
            wi_init=wi,   
            wz_init=self.wz,
            wrec_init=self.wrec, 
            h0_init=self.h0, 
            si_init=si, 
            b_init=self.b,
            non_linearity=self.non_linearity
        )
        return new_net

    
class MyAltRNN(nn.Module):
    """
    Has a new option for re-scaling behavioral outputs
    """
    def __init__(self, 
                 input_size,
                 observed_size,
                 hidden_size,
                 input_expansion_size=None,
                 noise_std=0.05,
                 alpha=0.2, 
                 rho=0.1,
                 train_wi=False, 
                 train_wrec=True, 
                 train_h0=False, 
                 train_si=False, 
                 train_wz=False,
                 train_b=True,
                 wi_init=None,   
                 wz_init=None,
                 wrec_init=None, 
                 h0_init=None, 
                 si_init=None,  
                 b_init=None, 
                 non_linearity=torch.nn.functional.relu,
                 behavior_outputs='sign'):
        """
        :param input_size: int
        :param observed_size: int
        :param hidden_size: int
        :param input_expansion_size: int or None
        :param noise_std: float
        :param alpha: float, value of dt / tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_wz: bool
        :param train_b: bool
        :param wi_init: 
            tuple of torch tensors of shapes (input_size, input_expansion_size) and (input_expansion_size, observed_size + hidden_size) 
            - or -
            torch tensor of shape (input_size, observed_size + hidden_size)
            - or -
            None
        :param wz_init: torch tensor of shape (observed_size + hidden_size, 1) or None
        :param wrec_init: torch tensor of shape (observed_size + hidden_size, observed_size + hidden_size) or None
        :param h0_init: torch tensor of shape (observed_size + hidden_size) or None
        :param si_init: 
            input scaling
            tuple of torch tensors of shapes (input_size) and (input_expansion_size) 
            - or -
            torch tensor of shape (input_size)
            - or -
            None
        :param b_init: torch tensor of shape (observed_size + hidden_size) or None
        :param non_linearity: torch module with a forward method
        """
        super(MyAltRNN, self).__init__()
        self.input_size = input_size
        self.observed_size = observed_size
        self.hidden_size = hidden_size
        self.network_size = observed_size + hidden_size
        self.input_expansion_size = input_expansion_size
        self.has_expansion_layer = input_expansion_size is not None
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_wz = train_wz
        self.train_b = train_b
        self.non_linearity = non_linearity
        self.behavior_outputs = behavior_outputs

        # Define parameters
        network_size = observed_size + hidden_size
        if not self.has_expansion_layer:
            self.wi = nn.Parameter(torch.Tensor(input_size, network_size))
            self.si = nn.Parameter(torch.Tensor(input_size))
            if train_wi:
                self.si.requires_grad = False
            else:
                self.wi.requires_grad = False
            if not train_si:
                self.si.requires_grad = False
        else:
            self.wi_1 = nn.Parameter(torch.Tensor(input_size, input_expansion_size))
            self.si_1 = nn.Parameter(torch.Tensor(input_size))
            self.wi_2 = nn.Parameter(torch.Tensor(input_expansion_size, network_size))
            self.si_2 = nn.Parameter(torch.Tensor(input_expansion_size))
            if train_wi:
                self.si_1.requires_grad = False
                self.si_2.requires_grad = False
            else:
                self.wi_1.requires_grad = False
                self.wi_2.requires_grad = False
            if not train_si:
                self.si_1.requires_grad = False
                self.si_2.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(network_size, network_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(network_size))
        if not train_b:
            self.b.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(network_size))
        if not train_h0:
            self.h0.requires_grad = False
        self.wz = nn.Parameter(torch.Tensor(network_size, 1))
        if not train_wz:
            self.wz.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                if not self.has_expansion_layer:
                    self.wi.normal_()
                else:
                    self.wi_1.normal_()
                    self.wi_2.normal_()
            else:
                if not self.has_expansion_layer:
                    self.wi.copy_(wi_init)
                else:
                    self.wi_1.copy_(wi_init[0])
                    self.wi_2.copy_(wi_init[1])
            if si_init is None:
                if not self.has_expansion_layer:
                    self.si.set_(torch.ones_like(self.si))
                else:
                    self.si_1.set_(torch.ones_like(self.si_1))
                    self.si_2.set_(torch.ones_like(self.si_2))
            else:
                if not self.has_expansion_layer:
                    self.si.copy_(si_init)
                else:
                    self.si_1.copy_(si_init[0])
                    self.si_2.copy_(si_init[1])
            if wrec_init is None:
                # self-connections are not allowed
                W_init = (rho / sqrt(network_size)) * torch.randn(network_size, network_size)
                W_init = W_init * (torch.ones(network_size, network_size) - torch.eye(network_size));
            else:
                W_init = wrec_init
                W_init = W_init * (torch.ones(network_size, network_size) - torch.eye(network_size))
            self.wrec.copy_(W_init)
            if b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(b_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
            if wz_init is None:
                self.wz.normal_(std=1 / network_size)
            else:
                self.wz.copy_(wz_init)
        self.wi_full = None
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        if not self.has_expansion_layer:
            self.wi_full = (self.wi.t() * self.si).t()
        else:
            self.wi_1_full = (self.wi_1.t() * self.si_1).t()
            self.wi_2_full = (self.wi_2.t() * self.si_2).t()

    def forward(self, input_, return_dynamics=False, initial_states=None, silenced_neurons=None):
        """
        :param input_: tensor of shape (batch_size, #timesteps, input_size)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial if desired
        :param silenced_neurons: None or list/array of integer indices of neurons to ablate if desired
        :return: if return_dynamics=False:
                     output tensor of shape (batch_size, #timesteps, observed_size + 1)
                 if return_dynamics=True:
                     (output tensor, trajectories tensor of shape (batch_size, #timesteps, observed_size + hidden_size))
        """
        batch_size = input_.shape[0]
        seq_len = input_.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(h + self.b)
        if silenced_neurons is not None:
            r[silenced_neurons] = 0
        z = torch.zeros(1)
        c = torch.zeros(1)
        self._define_proxy_parameters()
        W_out = torch.zeros(self.network_size, self.observed_size)
        W_out[:self.observed_size, :] = torch.eye(self.observed_size)
        noise = torch.randn(batch_size, seq_len, self.network_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.observed_size + 1, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.network_size, device=self.wrec.device)
            trajectories[:, 0, :] = h.clone()

        # simulation loop
        for i in range(seq_len):
            
            # stimulus
            if self.has_expansion_layer:
                stim_expand = torch.tanh( input_[:, i, :].matmul(self.wi_1_full) )
                stim = stim_expand.matmul(self.wi_2_full)
            else:
                stim = input_[:, i, :].matmul(self.wi_full)
                
            # current
            W = self.wrec * (torch.ones_like(self.wrec) - torch.eye(self.wrec.shape[0])) # ensure self-connections are not trained
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(W.t()) + stim)
           
            # firing rate
            r = self.non_linearity(h + self.b)
            if silenced_neurons is not None:
                r[:, silenced_neurons] = 0
            
            # behavior
            z = z + self.alpha * (-z + r @ self.wz)
            if self.behavior_outputs == 'sign':
                c = torch.tanh(z)
            elif self.behavior_outputs == 'probability':
                c = (1. + torch.tanh(z)) / 2.

            # output
            output[:, i, :self.observed_size] = r @ W_out
            output[:, i, -1] = c.flatten()

            if return_dynamics:
                trajectories[:, i + 1, :] = h.clone()
        
        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        if self.has_expansion_layer:
            wi = (self.wi_1, self.wi_2)
            si = (self.si_1, self.si_2)
        else:
            wi = self.wi
            si = self.si
        new_net = MyRNN(
            input_size=self.input_size,
            observed_size=self.observed_size,
            hidden_size=self.hidden_size,
            input_expansion_size=self.input_expansion_size,
            noise_std=self.noise_std,
            alpha=self.alpha, 
            rho=self.rho,
            train_wi=self.train_wi,  
            train_wrec=self.train_wrec, 
            train_h0=self.train_h0, 
            train_si=self.train_si, 
            train_wz=self.train_wz,
            train_b=self.train_b,
            wi_init=wi,   
            wz_init=self.wz,
            wrec_init=self.wrec, 
            h0_init=self.h0, 
            si_init=si, 
            b_init=self.b,
            non_linearity=self.non_linearity
        )
        return new_net    
    
    
class ReLUX(nn.Module):
    def __init__(self, max_):
        super(ReLUX, self).__init__()
        self._max = max_
        
    def forward(self, x):
        return torch.clamp(x, 0, self._max)

    
def my_loss_mse(output, target, mask, decision_time, lambda_neural, lambda_behavioral, lambda_bias_correction, normalize=False):
    """
    modified from 'loss_mse' in https://github.com/adrian-valente/lowrank_inference/blob/main/low_rank_rnns/modules.py
    """
    if normalize:
        # normalize by time-series 2-norms so every neuron contributes the same to loss
        scale = torch.sqrt(torch.sum(target.pow(2), dim=1, keepdim=True))
        scale[scale == 0] = 1
        loss_tensor = (mask * (target - output) / scale).pow(2)
        loss_neural = lambda_neural * loss_tensor[:, :, :-1].sum() / loss_tensor.shape[0] / (loss_tensor.shape[2] - 1)
        loss_behavioral = lambda_behavioral * loss_tensor[:, :, -1].sum() / loss_tensor.shape[0]
    else:
        # mean squared error (no normalization, high firing rate neurons contribute more loss)
        loss_tensor = (mask * (target - output)).pow(2)
        scale_neural = max(mask[:, :, :-1].sum().item(), 1)
        scale_behavioral = max(mask[:, :, -1].sum().item(), 1)
        loss_neural = lambda_neural * loss_tensor[:, :, :-1].sum() / scale_neural
        loss_behavioral = lambda_behavioral * loss_tensor[:, :, -1].sum() / scale_behavioral
    loss_bias = lambda_bias_correction * output[:, decision_time, -1].mean().pow(2)
    return loss_neural + loss_behavioral + loss_bias, loss_neural, loss_behavioral


def train(
    net, 
    _input, 
    _target, 
    _mask, 
    n_epochs, 
    decision_time,
    lambda_neural,
    lambda_behavioral,
    lambda_bias_correction,
    lr=1e-2, 
    batch_size=32, 
    plot_learning_curve=False, 
    plot_gradient=False, 
    clip_gradient=None, 
    early_stop=None, 
    keep_best=False, 
    cuda=False, 
    resample=False, 
    initial_states=None, 
    normalize_loss=False,
    print_every=1,
    verbose=True):
    """
    modified from 'train' in https://github.com/adrian-valente/lowrank_inference/blob/main/low_rank_rnns/modules.py
    """
    
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            if cuda == True:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda}')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input_ = _input.to(device=device, dtype=torch.float32)   # TODO do we need _input
    target = _target.to(device=device, dtype=torch.float32)
    mask = _mask.to(device=device, dtype=torch.float32)
    if initial_states is not None:
        initial_states = initial_states.to(device=device, dtype=torch.float32)

    # Initialize setup to keep best network
    with torch.no_grad():
        output = net(input_, initial_states=initial_states)
        initial_loss_total, initial_loss_neural, initial_loss_behavioral = my_loss_mse(
            output, target, mask, decision_time, 
            lambda_neural, lambda_behavioral, lambda_bias_correction, normalize=normalize_loss)
        if verbose:
            print("initial loss: neural: %.3f behavioral: %.3f" % 
                  (initial_loss_neural.item(), initial_loss_behavioral.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss_total.item()

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        losses = []  # losses over the whole epoch
        losses_neural = []
        losses_behavioral = []
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input_[random_batch_idx]
            if initial_states is not None:
                output = net(batch, initial_states=initial_states[random_batch_idx])
            else:
                output = net(batch)
            loss, loss_neural, loss_behavioral = my_loss_mse(
                output, target[random_batch_idx], mask[random_batch_idx], decision_time, 
                lambda_neural, lambda_behavioral, lambda_bias_correction, normalize=normalize_loss)
            losses.append(loss.item())
            losses_neural.append(loss_neural.item())
            losses_behavioral.append(loss_behavioral.item())
            all_losses.append(loss.item())
            net_before_update = net.clone()
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These lines important to prevent memory leaks
            loss.detach_()
            loss_neural.detach_()
            loss_behavioral.detach_()
            output.detach_()
            if resample:
                net.resample_basis()

        if keep_best and np.mean(losses) < best_loss:
            best = net_before_update.clone()
            best_loss = np.mean(losses)
            if verbose and ((epoch % print_every) == 0):
                print("epoch %d:  loss neural=%.3f loss behavioral=%.3f (took %.2f s) *" % 
                      (epoch, np.mean(losses_neural), np.mean(losses_behavioral), time.time() - begin))
        else:
            if verbose and ((epoch % print_every) == 0):
                print("epoch %d:  loss neural=%.3f loss behavioral=%.3f (took %.2f s)" % 
                      (epoch, np.mean(losses_neural), np.mean(losses_behavioral), time.time() - begin))
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())
    else:
        net.load_state_dict(net_before_update.state_dict())
        
    print('Done.')
    

def multi_model_simulation(
    all_models, 
    inputs, 
    unique_stims, 
    t_decision_window,
    n_trials_per_stim, 
    sigma,
    silenced=None,
    label_data=None, 
    responses_over_time=False,
    fit_options=None,
    inds_L=None, 
    inds_R=None,
    responses_beginning_end=False,
    window_beginning=None,
    window_end=None,
    random_seed=9999995):
    """
    Simulate trials (optionally, with noise) for all trained models (optionally, with specific sub-populations ablated)
    Optionally, re-run the coding type analysis for each
    """
    ## parse inputs
    if (silenced is not None) and (label_data is None):
        raise ValueError("If a 'silenced' population is provided, you must also provide 'label_data'")
    if (responses_over_time or responses_beginning_end) and (fit_options is None):
        raise ValueError("Response profile analysis was requested but no 'fit_options' were provided")
    if responses_over_time and any([inds_L is None, inds_R is None]):
        raise ValueError("If 'responses_over_time' is True, you must provide 'inds_L' and 'inds_R'")
    if responses_beginning_end and any([window_beginning is None, window_end is None]):
        raise ValueError("If 'responses_beginning_end' is True, you must provide 'window_beginning' and 'window_end'")
    n_bins = inputs.shape[1]
    n_stims = len(unique_stims)
    if type(sigma) in [int, float]:
        sigma_all = [sigma for i in range(len(all_models))]
    else:
        sigma_all = sigma
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed);
    
    ## init
    Xs = []
    all_choice_trajs = []
    y_stims = []
    y_lefts = []
    y_outcomes = []
    all_labels = []
    all_labels_beginning = []
    all_labels_end = []
    all_classifications = []
    all_classifications_beginning = []
    all_classifications_end = []
    all_p_left = []
    all_accuracies = []
    all_psths_correct = [] # average over correct trials
    all_psths_error = [] # average over error trials
    all_psths = [] # average over all trials

    for i_net, net in enumerate(all_models):

        r_out = np.zeros((n_stims * n_trials_per_stim, net.network_size, n_bins))
        choice_trajs = np.zeros((n_stims * n_trials_per_stim, n_bins))
        y_conc_out = np.zeros(n_stims * n_trials_per_stim)
        y_choice_out = np.zeros(n_stims * n_trials_per_stim)
        y_outcome_out = np.zeros(n_stims * n_trials_per_stim)
        n_left = np.zeros(n_stims)
        n_correct = np.zeros(n_stims)
        psths = [np.zeros((n_stims, n_bins)) for i_neuron in range(net.network_size)]
        psths_correct = [np.zeros((n_stims, n_bins)) for i_neuron in range(net.network_size)]
        psths_error = [np.zeros((n_stims, n_bins)) for i_neuron in range(net.network_size)]
        i_trial = 0
        
        for rep in range(n_trials_per_stim):
            noise = sigma_all[i_net] * torch.randn_like(inputs)
            output_new, traj_new = net(inputs + noise, initial_states=None, return_dynamics=True, 
                                       silenced_neurons=(None if silenced is None else label_data[i_net][silenced]))
            r_block = net.non_linearity(traj_new.detach()[:, 1:, :] + net.b.detach()).numpy()
            choice_traj_block = output_new.detach().numpy()[:, :, -1]
            if silenced is not None:
                r_block[:, :, label_data[i_net][silenced]] = 0
            for i_stim, stim in enumerate(unique_stims):
                target = (1.0 if (stim > 50) else 0.0)
                choice_traj = choice_traj_block[i_stim, :]
                decision = float(choice_traj[t_decision_window].mean().item() > 0)
                n_left[i_stim] += decision
                if (decision == target): 
                    y_outcome_out[i_trial] = 1
                    n_correct[i_stim] += 1
                    for i_neuron in range(net.network_size):
                        output_neuron = r_block[i_stim, :, i_neuron]
                        psths_correct[i_neuron][i_stim, :] += output_neuron
                        psths[i_neuron][i_stim, :] += output_neuron
                else:
                    y_outcome_out[i_trial] = 0
                    for i_neuron in range(net.network_size):
                        output_neuron = r_block[i_stim, :, i_neuron]
                        psths_error[i_neuron][i_stim, :] += output_neuron
                        psths[i_neuron][i_stim, :] += output_neuron
                r_out[i_trial, :, :] = r_block.copy()[i_stim, :, :].T
                choice_trajs[i_trial, :] = choice_traj.copy()
                y_conc_out[i_trial] = stim
                y_choice_out[i_trial] = decision
                i_trial += 1
        p_left = n_left / n_trials_per_stim
        all_p_left.append(p_left)
        p_correct = p_left * (unique_stims > 50) + (1 - p_left) * (unique_stims < 50)
        all_accuracies.append(p_correct.mean())
        for i_neuron in range(net.network_size):
            for i, n in enumerate(n_correct):
                if (n != 0):
                    psths_correct[i_neuron][i, :] /= n
                else:
                    psths_correct[i_neuron][i, :] *= np.nan
                if ((n_trials_per_stim - n) != 0):
                    psths_error[i_neuron][i, :] /= (n_trials_per_stim - n)
                else:
                    psths_error[i_neuron][i, :] *= np.nan
            psths[i_neuron] /= n_trials_per_stim
        all_psths_correct.append(psths_correct.copy())
        all_psths_error.append(psths_error.copy())
        all_psths.append(psths.copy())
        Xs.append(r_out.copy())
        all_choice_trajs.append(choice_trajs.copy())
        y_stims.append(y_conc_out.copy())
        y_lefts.append(y_choice_out.copy())
        y_outcomes.append(y_outcome_out.copy())

        ## response profiles over time
        if responses_over_time:
            classifications_over_time = []
            labels_over_time = []
            for ind_L, ind_R in zip(inds_L, inds_R):
                classifications = []
                labels = []
                for i_neuron in range(net.network_size):
                    response_profile_correct = np.full(len(unique_stims), np.nan)
                    response_profile_error = np.full(len(unique_stims), np.nan)
                    for i_stim, stim in enumerate(unique_stims):
                        psth_correct = psths_correct[i_neuron][i_stim, :]
                        response_profile_correct[i_stim] = psth_correct[ind_L:ind_R].mean()
                        psth_error = psths_error[i_neuron][i_stim, :]
                        response_profile_error[i_stim] = psth_error[ind_L:ind_R].mean()
                    fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
                    classification = classify_fits(fits)
                    classifications.append(classification)
                    label = parse_label(classification, response_profile_correct, response_profile_error)
                    labels.append(label)
                classifications_over_time.append(classifications)
                labels_over_time.append(labels)
            all_classifications.append(classifications_over_time)
            all_labels.append(labels_over_time)
            
        ## response profiles beginning vs. end
        if responses_beginning_end:
            classifications_beginning = []
            labels_beginning= []
            classifications_end = []
            labels_end = []
            for i_neuron in range(net.network_size):
                # beginning
                response_profile_correct = np.full(n_stims, np.nan)
                response_profile_error = np.full(n_stims, np.nan)
                for i_stim, stim in enumerate(unique_stims):
                    psth_correct = psths_correct[i_neuron][i_stim, :]
                    response_profile_correct[i_stim] = psth_correct[window_beginning].mean()
                    psth_error = psths_error[i_neuron][i_stim, :]
                    response_profile_error[i_stim] = psth_error[window_beginning].mean()
                fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
                classification = classify_fits(fits)
                classifications_beginning.append(classification)
                label = parse_label(classification, response_profile_correct, response_profile_error)
                labels_beginning.append(label)
                # end
                response_profile_correct = np.full(n_stims, np.nan)
                response_profile_error = np.full(n_stims, np.nan)
                for i_stim, stim in enumerate(unique_stims):
                    psth_correct = psths_correct[i_neuron][i_stim, :]
                    response_profile_correct[i_stim] = psth_correct[window_end].mean()
                    psth_error = psths_error[i_neuron][i_stim, :]
                    response_profile_error[i_stim] = psth_error[window_end].mean()
                fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
                classification = classify_fits(fits)
                classifications_end.append(classification)
                label = parse_label(classification, response_profile_correct, response_profile_error)
                labels_end.append(label)
            all_classifications_beginning.append(classifications_beginning)
            all_labels_beginning.append(labels_beginning)
            all_classifications_end.append(classifications_end)
            all_labels_end.append(labels_end)
        
    output_dict = {
        'Xs': Xs,
        'y_stims': y_stims,
        'y_lefts': y_lefts,
        'y_outcomes': y_outcomes,
        'all_choice_trajs': all_choice_trajs,
        'all_labels': all_labels,
        'all_labels_beginning': all_labels_beginning,
        'all_labels_end': all_labels_end,
        'all_classifications': all_classifications,
        'all_classifications_beginning': all_classifications_beginning,
        'all_classifications_end': all_classifications_end,
        'all_p_left': all_p_left,
        'all_accuracies': all_accuracies,
        'all_psths': all_psths,
        'all_psths_correct': all_psths_correct,
        'all_psths_error': all_psths_error
    }
    return output_dict
        
    
def model_simulation(
    model, 
    inputs, 
    unique_stims, 
    t_decision_window,
    n_trials_per_stim, 
    sigma,
    silenced=None,
    label_data=None, 
    responses_over_time=False,
    fit_options=None,
    inds_L=None, 
    inds_R=None,
    responses_beginning_end=False,
    window_beginning=None,
    window_end=None,
    random_seed=9999995):
    """
    Simulate trials (optionally, with noise) for a trained model (optionally, with specific sub-populations ablated)
    Optionally, re-run the coding type analysis
    """
    ## parse inputs
    if (silenced is not None) and (label_data is None):
        raise ValueError("If a 'silenced' population is provided, you must also provide 'label_data'")
    if (responses_over_time or responses_beginning_end) and (fit_options is None):
        raise ValueError("Response profile analysis was requested but no 'fit_options' were provided")
    if responses_over_time and any([inds_L is None, inds_R is None]):
        raise ValueError("If 'responses_over_time' is True, you must provide 'inds_L' and 'inds_R'")
    if responses_beginning_end and any([window_beginning is None, window_end is None]):
        raise ValueError("If 'responses_beginning_end' is True, you must provide 'window_beginning' and 'window_end'")
    n_bins = inputs.shape[1]
    n_stims = len(unique_stims)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed);
    
    ## init
    X = np.zeros([n_stims * n_trials_per_stim, n_bins, model.network_size])
    choice_trajs = np.zeros([n_stims * n_trials_per_stim, n_bins])
    y_stim = []
    y_left = []
    y_outcome = []
    labels_over_time = []
    labels_beginning = []
    labels_end = []
    classifications_over_time = []
    classifications_beginning = []
    classifications_end = []
    
    i_trial = 0
    for rep in range(n_trials_per_stim):
        y_stim += list(unique_stims)
        noise = sigma * torch.randn_like(inputs)
        output, traj = model(inputs + noise, initial_states=None, return_dynamics=True, 
                                   silenced_neurons=(None if silenced is None else label_data[silenced]))
        r_traj = model.non_linearity(traj.detach()[:, 1:, :] + model.b.detach()).numpy()
        choice_traj = output.detach().numpy()[:, :, -1]
        if silenced is not None:
            r_traj[:, :, label_data[silenced]] = 0
        X[i_trial:(i_trial + n_stims), :, :] = r_traj.copy()
        choice_trajs[i_trial:(i_trial + n_stims), :] = choice_traj.copy()
        for i_stim, stim in enumerate(unique_stims):
            target = (1.0 if (stim > 50) else 0.0)
            decision = float(choice_traj[i_stim, t_decision_window].mean().item() > 0)
            y_left.append(decision)
            if (decision == target): 
                y_outcome.append(1)
            else:
                y_outcome.append(0)
        i_trial += n_stims
    y_stim = np.array(y_stim)
    y_left = np.array(y_left)
    y_outcome = np.array(y_outcome)
        
    p_left = [y_left[y_stim == stim].sum() / n_trials_per_stim for stim in unique_stims]
    accuracy = 100 * y_outcome.sum() / (n_stims * n_trials_per_stim)
    
    psths_correct = [] # average over correct trials
    psths_error = [] # average over error trials
    psths = [] # average over all trials
    for i_neuron in range(model.network_size):
        psth_correct = np.full([n_stims, n_bins], np.nan)
        psth_error = np.full([n_stims, n_bins], np.nan)
        psth = np.full([n_stims, n_bins], np.nan)
        for i_stim, stim in enumerate(unique_stims):
            mask_correct = (y_stim == stim) & (y_outcome == 1)
            n_correct = mask_correct.sum()
            mask_error = (y_stim == stim) & (y_outcome == 0)
            n_error = mask_error.sum()
            psth[i_stim, :] = X[y_stim == stim, :, i_neuron].mean(axis=0)
            if (n_correct != 0):
                psth_correct[i_stim, :] = X[mask_correct, :, i_neuron].mean(axis=0)
            if (n_error != 0):
                psth_error[i_stim, :] = X[mask_error, :, i_neuron].mean(axis=0)
        psths.append(psth)
        psths_correct.append(psth_correct)
        psths_error.append(psth_error)

    ## response profiles over time
    if responses_over_time:
        for ind_L, ind_R in zip(inds_L, inds_R):
            classifications = []
            labels = []
            for i_neuron in range(model.network_size):
                response_profile_correct = np.full(n_stims, np.nan)
                response_profile_error = np.full(n_stims, np.nan)
                for i_stim, stim in enumerate(unique_stims):
                    psth_correct = psths_correct[i_neuron][i_stim, :]
                    response_profile_correct[i_stim] = psth_correct[ind_L:ind_R].mean()
                    psth_error = psths_error[i_neuron][i_stim, :]
                    response_profile_error[i_stim] = psth_error[ind_L:ind_R].mean()
                fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
                classification = classify_fits(fits)
                classifications.append(classification)
                label = parse_label(classification, response_profile_correct, response_profile_error)
                labels.append(label)
            classifications_over_time.append(classifications)
            labels_over_time.append(labels)

    ## response profiles beginning vs. end
    if responses_beginning_end:
        for i_neuron in range(model.network_size):
            # beginning
            response_profile_correct = np.full(n_stims, np.nan)
            response_profile_error = np.full(n_stims, np.nan)
            for i_stim, stim in enumerate(unique_stims):
                psth_correct = psths_correct[i_neuron][i_stim, :]
                response_profile_correct[i_stim] = psth_correct[window_beginning].mean()
                psth_error = psths_error[i_neuron][i_stim, :]
                response_profile_error[i_stim] = psth_error[window_beginning].mean()
            fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
            classification = classify_fits(fits)
            classifications_beginning.append(classification)
            label = parse_label(classification, response_profile_correct, response_profile_error)
            labels_beginning.append(label)
            # end
            response_profile_correct = np.full(n_stims, np.nan)
            response_profile_error = np.full(n_stims, np.nan)
            for i_stim, stim in enumerate(unique_stims):
                psth_correct = psths_correct[i_neuron][i_stim, :]
                response_profile_correct[i_stim] = psth_correct[window_end].mean()
                psth_error = psths_error[i_neuron][i_stim, :]
                response_profile_error[i_stim] = psth_error[window_end].mean()
            fits = fit_shape_templates(unique_stims, response_profile_correct, fit_options)
            classification = classify_fits(fits)
            classifications_end.append(classification)
            label = parse_label(classification, response_profile_correct, response_profile_error)
            labels_end.append(label)
        
    output_dict = {
        'X': X,
        'choice_trajs': choice_trajs,
        'y_stim': y_stim,
        'y_left': y_left,
        'y_outcome': y_outcome,
        'p_left': p_left,
        'accuracy': accuracy,
        'psths': psths,
        'psths_correct': psths_correct,
        'psths_error': psths_error,
        'labels_over_time': labels_over_time,
        'classifications_over_time': classifications_over_time,
        'labels_beginning': labels_beginning,
        'classifications_beginning': classifications_beginning,
        'labels_end': labels_end,
        'classifications_end': classifications_end,
    }
    return output_dict