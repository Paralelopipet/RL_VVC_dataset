from cmath import phase
from distutils.command.config import config
from statistics import mean
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from algos.writer import writeAgent

from typing import Tuple
from .algo_utils import int2D_to_grouponehot

from scipy.stats import norm

class Agent:
    def __init__(self, config, env):

        torch.manual_seed(config['seed'])

        self.lr = config['algo']['lr']
        self.smooth = config['algo']['smooth']
        self.discount = config['algo']['discount']
        self.alpha = config['algo']['alpha']
        self.batch_size = config['algo']['batch_size']
        self.dims_hidden_neurons = config['algo']['dims_hidden_neurons']

        self.dim_state = env.dim_state
        self.dims_action = env.dims_action
        self.num_device = len(self.dims_action)

        self.actor = ActorNet(dim_state=self.dim_state,
                              dims_action=self.dims_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1 = QCriticNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2 = QCriticNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.V = VCriticNet(dim_state=self.dim_state,
                            dims_hidden_neurons=self.dims_hidden_neurons)
        self.V_tar = VCriticNet(dim_state=self.dim_state,
                                dims_hidden_neurons=self.dims_hidden_neurons)

        self.Qc1 = QConstraintNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Qc2 = QConstraintNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Vc = VConstraintNet(dim_state=self.dim_state,
                            dims_hidden_neurons=self.dims_hidden_neurons)
        self.Vc_tar = VConstraintNet(dim_state=self.dim_state,
                                dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)
        self.optimizer_V = torch.optim.Adam(self.V.parameters(), lr=self.lr)
        self.optimizer_Qc1 = torch.optim.Adam(self.Qc1.parameters(), lr=self.lr)
        self.optimizer_Qc2 = torch.optim.Adam(self.Qc2.parameters(), lr=self.lr)
        self.optimizer_Vc = torch.optim.Adam(self.Vc.parameters(), lr=self.lr)


        self.lagrange_multiplier = torch.tensor(config['algo']['lagrange_multiplier'], requires_grad=True)#
        self.lagrange_optimiser = torch.optim.Adam([self.lagrange_multiplier], lr=config['algo']['step_lagrange'])
        self.step_policy = config['algo']['step_policy']
        self.algo = config['algo']['algo']

        self.writer_counter = 0

    def update(self, replay):
        t = replay.sample(self.batch_size)

        # sample action
        action_sample, action_sample_prob = self.sample_action_with_prob(t.state)
        action_sample_onehot = int2D_to_grouponehot(indices=action_sample, depths=self.dims_action)
        
        Vcp = self.Vc_tar(t.next_state).detach()
        Vp = self.V_tar(t.next_state).detach()

        log_action_sample_prob = torch.zeros_like(Vp)
        for ii in range(self.num_device):
            log_action_sample_prob += torch.log(action_sample_prob[:, ii:ii+1] + 1e-10)

        # compute Q target and V target
        with torch.no_grad():
            action_onehot = int2D_to_grouponehot(indices=t.action.long(), depths=self.dims_action)
            Q_target = t.reward_loss - self.lagrange_multiplier * t.reward_constraint + self.discount*(~t.done)*Vp
            Qc_target = t.reward_constraint + self.discount*(~t.done)*Vcp
            V_target = torch.min(self.Q1(t.state, action_sample_onehot),
                                      self.Q2(t.state, action_sample_onehot)) - \
                       self.alpha * log_action_sample_prob
            Vc_target = torch.max(self.Q1(t.state, action_sample_onehot),
                                      self.Q2(t.state, action_sample_onehot)) - \
                       self.alpha * log_action_sample_prob

       

        # construct loss functions
        loss_Q1 = torch.mean((self.Q1(t.state, action_onehot) - Q_target) ** 2)
        loss_Qc1 = torch.mean((self.Qc1(t.state, action_onehot) - Qc_target) ** 2)
        loss_Q2 = torch.mean((self.Q2(t.state, action_onehot) - Q_target) ** 2)
        loss_Qc2 = torch.mean((self.Qc2(t.state, action_onehot) - Qc_target) ** 2)
        loss_V = torch.mean((self.V(t.state) - V_target) ** 2)
        loss_Vc = torch.mean((self.Vc(t.state) - Vc_target) ** 2)
        with torch.no_grad():
            V = self.V(t.state)
            Q = self.Q1(t.state, action_sample_onehot)

        ########### WCSAC CHANGES ##############
        # Gamma function
        self.gamma = Q + self.alpha**(-1) * norm.pdf(norm.ppf(self.alpha)) * np.sqrt(V)

        Q_probabilities = 0
        for 
        with torch.no_grad():
            Q_target = t.reward_constraint + self.discount* + self.lagrange_multiplier *  + self.discount*(~t.done)*Vp
            





        #########################################
        
        objective_actor = torch.mean(log_action_sample_prob * (Q - V - self.alpha * log_action_sample_prob.detach()))
        loss_actor = - self.step_policy * objective_actor

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.optimizer_Q1.zero_grad()
        loss_Q1.backward()
        self.optimizer_Q1.step()

        self.optimizer_Qc1.zero_grad()
        loss_Qc1.backward()
        self.optimizer_Qc1.step()

        self.optimizer_Q2.zero_grad()
        loss_Q2.backward()
        self.optimizer_Q2.step()

        self.optimizer_Qc2.zero_grad()
        loss_Qc2.backward()
        self.optimizer_Qc2.step()

        self.optimizer_V.zero_grad()
        loss_V.backward()
        self.optimizer_V.step()

        self.optimizer_Vc.zero_grad()
        loss_Vc.backward()
        self.optimizer_Vc.step()

        # update V target and Vc target parameters
        with torch.no_grad():
            for p, p_tar in zip(self.V.parameters(), self.V_tar.parameters()):
                p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))
            for p, p_tar in zip(self.Vc.parameters(), self.Vc_tar.parameters()):
                p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))

        # number of expected volatage violations
        VcExp = 0
        # update lagrange multiplier   
        self.lagrange_optimiser.zero_grad()
        #lambda loss calc
        lambda_loss = torch.mean(self.Vc(t.state) - VcExp)
        self.log_lam = torch.nn.functional.softplus(self.lagrange_multiplier)
        lambda_loss =  self.log_lam*lambda_loss.detach()
        lambda_loss = lambda_loss.sum(dim=-1)
        lambda_loss.backward()
        self.lagrange_optimiser.step()

        # Safety measure - Gamma - Γ_policy
        self.gamma = 0
        
        #add to tensorboard
        writeAgent(self.lagrange_multiplier, self.writer_counter, self.algo)    
        self.writer_counter = self.writer_counter + 1     

    def sample_action_with_prob(self, state: torch.Tensor):
        prob_all = self.actor(state)
        a = []
        prob = []
        for ii in range(self.num_device):
            samples = Categorical(prob_all[ii]).sample()
            a.append(samples)
            prob.append(prob_all[ii][range(torch.numel(samples)), samples])
        return torch.stack(a, dim=1), torch.stack(prob, dim=1)

    def act_probabilistic(self, state: torch.Tensor):
        prob_all = self.actor(state)
        a = []
        for ii in range(self.num_device):
            samples = Categorical(prob_all[ii]).sample().item()
            a.append(samples)
        return np.array(a)

    def act_deterministic(self, state: torch.Tensor):
        prob_all = self.actor(state)
        a = []
        for ii in range(self.num_device):
            samples = torch.argmax(prob_all[ii])
            a.append(samples)
        return np.array(a)


class ActorNet(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_action: Tuple[int],
                 dims_hidden_neurons: Tuple[int]):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dims_action = dims_action
        self.num_device = len(self.dims_action)

        n_neurons = (dim_state,) + dims_hidden_neurons + (sum(dims_action),)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.logits = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.logits.weight)
        torch.nn.init.zeros_(self.logits.bias)

        self.softmax = nn.Softmax(dim=1)  # dim is the dimension of features

    def forward(self, state: torch.Tensor):
        # ordinal encoding policy network
        x = state
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        logits = self.logits(x)
        sig_logits = torch.sigmoid(logits)
        sig_logits_per_device = torch.split(sig_logits, self.dims_action, dim=1)  # a tuple of torch.tensor
        transformed_logits_per_device = ()
        output_per_device = ()
        for ii in range(self.num_device):
            transformed_logits = torch.cumsum(torch.log(sig_logits_per_device[ii] + 1e-10), dim=1) + \
                                 torch.flip(torch.cumsum(torch.flip(torch.log(1 - sig_logits_per_device[ii] + 1e-10),
                                                                    dims=[1]),
                                                         dim=1), dims=[1]) - \
                                 torch.log(1 - sig_logits_per_device[ii] + 1e-10)
            transformed_logits_per_device += (transformed_logits,)
            output_per_device += (self.softmax(transformed_logits),)
        return output_per_device


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_action: Tuple[int],
                 dims_hidden_neurons: Tuple[int]):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dims_action = dims_action

        n_neurons = (dim_state + sum(dims_action),) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat((state, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


class VCriticNet(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_hidden_neurons: Tuple[int]):
        super(VCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)

        n_neurons = (dim_state,) + dims_hidden_neurons + (1,)
        for i, (fan_in, fan_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(fan_in, fan_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor):
        x = state
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)




class QConstraintNet(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_action: Tuple[int],
                 dims_hidden_neurons: Tuple[int]):
        super(QConstraintNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dims_action = dims_action

        n_neurons = (dim_state + sum(dims_action),) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat((state, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


class VConstraintNet(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_hidden_neurons: Tuple[int]):
        super(VConstraintNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)

        n_neurons = (dim_state,) + dims_hidden_neurons + (1,)
        for i, (fan_in, fan_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(fan_in, fan_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
    def forward(self, state: torch.Tensor):
        x = state
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)
