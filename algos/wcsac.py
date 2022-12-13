from cmath import phase
import copy
from distutils.command.config import config
from statistics import mean
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from algos.writer import writeAgent

from typing import Tuple
from .algo_utils import int2D_to_grouponehot
from itertools import chain

from scipy.stats import norm
import torch.distributions as tdist


class Agent:
    def __init__(self, config, env):

        torch.manual_seed(config['seed'])

        self.lr = config['algo']['lr']
        self.smooth = config['algo']['smooth']
        self.discount = config['algo']['discount']
        self.alpha = config['algo']['alpha']
        self.batch_size = config['algo']['batch_size']
        self.dims_hidden_neurons = config['algo']['dims_hidden_neurons']
        self.damp_scale = config['algo']['damp_scale']
        self.cost_limit = config['algo']['cost_limit']
        self.init_temperature = config['algo']['init_temperature']
        self.betas = config['algo']['betas']
        self.cost_lr_scale = config['algo']['lr_scale']

        self.dim_state = env.dim_state
        self.dims_action = env.dims_action
        self.num_device = len(self.dims_action)
        self.online_training_steps = config['algo']['online_training_steps']
        self.max_episode_len = config['algo']['max_episode_len']

        self.actor = ActorNet(dim_state=self.dim_state,
                              dims_action=self.dims_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1 = QCriticNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2 = QCriticNet(dim_state=self.dim_state,
                             dims_action=self.dims_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1_tar = QCriticNet(dim_state=self.dim_state,
                                 dims_action=self.dims_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2_tar = QCriticNet(dim_state=self.dim_state,
                                 dims_action=self.dims_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.critic = QCriticNet(dim_state=self.dim_state,
                                 dims_action=self.dims_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.Qc = QConstraintNet(dim_state=self.dim_state,
                                 dims_action=self.dims_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.Qc_tar = QConstraintNet(dim_state=self.dim_state,
                                     dims_action=self.dims_action,
                                     dims_hidden_neurons=self.dims_hidden_neurons)
        self.Vc = VConstraintNet(dim_state=self.dim_state,
                                 dims_hidden_neurons=self.dims_hidden_neurons)
        self.Vc_tar = copy.deepcopy(self.Vc)

        self.safety_critic = QConstraintNet(dim_state=self.dim_state,
                                                 dims_action=self.dims_action,
                                                 dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.lr, betas=self.betas)

        self.all_critics_optimizer = torch.optim.AdamW(chain(self.critic.parameters(), self.safety_critic.parameters()),
                                                       lr=self.lr, betas=self.betas)

        self.optimizer_Q1 = torch.optim.AdamW(self.Q1.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_Q2 = torch.optim.AdamW(self.Q2.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_Qc = torch.optim.AdamW(self.Qc.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_Vc = torch.optim.AdamW(self.Vc.parameters(), lr=self.lr, betas=self.betas)

        # self.step_policy = config['algo']['step_policy']
        self.algo = config['algo']['algo']
        self.writer_counter = 0

        # beta in paper - adaptive entropy
        self.log_beta = torch.tensor(np.log(np.clip(self.init_temperature, 1e-8, 1e8)))
        self.log_beta.requires_grad = True
        # kappa in paper - safety weights
        self.log_kappa = torch.tensor(np.log(np.clip(self.init_temperature, 1e-8, 1e8)))
        self.log_kappa.requires_grad = True

        # beta (entropy) and kappa (safty) optimizers
        self.beta_optimizer = torch.optim.AdamW([self.log_beta], lr=self.lr, betas=self.betas)
        self.kappa_optimizer = torch.optim.AdamW([self.log_kappa], lr=self.lr * self.cost_lr_scale, betas=self.betas)

        # Set target entropy to -|A|
        self.target_entropy = -self.num_device

        # Set target cost
        self.target_cost = (
            # max max_episode_len =1000 in original paper
                self.cost_limit * (1 - self.discount ** self.max_episode_len) / (
                    1 - self.discount) / self.max_episode_len
        )

    def update(self, replay):
        t = replay.sample(self.batch_size)

        # sample action
        action_sample, action_sample_prob = self.sample_action_with_prob(t.state)
        action_sample_onehot = int2D_to_grouponehot(indices=action_sample, depths=self.dims_action)

        # sample next action
        next_action_sample, next_action_sample_prob = self.sample_action_with_prob(t.next_state)
        next_action_sample_onehot = int2D_to_grouponehot(indices=next_action_sample, depths=self.dims_action)

        Vcp = self.Vc_tar(t.next_state).detach()
        Vcp = torch.clamp(Vcp, min=1e-8, max=1e8)

        # log sample action in next state
        log_action_sample_prob = torch.zeros_like(Vcp)
        for ii in range(self.num_device):
            log_action_sample_prob += torch.log(next_action_sample_prob[:, ii:ii + 1] + 1e-10)

        # compute Q target and V target
        with torch.no_grad():
            action_onehot = int2D_to_grouponehot(indices=t.action.long(), depths=self.dims_action)
            # previos code
            # Q_target = t.reward_loss - self.lagrange_multiplier * t.reward_constraint + self.discount*(~t.done)*Vp
            # Qc_target = t.reward_constraint + self.discount*(~t.done)*Vcp
            # V_target = torch.min(self.Q1(t.state, action_sample_onehot),
            #                           self.Q2(t.state, action_sample_onehot)) - \
            #            self.alpha * log_action_sample_prob
            # Vc_target = torch.max(self.Q1(t.state, action_sample_onehot),
            #                           self.Q2(t.state, action_sample_onehot)) - \
            #            self.alpha * log_action_sample_prob

            # new code
            Q1_target = self.Q1_tar(t.next_state, next_action_sample_onehot)
            Q2_target = self.Q2_tar(t.next_state, next_action_sample_onehot)

            # V target is estimated from min Q_target
            V_target = torch.min(Q1_target, Q2_target) - self.log_beta.exp().detach() * log_action_sample_prob
            Q_target = t.reward_loss + self.discount * (~t.done) * V_target
            # Q_target = Q_target.detach()

            # get current and next Qc values EQ (7)
            Qc_current = self.Qc(t.state, action_onehot)
            Qc_next = self.Qc_tar(t.next_state, next_action_sample_onehot)

            # get Vc values
            Vc_next = self.Vc_tar(t.next_state)
            Vc_next = torch.clamp(Vc_next, min=1e-8, max=1e8)

            # calculate constraint rewards EQ (8) - WCSAC paper
            Vc_target = t.reward_constraint ** 2 \
                        - Qc_current ** 2 \
                        + 2 * self.discount * t.reward_constraint * Qc_next \
                        + self.discount ** 2 * Vc_next \
                        + self.discount ** 2 * Qc_next ** 2

            Vc_target = torch.clamp(Vc_target.detach(), min=1e-8, max=1e8)

            Qc_target = t.reward_constraint + (self.discount * (~t.done) * Qc_next)
            Qc_target = Qc_target.detach()

        ############### update actor and alpha beta ###############

        # Reward Critic
        Q1_actor = self.Q1(t.state, action_sample_onehot)
        Q1_actor = torch.clamp(Q1_actor, min=-1e8, max=1e8)
        Q2_actor = self.Q2(t.state, action_sample_onehot)
        Q2_actor = torch.clamp(Q2_actor, min=-1e8, max=1e8)
        Q_actor = torch.min(Q1_actor, Q2_actor)
        Q_actor = torch.clamp(Q_actor, min=-1e8, max=1e8)

        # Safety Critic with actor actions
        Qc_actor = self.Qc(t.state, action_sample_onehot)
        Vc_actor = self.Vc(t.state)
        Vc_actor = torch.clamp(Vc_actor, min=1e-8, max=1e8)

        Vc_current = self.Vc(t.state)
        Vc_current = torch.clamp(Vc_current, min=1e-8, max=1e8)

        # norm is Normal(mean = 0,sigma = 1), norm.ppf - inverse cdf
        # pdf_cdf = self.alpha**(-1) * norm.pdf(norm.ppf(self.alpha)) # maybe try logpdf (Dmajan Code)
        normal = tdist.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        pdf_cdf = normal.log_prob(normal.icdf(torch.tensor(self.alpha))).exp() / self.alpha
        # original impl normal.log_prob(normal.icdf(torch.tensor(self.risk_level))).exp() / self.risk_level (Orignal code)
        # WCSAC paper Equation (9)
        cvar = Qc_current + pdf_cdf * torch.sqrt(Vc_current)

        # damp is introduced in
        damp = self.damp_scale * torch.mean(self.target_cost - cvar)

        # calculate log probability of sample action
        log_action_prob = torch.zeros_like(Vcp)
        for ii in range(self.num_device):
            log_action_prob += torch.log(action_sample_prob[:, ii:ii + 1] + 1e-10)

        # Actor Loss
        loss_actor = torch.mean(
            self.log_beta.exp().detach() * log_action_prob
            - Q_actor
            + (self.log_kappa.exp().detach() - damp) * (Qc_actor + pdf_cdf * torch.sqrt(Vc_actor))
        )

        # Reward Critic Loss
        loss_Q1 = torch.mean((self.Q1(t.state, action_onehot) - Q_target) ** 2)
        loss_Q2 = torch.mean((self.Q2(t.state, action_onehot) - Q_target) ** 2)
        critic_loss = loss_Q1 + loss_Q2

        # Safety Critic Loss
        loss_Qc = torch.mean((self.Qc(t.state, action_onehot) - Qc_target) ** 2)
        # loss_Vc = torch.mean(Vc_current + Vc_target - torch.sign(Vc_target * Vc_current) * 2*torch.sqrt(abs(Vc_target * Vc_current)))
        loss_Vc = torch.mean(Vc_current + Vc_target - 2 * torch.sqrt(abs(Vc_target * Vc_current)))
        safety_critic_loss = loss_Qc + loss_Vc

        # Jointly optimize Reward and Safety Critics
        total_loss = critic_loss + safety_critic_loss

        self.all_critics_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.all_critics_optimizer.step()

        self.optimizer_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        self.optimizer_actor.step()

        self.optimizer_Q1.zero_grad()
        loss_Q1.backward()
        self.optimizer_Q1.step()

        self.optimizer_Qc.zero_grad()
        loss_Qc.backward()
        self.optimizer_Qc.step()

        self.optimizer_Q2.zero_grad()
        loss_Q2.backward()
        self.optimizer_Q2.step()

        self.optimizer_Vc.zero_grad()
        loss_Vc.backward()
        self.optimizer_Vc.step()

        # train beta and kappa
        self.beta_optimizer.zero_grad()
        beta_loss = torch.mean(self.log_beta.exp() * (-log_action_prob - self.target_entropy).detach())
        beta_loss.backward()
        self.beta_optimizer.step()

        self.kappa_optimizer.zero_grad()
        kappa_loss = torch.mean(self.log_kappa.exp() * (self.target_cost - cvar).detach())
        kappa_loss.backward()
        self.kappa_optimizer.step()

        # update V target and Vc target parameters
        # with torch.no_grad():
        # for p, p_tar in zip(self.Q1.parameters(), self.Q1_tar.parameters()):
        # p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))
        # for p, p_tar in zip(self.Q2.parameters(), self.Q2_tar.parameters()):
        # p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))
        # for p, p_tar in zip(self.Vc.parameters(), self.Vc_tar.parameters()):
        # p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))
        # for p, p_tar in zip(self.Qc.parameters(), self.Qc_tar.parameters()):
        # p_tar.data.copy_(p_tar * self.smooth + p * (1-self.smooth))

        # add to tensorboard
        writeAgent(self.log_beta, self.writer_counter, self.algo, 'log beta')
        writeAgent(self.log_kappa, self.writer_counter, self.algo, 'log kappa')
        writeAgent(torch.mean(Vc_current), self.writer_counter, self.algo, 'Vc_current mean')
        writeAgent(torch.min(Vc_current), self.writer_counter, self.algo, 'Vc_current min')
        writeAgent(torch.mean(Vc_target), self.writer_counter, self.algo, 'Vc_target mean')
        writeAgent(torch.min(Vc_target), self.writer_counter, self.algo, 'Vc_target min')
        writeAgent(torch.mean(Qc_actor), self.writer_counter, self.algo, 'Qc_actor mean')
        writeAgent(torch.mean(Q_actor), self.writer_counter, self.algo, 'Q_actor mean')
        writeAgent(torch.mean(Vc_actor), self.writer_counter, self.algo, 'Vc_actor mean')
        writeAgent(torch.mean(Qc_target), self.writer_counter, self.algo, 'Qc_target mean')
        writeAgent(loss_Vc, self.writer_counter, self.algo, 'loss Vc')
        writeAgent(loss_Qc, self.writer_counter, self.algo, 'loss Qc')
        writeAgent(loss_actor, self.writer_counter, self.algo, 'loss actor')
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
