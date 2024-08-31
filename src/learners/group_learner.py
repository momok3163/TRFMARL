import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.group import Mixer as GroupMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from utils.th_utils import get_parameters_num

import torch
import torch.optim as optim
from torch.nn import functional as F


class GROUPLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.params = list(self.mac.parameters())

        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')

        if args.mixer == "group":
            self.mixer = GroupMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0
#la-smac
        self.start_anneal_time = 5e6
        self.init_anneal_time = False
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies

        self.eval_model_env = Predict_Network(args,
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)
        self.target_model_env = Predict_Network(args,
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)

        self.target_model_env.load_state_dict(self.eval_model_env.state_dict())
        self.Target_update = False
        if args.use_cuda:
            self.eval_model_env.cuda()
            self.target_model_env.cuda()
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        state = batch["state"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        visible = batch['visible_matrix'][:, :-1]

        b, t, a, _ = batch["obs"][:, :-1].shape
        actions_onehot = (batch["actions_onehot"][:, :-1])
        model_s = th.cat((state, actions_onehot.reshape(b, t, -1)), dim=-1)
        model_opp_s = batch['extrinsic_state'][:, 1:]
        intrinsic_mask = mask.clone()

        loss_model_list = []
        for _ in range(self.args.predict_epoch):
            loss_model = self.eval_model_env.update(
                model_s, model_opp_s, mask)
            loss_model_list.append(loss_model)
            
        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden = []
        mac_group_state = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_hidden.append(self.mac.hidden_states)
            mac_group_state.append(self.mac.group_states)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)
        mac_hidden = th.stack(mac_hidden, dim=1)
        mac_group_state = th.stack(mac_group_state, dim=1)
        mac_hidden = mac_hidden.detach()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Mixer
        chosen_action_qvals, w1_avg_list, sd_loss = self.mixer(chosen_action_qvals, batch["state"][:, :-1], mac_hidden[:, :-1], mac_group_state[:, :-1], "eval")

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            target_mac_hidden = []
            target_mac_group_state = []

            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_hidden.append(self.target_mac.hidden_states)
                target_mac_group_state.append(self.target_mac.group_states)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_hidden = th.stack(target_mac_hidden, dim=1)
            target_mac_group_state = th.stack(target_mac_group_state, dim=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"], target_mac_hidden, target_mac_group_state, "target")

            agent_visible = visible[..., :self.n_agents]
            enemies_visible = visible[..., self.n_agents:]
            agent_alive = (agent_visible * (torch.eye(self.n_agents).to(agent_visible.device))).sum(dim=-1)
            agent_alive_mask = torch.bmm(agent_alive.reshape(-1, self.n_agents, 1),
                                         agent_alive.reshape(-1, 1, self.n_agents)).reshape(b, t, self.n_agents,
                                                                                            self.n_agents)
            enemies_visible = enemies_visible.unsqueeze(-1).repeat(1, 1, 1, 1, self.args.enemy_shape)
            enemies_visible = enemies_visible.reshape(b, t, self.n_agents, -1)
            mask_env = mask.clone().roll(dims=-2, shifts=-1)
            mask_env[:, -1, :] = 0
            # Opp_mse_Exp = self.target_model_env.get_log_pi(model_s, model_opp_s) * mask_env
            ac = avail_actions[:, :-1]
            ac = (1 - actions_onehot) * avail_actions[:, :-1]
            lazy_avoid_intrinsic,team_intrinsic, enemy_ate = self.target_model_env.get_opp_intrinsic(model_s.clone(), state.clone(),
                                                                                        actions_onehot,
                                                                                        enemies_visible, ac)

            lazy_avoid_intrinsic = lazy_avoid_intrinsic.clamp(max=self.args.i_one_clip)
            mean_rewards = rewards.sum() / mask.sum()
            lazy_avoid_intrinsic = lazy_avoid_intrinsic * agent_alive
            if not self.args.cuda_save:
                CDI =team_intrinsic.clamp(max=0.1)/100
            else:
                old_extrin_s = batch['extrinsic_state'][:, :-1]
                new_extrin_s = batch['extrinsic_state'][:, 1:]
                s_transition = (old_extrin_s - new_extrin_s) ** 2
                CDI = s_transition.sum(dim=-1).clamp(max=0.15).unsqueeze(-1) / 100

            IDI = (lazy_avoid_intrinsic.sum(dim=-1).unsqueeze(-1))
            CDI, IDI = CDI * intrinsic_mask, IDI * intrinsic_mask  # ()
            CDI=CDI.clamp(max=0.06)
            intrinsic = self.args.beta2 * CDI + self.args.beta1 * IDI
            intrinsic = intrinsic.clamp(max=self.args.itrin_two_clip)
            mean_alive = (agent_alive * terminated).sum(dim=-1).sum(dim=-1).mean()
            enemy_alive = (((batch['extrinsic_state'][:, 1:]) * terminated).sum(-2).reshape(b, self.n_enemies, 3)[
                               ..., 0] > 0).float().sum(-1).mean()
            if not self.init_anneal_time and mean_rewards > 0.00:
                self.init_anneal_time = True
                self.start_anneal_time = t_env
            if t_env > self.start_anneal_time and self.args.env_args['reward_sparse'] and self.args.anneal_intrin:
                intrinsic = max(1 - (
                        t_env - self.start_anneal_time) / self.args.anneal_speed, 0) * intrinsic

            rewards_new = rewards + intrinsic  # +intrinsic
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards_new, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards_new, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

#            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
#                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)
        
        # lasso_alpha
        lasso_alpha = []
        for i in range(len(w1_avg_list)):
            lasso_alpha_time = self.args.lasso_alpha_start * (self.args.lasso_alpha_anneal ** (t_env//self.args.lasso_alpha_anneal_time))
            lasso_alpha.append(lasso_alpha_time)

        # lasso loss
        lasso_loss = 0
        for i in range(len(w1_avg_list)):
            group_w1_sum = th.sum(w1_avg_list[i])
            lasso_loss += group_w1_sum * lasso_alpha[i]
        
        sd_loss = sd_loss * mask
        sd_loss = self.args.sd_alpha * sd_loss.sum() / mask.sum()

        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        loss = td_loss + lasso_loss + sd_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.logger.log_stat("total_loss", loss.item(), t_env)
            self.logger.log_stat("lasso_loss", lasso_loss.item(), t_env)
            self.logger.log_stat("sd_loss", sd_loss.item(), t_env)
            
            self.log_stats_t = t_env
    
    def change_group(self, batch: EpisodeBatch, change_group_i: int):
        if change_group_i == 0:
            self.agent_w1_avg = 0

        mac_hidden = []

        with th.no_grad():
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_hidden.append(self.mac.hidden_states)
            mac_hidden = th.stack(mac_hidden, dim=1)
            
            w1_avg = self.mixer.get_w1_avg(mac_hidden[:, :-1])
            self.agent_w1_avg += w1_avg

        if change_group_i == self.args.change_group_batch_num - 1:
            
            self.agent_w1_avg /= self.args.change_group_batch_num
            group_now = copy.deepcopy(self.mixer.group)
            group_nxt = copy.deepcopy(self.mixer.group)
            for group_index, group_i in enumerate(group_now):
                group_w1_avg = self.agent_w1_avg[group_i]

                group_avg = th.mean(group_w1_avg)
                relative_lasso_threshold = group_avg * self.args.change_group_value
                indices = th.where(group_w1_avg < relative_lasso_threshold)[0]

                if len(group_i) < 3:
                    continue
                
                if group_index+1 == len(group_now) and len(indices) != 0:
                    tmp = []
                    group_nxt.append(tmp)
                    self.mixer.add_new_net()
                    self.target_mixer.add_new_net()
                
                for i in range(len(indices)-1, -1, -1):
                    idx = group_now[group_index][indices[i]]
                    group_nxt[group_index+1].append(idx)
                    del group_nxt[group_index][indices[i]]
                    for m in self.mixer.hyper_w1[idx]:
                        if type(m) != nn.ReLU:
                            m.reset_parameters()
            
            whether_group_changed = True if group_now != group_nxt else False
            
            if not whether_group_changed:
                return
            
            for i in range(len(group_nxt)-1, -1, -1):
                if group_nxt[i] == []:
                    del group_nxt[i]
                    self.mixer.del_net(i)
                    self.target_mixer.del_net(i)
            
            self.mixer.update_group(group_nxt)
            self.target_mixer.update_group(group_nxt)
            self._update_targets()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            #arr = np.asarray(self.mixer.group, dtype = object)可以忽略mixersize不匹配正常保存模型，但是load时会不匹配
            #np.save("{}/group.npy".format(path), arr)可以忽略mixersize不匹配正常保存模型，但是load时会不匹配
            np.save("{}/group.npy".format(path), self.mixer.group)
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.group = np.load("{}/group.npy".format(path),allow_pickle=True)
            for i in range(len(self.mixer.group)-1):
                self.mixer.add_new_net()
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class Predict_Network(nn.Module):

    def __init__(self,args, num_inputs, hidden_dim, num_outputs, lr=3e-4):
        super(Predict_Network, self).__init__()

        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)

        self.hideen_dim = hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            num_layers=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.last_fc = nn.Linear(hidden_dim, num_outputs)
        self.args=args
        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        b, t, _ = input.shape
        hidden = torch.zeros((1, b, self.hideen_dim)).to(input.device)
        h1 = F.relu(self.linear1(input))
        hrnn, _ = self.rnn(h1, hidden)
        x = self.last_fc(hrnn)
        return x, hrnn

    def counterfactual(self, input, h):
        b, t, n_a, _ = input.shape
        input = input.reshape(b * t * n_a, 1, -1)
        h = h.reshape(1, b * t * n_a, -1)
        h1 = F.relu(self.linear1(input))
        hrnn, _ = self.rnn(h1, h)
        x = self.last_fc(hrnn)
        return x.reshape(b, t, n_a, -1)

    def get_log_pi(self, own_variable, other_variable):
        predict_variable, _ = self.forward(own_variable)
        log_prob = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def get_opp_intrinsic(self, s_a, s, a, enemies_visible, avail_u=None):
        b, t, n_agents, n_actions = a.shape

        p_s_a, h = self.forward(s_a)

        h_new = torch.zeros_like(h).to(h.device)
        h_new[:, 1:] = h[:, :-1]
        full_actions = torch.ones((b, t, n_agents, n_actions, n_actions)) * torch.eye(n_actions)
        full_actions = full_actions.type_as(s).to(a.device)
        full_s = s.unsqueeze(-2).repeat(1, 1, n_actions, 1)
        full_a = a.unsqueeze(-2).repeat(1, 1, 1, n_actions, 1)
        full_h = h_new.unsqueeze(-2).repeat(1, 1, n_actions, 1)
        intrinsic_1 = torch.zeros((b, t, n_agents)).to(a.device)
        Enemy = torch.zeros((b, t, n_agents, p_s_a.shape[-1])).to(a.device)
        if not self.args.cuda_save:
            sample_size=self.args.sample_size
            random_ = torch.rand(b, t, sample_size, n_agents, n_actions).type_as(s)*(avail_u.unsqueeze(-3))
            sample_a=torch.zeros_like(random_)
            values, indices = random_.topk(1, dim=-1, largest=True, sorted=True)
            random_=(random_==values).type_as(s)*(avail_u.unsqueeze(-3))
            random_full_s = s.unsqueeze(-2).repeat(1, 1, sample_size, 1)
            random_s_a=torch.cat((random_full_s,sample_a.reshape(b, t, sample_size, -1)),dim=-1)
            random_full_h = h_new.unsqueeze(-2).repeat(1, 1, sample_size, 1)
            s_enemy_visible=enemies_visible.sum(dim=-2).clamp(min=0,max=1)
            p_s_random = self.counterfactual(random_s_a, random_full_h).mean(dim=-2)
            ATE_enemy_joint=s_enemy_visible * F.mse_loss(p_s_random, p_s_a, reduction='none')
            intrinsic_2=ATE_enemy_joint.sum(dim=-1).unsqueeze(-1)
        else:
            intrinsic_2 =torch.zeros((b, t,1))
        if avail_u == None:
            avail_u = torch.ones_like(a).type_as(a)
        for i in range(n_agents):
            ATE_a = (full_a.clone())
            ATE_a[..., i, :, :] = full_actions[..., i, :, :]
            ATE_a = ATE_a.transpose(-2, -3).reshape(b, t, n_actions, -1)
            s_a_noi = torch.cat((full_s, ATE_a), dim=-1)
            p_s_a_noi = self.counterfactual(s_a_noi, full_h)
            p_s_a_noi = p_s_a_noi * (avail_u[..., i, :].unsqueeze(-1))
            p_s_a_mean_noi = p_s_a_noi.sum(dim=-2) / (avail_u[..., i, :].sum(dim=-1).unsqueeze(-1) + 1e-6)
            ATE_enemy_i = enemies_visible[..., i, :] * F.mse_loss(p_s_a_mean_noi, p_s_a, reduction='none')
            # ATE_enemy_i=enemies_visible[...,i,:]*torch.abs(p_s_a_mean_noi-p_s_a)
            ATE_i = ATE_enemy_i.sum(dim=-1)
            intrinsic_1[..., i] = ATE_i
            Enemy[..., i, :] = ATE_enemy_i
        return intrinsic_1,intrinsic_2, Enemy

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable, _ = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None
