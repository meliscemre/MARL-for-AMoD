import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.algos.layers import GNNCritic4, VF
import random
import json

SMALL_NUMBER = 1e-6

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        # Takes input from the environemnt and returns a graph (with node features and connectivity)
        # Here we aggregate environment observations into node-wise features
        # In order, x is a collection of the following information:
        # 1) current availability scaled by factor, 
        # 2) Estimated availability (T timestamp) scaled by factor, 
        # 3) Estimated revenue (T timestamp) scaled by factor
        x = (
            torch.cat(
                (
                    # Current availability
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] *
                            self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Estimated availability
                    torch.tensor(
                        [
                            [
                                (obs[0][n][self.env.time + 1] +
                                 self.env.dacc[n][t])
                                * self.s
                                for n in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                    # Queue length
                    torch.tensor(
                        [
                            len(self.env.queue[n]) * self.s for n in self.env.region
                        ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Current demand
                    torch.tensor(
                            [
                                sum(
                                    [
                                        (self.env.demand[i, j][self.env.time])
                                        # * (self.env.price[i, j][self.env.time])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Current price
                    torch.tensor(
                            [
                                sum(
                                    [
                                        (self.env.price[i, j][self.env.time])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),                    
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + 1 + 1 + 1, self.env.nregion)
            .T
        )       
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long()
        data = Data(x, edge_index)
        return data

#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=0, edges=None):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        if mode == 0:
            self.lin3 = nn.Linear(hidden_size, 1)
        elif mode == 1:
            self.lin3 = nn.Linear(hidden_size, 2)
        else:
            self.lin3 = nn.Linear(hidden_size, 3)

    def forward(self, state, edge_index, deterministic=False, return_D=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if deterministic:
            if self.mode == 0:
                action = (concentration) / (concentration.sum() + 1e-20)
            elif self.mode == 1:
                action_o = (concentration[:,:,0]-1)/(concentration[:,:,0] + concentration[:,:,1] -2 + 1e-10)
                action_o[action_o<0] = 0
                action = action_o.squeeze(0).unsqueeze(-1)
            else:
                action_o = (concentration[:,:,0]-1)/(concentration[:,:,0] + concentration[:,:,1] -2 + 1e-10)
                action_o[action_o<0] = 0
                action_reb = (concentration[:,:,2]) / (concentration[:,:,2].sum() + 1e-10)
                action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_reb.squeeze(0).unsqueeze(-1)),-1)
        else:
            if self.mode == 0:
                m = Dirichlet(concentration + 1e-20)
                action = m.rsample()
                action = action.squeeze(0).unsqueeze(-1)
            elif self.mode == 1:
                m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
                action_o = m_o.rsample()
                action = action_o.squeeze(0).unsqueeze(-1)             
            else:        
                m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
                action_o = m_o.rsample()
                # Rebalancing desired distribution
                m_reb = Dirichlet(concentration[:,:,-1] + 1e-10)
                action_reb = m_reb.rsample()              
                action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_reb.squeeze(0).unsqueeze(-1)),-1)       

                if return_D:
                    return (m_reb,m_o)
        
        return action

#########################################
############## A2C AGENT ################
#########################################


class IQL(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        hidden_size=32,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        batch_size=128,
        p_lr=3e-4,
        q_lr=1e-3,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        clip=10,
        quantile=0.5,
        temperature=1.0,
        clip_score=100,
        mode = 1,
        json_file=None,
    ):
        super(IQL, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.nnodes = env.nregion
        self.mode = mode

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma

        self.num_random = 10
        self.temp = 1.0
        self.clip = clip
        self.clip_score = clip_score
        self.quantile = quantile
        self.temperature = temperature

        self.step = 0

        # nnets
        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.nnodes, mode=mode)
    
        self.critic1 = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.nnodes, mode=mode
        )
        self.critic2 = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.nnodes, mode=mode
        )
        assert self.critic1.parameters() != self.critic2.parameters()
   

        self.critic1_target = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.nnodes, mode=mode
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.nnodes, mode=mode
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.vf = VF(in_channels=self.input_size, hidden_size=self.hidden_size, nnodes=self.nnodes)

        self.obs_parser = GNNParser(self.env, json_file=json_file, T=6)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            a = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy().tolist()
        return a

    def update(self, data, only_q=False):
        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
            done_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.nnodes, 2),
            data.done.float(),
        )        

        q1_pred = self.critic1(state_batch, edge_index, action_batch)
        q2_pred = self.critic2(state_batch, edge_index, action_batch)

        target_vf_pred = self.vf(next_state_batch, edge_index2).detach()

        with torch.no_grad():
            q_target = reward_batch + (1 - done_batch) * self.gamma * target_vf_pred
        q_target = q_target.detach()

        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)

        q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, action_batch)
        q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, action_batch)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).detach()
        vf_pred = self.vf(state_batch, edge_index)

        vf_err = q_pi_targ - vf_pred

        weight = torch.where(vf_err > 0, self.quantile, (1 - self.quantile))
        vf_loss = weight * (vf_err**2)
        vf_loss = vf_loss.mean()
        if not only_q:
            D, G = self.actor(state_batch, edge_index, return_D=True)

            price_act = action_batch[:, :, 0]

            reb_act = action_batch[:, :, 1]

            price_act = torch.clamp(price_act, 0.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
            # reb_act = torch.clamp(reb_act, 0.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)

            price_log_prob = G.log_prob(price_act.squeeze(-1)).sum(dim=-1)

            Dirichlet_log_prob = D.log_prob(reb_act.squeeze(-1))

            log_prob = price_log_prob + Dirichlet_log_prob

            adv = q_pi_targ - vf_pred

            exp_adv = torch.exp(adv * self.temperature)

            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

            weights = exp_adv.detach()

            loss_pi = -(log_prob * weights).mean()

        # log metrics
        log = {"loss Q1":loss_q1.item(), "loss Q2":loss_q2.item(), "loss V":vf_loss.item(),
                "Q1":torch.mean(q1_pred).item(),"Q2":torch.mean(q2_pred).item(),}

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        self.optimizers["v_optimizer"].zero_grad()
        vf_loss.backward()
        self.optimizers["v_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        if not only_q:
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi.backward(retain_graph=False)
            # actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True
        
        return log


    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        v_params = list(self.vf.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)
        optimizers["v_optimizer"] = torch.optim.Adam(v_params, lr=self.q_lr)


        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            action_rl = [0]*env.nregion
            done = False
            while not done:

                if env.mode == 0:
                    obs, paxreward, done, info, _, _ = env.match_step_simple()
                    # obs, paxreward, done, info, _, _ = env.pax_step(
                    #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                    #             )

                    o = self.parse_obs(obs=obs).to(self.device)
                    eps_reward += paxreward

                    action_rl = self.select_action(o, deterministic=True)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[0][i] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }
                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(
                        env,
                        "scenario_san_francisco4",
                        desiredAcc,
                        cplexpath,
                        directory, 
                    )
                    # Take rebalancing action in environment
                    _, rebreward, done, _, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward

                elif env.mode == 1:
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                    o = self.parse_obs(obs=obs).to(self.device)

                    eps_reward += paxreward

                    action_rl = self.select_action(o, deterministic=True)  

                    env.matching_update()
                elif env.mode == 2:
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                    o = self.parse_obs(obs=obs).to(self.device)
                    eps_reward += paxreward

                    action_rl = self.select_action(o, deterministic=True)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }
                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(
                        env,
                        "scenario_san_francisco4",
                        desiredAcc,
                        cplexpath,
                        directory, 
                    )
                    # Take rebalancing action in environment
                    _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward                
                else:
                    raise ValueError("Only mode 0, 1, and 2 are allowed")  
                
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
