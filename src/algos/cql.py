import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.distributions import Dirichlet, Uniform
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.algos.layers import GNNActor, GNNActor1, MLPActor, MLPActor1, GNNCritic1, GNNCritic2, GNNCritic3, GNNCritic4, GNNCritic4_1, GNNCritic5, GNNCritic6, MLPCritic4, MLPCritic4_1
import random
import json


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
############## A2C AGENT ################
#########################################


class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        hidden_size=32,
        alpha=0.2,
        gamma=0.97,
        polyak=0.995,
        batch_size=128,
        p_lr=3e-4,
        q_lr=1e-3,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=-1,
        min_q_weight=1,
        deterministic_backup=True,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        min_q_version=3,
        clip=200,
        mode = 1,
        q_lag = 10,
        json_file = None
    ):
        super(SAC, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion
        self.mode = mode

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.clip = clip
        self.lag= 0
        self.q_lag = q_lag

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup
        self.step = 0
        self.nodes = env.nregion

        # nnets
        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode)
    
        self.critic1 = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        self.critic2 = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        assert self.critic1.parameters() != self.critic2.parameters()
   

        self.critic1_target = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic4(
            self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.obs_parser = GNNParser(self.env, json_file=json_file, T=6)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.p_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy().tolist()
        return a
    
    def _get_action_and_values(self, data, batch_size, action_dim):

        alpha = torch.ones(action_dim)
        p = Uniform(0, 2)
        d = Dirichlet(alpha)

        # Random actions
        random_price = p.sample((batch_size*self.num_random, action_dim))
        random_distribution = d.sample((batch_size*self.num_random,))
        random_log_prob = (
            p.log_prob(random_price).sum(axis=-1).view(
                batch_size, self.num_random, 1).to(self.device)
        ) + (
            d.log_prob(random_distribution).view(
                batch_size, self.num_random, 1).to(self.device)
        )
        random_actions = torch.cat([random_price.unsqueeze(-1), random_distribution.unsqueeze(-1)], dim=-1)
        random_actions = random_actions.to(self.device)

        # Actions from current policy
        data_list = data.to_data_list()
        data_list = data_list * self.num_random
        batch_temp = Batch.from_data_list(data_list).to(self.device)

        current_actions, current_log = self.actor(
            batch_temp.x_s, batch_temp.edge_index_s
        )
        current_log = current_log.view(batch_size, self.num_random, 1)

        next_actions, next_log = self.actor(
            batch_temp.x_t, batch_temp.edge_index_t)
        next_log = next_log.view(batch_size, self.num_random, 1)

        # Q value of random policy and current policy
        q1_rand = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, self.num_random, 1)

        q2_rand = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, self.num_random, 1)

        q1_current = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, self.num_random, 1)

        q2_current = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, self.num_random, 1)

        q1_next = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, self.num_random, 1)

        q2_next = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, self.num_random, 1)

        return (
            random_log_prob,
            current_log,
            next_log,
            q1_rand,
            q2_rand,
            q1_current,
            q2_current,
            q1_next,
            q2_next,
        )


    def compute_loss_q(self, data, conservative=False, enable_calql=False):
        if enable_calql:
            (
                state_batch,
                edge_index,
                next_state_batch,
                edge_index2,
                reward_batch,
                action_batch,
                mc_returns,
            ) = (
                data.x_s,
                data.edge_index_s,
                data.x_t,
                data.edge_index_t,
                data.reward,
                data.action.reshape(-1, self.nodes, max(self.mode,1)),
                data.mc_returns,
            )
        else:
            (
                state_batch,
                edge_index,
                next_state_batch,
                edge_index2,
                reward_batch,
                action_batch,
            ) = (
                data.x_s,
                data.edge_index_s,
                data.x_t,
                data.edge_index_t,
                data.reward,
                data.action.reshape(-1, self.nodes, max(self.mode,1)),
            )

        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if not self.deterministic_backup:
                backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)
            else:
                backup = reward_batch + self.gamma * q_pi_targ

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        if conservative:
            batch_size = action_batch.shape[0]
            action_dim = action_batch.shape[1]

            (
                random_log_prob,
                current_log,
                next_log,
                q1_rand,
                q2_rand,
                q1_current,
                q2_current,
                q1_next,
                q2_next,
            ) = self._get_action_and_values(data, batch_size, action_dim)

            if enable_calql:
                """Cal-QL: prepare for Cal-QL, and calculate how much data will be lower bounded for logging"""
                lower_bounds = (
                    mc_returns.unsqueeze(-1)
                    .repeat(1, q1_current.shape[1])
                    .unsqueeze(-1)
                )

            """ Cal-QL: bound Q-values with MC return-to-go """
            if enable_calql:
                q1_current = torch.maximum(q1_current, lower_bounds)
                q2_current = torch.maximum(q2_current, lower_bounds)
                q1_next = torch.maximum(q1_next, lower_bounds)
                q2_next = torch.maximum(q2_next, lower_bounds)

            if self.min_q_version == 1:
                cat_q1 = q1_current - np.log(10)
                cat_q2 = q2_current - np.log(10)

            if self.min_q_version == 2:
                cat_q1 = torch.cat(
                    [q1_rand, q1.unsqueeze(1).unsqueeze(
                        1), q1_next, q1_current], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand, q2.unsqueeze(1).unsqueeze(
                        1), q2_next, q2_current], 1
                )

            if self.min_q_version == 3:
                # importance sampled version
                # random_density = np.log(0.5**action_dim)
                cat_q1 = torch.cat(
                    [
                        q1_rand - random_log_prob.detach(),
                        q1_next - next_log.detach(),
                        q1_current - current_log.detach(),
                    ],
                    1,
                )
                cat_q2 = torch.cat(
                    [
                        q2_rand - random_log_prob.detach(),
                        q2_next - next_log.detach(),
                        q2_current - current_log.detach(),
                    ],
                    1,
                )

            min_qf1_loss = (
                torch.logsumexp(
                    cat_q1 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )
            min_qf2_loss = (
                torch.logsumexp(
                    cat_q2 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2.mean() * self.min_q_weight

            # log metrics
            log = {"Bellman loss Q1":loss_q1.item(),"Regularizor loss Q1":(min_qf1_loss/self.min_q_weight).item(),
                   "Bellman loss Q2":loss_q2.item(),"Regularizor loss Q2":(min_qf2_loss/self.min_q_weight).item(),
                   "Q1":torch.mean(q1).item(),"Q2":torch.mean(q2).item(),}

            if self.with_lagrange:
                alpha_prime = torch.clamp(
                    torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
                )
                min_qf1_loss = alpha_prime * \
                    (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * \
                    (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            loss_q1 = loss_q1 + min_qf1_loss
            loss_q2 = loss_q2 + min_qf2_loss        

        return loss_q1, loss_q2, q1, q2, log

    def compute_loss_pi(self, data):
        state_batch, edge_index = (
            data.x_s,
            data.edge_index_s,
        )

        actions, logp_a = self.actor(state_batch, edge_index)
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()

        return loss_pi

    def update(self, data, conservative=False, enable_calql=False):
        self.lag += 1

        loss_q1, loss_q2, _, _, log = self.compute_loss_q(data, conservative, enable_calql)

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward(retain_graph=False)
        # actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # Update target networks by polyak averaging.
        if self.lag == self.q_lag:
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
                    
            self.lag = 0
        
        return log


    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)


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
