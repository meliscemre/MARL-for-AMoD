from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env_single_agent import Scenario, AMoD
from src.algos.sac_single_agent import SAC
from src.algos.reb_flow_solver_single_agent import solveRebFlow
from src.misc.utils import dictsum, nestdictsum
import json, pickle
from torch_geometric.data import Data
import copy, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import wandb
import matplotlib.pyplot as plt
from visualizations import plot_mean_rebalancing_per_region, plot_mean_origin_demand_per_region, plot_mean_price_scalar_per_region


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
                    # Current price sum
                    #torch.tensor(
                    #        [
                    #            sum(
                    #                [
                    #                    (self.env.price[i, j][self.env.time])
                    #                    * self.s
                    #                    for j in self.env.region
                    #                ]
                    #            )
                    #            for i in self.env.region
                    #        ]
                    #)
                    #.view(1, 1, self.env.nregion)
                    #.float(),
                    #prices od
                    torch.tensor(
                        [
                            [self.env.price[i, j][self.env.time] * self.s for j in self.env.region]
                            for i in self.env.region
                        ]
                    )
                    .view(1, self.env.nregion, self.env.nregion)
                    .float(),                    
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + 1 + 1 + self.env.nregion, self.env.nregion)
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


parser = argparse.ArgumentParser(description="SAC-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--demand_ratio",
    type=float,
    default=18,
    metavar="S",
    help="demand_ratio (default: 1)",
)
parser.add_argument(
    "--json_hr", type=int, default=19, metavar="S", help="json_hr"
)
parser.add_argument(
    "--json_tstep",
    type=int,
    default=3,
    metavar="S",
    help="minutes per timestep (default: 3min)",
)
parser.add_argument(
    '--mode', 
    type=int, 
    default=0,
    help='rebalancing mode. (0:manul, 1:pricing, 2:both. default 1)',
)

parser.add_argument(
    "--beta",
    type=int,
    default=0.3,
    metavar="S",
    help="cost of rebalancing (default: 0.5)",
)

parser.add_argument(
    "--strategy",
    type=str,
    default="learned",
    choices=["learned", "equal", "none"],
    help="Rebalancing strategy: learned, equal, or none"
)


# Model parameters
#parser.add_argument(
#    "--test", type=bool, default=False, help="activates test mode for agent evaluation"
#)
parser.add_argument("--test", action="store_true", help="activates test mode for agent evaluation")
parser.add_argument(
    "--cplexpath",
    type=str,
    default="/apps/cplex/cplex1210/opl/bin/x86-64_linux/",
    help="defines directory of the CPLEX installation",
)
parser.add_argument(
    "--directory",
    type=str,
    default=None,
    help="defines directory where to save files",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=10000,
    metavar="N",
    help="number of episodes to train agent (default: 16k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=20,
    metavar="N",
    help="number of steps per episode (default: T=20)",
)
parser.add_argument(
    "--no-cuda", 
    type=bool, 
    default=True,
    help="disables CUDA training",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch size for training (default: 100)",
)
parser.add_argument(
    "--jitter",
    type=int,
    default=1,
    help="jitter for demand 0 (default: 1)",
)
parser.add_argument(
    "--maxt",
    type=int,
    default=2,
    help="maximum passenger waiting time (default: 6mins)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,
    help="entropy coefficient (default: 0.3)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="hidden size of neural networks (default: 256)",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="name of checkpoint file to save/load (default: SAC)",
)
#parser.add_argument(
#    "--load",
#    type=bool,
#    default=False,
#    help="either to start training from checkpoint (default: False)",
#)
parser.add_argument(
    "--load",
    action="store_true",
    help="load checkpoint if specified",
)
parser.add_argument(
    "--clip",
    type=int,
    default=500,
    help="clip value for gradient clipping (default: 500)",
)
parser.add_argument(
    "--p_lr",
    type=float,
    default=1e-3,
    help="learning rate for policy network (default: 1e-4)",
)
parser.add_argument(
    "--q_lr",
    type=float,
    default=1e-3,
    help="learning rate for Q networks (default: 4e-3)",
)
parser.add_argument(
    "--q_lag",
    type=int,
    default=1,
    help="update frequency of Q target networks (default: 10)",
)
parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
    help="city to train on",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.05,
    help="reward scaling factor (default: 0.1)",
)
parser.add_argument(
    "--critic_version",
    type=int,
    default=4,
    help="critic version (default: 4)",
)
parser.add_argument(
    "--price_version",
    type=str,
    default="GNN-origin",
    help="price network version",
)
parser.add_argument(
    "--impute",
    type=int,
    default=0,
    help="Whether impute the zero price (default: False)",
)
parser.add_argument(
    "--supply_ratio",
    type=float,
    default=1.0,
    help="supply scaling factor (default: 1)",
)
parser.add_argument(
    "--small",
    type=bool,
    default=False,
    help="whether to run the small hypothetical case (default: False)",
)

args = parser.parse_args()
if args.directory is None:
    args.directory = f"saved_files_mode{args.mode}_single_agent_od_price_500"
if args.checkpoint_path is None:
    args.checkpoint_path = f"SAC_mode{args.mode}_single_agent_od_price_500"
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city

#wandb

wandb.init(
    project="amod-rl",
    entity="meliscemre-technical-university-of-denmark",
    name=f"{args.checkpoint_path}",
    config=vars(args)
)


#choice model price multiplier vs rejection rate function

def sweep_choice_price_mult(scenario, base_args, sweep_values, steps=20, episodes=1):
    results = []

    for mult in sweep_values:
        acc_rate_all = []

        for _ in range(episodes):
            scenario_copy = copy.deepcopy(scenario) 
            env = AMoD(
                scenario=scenario_copy,
                mode=base_args.mode,
                beta=base_args.beta,
                jitter=base_args.jitter,
                max_wait=base_args.maxt,
                choice_price_mult=mult
            )

            env.reset()

            acc_rate_episode = []
            for _ in range(steps):
                obs, _, done, info, _, _ = env.match_step_simple()
                acc_rate_episode.append(1 - info["rejection_rate"])
                env.matching_update()
                if done:
                    break

            acc_rate_all.append(np.mean(acc_rate_episode))

        results.append(np.mean(acc_rate_all))
        print(f"choice_price_mult={mult:.2f} | acceptance_rate={results[-1]:.4f}")
        
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_values, results, marker='o')
    plt.xlabel("Price Multiplier")
    plt.ylabel("Acceptance Rate")
    plt.title("Passenger Acceptance Rate vs Price Multiplier")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/acceptance_rate_vs_choice_price_mult_{args.checkpoint_path}.png")
    plt.show()

    return sweep_values, results
    
    
if not args.test:
    if not args.small:
        scenario = Scenario(
            json_file=f"data/scenario_nyc_brooklyn.json",
            demand_ratio=args.demand_ratio,
            json_hr=args.json_hr,
            sd=args.seed,
            json_tstep=args.json_tstep,
            tf=args.max_steps,
            impute=args.impute,
            supply_ratio=args.supply_ratio,
        )

        simulation_end_time = scenario.tf
        
        # --- Aggregate demand ---
        aggregated_demand = defaultdict(float)
        for (o, d, t, demand, _) in scenario.tripAttr:
            if 0 <= t < simulation_end_time:
                aggregated_demand[(o, d, t)] += demand

        total_demand_tf = sum(aggregated_demand.values())
        print(f"\n Total demand from 0 to {simulation_end_time} : {total_demand_tf:,}")

        if aggregated_demand:
            origins = sorted(set(o for (o, d, t) in aggregated_demand))
            destinations = sorted(set(d for (o, d, t) in aggregated_demand))
            demand_matrix = np.zeros((len(origins), len(destinations)))

            for (o, d, t), val in aggregated_demand.items():
                i, j = origins.index(o), destinations.index(d)
                demand_matrix[i, j] += val

            plt.figure(figsize=(12, 10))
            sns.heatmap(demand_matrix, annot=True, fmt=".1f", cmap="coolwarm",
                        xticklabels=destinations, yticklabels=origins)
            plt.xlabel("Destination")
            plt.ylabel("Origin")
            plt.title(f"Origin-Destination Demand Heatmap (0 to {simulation_end_time})")
            plt.savefig(f"visualizations/demand_heatmap_melis_{args.checkpoint_path}.png")

            plt.figure(figsize=(12, 10))
            sns.heatmap(np.log1p(demand_matrix), annot=True, fmt=".1f", cmap="coolwarm",
                        xticklabels=destinations, yticklabels=origins)
            plt.xlabel("Destination")
            plt.ylabel("Origin")
            plt.title("Log-Scaled Origin-Destination Demand Heatmap)")
            plt.savefig(f"visualizations/demand_heatmap_log_melis_{args.checkpoint_path}.png")

        # --- Aggregate prices ---
        aggregated_prices = defaultdict(list)
        for (o, d, t, demand, price) in scenario.tripAttr:
            if 0 <= t < simulation_end_time and demand > 0:
                aggregated_prices[(o, d)].append(price)

        average_prices = {od: np.mean(prices) for od, prices in aggregated_prices.items()}
        origins = sorted(list(scenario.G.nodes))  # or scenario.region if you have that
        destinations = sorted(list(scenario.G.nodes))
        price_matrix = np.zeros((len(origins), len(destinations)))

        origin_idx = {o: i for i, o in enumerate(origins)}
        dest_idx = {d: j for j, d in enumerate(destinations)}
        
        for (o, d), avg_price in average_prices.items():
            i = origin_idx[o]
            j = dest_idx[d]
            price_matrix[i, j] = avg_price

        plt.figure(figsize=(12, 10))
        sns.heatmap(price_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=destinations, yticklabels=origins)
        plt.xlabel("Destination j")
        plt.ylabel("Origin i")
        plt.title(f"Average Cost to Travel from Origin i to Destination j (0 to {simulation_end_time})")
        plt.savefig("avg_price_heatmap_melis.png")

        plt.figure(figsize=(12, 10))
        sns.heatmap(np.log1p(price_matrix), annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=destinations, yticklabels=origins)
        plt.xlabel("Destination j")
        plt.ylabel("Origin i")
        plt.title(f"Log-Scaled Average Cost (Origin i → Destination j, 0 to {simulation_end_time})")
        plt.savefig(f"visualizations/avg_price_heatmap_log_melis_{args.checkpoint_path}.png")
        

    # --- END: Demand & Price Heatmaps and Plots ---

        
    else:
        d = {
        (2, 3): 6,
        (2, 0): 4,
        (0, 3): 4,
        "default": 1,
        }
        r = {
        0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
        1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
        3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
        }
        scenario = Scenario(tf=20, demand_input=d, demand_ratio=r, ninit=30, N1=2, N2=2)

    #choice model price multiplier vs rejection rate
    sweep_range = np.arange(0.2, 2.01, 0.2)
    sweep_choice_price_mult(scenario, args, sweep_range, steps=20, episodes=1)
    
    env = AMoD(scenario, args.mode, beta=args.beta, jitter=args.jitter, max_wait=args.maxt, choice_price_mult=1.0)
    
    total_acc = sum(env.G.nodes[i]['accInit'] for i in env.region)
    
    if not args.small:
        parser = GNNParser(
            env, T=6, json_file=f"data/scenario_nyc_brooklyn.json"
        )  # Timehorizon T=6 (K in paper)
    else:
        parser = GNNParser(
            env, T=6
        )  # Timehorizon T=6 (K in paper)

    model = SAC(
        env=env,
        input_size=27,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version = args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    if args.load:
        print("load checkpoint")
        model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    model.train()  # set model in train mode

    log_lines = []

    # Check metrics
    epoch_demand_list = []
    epoch_reward_list = []
    #epoch_waiting_list = []
    epoch_servedrate_list = []
    epoch_rebalancing_cost = []
    epoch_value1_list = []
    epoch_value2_list = []
    
    price_history = []

    for i_episode in epochs:
        obs = env.reset()  # initialize environment
        # Save original demand for reference
        demand_ori = nestdictsum(env.demand)
        if i_episode == train_episodes - 1:
            export = {"demand_ori":copy.deepcopy(env.demand)}
        action_rl = [0]*env.nregion
        episode_reward = 0
        episode_served_demand = 0
        episode_unserved_demand = 0
        episode_rebalancing_cost = 0
        episode_rejected_demand = 0
        episode_total_revenue = 0
        episode_total_operating_cost = 0
        episode_waiting = 0
        actions = []
        episode_rejection_rates = []



        current_eps = []
        done = False
        step = 0
        while not done:
            # take matching step (Step 1 in paper)
            if step > 0:
                obs1 = copy.deepcopy(o)
            
            #mode 0: only rebalancing
            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                episode_rejection_rates.append(info["rejection_rate"])
                # obs, paxreward, done, info, _, _ = env.pax_step(
                #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                #             )

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memory
                    rl_reward = paxreward + rebreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                ##no rebalancing
                #action_rl = [0.0] * env.nregion
                #desiredAcc = {region: 0 for region in env.region}
                
                #equal distribution
                #action_rl = [1 / env.nregion for _ in range(env.nregion)]
                #desiredAcc = {
                #    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time +1))
                #    for i in range(len(env.region))
                #}
                
                #original
                #action_rl = model.select_action(o)
                
                if args.strategy == "learned":
                    action_rl = model.select_action(o)
                elif args.strategy == "equal":
                    action_rl = [1 / env.nregion for _ in range(env.nregion)]
                elif args.strategy == "none":
                    action_rl = [0.0 for _ in range(env.nregion)]
                else:
                    raise ValueError("Invalid strategy")

                # Unwrap if it's nested
                if isinstance(action_rl[0], list) and len(action_rl) == 1:
                    action_rl = action_rl[0]
                                
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env,
                    "scenario_san_francisco4",
                    desiredAcc,
                    args.cplexpath,
                    args.directory, 
                )
                # Take rebalancing action in environment
                new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
                
            # mode 1: only pricing
            
            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                episode_rejection_rates.append(info["rejection_rate"])
                
                o = parser.parse_obs(obs=obs)

                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o)  

                env.matching_update()

            #mode 2: rebalancing & pricing

            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                episode_rejection_rates.append(info["rejection_rate"])
                
                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o)

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
                    args.cplexpath,
                    args.directory, 
                )
                # Take rebalancing action in environment
                new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward                
            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed")                    

            # track performance over episode
            episode_unserved_demand += info["unserved_demand"]
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_rejected_demand += info["rejected_demand"]
            episode_total_revenue += info["revenue"]
            episode_total_operating_cost += info["operating_cost"]
            episode_waiting += info['served_waiting']
            actions.append(action_rl)

            step += 1
            if i_episode > 10:
                # sample from memory and update model
                batch = model.replay_buffer.sample_batch(
                    args.batch_size, norm=False)
                grad_norms = model.update(data=batch)  
            else:
                grad_norms = {"actor_grad_norm":0, "critic1_grad_norm":0, "critic2_grad_norm":0, "actor_loss":0, "critic1_loss":0, "critic2_loss":0, "Q1_value":0, "Q2_value":0}
            

            # Keep track of loss
            epoch_value1_list.append(grad_norms["Q1_value"])
            epoch_value2_list.append(grad_norms["Q2_value"])

        # Keep metrics
        epoch_reward_list.append(episode_reward)
        epoch_demand_list.append(env.arrivals)
        #epoch_waiting_list.append(episode_waiting/episode_served_demand)
        if env.arrivals > 0:
            epoch_servedrate_list.append(episode_served_demand / env.arrivals)
        else:
            epoch_servedrate_list.append(0)
        epoch_rebalancing_cost.append(episode_rebalancing_cost)

        # Keep price (only needed for pricing training)
        price_history.append(actions)

        #wandb
        wandb.log({
            "episode": i_episode,
            "episode_reward": episode_reward,
            "episode_unserved_demand": episode_unserved_demand,
            "episode_served_demand": episode_served_demand,
            "episode_waiting_time": episode_waiting / episode_served_demand if episode_served_demand > 0 else 0,
            "episode_total_revenue": episode_total_revenue,
            "episode_operating_cost": episode_total_operating_cost,
            "episode_rebalancing_cost": episode_rebalancing_cost,
            "episode_rejection_rate": np.mean(episode_rejection_rates),
            "actor_loss": grad_norms["actor_loss"],
            "critic1_loss": grad_norms["critic1_loss"],
            "critic2_loss": grad_norms["critic2_loss"],
            "Q1_value": grad_norms["Q1_value"],
            "Q2_value": grad_norms["Q2_value"]
        })
        
        log_line = (
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Revenue: {episode_total_revenue:.2f} | "
            f"Reb. Cost: {episode_rebalancing_cost:.2f} | Op. Cost: {episode_total_operating_cost:.2f} | "
            f"Tot.Demand: {env.arrivals} | ServedD: {episode_served_demand} | "
            f"UnservedD: {episode_unserved_demand} | Wait.: {sum(len(q) for q in env.queue.values())} | "
            f"Rej. Rate: {np.mean(episode_rejection_rates)} | RejectedD: {episode_rejected_demand}"
        )
        epochs.set_description(log_line)
        log_lines.append(log_line)
        wandb.termlog(log_line)


        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
        if i_episode % 100 == 0:
            test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                10, env, args.cplexpath, args.directory, parser=parser
            )
            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                model.save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}_test.pth")
                 
    # Save metrics file
    metricPath = f"{args.directory}/train_logs/"
    if not os.path.exists(metricPath):
        os.makedirs(metricPath)
    np.save(f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", np.array([epoch_reward_list,epoch_servedrate_list,epoch_demand_list,epoch_rebalancing_cost]))
    np.save(f"{args.directory}/train_logs/{city}_price_mode{args.mode}_{train_episodes}.npy", np.array(price_history))
    np.save(f"{args.directory}/train_logs/{city}_q_mode{args.mode}_{train_episodes}.npy", np.array([epoch_value1_list,epoch_value2_list]))

    export["avail_distri"] = env.acc
    export["demand_scaled"] = env.demand
    with open(f"{args.directory}/train_logs/{city}_export_mode{args.mode}_{train_episodes}.pickle", 'wb') as f:
        pickle.dump(export, f) 
else:
    if not args.small:
        scenario = Scenario(
            json_file=f"data/scenario_nyc_brooklyn.json",
            demand_ratio=args.demand_ratio,
            json_hr=args.json_hr,
            sd=args.seed,
            json_tstep=args.json_tstep,
            tf=args.max_steps,
            impute=args.impute,
            supply_ratio=args.supply_ratio
        )
    else:

        d = {
        (2, 3): 6,
        (2, 0): 4,
        (0, 3): 4,
        "default": 1,
        }
        r = {
        0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
        1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
        3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
        }
        scenario = Scenario(tf=20, demand_input=d, demand_ratio=r, ninit=30, N1=2, N2=2)

    env = AMoD(scenario, args.mode, beta=args.beta, jitter=args.jitter, max_wait=args.maxt, choice_price_mult=1.0)

    if not args.small:
        parser = GNNParser(
            env, T=6, json_file=f"data/scenario_nyc_brooklyn.json"
        )  # Timehorizon T=6 (K in paper)
    else:
        parser = GNNParser(
            env, T=6
        )  # Timehorizon T=6 (K in paper)

    model = SAC(
        env=env,
        input_size=27,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version = args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)
    

    if args.load:
        print("Loading checkpoint...")
        model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
    else:
        print("Training from scratch, no checkpoint loaded.")
#    print("load model")
#    model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

    rewards = []
    demands = []
    costs = []
    arrivals = []
    rejection_rates = []
    demand_original_steps = []
    demand_scaled_steps = []
    reb_steps = []
    reb_ori_steps = []
    reb_num = []
    pax_steps = []
    pax_wait = []
    actions_step = []
    price_mean = []
    available_steps = []
    rebalancing_cost_steps = []
    price_original_steps = []
    queue_steps = []
    waiting_steps = []
    rejection_rate_steps = []
    accepted_demand_by_origin_steps = []
    price_scalar_steps = []


    for episode in range(10):
        actions = []
        actions_price = []
        rebalancing_cost = []
        rebalancing_num = []
        queue = []
        accepted_demand_ij = defaultdict(float)
        episode_reward = 0
        episode_served_demand = 0
        episode_price = []
        episode_rebalancing_cost = 0
        episode_waiting = 0
        obs = env.reset()
        # Original demand and price
        demand_original_steps.append(env.demand)
        price_original_steps.append(env.price)

        action_rl = [0]*env.nregion        
        done = False
        while not done:

            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                t = env.time
                for (i, j), flow_dict in env.demand.items():
                    if t in flow_dict:
                        accepted_demand_ij[(i, j)] += flow_dict[t]

                rejection_rates.append(info["rejection_rate"])

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward

                if args.strategy == "learned":
                    action_rl = model.select_action(o, deterministic=True)
                elif args.strategy == "equal":
                    action_rl = [1 / env.nregion for _ in range(env.nregion)]
                elif args.strategy == "none":
                    action_rl = [0.0 for _ in range(env.nregion)]
                else:
                    raise ValueError("Invalid strategy")
                
                if isinstance(action_rl[0], list) and len(action_rl) == 1:
                    action_rl = action_rl[0]

                actions.append(action_rl)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # print(desiredAcc)
                # print({env.region[i]: env.acc[env.region[i]][env.time+1] for i in range(len(env.region))})
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env,
                    "scenario_san_francisco4",
                    desiredAcc,
                    args.cplexpath,
                    args.directory, 
                )
                # Take rebalancing action in environment
                _, rebreward, done, _, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward

            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                t = env.time
                for (i, j), flow_dict in env.demand.items():
                    if t in flow_dict:
                        accepted_demand_ij[(i, j)] += flow_dict[t]
                rejection_rates.append(info["rejection_rate"])

                o = parser.parse_obs(obs=obs)

                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)  

                env.matching_update()
            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                t = env.time
                for (i, j), flow_dict in env.demand.items():
                    if t in flow_dict:
                        accepted_demand_ij[(i, j)] += flow_dict[t]
                rejection_rates.append(info["rejection_rate"])

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)

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
                    args.cplexpath,
                    args.directory, 
                )
                # Take rebalancing action in environment
                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed")  
            
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_waiting += info['served_waiting']
            
            actions.append(action_rl)
            if args.mode == 1:
                actions_price.append(np.mean(2*np.array(action_rl)))
            elif args.mode == 2:
                actions_price.append(np.mean(2*np.array(action_rl)[:,0]))
            rebalancing_cost.append(info["rebalancing_cost"])
            # queue.append([len(env.queue[i]) for i in sorted(env.queue.keys())])
            queue.append(np.mean([len(env.queue[i]) for i in env.queue.keys()]))
            
            if args.mode in [1, 2]:
                price_scalar_per_region = {
                    i: 2 * action_rl[i] if args.mode == 1 else 2 * action_rl[i][0]
                    for i in range(env.nregion)
                }
                price_scalar_steps.append(price_scalar_per_region)

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        mean_rejection = np.mean(rejection_rates) if rejection_rates else 0
        print(f"→ Rejection rate: {mean_rejection:.4f}")
    
        
        accepted_demand_by_origin = defaultdict(float)
        for (i, j), val in accepted_demand_ij.items():
            accepted_demand_by_origin[i] += val
        accepted_demand_by_origin_steps.append(accepted_demand_by_origin)

        # Log KPIs
        demand_scaled_steps.append(copy.deepcopy(env.demand))
        available_steps.append(env.acc)
        reb_steps.append(env.rebFlow)
        reb_ori_steps.append(env.rebFlow_ori)
        pax_steps.append(env.paxFlow)
        pax_wait.append(env.paxWait)
        rejection_rate_steps.append(mean_rejection)
        reb_od = 0
        for (o,d),flow in env.rebFlow.items():
            reb_od += sum(flow.values())
        reb_num.append(reb_od)
        actions_step.append(actions)
        if args.mode != 0:
            price_mean.append(np.mean(actions_price))
        
        rebalancing_cost_steps.append(rebalancing_cost)
        queue_steps.append(np.mean(queue))
        waiting_steps.append(episode_waiting/episode_served_demand)

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)
        arrivals.append(env.arrivals)


    # Save metrics file
    np.save(f"{args.directory}/{city}_actions_mode{args.mode}.npy", np.array(actions_step))
    np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(queue_steps))
    
    with open(f"{args.directory}/{city}_demand_origin_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(accepted_demand_by_origin_steps, f)

    if args.mode in [1, 2]:
        with open(f"{args.directory}/{city}_price_scalar_per_region_mode{args.mode}.pickle", "wb") as f:
            pickle.dump(price_scalar_steps, f)

    # np.save(f"{args.directory}/{city}_served_mode{args.mode}.npy", np.array([demands,arrivals]))
    if env.mode != 1: 
        # np.save(f"{args.directory}/{city}_cost_mode{args.mode}.npy", np.array(rebalancing_cost_steps))
        with open(f"{args.directory}/{city}_reb_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_steps, f)
        with open(f"{args.directory}/{city}_reb_ori_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_ori_steps, f)

    with open(f"{args.directory}/{city}_pax_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_steps, f)
    with open(f"{args.directory}/{city}_pax_wait_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_wait, f)                     
    
    # with open(f"{args.directory}/{city}_demand_ori_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(demand_original_steps, f)
    # with open(f"{args.directory}/{city}_price_ori_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(price_original_steps, f)

    with open(f"{args.directory}/{city}_demand_scaled_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(demand_scaled_steps, f)    
    # with open(f"{args.directory}/{city}_acc_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(available_steps, f)
    
    print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))
    print("Waiting time (mean, std):", np.mean(waiting_steps), np.std(waiting_steps))
    print("Queue length (mean, std):", np.mean(queue_steps), np.std(queue_steps))
    print("Arrivals (mean, std):", np.mean(arrivals), np.std(arrivals))
    print("Rebalancing trips (mean, std):", np.mean(reb_num), np.std(reb_num))
    print("Rejection rate (mean, std):", np.mean(rejection_rate_steps), np.std(rejection_rate_steps))
    if args.mode != 0:
        print("Price scalar (mean, std):", np.mean(price_mean), np.std(price_mean))
    
    # === Plot Heatmaps ===

    if args.mode != 1:
        plot_mean_rebalancing_per_region(
            reb_steps=reb_steps,
            geojson_path="data/nyc_zones.geojson",
            output_path=f"visualizations/rebalancing_map_{args.checkpoint_path}.png"
        )
    
    plot_mean_origin_demand_per_region(
        origin_steps=accepted_demand_by_origin_steps,
        geojson_path="data/nyc_zones.geojson",
        output_path=f"visualizations/demand_map_{args.checkpoint_path}.png"
    )
    
    if args.mode != 0:
        plot_mean_price_scalar_per_region(
            price_steps=price_scalar_steps,
            geojson_path="data/nyc_zones.geojson",
            output_path=f"visualizations/price_scalar_map_{args.checkpoint_path}.png"
        )
    