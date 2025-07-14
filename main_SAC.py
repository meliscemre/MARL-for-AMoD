from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
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
import random
from visualizations_multi_agent import (
    plot_mean_rebalancing_per_region_agents,
    plot_mean_accepted_demand_per_region_agents,
    plot_mean_price_scalar_per_region_agents
)



class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    Supports multi-agent inputs by taking agent_id and accessing agent-specific env state.
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs, agent_id):
        acc, time, dacc, demand = obs  # obs = self.obs[agent_id] from env
        opponent_id = 1 - agent_id
        region = self.env.region
        #acc, time, dacc, demand, opponent_price = obs  # obs = self.obs[agent_id] from env
        region = self.env.region
        T = self.T
        s = self.s

        x = (
            torch.cat(
                (
                    # Current availability at t+1
                    torch.tensor(
                        [acc[n].get(time + 1, 0) * s for n in region]
                    )
                    .view(1, 1, len(region))
                    .float(),

                    # Estimated availability for T future steps (acc + dacc)
                    torch.tensor(
                        [
                            [
                                (acc[n].get(time + 1, 0) + dacc[n].get(t, 0)) * s
                                for n in region
                            ]
                            for t in range(time + 1, time + T + 1)
                        ]
                    )
                    .view(1, T, len(region))
                    .float(),

                    # Queue length at region
                    torch.tensor(
                        [
                            len(self.env.queue_agents[agent_id][n]) * s
                            for n in region
                        ]
                    )
                    .view(1, 1, len(region))
                    .float(),

                    ## Current demand at time t
                    torch.tensor(
                        [
                            sum(
                                [
                                    demand[i, j].get(time, 0) * s
                                    for j in region
                                ]
                            )
                            for i in region
                        ]
                    )
                    .view(1, 1, len(region))
                    .float(),
                    
                    #OD Demand
                    #torch.tensor(
                    #    [
                    #        [
                    #            demand[i, j].get(time, 0) * s
                    #            for j in region
                    #        ]
                    #        for i in region
                    #    ]
                    #)
                    #.view(1, len(region), len(region))
                    #.float()

                    
                    # Own OD prices
                    torch.tensor(
                        [
                            [self.env.price_agents[agent_id][i, j].get(time, 0) * self.s for j in region]
                            for i in region
                        ]
                    )
                    .view(1, len(region), len(region))
                    .float(),


                    # Own origin sum prices
                    #torch.tensor(
                    #    [
                    #        sum(
                    #            [
                    #                self.env.price_agents[agent_id][i, j].get(time, 0) * s
                    #                for j in region
                    #            ]
                    #        )
                    #        for i in region
                    #    ]
                    #)
                    #.view(1, 1, len(region))
                    #.float(),
                    
                    # Opponent's OD prices
                    
                    torch.tensor(
                        [
                            [self.env.price_agents[opponent_id][i, j].get(time, 0) * self.s for j in region]
                            for i in region
                        ]
                    ).view(1, len(region), len(region)).float()
                    
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + T + 1 + 1 + len(region) + len(region), len(region))
            .T
        )

        # Build edge index
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
                    torch.arange(len(region)).view(1, len(region)),
                    torch.arange(len(region)).view(1, len(region)),
                ),
                dim=0,
            ).long()

        return Data(x, edge_index)



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
    choices=[0, 1, 2], 
    help='Rebalancing mode. (0: manual, 1: pricing, 2: both. default 0)'
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
    default=0.1,
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
    args.directory = f"saved_files_two_agents_mode{args.mode}_od_information_sharing_wage_25_monopoly_to_competititon_400_500"
if args.checkpoint_path is None:
    args.checkpoint_path = f"SAC_two_agents_mode{args.mode}_od_information_sharing_wage_25_monopoly_to_competititon_400_500"
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city

#wandb

wandb.init(
    project="amod-rl",
    entity="meliscemre-technical-university-of-denmark",
    name=f'{args.checkpoint_path}',
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
                acc_rate_episode.append(1 - np.mean([info[a]["rejection_rate"] for a in [0, 1]]))
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
            plt.title(f"Log-Scaled Origin-Destination Demand Heatmap (0 to {simulation_end_time})")
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
        plt.savefig(f"visualizations/avg_price_heatmap_melis_{args.checkpoint_path}.png")

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
    
    total_acc = env.scenario.totalAcc
    
    if not args.small:
        parser = GNNParser(
            env, T=6, json_file=f"data/scenario_nyc_brooklyn.json"
        )  # Timehorizon T=6 (K in paper)
    else:
        parser = GNNParser(
            env, T=6
        )  # Timehorizon T=6 (K in paper)

    model_agents = {
    0: SAC(
        env=env,
        input_size=45,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version=args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device),
    1: SAC(
        env=env,
        input_size=45,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version=args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)
}

    if args.load:
        print("Loading checkpoints for agents 0 and 1")
        for a in [0, 1]:
            model_agents[a].load_checkpoint(path=f"ckpt/{args.checkpoint_path}_agent{a}.pth")


    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    for a in [0, 1]:
        model_agents[a].train()


    log_lines = []

    # Check metrics
    epoch_demand_list = []
    epoch_reward_list = []
    #epoch_waiting_list = []
    epoch_servedrate_list = []
    epoch_rebalancing_cost = []
    epoch_value1_list = []
    epoch_value2_list = []
    total_episode_reward_appended = []
    price_history = []

    all_rejection_rates_0 = []
    all_rejection_rates_1 = []
        
    for i_episode in epochs:
        obs = env.reset()
        
        action_rl = {
            a: [0.0] * env.nregion for a in [0, 1]
        }

        # Save original demand for reference
        demand_ori = nestdictsum(env.demand)
        if i_episode == train_episodes - 1:
            export = {"demand_ori":copy.deepcopy(env.demand)}
        #action_rl = [0]*env.nregion
        
        episode_reward = {0: 0, 1: 0}
        episode_served_demand = {0: 0, 1: 0}
        episode_unserved_demand = {0: 0, 1: 0}
        episode_rebalancing_cost = {0: 0, 1: 0}
        episode_rejected_demand = {0: 0, 1: 0}
        episode_total_revenue = {0: 0, 1: 0}
        episode_total_operating_cost = {0: 0, 1: 0}
        episode_waiting = {0: 0, 1: 0}
        episode_rejection_rates = {0: [], 1: []}
        actions = {0: [], 1: []}
        

        done = False
        step = 0
        while not done:
            
            #mode 0: only rebalancing
            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                o_next = {
                    0: parser.parse_obs(obs=obs[0], agent_id=0),
                    1: parser.parse_obs(obs=obs[1], agent_id=1)
                }
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                if step > 0:
                    for a in [0,1]:
                        rl_reward = paxreward[a] + rebreward[a]
                        model_agents[a].replay_buffer.store(
                            o_prev[a], action_rl[a], args.rew_scale * rl_reward, o_next[a]
                        )

                action_rl = {
                    a: model_agents[a].select_action(o_next[a]) if args.strategy == "learned" else (
                        [1/env.nregion]*env.nregion if args.strategy == "equal" else [0.0]*env.nregion
                    )
                    for a in [0,1]
                }
                
                #to try one agent equal distribution
                #action_rl[0] = [1 / env.nregion] * env.nregion
                
                
                desiredAcc = {
                    a: {
                        env.region[i]: int(action_rl[a][i] * dictsum(env.acc_agents[a], env.time + 1))
                        for i in range(env.nregion)
                    }
                    for a in [0, 1]
                }


                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory)
                    for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }

                new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)

                o_prev = o_next
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}



            # mode 1: only pricing
            
            if env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o_next = {
                a: parser.parse_obs(obs=obs[a], agent_id=a)
                for a in [0, 1]
            }

                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}


                if step > 0:
                    for a in [0, 1]:
                        rl_reward = paxreward[a]
                        model_agents[a].replay_buffer.store(
                            o_prev[a], action_rl[a], args.rew_scale * rl_reward, o_next[a]
                        )

                action_rl = {
                    a: model_agents[a].select_action(o_next[a]) for a in [0, 1]
                }


                # Save current obs for next transition
                o_prev = o_next

                # Matching update (global step)
                env.matching_update()


            #mode 2: rebalancing & pricing

            if env.mode == 2:
                # --- Matching step ---
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
            
                o_next = {
                    0: parser.parse_obs(obs=obs[0], agent_id=0),
                    1: parser.parse_obs(obs=obs[1], agent_id=1)
                }
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            
                if step > 0:
                    for a in [0,1]:
                        rl_reward = paxreward[a] + rebreward[a]
                        model_agents[a].replay_buffer.store(
                            o_prev[a], action_rl[a], args.rew_scale * rl_reward, o_next[a]
                        )
            
                # --- Action selection ---
                action_rl = {
                    a: model_agents[a].select_action(o_next[a]) for a in [0,1]
                }
                

                # Force agent 0's rebalancing to equal distribution, keep its learned pricing
                #for i in range(env.nregion):
                #    price = action_rl[0][i][0]              # keep original learned price scalar
                #    action_rl[0][i] = (price, 1.0 / env.nregion)  # override rebalancing to equal

                            
                # --- Desired Acc computation ---
                desiredAcc = {
                    a: {
                        env.region[i]: int(action_rl[a][i][-1] * dictsum(env.acc_agents[a], env.time + 1))
                        for i in range(env.nregion)
                    } for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }
                
                
                # --- Rebalancing step ---
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory)
                    for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }
            
                new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            
                # --- Save for next step ---
                o_prev = o_next
            
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
                

            # track performance over episode
            for a in [0, 1]:
                episode_served_demand[a] += info[a]["served_demand"]
                episode_unserved_demand[a] += info[a]["unserved_demand"]
                episode_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                episode_total_revenue[a] += info[a]["revenue"]
                episode_rejected_demand[a] += info[a]["rejected_demand"]
                episode_total_operating_cost[a] += info[a]["operating_cost"]
                episode_waiting[a] += info[a]["served_waiting"]
                episode_rejection_rates[a].append(info[a]["rejection_rate"])

            for a in [0, 1]:
                actions[a].append(action_rl[a])
                
            #if args.mode in [1, 2]:
            #    for a in [0, 1]:
            #        # Assuming action_rl[a][i] = (price_scalar, reb_scalar)
            #        mean_price_scalar = np.mean([2 * action_rl[a][i][0] for i in range(env.nregion)])
            #        wandb.log({f"price_scalar_agent{a}": mean_price_scalar})


            step += 1
            
            grad_norms_agents = {}

            if i_episode > 10:
                for a in [0,1]:
                    batch = model_agents[a].replay_buffer.sample_batch(args.batch_size, norm=False)
                    grad_norms_agents[a] = model_agents[a].update(data=batch)
            else:
                grad_norms_agents = {
                    a: {
                        "actor_grad_norm": 0,
                        "critic1_grad_norm": 0,
                        "critic2_grad_norm": 0,
                        "actor_loss": 0,
                        "critic1_loss": 0,
                        "critic2_loss": 0,
                        "Q1_value": 0,
                        "Q2_value": 0,
                    } for a in [0,1]
                }


            # Keep track of loss
            epoch_value1_list.append((grad_norms_agents[0]["Q1_value"] + grad_norms_agents[1]["Q1_value"]) / 2)
            epoch_value2_list.append((grad_norms_agents[0]["Q2_value"] + grad_norms_agents[1]["Q2_value"]) / 2)
            
            ##to try one agent equal distribution
            #epoch_value1_list.append(grad_norms_agents[1]["Q1_value"])
            #epoch_value2_list.append(+ grad_norms_agents[1]["Q2_value"])


        # Keep metrics
        epoch_reward_list.append(episode_reward)
        epoch_demand_list.append(env.arrivals)
        #epoch_waiting_list.append(episode_waiting/episode_served_demand)
        
        if env.arrivals > 0:
            total_served = sum(episode_served_demand.values())
            epoch_servedrate_list.append(total_served / env.arrivals)
        else:
            epoch_servedrate_list.append(0)

        epoch_rebalancing_cost.append(episode_rebalancing_cost)

        # Keep price (only needed for pricing training)
        price_history.append(actions)
        
        if args.mode == 2:
            # action_rl[a][i] is [price, reb]
            price_scalar_agent0 = np.mean([2 * action_rl[0][i][0] for i in range(env.nregion)])
            price_scalar_agent1 = np.mean([2 * action_rl[1][i][0] for i in range(env.nregion)])
        elif args.mode == 1:
            # action_rl[a][i] is just price scalar
            price_scalar_agent0 = np.mean([2 * action_rl[0][i] for i in range(env.nregion)])
            price_scalar_agent1 = np.mean([2 * action_rl[1][i] for i in range(env.nregion)])
        else:
            price_scalar_agent0 = None
            price_scalar_agent1 = None


        #wandb
        wandb.log({
            "episode": i_episode,
            "episode_reward_total": sum(episode_reward.values()),
            "episode_reward_agent0": episode_reward[0],
            "episode_reward_agent1": episode_reward[1],
            "served_demand_agent0": episode_served_demand[0],
            "served_demand_agent1": episode_served_demand[1],
            "unserved_demand_agent0": episode_unserved_demand[0],
            "unserved_demand_agent1": episode_unserved_demand[1],
            "rebalancing_cost_agent0": episode_rebalancing_cost[0],
            "rebalancing_cost_agent1": episode_rebalancing_cost[1],
            "rejected_demand_agent0": episode_rejected_demand[0],
            "rejected_demand_agent1": episode_rejected_demand[1],
            "revenue_agent0": episode_total_revenue[0],
            "revenue_agent1": episode_total_revenue[1],
            "operating_cost_agent0": episode_total_operating_cost[0],
            "operating_cost_agent1": episode_total_operating_cost[1],
            "waiting_time_agent0": episode_waiting[0] / episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
            "waiting_time_agent1": episode_waiting[1] / episode_served_demand[1] if episode_served_demand[1] > 0 else 0,
            "rejection_rate_agent0": np.mean(episode_rejection_rates[0]) if episode_rejection_rates[0] else 0,
            "rejection_rate_agent1": np.mean(episode_rejection_rates[1]) if episode_rejection_rates[1] else 0,
            "actor_loss_agent0": grad_norms_agents[0]["actor_loss"],
            "actor_loss_agent1": grad_norms_agents[1]["actor_loss"],
            "critic1_loss_agent0": grad_norms_agents[0]["critic1_loss"],
            "critic1_loss_agent1": grad_norms_agents[1]["critic1_loss"],
            "critic2_loss_agent0": grad_norms_agents[0]["critic2_loss"],
            "critic2_loss_agent1": grad_norms_agents[1]["critic2_loss"],
            "Q1_value_agent0": grad_norms_agents[0]["Q1_value"],
            "Q1_value_agent1": grad_norms_agents[1]["Q1_value"],
            "Q2_value_agent0": grad_norms_agents[0]["Q2_value"],
            "Q2_value_agent1": grad_norms_agents[1]["Q2_value"],
            "price_scalar_agent0": price_scalar_agent0,
            "price_scalar_agent1": price_scalar_agent1
        }, step=i_episode, commit=False)
        
        mean_rej_rate_0 = np.mean(episode_rejection_rates[0]) if episode_rejection_rates[0] else 0
        mean_rej_rate_1 = np.mean(episode_rejection_rates[1]) if episode_rejection_rates[1] else 0

        all_rejection_rates_0.append(mean_rej_rate_0)
        all_rejection_rates_1.append(mean_rej_rate_1)


        episode_total_revenue_sum = sum(episode_total_revenue.values())
        total_episode_reward = sum(episode_reward.values())
        episode_rebalancing_cost_sum = sum(episode_rebalancing_cost.values())
        episode_total_operating_cost_sum = sum(episode_total_operating_cost.values())
        total_episode_reward_appended.append(total_episode_reward)

        log_line = (
            f"Episode {i_episode+1} | Reward: {total_episode_reward:.2f} | Agents Reward: {{0: {episode_reward[0]:.2f}, 1: {episode_reward[1]:.2f}}} | Revenue: {{0: {episode_total_revenue[0]:.2f}, 1: {episode_total_revenue[1]:.2f}}} | "
            f"Reb. Cost: {episode_rebalancing_cost_sum:.2f} | Agent Reb. Cost: {{0: {episode_rebalancing_cost[0]:.2f}, 1: {episode_rebalancing_cost[1]:.2f}}} | "
            f"Op. Cost: {episode_total_operating_cost_sum:.2f} | Agent Op. Cost: {{0: {episode_total_operating_cost[0]:.2f}, 1: {episode_total_operating_cost[1]:.2f}}} | "
            f"Tot.Demand: {env.arrivals} | ServedD: {episode_served_demand} | "
            f"UnservedD: {episode_unserved_demand} | Wait.: {sum(len(q) for agent_queue in env.queue_agents.values() for q in agent_queue.values())} | "
            f"Rej. Rate: {np.mean(episode_rejection_rates[0])} | RejectedD: {episode_rejected_demand[0]}"
        )
        epochs.set_description(log_line)
        log_lines.append(log_line)
        wandb.termlog(log_line)
        


        # Checkpoint best performing model
        # Checkpoint best performing model based on total training reward
        if total_episode_reward >= best_reward:
            for a in [0, 1]:
                model_agents[a].save_checkpoint(path=f"ckpt/{args.checkpoint_path}_agent{a}_sample.pth")
            best_reward = total_episode_reward

        # Always save the most recent model
        for a in [0, 1]:
            model_agents[a].save_checkpoint(path=f"ckpt/{args.checkpoint_path}_agent{a}_running.pth")

        # Every 100 episodes, run multi-agent test eval
        if i_episode % 100 == 0:
            test_reward = model_agents[0].test_agent(
                test_episodes=10,
                env=env,
                model_agents=model_agents,
                parser=parser,
                args=args,
                cplexpath=args.cplexpath,
                directory=args.directory
            )
            
            wandb.log({
                "test_every100_total_reward": test_reward
            }, step=i_episode)  # default commit=True

            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                for a in [0, 1]:
                    model_agents[a].save_checkpoint(path=f"ckpt/{args.checkpoint_path}_agent{a}_test.pth")
        else:
            wandb.log({}, step=i_episode)
                                
                    
    mean_rej_rate_0_all = np.mean(all_rejection_rates_0)
    mean_rej_rate_1_all = np.mean(all_rejection_rates_1)
    
    print(f"\nMean rejection rate over {train_episodes} episodes:")
    print(f"  Agent 0: {mean_rej_rate_0_all:.4f}")
    print(f"  Agent 1: {mean_rej_rate_1_all:.4f}")

    
    mean_total_episode_reward = np.mean(total_episode_reward_appended)
    print(f"\nMean total episode reward over {train_episodes} episodes: {mean_total_episode_reward:.2f}")

    # Save metrics file
    metricPath = f"{args.directory}/train_logs/"
    if not os.path.exists(metricPath):
        os.makedirs(metricPath)
    np.save(f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", np.array([epoch_reward_list,epoch_servedrate_list,epoch_demand_list,epoch_rebalancing_cost]))
    np.save(f"{args.directory}/train_logs/{city}_price_mode{args.mode}_{train_episodes}.npy", np.array(price_history))
    np.save(f"{args.directory}/train_logs/{city}_q_mode{args.mode}_{train_episodes}.npy", np.array([epoch_value1_list,epoch_value2_list]))

    export["avail_distri"] = env.acc_agents
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
        d = {(2, 3): 6, (2, 0): 4, (0, 3): 4, "default": 1}
        r = {
            0: [1]*12, 1: [1]*12, 2: [1]*12, 3: [1]*12,
        }
        scenario = Scenario(tf=20, demand_input=d, demand_ratio=r, ninit=30, N1=2, N2=2)
        
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
        plt.title(f"Origin-Destination Demand Heatmap")
        plt.savefig(f"visualizations/demand_heatmap_melis_{args.checkpoint_path}.png")
        plt.figure(figsize=(12, 10))
        sns.heatmap(np.log1p(demand_matrix), annot=True, fmt=".1f", cmap="coolwarm",
                    xticklabels=destinations, yticklabels=origins)
        plt.xlabel("Destination")
        plt.ylabel("Origin")
        plt.title(f"Log-Scaled Origin-Destination Demand Heatmap")
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
    plt.title(f"Average Travel Cost")
    plt.savefig(f"visualizations/avg_price_heatmap_melis_{args.checkpoint_path}.png")
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.log1p(price_matrix), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=destinations, yticklabels=origins)
    plt.xlabel("Destination j")
    plt.ylabel("Origin i")
    plt.title(f"Log-Scaled Average Travel Cost")
    plt.savefig(f"visualizations/avg_price_heatmap_log_melis_{args.checkpoint_path}.png")
        

    # --- END: Demand & Price Heatmaps and Plots ---
    
    env = AMoD(scenario, args.mode, beta=args.beta, jitter=args.jitter, max_wait=args.maxt, choice_price_mult=1.0)
    
    parser = GNNParser(env, T=6, json_file=f"data/scenario_nyc_brooklyn.json" if not args.small else None)
    
    model_agents = {
        a: SAC(
            env=env,
            input_size=45,
            hidden_size=args.hidden_size,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            alpha=args.alpha,
            batch_size=args.batch_size,
            use_automatic_entropy_tuning=False,
            clip=args.clip,
            critic_version=args.critic_version,
            price_version=args.price_version,
            mode=args.mode,
            q_lag=args.q_lag
        ).to(device)
        for a in [0, 1]
    }
    
    if args.load:
        for a in [0, 1]:
            model_agents[a].load_checkpoint(path=f"ckpt/{args.checkpoint_path}_agent{a}_test.pth")
    
    test_episodes = 10
    epochs = trange(test_episodes)
    
    SEED = args.seed if hasattr(args, 'seed') else 42 

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    
    results = {
        "total_reward": [],
        "agent_reward_0": [],
        "agent_reward_1": [],
        "rebalancing_cost_0": [],
        "rebalancing_cost_1": [],
        "operating_cost": [],
        "operating_cost_0": [],
        "operating_cost_1": [],
        "waiting_steps": [],
        "queue_steps": [],
        "reb_num": [],
        "actions_step": [],
        "served_demand": [],
        "served_demand_0": [],
        "served_demand_1": [],
        "rebalancing_cost": [],
        "total_demand" : [],
        "rejection_rate" : [],
        "reb_num_agent0": [],
        "reb_num_agent1": [],
        "price_scalar_agent_0": [],
        "price_scalar_agent_1": [],
         "queue_steps_agent_0": [],
        "queue_steps_agent_1": [],
        "waiting_steps_agent_0": [],
        "waiting_steps_agent_1": [],
        "accepted_demand_agent0": [],
        "accepted_demand_agent1": [],
        "price_scalar_per_region_agent0": [],
        "price_scalar_per_region_agent1": [],

    }
   
    reb_steps_agent_0 = []
    reb_steps_agent_1 = []
    
    for episode in range(test_episodes):
        obs = env.reset()
        done = False
        step = 0

        episode_reward = {0: 0, 1: 0}
        episode_served_demand = {0: 0, 1: 0}
        episode_unserved_demand = {0: 0, 1: 0}
        episode_rebalancing_cost = {0: 0, 1: 0}
        episode_total_revenue = {0: 0, 1: 0}
        episode_total_operating_cost = {0: 0, 1: 0}
        episode_rejected_demand = {0: 0, 1: 0}
        episode_waiting = {0: 0, 1: 0}
        episode_rejection_rates = {0: [], 1: []}
        actions = {a: [] for a in [0, 1]}
        action_rl = {
            a: [0.0] * env.nregion for a in [0, 1]
        }
    
        while not done:

            if env.mode == 0:
                # === REBALANCING ONLY ===
                obs, paxreward, done, info, _, _ = env.match_step_simple()

                # Parse new observation
                o_next = {
                    a: parser.parse_obs(obs=obs[a], agent_id=a)
                    for a in [0, 1]
                }
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                action_rl = {}
                for a in [0,1]:
                    if args.strategy == "learned":
                        action_rl[a] = model_agents[a].select_action(o_next[a], deterministic=True)
                    elif args.strategy == "equal":
                        action_rl[a] = [1 / env.nregion] * env.nregion
                    elif args.strategy == "none":
                        action_rl[a] = [0.0] * env.nregion
                    else:
                        raise ValueError(f"Unknown strategy: {args.strategy}")
                    
                #action_rl[0] = [1 / env.nregion] * env.nregion
                    
                # Desired rebalancing
                desiredAcc = {
                    a: {
                        env.region[i]: int(action_rl[a][i] * dictsum(env.acc_agents[a], env.time + 1))
                        for i in range(env.nregion)
                    } for a in [0, 1]
                }

                # Solve rebalance & apply
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory)
                    for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }

                obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}

            elif env.mode == 1:
                # === PRICING ONLY ===
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o_next = {
                    a: parser.parse_obs(obs=obs[a], agent_id=a)
                    for a in [0, 1]
                }
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}
                
                action_rl = {
                    a: model_agents[a].select_action(o_next[a], deterministic=True) for a in [0, 1]
                }

                env.matching_update()

            elif env.mode == 2:
                # === REBALANCING + PRICING ===
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o_next = {
                    a: parser.parse_obs(obs=obs[a], agent_id=a)
                    for a in [0, 1]
                }
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                action_rl = {
                    a: model_agents[a].select_action(o_next[a], deterministic=True) for a in [0, 1]
                }
                
                
                #for i in range(env.nregion):
                #    price = action_rl[0][i][0]
                #    action_rl[0][i] = (price, 1.0 / env.nregion)

                desiredAcc = {
                    a: {
                        env.region[i]: int(action_rl[a][i][-1] * dictsum(env.acc_agents[a], env.time + 1))
                        for i in range(env.nregion)
                    } for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }

                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory)
                    for a in [0, 1] if dictsum(env.acc_agents[a], env.time + 1) > 0
                }

                obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}

            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed.")

            # === COMMON METRIC TRACKING ===
            for a in [0, 1]:
                episode_served_demand[a] += info[a]["served_demand"]
                episode_unserved_demand[a] += info[a]["unserved_demand"]
                episode_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                episode_waiting[a] += info[a]["served_waiting"]
                episode_rejected_demand[a] += info[a]["rejected_demand"]
                episode_total_revenue[a] += info[a]["revenue"]
                episode_total_operating_cost[a] += info[a]["operating_cost"]
                episode_rejection_rates[a].append(info[a]["rejection_rate"])
                actions[a].append(action_rl[a])
                
            if args.mode in [1, 2]:
                for a in [0, 1]:
                    price_region = {i: 2 * action_rl[a][i][0] if args.mode == 2 else 2 * action_rl[a][i]
                                    for i in range(env.nregion)}
                    results[f"price_scalar_per_region_agent{a}"].append(price_region)

                

            step += 1
        
        # === Post-episode logging: mirror training ===
        episode_total_revenue_sum = sum(episode_total_revenue.values())
        episode_total_operating_cost_sum = sum(episode_total_operating_cost.values())
        episode_rebalancing_cost_sum = sum(episode_rebalancing_cost.values())
        total_episode_reward = sum(episode_reward.values())
        results["total_reward"].append(total_episode_reward)
        results["served_demand"].append(sum(episode_served_demand.values()))
        results["rebalancing_cost"].append(episode_rebalancing_cost_sum)
        results["total_demand"].append(env.arrivals)
        results["rejection_rate"].append(np.mean([np.mean(episode_rejection_rates[a]) for a in [0, 1]]))
        results["agent_reward_0"].append(episode_reward[0])
        results["agent_reward_1"].append(episode_reward[1])
        results["rebalancing_cost_0"].append(episode_rebalancing_cost[0])
        results["rebalancing_cost_1"].append(episode_rebalancing_cost[1])
        results["operating_cost_0"].append(episode_total_operating_cost[0])
        results["operating_cost_1"].append(episode_total_operating_cost[1])
        results["served_demand"].append(sum(episode_served_demand.values()))
        results["served_demand_0"].append(episode_served_demand[0])
        results["served_demand_1"].append(episode_served_demand[1])
        results["rebalancing_cost"].append(sum(episode_rebalancing_cost.values()))
        results["total_demand"].append(env.arrivals)
        results["rejection_rate"].append(np.mean([np.mean(episode_rejection_rates[a]) for a in [0, 1]]))
        results["operating_cost"].append(episode_total_operating_cost_sum)
        results["accepted_demand_agent0"].append(copy.deepcopy(env.accepted_demand_agents[0]))
        results["accepted_demand_agent1"].append(copy.deepcopy(env.accepted_demand_agents[1]))

        reb_steps_agent_0.append(copy.deepcopy(env.rebFlow_agents[0]))
        reb_steps_agent_1.append(copy.deepcopy(env.rebFlow_agents[1]))


        log_line = (
            f"Episode {episode+1} | Reward: {total_episode_reward:.2f} | Agents Reward: {{0: {episode_reward[0]:.2f}, 1: {episode_reward[1]:.2f}}} | Revenue: {episode_total_revenue_sum:.2f} | "
            f"Reb. Cost: {episode_rebalancing_cost_sum:.2f} | Agent Reb. Cost: {{0: {episode_rebalancing_cost[0]:.2f}, 1: {episode_rebalancing_cost[1]:.2f}}} | "
            f"Op. Cost: {episode_total_operating_cost_sum:.2f} | Agent Op. Cost: {{0: {episode_total_operating_cost[0]:.2f}, 1: {episode_total_operating_cost[1]:.2f}}} | "
            f"Tot.Demand: {env.arrivals} | ServedD: {episode_served_demand} | "
            f"UnservedD: {episode_unserved_demand} | Wait.: {sum(len(q) for agent_queue in env.queue_agents.values() for q in agent_queue.values())} | "
            f"Rej. Rate: {np.mean(episode_rejection_rates[0]):.4f} | RejectedD: {episode_rejected_demand[0]}"
        )
        epochs.set_description(log_line)
        wandb.termlog(log_line)

        wandb.log({
            "test_episode": episode,
            "test_episode_reward_total": total_episode_reward,
            "test_episode_reward_agent0": episode_reward[0],
            "test_episode_reward_agent1": episode_reward[1],
            "test_served_demand_agent0": episode_served_demand[0],
            "test_served_demand_agent1": episode_served_demand[1],
            "test_unserved_demand_agent0": episode_unserved_demand[0],
            "test_unserved_demand_agent1": episode_unserved_demand[1],
            "test_rebalancing_cost_agent0": episode_rebalancing_cost[0],
            "test_rebalancing_cost_agent1": episode_rebalancing_cost[1],
            "test_rejected_demand_agent0": episode_rejected_demand[0],
            "test_rejected_demand_agent1": episode_rejected_demand[1],
            "test_revenue_agent0": episode_total_revenue[0],
            "test_revenue_agent1": episode_total_revenue[1],
            "test_operating_cost_agent0": episode_total_operating_cost[0],
            "test_operating_cost_agent1": episode_total_operating_cost[1],
            "test_waiting_time_agent0": episode_waiting[0] / episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
            "test_waiting_time_agent1": episode_waiting[1] / episode_served_demand[1] if episode_served_demand[1] > 0 else 0,
            "test_rejection_rate_agent0": np.mean(episode_rejection_rates[0]) if episode_rejection_rates[0] else 0,
            "test_rejection_rate_agent1": np.mean(episode_rejection_rates[1]) if episode_rejection_rates[1] else 0,
        })
        

    
        wait_avg = np.mean([
            episode_waiting[a] / episode_served_demand[a] if episode_served_demand[a] > 0 else 0
            for a in [0, 1]
        ])
        
        for a in [0, 1]:
            if episode_served_demand[a] > 0:
                mean_wait = episode_waiting[a] / episode_served_demand[a]
            else:
                mean_wait = 0
            results[f"waiting_steps_agent_{a}"].append(mean_wait)
            
        queue_avg = np.mean([
            np.mean([len(env.queue_agents[a][n]) for n in env.region])
            for a in [0, 1]
        ])
        
        for a in [0, 1]:
            mean_queue = np.mean([len(env.queue_agents[a][n]) for n in env.region])
            results[f"queue_steps_agent_{a}"].append(mean_queue)

    
        results["waiting_steps"].append(wait_avg)
        results["queue_steps"].append(queue_avg)
    
        if args.mode in [1, 2]:
            for a in [0, 1]:
                key = f"price_scalar_values_agent{a}"
                if key not in results:
                    results[key] = []
        
                for step in actions[a]:
                    for i, action in enumerate(step):
                        if args.mode == 2:
                            try:
                                price_scalar, _ = action
                                results[key].append(2 * price_scalar)
                            except Exception as e:
                                print(f"Malformed action at region {i}: {action}")
                        elif args.mode == 1:
                            results[key].append(2 * action)



    
        reb_num_agent0 = sum(sum(f.values()) for f in env.rebFlow_agents[0].values())
        reb_num_agent1 = sum(sum(f.values()) for f in env.rebFlow_agents[1].values())
        results["reb_num"].append(reb_num_agent0 + reb_num_agent1)
        results["reb_num_agent0"].append(reb_num_agent0)
        results["reb_num_agent1"].append(reb_num_agent1)

        results["actions_step"].append(actions)
    
        #epochs.set_description(
        #    f"Episode {episode+1} | Reward: {total_reward:.2f} | Served: {total_demand} | Rebal. Cost: {total_cost:.2f}"
        #)
    
    # Save logs
    np.save(f"{args.directory}/{city}_rewards_mode{args.mode}.npy", np.array(results["total_reward"]))
    np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(results["queue_steps"]))
    np.save(f"{args.directory}/{city}_waiting_mode{args.mode}.npy", np.array(results["waiting_steps"]))
    np.save(f"{args.directory}/{city}_queue_agent0_mode{args.mode}.npy", np.array(results["queue_steps_agent_0"]))
    np.save(f"{args.directory}/{city}_queue_agent1_mode{args.mode}.npy", np.array(results["queue_steps_agent_1"]))
    np.save(f"{args.directory}/{city}_waiting_agent0_mode{args.mode}.npy", np.array(results["waiting_steps_agent_0"]))
    np.save(f"{args.directory}/{city}_waiting_agent1_mode{args.mode}.npy", np.array(results["waiting_steps_agent_1"]))

    if args.mode in [1, 2]:
        for a in [0, 1]:
            np.save(
                f"{args.directory}/{city}_price_mean_agent{a}_mode{args.mode}.npy",
                np.array(results[f"price_scalar_agent_{a}"]))
        
    with open(f"{args.directory}/{city}_actions_mode{args.mode}.pickle", "wb") as f:
        pickle.dump(results["actions_step"], f)
        
    
    print("Test complete.")
    print("Rewards (mean, std):", np.mean(results["total_reward"]), np.std(results["total_reward"]))
    print("Reward Agent 0 (mean, std):", np.mean(results["agent_reward_0"]), np.std(results["agent_reward_0"]))
    print("Reward Agent 1 (mean, std):", np.mean(results["agent_reward_1"]), np.std(results["agent_reward_1"]))
    print("Served Demand (mean, std):", np.mean(results["served_demand"]), np.std(results["served_demand"]))
    print("Served Demand Agent 0 (mean, std):", np.mean(results["served_demand_0"]), np.std(results["served_demand_0"]))
    print("Served Demand Agent 1 (mean, std):", np.mean(results["served_demand_1"]), np.std(results["served_demand_1"]))
    print("Rebalancing Cost (mean, std):", np.mean(results["rebalancing_cost"]), np.std(results["rebalancing_cost"]))
    print("Rebalancing Cost Agent 0 (mean, std):", np.mean(results["rebalancing_cost_0"]), np.std(results["rebalancing_cost_0"]))
    print("Rebalancing Cost Agent 1 (mean, std):", np.mean(results["rebalancing_cost_1"]), np.std(results["rebalancing_cost_1"]))
    print("Operating Cost (mean, std):", np.mean(results["operating_cost"]), np.std(results["operating_cost"]))
    print("Operating Cost Agent 0 (mean, std):", np.mean(results["operating_cost_0"]), np.std(results["operating_cost_0"]))
    print("Operating Cost Agent 1 (mean, std):", np.mean(results["operating_cost_1"]), np.std(results["operating_cost_1"]))
    print("Waiting time (mean, std):", np.mean(results["waiting_steps"]), np.std(results["waiting_steps"]))
    print("Waiting Time Agent 0 (mean, std):", np.mean(results["waiting_steps_agent_0"]), np.std(results["waiting_steps_agent_0"]))
    print("Waiting Time Agent 1 (mean, std):", np.mean(results["waiting_steps_agent_1"]), np.std(results["waiting_steps_agent_1"]))
    print("Queue length (mean, std):", np.mean(results["queue_steps"]), np.std(results["queue_steps"]))
    print("Queue Agent 0 (mean, std):", np.mean(results["queue_steps_agent_0"]), np.std(results["queue_steps_agent_0"]))
    print("Queue Agent 1 (mean, std):", np.mean(results["queue_steps_agent_1"]), np.std(results["queue_steps_agent_1"]))
    print("Total demand (mean, std):", np.mean(results["total_demand"]), np.std(results["total_demand"]))
    print("Rebalancing trips (mean, std):", np.mean(results["reb_num"]), np.std(results["reb_num"]))
    print("Rebalancing Trips Agent 0 (mean, std):", np.mean(results["reb_num_agent0"]), np.std(results["reb_num_agent0"]))
    print("Rebalancing Trips Agent 1 (mean, std):", np.mean(results["reb_num_agent1"]), np.std(results["reb_num_agent1"]))
    print("Rejection rate (mean, std):", np.mean(results["rejection_rate"]), np.std(results["rejection_rate"]))
    if args.mode in [1, 2]:
        for a in [0, 1]:
            all_prices = results[f"price_scalar_values_agent{a}"]
            print(f"Price scalar Agent {a} (mean, std):", np.mean(all_prices), np.std(all_prices))
   
    if args.mode != 1:    
        plot_mean_rebalancing_per_region_agents(
            reb_steps_agent0=reb_steps_agent_0,
            reb_steps_agent1=reb_steps_agent_1,
            geojson_path="data/nyc_zones.geojson",
            out0=f"visualizations/rebalancing_map_{args.checkpoint_path}_agent0.png",
            out1=f"visualizations/rebalancing_map_{args.checkpoint_path}_agent1.png"
        )

    plot_mean_accepted_demand_per_region_agents(
        demand_agent0=results["accepted_demand_agent0"],
        demand_agent1=results["accepted_demand_agent1"],
        geojson_path="data/nyc_zones.geojson",
        out0=f"visualizations/demand_map_{args.checkpoint_path}_agent0.png",
        out1=f"visualizations/demand_map_{args.checkpoint_path}_agent1.png"
    )
    

    if args.mode in [1, 2]:
        plot_mean_price_scalar_per_region_agents(
            price_agent0=results["price_scalar_per_region_agent0"],
            price_agent1=results["price_scalar_per_region_agent1"],
            geojson_path="data/nyc_zones.geojson",
            out0=f"visualizations/price_scalar_map_{args.checkpoint_path}_agent0.png",
            out1=f"visualizations/price_scalar_map_{args.checkpoint_path}_agent1.png"
        )
        


