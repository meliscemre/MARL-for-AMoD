"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator.
"""
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import subprocess
import os
import networkx as nx
from src.misc.utils import mat2str
from src.misc.helper_functions import demand_update
from src.envs.structures import generate_passenger
from copy import deepcopy
import json
import random
from src.misc.wage_utils import generate_fixed_od_wages


class AMoD:
    # initialization
    # updated to take scenario and beta (cost for rebalancing) as input
    def __init__(self, scenario, mode, beta=0.2, jitter=0, max_wait=2, choice_price_mult=1.0):
        # I changed it to deep copy so that the scenario input is not modified by env
        self.scenario = deepcopy(scenario)
        self.mode = mode  # Mode of rebalancing
        self.jitter = jitter # Jitter for zero demand
        self.max_wait = max_wait # Maximum passenger waiting time
        self.G = scenario.G  # Road Graph: node - regiocon'dn, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0  # current time
        self.choice_price_mult = choice_price_mult
        self.tf = scenario.tf  # final time
        self.agents = [0, 1]
        self.tstep = scenario.tstep
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)  # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)
        self.price_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }
        self.arrivals_agents = {
            0: 0,
            1: 0
        }
        
        self.demand_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }

        self.accepted_demand_agents = {
            0: defaultdict(lambda: defaultdict(float)),
            1: defaultdict(lambda: defaultdict(float))
        }

        # number of vehicles within each region, key: i - region, t - time
        self.acc_agents = {
            0: defaultdict(lambda: defaultdict(int)),
            1: defaultdict(lambda: defaultdict(int))
        }

        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc_agents = {
            0: defaultdict(lambda: defaultdict(int)),
            1: defaultdict(lambda: defaultdict(int))
        }
        self.queue_agents = {
            0: defaultdict(list),
            1: defaultdict(list)
        }
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow_agents = {
            0: defaultdict(lambda: defaultdict(float)),
            1: defaultdict(lambda: defaultdict(float))
        }
        self.rebFlow_ori_agents = {
            0: defaultdict(lambda: defaultdict(float)),
            1: defaultdict(lambda: defaultdict(float))
        }

        self.paxWait_agents = {
            0: defaultdict(list),
            1: defaultdict(list)
        }
        self.servedDemand_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }
        self.unservedDemand_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }
        
        self.passenger = {
            0: defaultdict(lambda: defaultdict(list)),
            1: defaultdict(lambda: defaultdict(list))
        }

        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        # number of edges leaving each region
        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region]
        # set rebalancing time for each link
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            for a in self.agents:
                self.rebFlow_agents[a][i, j]  # triggers the defaultdict init
                self.rebFlow_ori_agents[a][i, j]
                
        total_vehicles = self.scenario.totalAcc
        vp0 = 400/900
        vp1 = 500/900
        regions = list(self.region)
        n = len(regions)

        # Total vehicles per agent
        agent0_vehicles = int(total_vehicles * vp0)
        agent1_vehicles = total_vehicles - agent0_vehicles

        vehicle_per_region_0 = agent0_vehicles // n
        vehicle_per_region_1 = agent1_vehicles // n

        remaining_0 = agent0_vehicles % n
        remaining_1 = agent1_vehicles % n

        # Initialize all region counts to base
        for i, region in enumerate(regions):
            extra_0 = 1 if i < remaining_0 else 0
            extra_1 = 1 if i < remaining_1 else 0
            self.acc_agents[0][region][0] = vehicle_per_region_0 + extra_0
            self.acc_agents[1][region][0] = vehicle_per_region_1 + extra_1

        

        self.od_wages = generate_fixed_od_wages(
            wage_mode="scaled_region",  # "normal25" or "fixed_mean" or "scaled_region"
            G=self.G,
            std=5,
            base_mean=25,  # used only if wage_mode="fixed_mean"
            income_array=np.array([
                65719, 95333, 46632, 54354, 46316, 127615, 56512, 48938, 58040, 73495,
                59266, 50865, 38677, 53048, 64020, 35020, 53491, 83101
            ]),
            seed=42  # ensures repeatability
        )



        # scenario.tstep: number of steps as one timestep
        self.beta = beta * scenario.tstep
        t = self.time

        self.N = len(self.region)  # total number of cells

        self.info_agents = {
            a: {
                'served_demand': 0,
                'unserved_demand': 0,
                'served_waiting': 0,
                'operating_cost': 0,
                'revenue': 0,
                'rebalancing_cost': 0,
                'rejected_demand': 0,
                'rejection_rate': 0,
            } for a in [0, 1]
        }

        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}
        # observation: current vehicle distribution, time, future arrivals, demand
        
        self.obs = {
            0: (self.acc_agents[0], self.time, self.dacc_agents[0], self.demand_agents[0]),
            1: (self.acc_agents[1], self.time, self.dacc_agents[1], self.demand_agents[1])
        }
        

    def match_step_simple(self, price=None):
        """
        A simple version of matching. Match vehicle and passenger in a first-come-first-serve manner. 

        price: list of price for eacj region. Default None.
        """
        t = self.time

        paxreward = {0: 0, 1: 0}

        self.info_agents = {
                    a: {
                        'served_demand': 0,
                        'unserved_demand': 0,
                        'served_waiting': 0,
                        'operating_cost': 0,
                        'revenue': 0,
                        'rebalancing_cost': 0,
                        'rejected_demand': 0,
                        'rejection_rate': 0,
                    } for a in [0, 1]
                }

        total_original_demand = 0
        total_rejected_demand = 0
        

        for n in self.region:

            # Update current queue
            for j in self.G[n]:
                d = self.demand[n, j][t]
                p = self.price_agents[0][n, j][t]
                
                
                if price is not None and (np.sum(price) != 0):
                    for agent_id in [0,1]:
                        if isinstance(price[agent_id][0], list):
                            if len(price[agent_id][0]) == len(price[agent_id]):
                                p = 2 * self.price_agents[agent_id][n, j][t] * price[agent_id][n][j]
                            else:
                                p = 2 * self.price_agents[agent_id][n, j][t] * price[agent_id][n][0]
                        else:
                            p = self.price_agents[agent_id][n, j][t] * price[agent_id][n] * 2

                        if p <= 1e-6:
                            p = self.jitter
                        self.price_agents[agent_id][n, j][t] = p

                d_original = d  # before applying choice model
                
                #--Choice Model--
                
                pr0 = self.price_agents[0][n, j][t]
                pr1 = self.price_agents[1][n, j][t]
                
                travel_time = self.demandTime[n, j][t]
                
                travel_time_in_hours = travel_time / 60
                U_reject = 0 
                
                supply0 = sum(self.acc_agents[0][n].values())
                supply1 = sum(self.acc_agents[1][n].values())
                
                exp_utilities = []
                labels = []
                
                
                
                wage = 25
                #wage = self.od_wages.get((n, j), 25)
                if wage < 0:
                    wage = 1e-3
                
                income_effect = 25 / wage

                if supply0 > 0:
                    U_0 = 13.5 - 0.71 * wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr0
                    exp_utilities.append(np.exp(U_0))
                    labels.append("agent0")
                
                if supply1 > 0:
                    U_1 = 13.5 - 0.71 * wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr1
                    exp_utilities.append(np.exp(U_1))
                    labels.append("agent1")
            
                exp_utilities.append(np.exp(U_reject))
                labels.append("reject")
                
                P = np.array(exp_utilities) / np.sum(exp_utilities)
                labels_array = np.array(labels)

                d0 = d1 = dr = 0

                if d_original > 0:
                    for _ in range(d_original):
                        choice = np.random.choice(labels_array, p=P)
                        if choice == "agent0":
                            d0 += 1
                        elif choice == "agent1":
                            d1 += 1
                        elif choice == "reject":
                            dr += 1
                
                
                self.accepted_demand_agents[0][(n, j)][t] += d0
                self.accepted_demand_agents[1][(n, j)][t] += d1

                pax0, self.arrivals = generate_passenger((n, j, t, d0, pr0), self.max_wait, self.arrivals)
                pax1, self.arrivals = generate_passenger((n, j, t, d1, pr1), self.max_wait, self.arrivals)
                self.arrivals_agents[0] += len(pax0)
                self.arrivals_agents[1] += len(pax1)

                self.passenger[0][n][t].extend(pax0)
                self.passenger[1][n][t].extend(pax1)
                
                rejected_demand = d_original - d0 - d1
                
                total_original_demand += d_original
                total_rejected_demand += rejected_demand

                self.demand[n, j][t] = d0 + d1
                self.demand_agents[0][n, j][t] = d0
                self.demand_agents[1][n, j][t] = d1


                # shuffle passenger list at station so that the passengers are not served in destination order
                random.Random(42).shuffle(self.passenger[0][n][t])
                random.Random(42).shuffle(self.passenger[1][n][t])


            for agent_id in [0, 1]:
                accCurrent = self.acc_agents[agent_id][n][t]

                # Add new entering passengers to this agent's queue
                new_enterq = [pax for pax in self.passenger[agent_id][n][t] if pax.enter()]
                queueCurrent = self.queue_agents[agent_id][n] + new_enterq
                self.queue_agents[agent_id][n] = queueCurrent

                matched_leave_index = []

                for i, pax in enumerate(queueCurrent):
                    if accCurrent > 0:
                        accept = pax.match(t)
                        if accept:
                            matched_leave_index.append(i)
                            accCurrent -= 1
                            wait_t = pax.wait_time
                            arr_t = t + self.demandTime[pax.origin, pax.destination][t]

                            self.paxFlow_agents[agent_id][pax.origin, pax.destination][arr_t] += 1
                            self.paxWait_agents[agent_id][pax.origin, pax.destination].append(wait_t)
                            self.dacc_agents[agent_id][pax.destination][arr_t] += 1
                            self.servedDemand_agents[agent_id][pax.origin, pax.destination][t] += 1

                            paxreward[agent_id] += pax.price - self.demandTime[pax.origin, pax.destination][t] * self.beta
                            self.ext_reward_agents[agent_id][n] += max(0, self.demandTime[pax.origin, pax.destination][t] * self.beta)

                            self.info_agents[agent_id]['revenue'] += pax.price
                            self.info_agents[agent_id]['served_demand'] += 1
                            self.info_agents[agent_id]['operating_cost'] += self.demandTime[pax.origin, pax.destination][t] * self.beta
                            self.info_agents[agent_id]['served_waiting'] += wait_t
                        else:
                            if pax.unmatched_update():
                                matched_leave_index.append(i)
                                self.unservedDemand_agents[agent_id][pax.origin, pax.destination][t] += 1
                                self.info_agents[agent_id]['unserved_demand'] += 1
                    else:
                        if pax.unmatched_update():
                            matched_leave_index.append(i)
                            self.unservedDemand_agents[agent_id][pax.origin, pax.destination][t] += 1
                            self.info_agents[agent_id]['unserved_demand'] += 1

                self.queue_agents[agent_id][n] = [
                    queueCurrent[i] for i in range(len(queueCurrent)) if i not in matched_leave_index
                ]
                self.acc_agents[agent_id][n][t+1] = accCurrent

        
        done = (self.tf == t + 1)

        ext_done = [done] * self.nregion

        self.obs = {
            0: (self.acc_agents[0], self.time, self.dacc_agents[0], self.demand_agents[0]),
            1: (self.acc_agents[1], self.time, self.dacc_agents[1], self.demand_agents[1])
        }


        rejection_rate = (
            total_rejected_demand / total_original_demand if total_original_demand > 0 else 0
        )

        for agent_id in [0, 1]:
            self.info_agents[agent_id]['rejection_rate'] = rejection_rate
            self.info_agents[agent_id]['rejected_demand'] += total_rejected_demand
            
        for agent_id in [0, 1]:
            for n in self.region:
                if t+1 not in self.acc_agents[agent_id][n]:
                    self.acc_agents[agent_id][n][t+1] = 0

        return self.obs, paxreward, done, self.info_agents, self.ext_reward_agents, ext_done

    def matching_update(self):
        """Update properties if there is no rebalancing after matching"""
        t = self.time
        # Update acc. Assuming arriving vehicle will only be availbe for the next timestamp.
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            for agent_id in [0, 1]:
                if (i, j) in self.paxFlow_agents[agent_id] and t in self.paxFlow_agents[agent_id][i, j]:
                    self.acc_agents[agent_id][j][t+1] += self.paxFlow_agents[agent_id][i, j][t]
        self.time += 1

    def matching(self, CPLEXPATH=None, PATH='', directory="saved_files", platform='linux'):
        t = self.time
        demandAttr = [(i, j, self.demand[i, j][t], self.price[i, j][t]) for i, j in self.demand
                      if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3]
        self.arrivals += sum([i[2] for i in demandAttr])
        accTuple = [(n, self.acc[n][t+1]) for n in self.acc]
        modPath = os.getcwd().replace('\\', '/')+'/src/cplex_mod/'
        matchingPath = os.getcwd().replace('\\', '/') + \
                      '/' + str(directory) + '/cplex_logs/matching/'+PATH + '/'
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath + 'data_{}.dat'.format(t)
        resfile = matchingPath + 'res_{}.dat'.format(t)
        with open(datafile, 'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        modfile = modPath+'matching.mod'
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio201/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        if platform == 'mac':
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = matchingPath + 'out_{}.dat'.format(t)
        with open(out_file, 'w') as output_f:
            subprocess.check_call(
                [CPLEXPATH+"oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()
        flow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        flow[int(i), int(j)] = float(f)
        paxAction = [flow[i, j] if (
            i, j) in flow else 0 for i, j in self.edges]
        return paxAction

    def pax_step(self, paxAction_agents, CPLEXPATH=None, directory="saved_files", PATH='', platform='linux'):
        t = self.time
        paxreward = {0: 0, 1: 0}
                            
        for agent_id in [0, 1]:
            for i in self.region:
                self.acc_agents[agent_id][i][t+1] = self.acc_agents[agent_id][i][t]
            self.info_agents[agent_id]['revenue'] = 0
            self.info_agents[agent_id]['served_demand'] = 0
            self.info_agents[agent_id]['operating_cost'] = 0

        for agent_id in [0, 1]:
            paxAction = paxAction_agents[agent_id]

            for i, j in self.edges:
                if (i, j) not in self.demand or t not in self.demand[i, j] or paxAction is None:
                    continue

                idx = self.edges.index((i, j))
                demand_val = self.demand[i, j][t]
                acc_val = self.acc_agents[agent_id][i][t+1]

                action_val = paxAction[idx] if idx < len(paxAction) else 0
                served = min(acc_val, action_val)
                paxAction[idx] = served
            
                assert served < acc_val + 1e-3, f"Agent {agent_id} attempted to serve more than available: served={served}, acc={acc_val}"

                if served < 1e-3:
                    continue

                self.paxFlow_agents[agent_id][i, j][t + self.demandTime[i, j][t]] = served
                self.dacc_agents[agent_id][j][t + self.demandTime[i, j][t]] += served
                self.servedDemand_agents[agent_id][i, j][t] = served

                self.acc_agents[agent_id][i][t+1] -= served

                paxreward[agent_id] += served * (self.price_agents[agent_id][i, j][t] - self.demandTime[i, j][t]*self.beta)
                self.ext_reward_agents[agent_id][i] += max(0, served * (self.price_agents[agent_id][i, j][t] - self.demandTime[i, j][t]*self.beta))

                self.info_agents[agent_id]['served_demand'] += served
                self.info_agents[agent_id]['operating_cost'] += self.demandTime[i, j][t] * self.beta * served
                self.info_agents[agent_id]['revenue'] += served * self.price_agents[agent_id][i, j][t]
                # Optional: you can estimate waiting time here if needed (not tracked in this step)

        self.obs = {
            0: (self.acc_agents[0], self.time, self.dacc_agents[0], self.demand_agents[0]),
            1: (self.acc_agents[1], self.time, self.dacc_agents[1], self.demand_agents[1])
        }

        done = (self.tf == t + 1)
        ext_done = [done] * self.nregion
        
        #for agent_id in [0, 1]:
        #    for n in self.region:
        #        if t+1 not in self.acc_agents[agent_id][n]:
        #            self.acc_agents[agent_id][n][t+1] = 0

        return self.obs, paxreward, done, self.info_agents, self.ext_reward_agents, ext_done

    def reb_step(self, rebAction_agents):
        t = self.time
        rebreward = {0: 0, 1: 0}
        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}
    
        # Initialize the info_agents dictionary
        for agent_id in [0, 1]:
            self.info_agents[agent_id]['rebalancing_cost'] = 0

    
        # Loop through agents
        for agent_id in [0, 1]:
            
            if agent_id not in rebAction_agents:
                continue
            
            rebAction = rebAction_agents[agent_id]
    
            # Loop through the edges for rebalancing
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) not in self.G.edges:
                    continue
    
                # Update rebalancing actions and flows
                rebAction[k] = min(self.acc_agents[agent_id][i][t+1], rebAction[k])
                self.rebFlow_agents[agent_id][i, j][t + self.rebTime[i, j][t]] = rebAction[k]
                self.rebFlow_ori_agents[agent_id][i, j][t] = rebAction[k]
    
                # Update the vehicle counts based on rebalancing actions
                self.acc_agents[agent_id][i][t+1] -= rebAction[k]
                self.dacc_agents[agent_id][j][t + self.rebTime[i, j][t]] += rebAction[k]
    
                # Calculate rebalancing costs for the agent
                rebalancing_cost = self.rebTime[i, j][t] * self.beta * rebAction[k]
                rebreward[agent_id] -= rebalancing_cost
                self.ext_reward_agents[agent_id][i] -= rebalancing_cost
    
                # Track rebalancing costs in info_agents
                self.info_agents[agent_id]['rebalancing_cost'] += rebalancing_cost
    
        # Vehicle arrivals from past rebalancing and passenger trips
        for agent_id in [0, 1]:
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) in self.rebFlow_agents[agent_id] and t in self.rebFlow_agents[agent_id][i, j]:
                    self.acc_agents[agent_id][j][t+1] += self.rebFlow_agents[agent_id][i, j][t]
                if (i, j) in self.paxFlow_agents[agent_id] and t in self.paxFlow_agents[agent_id][i, j]:
                    self.acc_agents[agent_id][j][t+1] += self.paxFlow_agents[agent_id][i, j][t]
    
        # Increment time step
        self.time += 1
    
        self.obs = {
            0: (self.acc_agents[0], self.time, self.dacc_agents[0], self.demand_agents[0]),
            1: (self.acc_agents[1], self.time, self.dacc_agents[1], self.demand_agents[1])
        }
    
        # Update rebalancing time on edges
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
    
        # Check if the episode is done
        done = (self.tf == t + 1)
        ext_done = [done] * self.nregion
    
        # Ensure vehicle counts are up-to-date for all agents
        #for agent_id in [0, 1]:
        #    for n in self.region:
        #        if t+1 not in self.acc_agents[agent_id][n]:
        #            self.acc_agents[agent_id][n][t+1] = 0
    
        # Return the observations and the updated rewards
        return self.obs, rebreward, done, self.info_agents, self.ext_reward_agents, ext_done


    def reset(self):
        # reset the episode
        for a in [0, 1]:
            self.acc_agents[a] = defaultdict(dict)
            self.dacc_agents[a] = defaultdict(lambda: defaultdict(int))
            self.queue_agents[a] = defaultdict(list)
            self.rebFlow_agents[a] = defaultdict(lambda: defaultdict(float))
            self.rebFlow_ori_agents[a] = defaultdict(lambda: defaultdict(float))
            self.paxFlow_agents[a] = defaultdict(dict)
            self.paxWait_agents[a] = defaultdict(list)
            self.servedDemand_agents[a] = defaultdict(dict)
            self.unservedDemand_agents[a] = defaultdict(dict)

            total_vehicles = self.scenario.totalAcc
            vp0 = 400/900
            vp1 = 500/900
            regions = list(self.region)
            n = len(regions)

            # Total vehicles per agent
            agent0_vehicles = int(total_vehicles * vp0)
            agent1_vehicles = total_vehicles - agent0_vehicles

            vehicle_per_region_0 = agent0_vehicles // n
            vehicle_per_region_1 = agent1_vehicles // n

            remaining_0 = agent0_vehicles % n
            remaining_1 = agent1_vehicles % n

            self.accepted_demand_agents[0].clear()
            self.accepted_demand_agents[1].clear()

            # Initialize all region counts to base
            for i, region in enumerate(regions):
                extra_0 = 1 if i < remaining_0 else 0
                extra_1 = 1 if i < remaining_1 else 0
                self.acc_agents[0][region][0] = vehicle_per_region_0 + extra_0
                self.acc_agents[1][region][0] = vehicle_per_region_1 + extra_1


            for i, j in self.edges:
                self.rebFlow_agents[a][i, j] = defaultdict(float)
                self.rebFlow_ori_agents[a][i, j] = defaultdict(float)
                self.paxFlow_agents[a][i, j] = defaultdict(float)
                self.paxWait_agents[a][i, j] = []
                self.servedDemand_agents[a][i, j] = defaultdict(float)
                self.unservedDemand_agents[a][i, j] = defaultdict(float)

        self.passenger = {
            0: defaultdict(lambda: defaultdict(list)),
            1: defaultdict(lambda: defaultdict(list))
        }
        
        self.demand_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }

        self.arrivals_agents = {0: 0, 1: 0}
        self.arrivals = 0
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))

        self.demand = defaultdict(dict)
        self.price_agents = {
            0: defaultdict(dict),
            1: defaultdict(dict)
        }

        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)

        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            self.price_agents[0][i, j][t] = p
            self.price_agents[1][i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        self.ext_reward_agents = {0: np.zeros(self.nregion), 1: np.zeros(self.nregion)}

        self.obs = {
            0: (self.acc_agents[0], self.time, self.dacc_agents[0], self.demand_agents[0]),
            1: (self.acc_agents[1], self.time, self.dacc_agents[1], self.demand_agents[1])
        }


class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None, supply_ratio=1,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.2, json_file=None, json_hr=9, json_tstep=3, varying_time=False, json_regions=None, impute=False):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distributionjson_tstep
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1*N2)
            self.G = self.G.to_directed()
            self.demandTime = defaultdict(dict)  # traveling time between nodes
            self.rebTime = defaultdict(dict)
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            self.tstep = json_tstep
            for i, j in self.edges:
                for t in range(tf*2):
                    self.demandTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)
                    self.rebTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)

            for n in self.G.nodes:
                # initial number of vehicles at station
                self.G.nodes[n]['accInit'] = int(ninit)
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            # demand mutiplier over time
            if demand_ratio == None or type(demand_ratio) == list or type(demand_ratio) == dict:
                for i, j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio)-1)), demand_ratio))+[demand_ratio[-1]]*tf
                    elif type(demand_ratio) == dict:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(0, tf+1, tf/(len(demand_ratio[i]) - 1)), demand_ratio[i]))+[demand_ratio[i][-1]]*tf
                    else:
                        self.demand_ratio[i, j] = [1]*(tf+tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio[i, j])-1)), demand_ratio[i, j]))+[1]*tf
                    else:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio['default'])-1)), demand_ratio['default']))+[1]*tf
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand()*2+1) * \
                        (self.demandTime[i, j][0]+1)
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand
        else:
            self.varying_time = varying_time
            self.is_json = True
            with open(json_file, "r") as file:
                data = json.load(file)           
            self.tstep = json_tstep
            self.N1 = data["nlat"]
            self.N2 = data["nlon"]
            self.demand_input = defaultdict(dict)
            self.json_regions = json_regions

            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif 'region' in data:
                self.G = nx.complete_graph(data['region'])
            else:
                self.G = nx.complete_graph(self.N1*self.N2)
            self.G = self.G.to_directed()
            self.p = defaultdict(dict)
            self.alpha = 0
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.json_start = json_hr * 60
            self.tf = tf
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            self.nregion = len(self.G)

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(int)
                self.rebTime[i, j] = 1

            matrix_demand = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            matrix_price_ori = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            for item in data["demand"]:
                t, o, d, v, tt, p = item["time_stamp"], item["origin"], item[
                    "destination"], item["demand"], item["travel_time"], item["price"]
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(
                        float), defaultdict(float), defaultdict(float)
                # set ALL demand, price, and traveling time for OD. price and traveling time will be averaged by demand after
                self.demand_input[o, d][(
                    t-self.json_start)//json_tstep] += v*demand_ratio
                self.p[o, d][(t-self.json_start) //
                             json_tstep] += p*v*demand_ratio
                self.demandTime[o, d][(t-self.json_start) //
                                      json_tstep] += tt*v*demand_ratio/json_tstep

                matrix_demand[(t-self.json_start) //
                                      json_tstep][o,d] += v*demand_ratio
                matrix_price_ori[(t-self.json_start) //
                                      json_tstep][o,d] += p*v*demand_ratio

            
            for o, d in self.edges:
                for t in range(0, tf*2):
                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(
                            int(round(self.demandTime[o, d][t])), 1)
                        matrix_price_ori[t][o,d] /= matrix_demand[t][o,d]
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0
            #print demand_input
            #for o, d in self.edges:
            #    for t in range(0, tf * 2):
            #        if (o, d) in self.demand_input and t in self.demand_input[o, d]:
            #            print(f"Demand at ({o}, {d}) at time {t}: {self.demand_input[o, d][t]}")
                        
            

            matrix_reb = np.zeros((self.nregion,self.nregion))
            for item in data["rebTime"]:
                hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if varying_time:
                    t0 = int((hr*60 - self.json_start)//json_tstep)
                    t1 = int((hr*60 + 60 - self.json_start)//json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(
                            int(round(rt/json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf+1):
                            self.rebTime[o, d][t] = max(
                                int(round(rt/json_tstep)), 1)
                            matrix_reb[o,d] = rt/json_tstep
            
            # KNN regression for each time step
            if impute:
                knn = defaultdict(lambda: KNeighborsRegressor(n_neighbors=3))
                for t in matrix_price_ori.keys():
                    reb = matrix_reb
                    price = matrix_price_ori[t]
                    X = []
                    y = []
                    for i in range(self.nregion):
                        for j in range(self.nregion):
                            if price[i,j] != 0:
                                X.append(reb[i,j])
                                y.append(price[i,j])
                    X_train = np.array(X).reshape(-1, 1)
                    y_train = np.array(y)
                    knn[t].fit(X_train, y_train)

                # Test point
                for o, d in self.edges:
                    for t in range(0, tf*2):
                        if self.p[o,d][t]==0 and t in knn.keys():
                            
                            knn_regressor = knn[t]

                            X_test = np.array([[matrix_reb[o,d]]])
                            # Predict the value for the test point
                            y_pred = knn_regressor.predict(X_test)[0]
                            self.p[o,d][t] = float(y_pred)

            # Initial vehicle distribution
            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr + int(round(json_tstep/2 * tf/60)):
                    self.totalAcc = int(supply_ratio * acc)  # <<<<<< SAVE TOTAL

            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(
                            self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = (np.random.rand(len(self.G))*self.alpha *
                           2+1-self.alpha)  # multiplyer of demand
            if type(self.demand_input) in [float, int, list, np.array]:

                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:  # demand in the format of each region
                    self.region_demand = region_rand * \
                        np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array(
                        [np.math.exp(-self.rebTime[i, j][0]*self.trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    for idx in range(len(J)):
                        # allocation of demand to OD pairs
                        self.static_demand[i, J[idx]
                                           ] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = self.demand_input[i, j] if (
                        i, j) in self.demand_input else self.demand_input['default']

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception(
                    "demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(
                        self.static_demand[i, j]*self.demand_ratio[i, j][t])
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(
                            2)+1)*self.demandTime[i, j][t]
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr