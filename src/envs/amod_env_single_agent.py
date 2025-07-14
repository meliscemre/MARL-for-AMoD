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
        # TODO: set tstep if the scenario is not based on real data
        self.tstep = scenario.tstep
        self.passenger = dict()  # passenger arrivals
        self.queue = defaultdict(list)  # passenger queue at each station
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)  # set of regions
        for i in self.region:
            self.passenger[i] = defaultdict(list)
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)

        self.price = defaultdict(dict)  # price
        self.arrivals = 0  # total number of added passengers
        # number of vehicles within each region, key: i - region, t - time
        self.acc = defaultdict(dict)
        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc = defaultdict(dict)
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow = defaultdict(dict)
        self.rebFlow_ori = defaultdict(dict)
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)
        self.paxWait = defaultdict(list)
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
            self.rebFlow[i, j] = defaultdict(float)
            self.rebFlow_ori[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
            self.paxWait[i, j] = []
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        # scenario.tstep: number of steps as one timestep
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        self.unservedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
            self.unservedDemand[i, j] = defaultdict(float)

        self.N = len(self.region)  # total number of cells

        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                  'rebalancing_cost', 'operating_cost', 'served_waiting'], 0)
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def match_step_simple(self, price=None):
        """
        A simple version of matching. Match vehicle and passenger in a first-come-first-serve manner. 

        price: list of price for eacj region. Default None.
        """
        t = self.time

        self.reward = 0
        self.ext_reward = np.zeros(self.nregion)

        self.info['served_demand'] = 0  # initialize served demand
        self.info['unserved_demand'] = 0
        self.info['served_waiting'] = 0
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        self.info['rejected_demand'] = 0


        total_original_demand = 0
        total_rejected_demand = 0

        for n in self.region:

            accCurrent = self.acc[n][t]
            # Update current queue
            for j in self.G[n]:
                d = self.demand[n, j][t]
                p = self.price[n, j][t]
                
                if (price is not None) and (np.sum(price) != 0):
                    p_ori = p
                    if p_ori != 0:
                        if isinstance(price[0], list):
                            if len(price[0]) == len(price):
                                p = 2 * p_ori * price[n][j]
                            else:
                                p = 2 * p_ori * price[n][0]
                        else:
                            p = p_ori * price[n] * 2                  
                    self.price[n, j][t] = p
               
                d_original = d 

                #--Choice Model--
               
                travel_time = self.demandTime[n, j][t]  
                travel_time_in_hours = travel_time / 60
                
                #Utilities
                U_smart = 14.25 - 0.71 * 25 * travel_time_in_hours - self.choice_price_mult * p
                U_reject = 0

                # Probabilities
                exp_U_smart = np.exp(U_smart)
                exp_U_reject = np.exp(U_reject)

                P_smart = exp_U_smart / (exp_U_smart + exp_U_reject)

                d = int(P_smart * d_original)
                rejected_demand = d_original - d

                
                total_original_demand += d_original
                total_rejected_demand += rejected_demand

                self.demand[n, j][t] = d

                newp, self.arrivals = generate_passenger(
                    (n, j, t, d, p), self.max_wait, self.arrivals)
                self.passenger[n][t].extend(newp)
                # shuffle passenger list at station so that the passengers are not served in destination order
                random.Random(42).shuffle(self.passenger[n][t])

            new_enterq = [pax for pax in self.passenger[n][t] if pax.enter()]
            queueCurrent = self.queue[n] + new_enterq
            self.queue[n] = queueCurrent
            # Match passenger in queue in order
            matched_leave_index = []  # Index of matched and leaving passenger in queue
            for i, pax in enumerate(queueCurrent):
                if accCurrent != 0:
                    accept = pax.match(t)
                    if accept:
                        matched_leave_index.append(i)
                        accCurrent -= 1
                        self.paxFlow[pax.origin, pax.destination][t + self.demandTime[pax.origin, pax.destination][t]] += 1
                        self.paxWait[pax.origin, pax.destination].append(pax.wait_time)
                        self.dacc[pax.destination][t + self.demandTime[pax.origin, pax.destination][t]] += 1
                        self.servedDemand[pax.origin, pax.destination][t] += 1
                        self.reward += pax.price - self.demandTime[pax.origin, pax.destination][t] * self.beta
                        self.ext_reward[n] += max(0, self.demandTime[pax.origin, pax.destination][t] * self.beta)
                        self.info['revenue'] += pax.price
                        self.info['served_demand'] += 1
                        self.info['operating_cost'] += self.demandTime[pax.origin, pax.destination][t] * self.beta
                        self.info['served_waiting'] += pax.wait_time
                    else:
                        leave = pax.unmatched_update()
                        if leave:
                            matched_leave_index.append(i)
                            self.unservedDemand[pax.origin, pax.destination][t] += 1
                            self.info['unserved_demand'] += 1
                else:
                    leave = pax.unmatched_update()
                    if leave:
                        matched_leave_index.append(i)
                        self.unservedDemand[pax.origin,
                                            pax.destination][t] += 1

                        self.info['unserved_demand'] += 1
            

            # Update queue
            self.queue[n] = [self.queue[n][i] for i in range(
                len(self.queue[n])) if i not in matched_leave_index]
            # Update acc
            self.acc[n][t+1] = accCurrent


        # for acc, the time index would be t+1, but for demand, the time index would be t
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        # if (self.mode == 0) | (self.mode == 2):
        #     done = False  # if rebalancing needs to be carried out after
        # elif self.mode == 1:
        #     done = (self.tf == t+1)
        done = (self.tf == t+1)

        ext_done = [done]*self.nregion
        self.info['rejection_rate'] = (
            total_rejected_demand / total_original_demand if total_original_demand > 0
            else 0
        )
        self.info['rejected_demand'] += total_rejected_demand



        return self.obs, max(0, self.reward), done, self.info, self.ext_reward, ext_done

    def matching_update(self):
        """Update properties if there is no rebalancing after matching"""
        t = self.time
        # Update acc. Assuming arriving vehicle will only be availbe for the next timestamp.
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t+1] += self.paxFlow[i, j][t]

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

    # pax step
    def pax_step(self, paxAction=None, CPLEXPATH=None, directory="saved_files", PATH='', platform='linux'):
        t = self.time
        self.reward = 0
        self.ext_reward = np.zeros(self.nregion)
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
        self.info['served_demand'] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
        if paxAction is None:
            paxAction = self.matching(
                CPLEXPATH=CPLEXPATH, directory=directory, PATH=PATH, platform=platform)
        self.paxAction = paxAction
        # serving passengers

        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.demand or t not in self.demand[i, j] or self.paxAction[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            self.paxAction[k] = min(self.acc[i][t+1], paxAction[k])
            assert paxAction[k] < self.acc[i][t+1] + 1e-3
            self.servedDemand[i, j][t] = self.paxAction[k]
            self.paxFlow[i, j][t+self.demandTime[i, j][t]] = self.paxAction[k]
            self.info["operating_cost"] += self.demandTime[i,
                                                           j][t]*self.beta*self.paxAction[k]
            self.acc[i][t+1] -= self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i, j][t]
            self.dacc[j][t+self.demandTime[i, j][t]
                         ] += self.paxFlow[i, j][t+self.demandTime[i, j][t]]
            self.reward += self.paxAction[k] * \
                (self.price[i, j][t] - self.demandTime[i, j][t]*self.beta)
            self.ext_reward[i] += max(0, self.paxAction[k] *
                                      (self.price[i, j][t] - self.demandTime[i, j][t]*self.beta))
            self.info['revenue'] += self.paxAction[k]*(self.price[i, j][t])

        # for acc, the time index would be t+1, but for demand, the time index would be t
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        done = False  # if passenger matching is executed first
        ext_done = [done]*self.nregion
        return self.obs, max(0, self.reward), done, self.info, self.ext_reward, ext_done

    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.ext_reward = np.zeros(self.nregion)
        self.rebAction = rebAction
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k])
            self.rebFlow[i, j][t+self.rebTime[i, j][t]] = self.rebAction[k]
            self.rebFlow_ori[i, j][t] = self.rebAction[k]
            self.acc[i][t+1] -= self.rebAction[k]
            self.dacc[j][t+self.rebTime[i, j][t]
                         ] += self.rebFlow[i, j][t+self.rebTime[i, j][t]]
            self.info['rebalancing_cost'] += self.rebTime[i, j][t] * \
                self.beta*self.rebAction[k]
            #self.info["operating_cost"] += self.rebTime[i, j][t] * \
            #    self.beta*self.rebAction[k]
            self.reward -= self.rebTime[i, j][t]*self.beta*self.rebAction[k]
            self.ext_reward[i] -= self.rebTime[i, j][t] * \
                self.beta*self.rebAction[k]
        
        #print(f"Step {self.time}: Reb Reward Update | Reward: {self.reward} | Reb. Cost: {self.info['rebalancing_cost']}")

        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t+1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
                self.acc[j][t+1] += self.paxFlow[i, j][t]

        self.time += 1
        # use self.time to index the next time step
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
        done = (self.tf == t+1)  # if the episode is completed
        ext_done = [done]*self.nregion
        return self.obs, self.reward, done, self.info, self.ext_reward, ext_done

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.rebFlow_ori = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.paxWait = defaultdict(list)
        self.passenger = dict()
        self.queue = defaultdict(list)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        for i in self.region:
            self.passenger[i] = defaultdict(list)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        self.arrivals = 0
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.rebFlow_ori[i,j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
            self.paxWait[i, j] = []
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
            self.unservedDemand[i, j] = defaultdict(float)
         # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs


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
                if hr == json_hr+int(round(json_tstep/2*tf/60)):
                    total_acc = int(supply_ratio * acc)
                    regions = list(self.G.nodes)
                    vehicles_per_region = total_acc // len(regions)
                    remainder = total_acc % len(regions)

                    for i, n in enumerate(regions):
                        extra = 1 if i < remainder else 0
                        self.G.nodes[n]['accInit'] = vehicles_per_region + extra
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