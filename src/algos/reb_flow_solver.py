"""
Minimal Rebalancing Cost 
------------------------
This file contains the specifications for the Min Reb Cost problem.
"""
import os
import math
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str


def solveRebFlow(env, res_path, desiredAcc, CPLEXPATH, directory, agent_id=0):
    t = env.time

    # Format desired and current availability tuples
    accRLTuple = [(n, int(desiredAcc[n])) for n in desiredAcc]
    accTuple = [(n, int(env.acc_agents[agent_id][n][t+1])) for n in env.acc_agents[agent_id]]

    edgeAttr = [(i, j, env.G.edges[i, j]['time']) for i, j in env.G.edges]

    # Prepare paths
    modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
    OPTPath = os.getcwd().replace('\\', '/') + '/' + directory + f'/cplex_logs/rebalancing/{res_path}/'

    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)

    datafile = OPTPath + f'data_agent{agent_id}_{t}.dat'
    resfile = OPTPath + f'res_agent{agent_id}_{t}.dat'

    # 1. Write CPLEX data file
    with open(datafile, 'w') as file:
        file.write(f'path="{resfile}";\r\n')
        file.write(f'edgeAttr={mat2str(edgeAttr)};\r\n')
        file.write(f'accInitTuple={mat2str(accTuple)};\r\n')
        file.write(f'accRLTuple={mat2str(accRLTuple)};\r\n')

    # 2. Solve with CPLEX
    modfile = modPath + 'minRebDistRebOnly.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH

    out_file = OPTPath + f'out_agent{agent_id}_{t}.dat'
    with open(out_file, 'w') as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile],
            stdout=output_f,
            env=my_env
        )

    # 3. Parse result file robustly
    flow = defaultdict(float)
    with open(resfile, 'r', encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(';').split('=')
            if item[0] == 'flow':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(',')
                    try:
                        flow[int(i), int(j)] = float(f)
                    except ValueError:
                        print(f"[Warning] Could not convert '{f}' to float for edge ({i}, {j}). Setting to 0.")
                        flow[int(i), int(j)] = 0.0


    # Convert edge flow into rebAction aligned with env.edges
    action = [flow[i, j] for i, j in env.edges]
    return action