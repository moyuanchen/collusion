import numpy as np
import torch
from itertools import product
from collections import defaultdict
from agents import *
# from config import Config
import psutil
import os
import time

def get_next_v(v_bar = 1, sigma_v = 1):
    return v_bar + np.random.normal(scale = sigma_v)
def log_resource_usage(logfile="resource_log.txt"):
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu = p.cpu_percent(interval=0.1)  # brief wait to get nonzero value
    mem = p.memory_info().rss / 1e9    # in GB
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile, "a") as f:
        f.write(f"[{timestamp}] CPU: {cpu:.2f}%, Memory: {mem:.2f} GB\n")


def simulate(
        T = 1000000, config = None,     # simulation time and config file
        continue_simulation = False,    # continue simulation if True
        save_path = None                # path to save the simulation
        ):
    assert config is not None, "Config file required"
    I = config.I
    Np = config.Np
    Nv = config.Nv
    Nx = config.Nx
    sigma_u = config.sigma_u
    if type(continue_simulation) == str:
        log = np.load(continue_simulation, allow_pickle=True).item() # change to torch later
        
        v_hist = np.zeros(T)
        p_hist = np.zeros(T)
        z_hist = np.zeros(T)
        x_hist = np.zeros((I, T))
        # y_hist = np.zeros((I, T))
        u_hist = np.zeros(T)
        profit_hist = np.zeros((I, T))
        t0 = 0

        informed_agents = log["agents"]["informed"]
        noise_agent = log["agents"]["noise"]
        preferred_habitat_agent = log["agents"]["preferred_habitat"]
        market_maker = log["agents"]["market_maker"]

        _state = log["last_state"]

        convergence_counter = log['convergence_counter']
    # elif type(continue_simulation) == dict:
    #     log = continue_simulation
    #     v_hist = log["v"]
    #     p_hist = log["p"]
    #     z_hist = log["z"]
    #     x_hist = log["x"]
    #     y_hist = log["y"]
    #     profit_hist = log["profit"]
    #     t0 = len(v_hist)
    #     informed_agents = log["agents"]["informed"]
    #     noise_agent = log["agents"]["noise"]
    #     preferred_habitat_agent = log["agents"]["preferred_habitat"]
    #     market_maker = log["agents"]["market_maker"]
    #     _state = log["last_state"]

    #     profit_hist = np.concatenate((profit_hist, np.zeros((I, T))), axis=1)
    #     v_hist = np.concatenate((v_hist, np.zeros(T)))
    #     p_hist = np.concatenate((p_hist, np.zeros(T)))
    #     z_hist = np.concatenate((z_hist, np.zeros(T)))
    #     x_hist = np.concatenate((x_hist, np.zeros((I, T))), axis=1)
    #     y_hist = np.concatenate((y_hist, np.zeros((I, T))), axis=1)

    #     convergence_counter = log['convergence_counter']
        
    elif continue_simulation == False:

        market_maker = AdaptiveMarketMaker(config)
        noise_agent = NoiseAgent(config)
        preferred_habitat_agent = PreferredHabitatAgent(config)
        informed_agents = [InformedAgent(config) for _ in range(I)]
        _state = (np.random.choice(Np), np.random.choice(Nv))

        # log histories
        v_hist = np.zeros(T)
        p_hist = np.zeros(T)
        z_hist = np.zeros(T)
        x_hist = np.zeros((I, T))
        # y_hist = np.zeros((I, T))
        u_hist = np.zeros(T)
        profit_hist = np.zeros((I, T))
        t0 = 0
        convergence_counter = 0
    else:
        raise ValueError("Invalid value for continue_simulation")
    for agent in informed_agents:
        agent.convergence_counter = convergence_counter
    if save_path is None:
        save_path = '/Users/moyuanchen/Documents/thesis/data.npy'

    for t in range(T):
        if t % 1000 == 0:
            log_resource_usage(logfile=save_path + "resource_log.txt")
        yt = []
        _p, _v = informed_agents[0].p_discrete[_state[0]], informed_agents[0].v_discrete[_state[1]]
        v_hist[t+t0] = _v
        p_hist[t+t0] = _p
        _x = []

        # generate a grid from x_n to x_m each step
        # x_n, x_m = config.chi_n * _v, config.chi_m * _v
        # x_diff = abs(x_m - x_n)
        # if _v > config.v_bar:
        #     x_disc = np.linspace(x_m - config.iota * x_diff, x_n + config.iota * x_diff, config.Nx)
        # else:
        #     x_disc = np.linspace(x_n - config.iota * x_diff, x_m + config.iota * x_diff, config.Nx)
        # print(x_disc)

        for idx, agent in enumerate(informed_agents):
            x = agent.get_action(_state)
            xd = agent.x_discrete[x]
            # xd = x_disc[x]
            yt.append(xd)
            _x.append(x)

            x_hist[idx, t + t0] = xd
            # y_hist[idx, t + t0] = yt[-1]
        ut = noise_agent.get_action()
        u_hist[t+t0] = ut
        yt_sum = np.sum(yt) + ut
        # print(yt_sum)
        

        pt = market_maker.determine_price(yt_sum)
        zt = preferred_habitat_agent.get_action(pt)

        z_hist[t+t0] = zt
        market_maker.update(_v, _p, zt, yt_sum)
        
        vt = get_next_v()
        next_state = informed_agents[0].continuous_to_discrete(pt, vt)
        for idx, agent in enumerate(informed_agents):
            reward = (_v - pt) * yt[idx]
            agent.update(_state, _x[idx], reward, next_state)
            profit_hist[idx, t + t0] = reward

        _state = next_state
    convergence = min([agent.convergence_counter for agent in informed_agents])
    log = {
        "v": v_hist,
        "p": p_hist,
        "z": z_hist,
        "x": x_hist,
        "u": u_hist,
        "profit": profit_hist,
        "last_state": _state,
        "convergence_counter": convergence
    }
    agents = {
        "informed": informed_agents,
        "noise": noise_agent,
        "preferred_habitat": preferred_habitat_agent,
        "market_maker": market_maker
    }
    log["agents"] = agents
    np.save(save_path, log)
    # print(max_c)
    return log, agents


def simulate_batch(
        T = 500000, B = 1000, config = None,     # simulation time and config file
        continue_simulation = False,    # continue simulation if True
        save_path = None,                # path to save the simulation
        device = 'cpu'                # device to run the simulation
        ):
    assert config is not None, "Config file required"
    device = torch.device(device)
    I = config.I
    Np = config.Np
    Nv = config.Nv
    Nx = config.Nx
    sigma_u = config.sigma_u
    if type(continue_simulation) == str:
        log = torch.load(continue_simulation)
        
        # v_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        p_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        z_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        # x_hist = torch.zeros((B, I, T), dtype = torch.float16, device = device)
        y_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        # u_hist = torch.zeros((B,T), dtype = torch.float16, device = device)
        u_path = torch.normal(config.v_bar, config.sigma_u, (B, T), device = device)
        v_path = torch.randint(0, Nv, (B, T), device = device)
        _p_init = torch.randint(0, Np, (B,), device = device)
        # _v_init = torch.randint(0, Nv, (B,), device = device)
        _v_init = v_path[:, 0]
        _state = torch.stack((_p_init, _v_init), dim = 1) # B x 2
        profit_hist = torch.zeros((B, I, T), dtype = torch.float16, device = device)
        # t0 = 0

        informed_agents = log["agents"]["informed"]
        # noise_agent = log["agents"]["noise"]
        preferred_habitat_agent = log["agents"]["preferred_habitat"]
        market_maker = log["agents"]["market_maker"]

        _state = log["last_state"]

        convergence_counter = log['convergence_counter']
        
    elif continue_simulation == False:

        market_maker = AdaptiveMarketMaker_Batch(config, B = B)
        # noise_agent = [NoiseAgent(config) for _ in range(B)]
        # noise = 
        preferred_habitat_agent = PreferredHabitatAgent(config)
        informed_agents = InformedAgents_Batch(config, B = B)

        # _state = (np.random.choice((B, Np)), np.random.choice((B, Nv)))

        # log histories
        # v_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        p_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        z_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        # x_hist = torch.zeros((B, I, T), dtype = torch.float16, device = device)
        y_hist = torch.zeros((B, T), dtype = torch.float16, device = device)
        # u_hist = torch.zeros((B,T), dtype = torch.float16, device = device)
        u_path = torch.normal(config.v_bar, config.sigma_u, (B, T), device = device)
        v_path = torch.randint(0, Nv, (B, T), device = device)

        _p_init = torch.randint(0, Np, (B,), device = device)
        # _v_init = torch.randint(0, Nv, (B,), device = device)
        _v_init = v_path[:, 0]
        _state = torch.stack((_p_init, _v_init), dim = 1) # B x 2

        profit_hist = torch.zeros((B, I, T), dtype = torch.float16, device = device)
        # t0 = 0
        convergence_counter = torch.zeros((B, I), dtype = torch.int16, device = device)
    else:
        raise ValueError("Invalid value for continue_simulation")
    # for b, agents_batch in enumerate(informed_agents):
    #     for agent in agents_batch:
    #         agent.convergence_counter = convergence_counter[b]
    informed_agents.convergence_counter = convergence_counter
    if save_path is None:
        raise ValueError("Save path required")

    for t in range(T - 1):
        # if t % 1000 == 0:
        #     log_resource_usage(logfile=save_path + "resource_log.txt")
        yt = []
        _p, _v = informed_agents.p_discrete[_state[:, 0]], informed_agents.v_discrete[_state[:, 1]]
        # v_hist[t+t0] = _v
        # print(_p)
        # print(_p.shape)
        p_hist[:, t] = _p
        # _x = []

        # generate a grid from x_n to x_m each step
        # x_n, x_m = config.chi_n * _v, config.chi_m * _v
        # x_diff = abs(x_m - x_n)
        # if _v > config.v_bar:
        #     x_disc = np.linspace(x_m - config.iota * x_diff, x_n + config.iota * x_diff, config.Nx)
        # else:
        #     x_disc = np.linspace(x_n - config.iota * x_diff, x_m + config.iota * x_diff, config.Nx)
        # print(x_disc)

        # for idx, agent in enumerate(informed_agents):
        #     x = agent.get_action(_state)
        #     xd = agent.x_discrete[x]
        #     # xd = x_disc[x]
        #     yt.append(xd)
        #     _x.append(x)

        #     x_hist[idx, t + t0] = xd
        action = informed_agents.get_action(_state)
        volume = informed_agents.x_discrete[action].sum(axis = -1)
            # y_hist[idx, t + t0] = yt[-1]
        # ut = noise_agent.get_action()
        # u_hist[t+t0] = ut
        yt_sum = volume + u_path[:, t] # B
        y_hist[:, t] = yt_sum
        # print(yt_sum)
        

        pt = market_maker.determine_price(yt_sum)
        zt = preferred_habitat_agent.get_action(pt)

        z_hist[:, t] = zt
        market_maker.update(_v, _p, zt, yt_sum)
        
        vt = v_path[:, t + 1]
        next_state = informed_agents.continuous_to_discrete(pt, vt)
        # for idx, agent in enumerate(informed_agents):
        #     reward = (_v - pt) * yt[idx]
        #     agent.update(_state, _x[idx], reward, next_state)
        #     profit_hist[idx, t + t0] = reward
        reward = (_v - pt) * action.T # (B - B) * I x B = I x B
        profit_hist[:, :, t] = reward.T
        informed_agents.update(_state, action, reward.T, next_state)


        _state = next_state
    convergence = informed_agents.convergence_counter.min()
    log = {
        "v": v_path,
        "p": p_hist,
        "z": z_hist,
        "y": y_hist,
        "u": u_path,
        "profit": profit_hist,
        "last_state": _state,
        "convergence_counter": convergence,
        "config": config
    }
    agents = {
        "informed": informed_agents,
        # "noise": noise_agent,
        "preferred_habitat": preferred_habitat_agent,
        "market_maker": market_maker
    }
    log["agents"] = agents
    torch.save(log, save_path)
    # print(max_c)
    return log, agents
# def simulate_constrained(
#         T = 1000000, config = None,     # simulation time and config file
#         continue_simulation = False,    # continue simulation if True
#         save_path = None                # path to save the simulation
#         ):
#     assert config is not None, "Config file required"
#     I = config.I
#     Np = config.Np
#     Nv = config.Nv
#     Nx = config.Nx
#     sigma_u = config.sigma_u
#     if type(continue_simulation) == str:
#         log = np.load(continue_simulation, allow_pickle=True).item()
        
#         v_hist = np.zeros(T)
#         p_hist = np.zeros(T)
#         z_hist = np.zeros(T)
#         x_hist = np.zeros((I, T))
#         # y_hist = np.zeros((I, T))
#         u_hist = np.zeros(T)
#         profit_hist = np.zeros((I, T))
#         t0 = 0

#         informed_agents = log["agents"]["informed"]
#         noise_agent = log["agents"]["noise"]
#         preferred_habitat_agent = log["agents"]["preferred_habitat"]
#         market_maker = log["agents"]["market_maker"]

#         _state = log["last_state"]

#         convergence_counter = log['convergence_counter']
#     # elif type(continue_simulation) == dict:
#     #     log = continue_simulation
#     #     v_hist = log["v"]
#     #     p_hist = log["p"]
#     #     z_hist = log["z"]
#     #     x_hist = log["x"]
#     #     y_hist = log["y"]
#     #     profit_hist = log["profit"]
#     #     t0 = len(v_hist)
#     #     informed_agents = log["agents"]["informed"]
#     #     noise_agent = log["agents"]["noise"]
#     #     preferred_habitat_agent = log["agents"]["preferred_habitat"]
#     #     market_maker = log["agents"]["market_maker"]
#     #     _state = log["last_state"]

#     #     profit_hist = np.concatenate((profit_hist, np.zeros((I, T))), axis=1)
#     #     v_hist = np.concatenate((v_hist, np.zeros(T)))
#     #     p_hist = np.concatenate((p_hist, np.zeros(T)))
#     #     z_hist = np.concatenate((z_hist, np.zeros(T)))
#     #     x_hist = np.concatenate((x_hist, np.zeros((I, T))), axis=1)
#     #     y_hist = np.concatenate((y_hist, np.zeros((I, T))), axis=1)

#     #     convergence_counter = log['convergence_counter']
        
#     elif continue_simulation == False:

#         market_maker = AdaptiveMarketMaker(config)
#         noise_agent = NoiseAgent(config)
#         preferred_habitat_agent = PreferredHabitatAgent(config)
#         informed_agents = [InformedAgent(config) for _ in range(I)]
#         _state = (np.random.choice(Np), np.random.choice(Nv))

#         # log histories
#         v_hist = np.zeros(T)
#         p_hist = np.zeros(T)
#         z_hist = np.zeros(T)
#         x_hist = np.zeros((I, T))
#         # y_hist = np.zeros((I, T))
#         u_hist = np.zeros(T)
#         profit_hist = np.zeros((I, T))
#         t0 = 0
#         convergence_counter = 0
#     else:
#         raise ValueError("Invalid value for continue_simulation")
#     for agent in informed_agents:
#         agent.convergence_counter = convergence_counter
#     if save_path is None:
#         save_path = '/Users/moyuanchen/Documents/thesis/data.npy'

#     for t in range(T):
#         yt = []
#         _p, _v = informed_agents[0].p_discrete[_state[0]], informed_agents[0].v_discrete[_state[1]]
#         v_hist[t+t0] = _v
#         p_hist[t+t0] = _p
#         _x = []

#         # generate a grid from x_n to x_m each step
#         x_n, x_m = config.chi_N * _v, config.chi_M * _v
#         x_diff = abs(x_m - x_n)
#         if _v > config.v_bar:
#             x_disc = np.linspace(x_m - config.iota * x_diff, x_n + config.iota * x_diff, config.Nx)
#         else:
#             x_disc = np.linspace(x_n - config.iota * x_diff, x_m + config.iota * x_diff, config.Nx)
#         # print(x_disc)

#         for idx, agent in enumerate(informed_agents):
#             x = agent.get_action(_state)
#             # xd = agent.x_discrete[x]
#             xd = x_disc[x]
#             yt.append(xd)
#             _x.append(x)

#             x_hist[idx, t + t0] = xd
#             # y_hist[idx, t + t0] = yt[-1]
#         ut = noise_agent.get_action()
#         u_hist[t+t0] = ut
#         yt_sum = np.sum(yt) + ut
#         # print(yt_sum)
#         zt = preferred_habitat_agent.get_action(_p)

#         z_hist[t+t0] = zt

#         pt = market_maker.determine_price(yt_sum)
        
#         market_maker.update(_v, _p, zt, yt_sum)
        
#         vt = get_next_v()
#         next_state = informed_agents[0].continuous_to_discrete(pt, vt)
#         for idx, agent in enumerate(informed_agents):
#             reward = (_v - pt) * yt[idx]
#             agent.update(_state, _x[idx], reward, next_state)
#             profit_hist[idx, t + t0] = reward

#         _state = next_state
#     convergence = min([agent.convergence_counter for agent in informed_agents])
#     log = {
#         "v": v_hist,
#         "p": p_hist,
#         "z": z_hist,
#         "x": x_hist,
#         "u": u_hist,
#         "profit": profit_hist,
#         "last_state": _state,
#         "convergence_counter": convergence
#     }
#     agents = {
#         "informed": informed_agents,
#         "noise": noise_agent,
#         "preferred_habitat": preferred_habitat_agent,
#         "market_maker": market_maker
#     }
#     log["agents"] = agents
#     np.save(save_path, log)
#     # print(max_c)
#     return log, agents