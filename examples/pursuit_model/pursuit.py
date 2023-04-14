"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import logging
import time
import logging as log
import numpy as np

from examples.battle_model.causal_inference.infer_utils import calculate_te

logging.basicConfig(level=logging.ERROR)


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=map_size * map_size * 0.0125)
    env.add_agents(handles[1], method="random", n=map_size * map_size * 0.025)


def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    obs = [None for _ in range(n_group)]
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    ids = [None for _ in range(n_group)]
    alive_idx = [None for _ in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            obs[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
            alive_idx[i] = np.array(list(map(lambda x: x % max_nums[i], ids[i])), dtype=int)

        #################
        # Choose action #
        #################
        for i in range(n_group):
            if 'causal' in models[i].name:
                former_act_prob[i] = np.tile(former_act_prob[i], (len(alive_idx[i]), 1)) if step_ct == 0 else former_act_prob[i][alive_idx[i]]  # filter dead agent
            else:  # former_act_prob[i] already has shape [len(obs[i][0]), n_action[i]]
                former_act_prob[i] = np.tile(former_act_prob[i], (len(alive_idx[i]), 1))
            acts[i][alive_idx[i]], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=True)

        for i in range(n_group):
            env.set_action(handles[i], acts[i][alive_idx[i]])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer_pred = {'state': obs[0], 'acts': acts[0][alive_idx[0]], 'rewards': rewards[0], 'alives': alives[0], 'ids': ids[0],
                  'prob': former_act_prob[0]}
        buffer_prey = {'state': obs[1], 'acts': acts[1][alive_idx[1]], 'rewards': rewards[1], 'alives': alives[1],
                       'ids': ids[1], 'prob': former_act_prob[1]}

        #############################
        # Calculate former_act_prob #
        #############################
        # obs, idx, te_arr may change its len every step, while acts, pi, former_act_prob, onehot_former_acts always keep its shape.
        for i in range(n_group):
            if 'causal' in models[i].name:
                onehot_former_acts = np.array(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])))  # t-1
                te_arr = calculate_te(n_action[i], obs[i], acts[i], models[i], max_nums[i], alive_idx[i], eps, infer_method='weight')
                former_act_prob[i] = np.zeros((max_nums[i], n_action[i]))
                former_act_prob[i][alive_idx[i]] = np.vstack([np.average(onehot_former_acts, axis=0, weights=te) for te in te_arr])
            else:
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])), axis=0, keepdims=True)

        if train:
            models[0].flush_buffer(**buffer_pred)
            models[1].flush_buffer(**buffer_prey)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6)}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards