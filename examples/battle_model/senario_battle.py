import copy
import random
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from examples.battle_model.causal_inference.infer_utils import calculate_te

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def vis_weight(alive_ids, pos, te_arr, vis_id, map_size, step_ct, t, algo):
    """
    alive_ids: [left_alice_ids, right_alive_ids] list of trackable_object of alive players
    pos: [left_pos, right_pos] list of positions of alive player
    te_arr: [left_te_arr, right_te_arr] list of te_arr of alive player
    vis_id: agent id of the left group to be visualized
    map_size: size of the map
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    if vis_id in alive_ids[0].tolist():
        index = alive_ids[0].tolist().index(vis_id)
        weights = te_arr[index, :]
        ax.scatter(np.array(pos[0][::2]) * map_size, np.array(pos[0][1::2]) * map_size, c='b', s=10, marker='s')
        ax.scatter(np.array(pos[1][::2]) * map_size, np.array(pos[1][1::2]) * map_size, c='r', s=10, marker='s')
        ax.scatter(pos[0][index*2] * map_size, pos[0][index*2+1] * map_size, c='g', s=10, marker='s')
        for i, (x,y,w) in enumerate(zip(np.array(pos[0][::2]) * map_size, np.array(pos[0][1::2]) * map_size, weights)):
            if i != index:
                ax.text(x-0.4, y+0.35, str(w)[1:5], fontsize=3)
        if not os.path.exists('./vis_weights_{}{}'.format(algo, t)):
            os.makedirs('./vis_weights_{}{}'.format(algo, t))
        plt.savefig('./vis_weights_{}{}/weight_vis_id{}_step{}.png'.format(algo, t, vis_id, step_ct), dpi=300)
        plt.savefig('./vis_weights_{}{}/weight_vis_id{}_step{}.svg'.format(algo, t, vis_id, step_ct), dpi=300)
        print('Step:{} Saved!'.format(step_ct))
    else:
        print('Step:{} vis_id {} is not alive'.format(step_ct, vis_id))


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


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
            former_act_prob[i] = np.tile(former_act_prob[i], (len(alive_idx[i]), 1))
            acts[i][alive_idx[i]], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=True)

        for i in range(n_group):
            env.set_action(handles[i], acts[i][alive_idx[i]])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': obs[0], 'acts': acts[0][alive_idx[0]], 'rewards': rewards[0], 
            'alives': alives[0], 'ids': ids[0], 
        }
  
        buffer['prob'] = former_act_prob[0]      

        #############################
        # Calculate former_act_prob #
        #############################
        # obs, idx, te_arr may change its len every step, while acts, pi, former_act_prob, onehot_former_acts always keep its shape.
        for i in range(n_group):
            if 'me' in models[i].name:
                former_act_prob[i] = np.sum(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])),axis=0, keepdims=True)
            else:
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])), axis=0, keepdims=True)
            
        if train:
            models[0].flush_buffer(**buffer)

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

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


# def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
#     """play a ground and train"""
#     env.reset()
#     generate_map(env, map_size, handles)
#
#     step_ct = 0
#     done = False
#
#     n_group = len(handles)
#     state = [None for _ in range(n_group)]
#     acts = [None for _ in range(n_group)]
#     ids = [None for _ in range(n_group)]
#
#     alives = [None for _ in range(n_group)]
#     rewards = [None for _ in range(n_group)]
#     nums = [env.get_num(handle) for handle in handles]
#     max_nums = nums.copy()
#
#     n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
#
#     print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
#     mean_rewards = [[] for _ in range(n_group)]
#     total_rewards = [[] for _ in range(n_group)]
#
#     former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
#
#     while not done and step_ct < max_steps:
#         # take actions for every model
#         for i in range(n_group):
#             state[i] = list(env.get_observation(handles[i]))
#             ids[i] = env.get_agent_id(handles[i])
#
#         for i in range(n_group):
#             former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
#             acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)
#
#         for i in range(n_group):
#             env.set_action(handles[i], acts[i])
#
#         # simulate one step
#         done = env.step()
#
#         for i in range(n_group):
#             rewards[i] = env.get_reward(handles[i])
#             alives[i] = env.get_alive(handles[i])
#
#         for i in range(n_group):
#             former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)
#
#         # stat info
#         nums = [env.get_num(handle) for handle in handles]
#
#         for i in range(n_group):
#             sum_reward = sum(rewards[i])
#             rewards[i] = sum_reward / nums[i]
#             mean_rewards[i].append(rewards[i])
#             total_rewards[i].append(sum_reward)
#
#         if render:
#             env.render()
#
#         # clear dead agents
#         env.clear_dead()
#
#         info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}
#
#         step_ct += 1
#
#         if step_ct % print_every == 0:
#             print("> step #{}, info: {}".format(step_ct, info))
#
#     for i in range(n_group):
#         mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
#         total_rewards[i] = sum(total_rewards[i])
#
#     return max_nums, nums, mean_rewards, total_rewards

def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    # generate_rand_map(env, map_size, density, handles)
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    obs = [None for _ in range(n_group)]
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    ids = [None for _ in range(n_group)]
    alive_idx = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
    t = time.time()
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
            acts[i][alive_idx[i]], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=False)

        for i in range(n_group):
            env.set_action(handles[i], acts[i][alive_idx[i]])

        # simulate one step
        done = env.step()

        #############################
        # Calculate former_act_prob #
        #############################
        # obs, idx, te_arr may change its len every step, while acts, pi, former_act_prob, onehot_former_acts always keep its shape.
        for i in range(n_group):
            if 'causal' in models[i].name:
                onehot_former_acts = np.array(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])))  # t-1
                te_arr = calculate_te(n_action[i], obs[i], acts[i], models[i], max_nums[i], alive_idx[i], eps, infer_method='weight')
                if i == 0:
                    te_arr_lf = te_arr
                former_act_prob[i] = np.zeros((max_nums[i], n_action[i]))
                # former_act_prob[i][alive_idx[i]] = np.vstack([np.average(onehot_former_acts, axis=0, weights=te) for te in te_arr])
            else:
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])), axis=0,
                                             keepdims=True)

        # Visualize the weight of each interaction
        if render:
            if step_ct != 0 and 'causal' in models[0].name:
                vis_pos = [[], []]
                for i in range(n_group):
                    for j, feature in enumerate(obs[i][1]):
                        vis_pos[i].append(feature[-2])
                        vis_pos[i].append(feature[-1])
                # print('vis_pos:{}'.format(vis_pos))
                vis_weight(ids, vis_pos, te_arr_lf, 30, map_size, step_ct, t, 'causal')

            elif step_ct != 0 and 'attention' in models[0].name:
                vis_pos = [[], []]
                for i in range(n_group):
                    for j, feature in enumerate(obs[i][1]):
                        vis_pos[i].append(feature[-2])
                        vis_pos[i].append(feature[-1])
                # print('vis_pos:{}'.format(vis_pos))
                weight = models[0].get_weight(state=obs[0])
                vis_weight(ids, vis_pos, weight, 30, map_size, step_ct, t, 'att')

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

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

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
