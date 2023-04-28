import numpy as np


def kl_div(p, q):
    """
    Calculate the KL(p||q) divergence between two probability distributions.
    """
    eps = 1e-16
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=1, keepdims=True)


def softmax(x, temperature=1.0):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1, keepdims=True)


def interv_weight(action_space, state, acts, model, max_num, ids, eps):
    """
    inference on the weights of weighted average of the actions.
    """
    former_onehot_acts = list(map(lambda x: np.eye(action_space)[x], acts[ids]))  # Per row according to an agent's action
    ave_act_prob = np.mean(former_onehot_acts, axis=0, keepdims=True)
    ave_act_prob = np.tile(ave_act_prob, (max_num, 1))
    _, pi = model.act(state=state, prob=ave_act_prob[ids], eps=eps, train=False)
    te_ls = []
    for agent_idx in range(len(ids)):
        # weights = np.zeros(len(ids)) * (1-gamma) / (len(ids)-1)
        # weights[agent_idx] = gamma
        # ave_act_prob = np.array(np.average(onehot_acts, axis=0, weights=weights))
        infered_act_prob = former_onehot_acts[agent_idx]
        infered_act_prob = np.tile(infered_act_prob, (max_num, 1))
        _, pi_inferred = model.act(state=state, prob=infered_act_prob[ids], eps=eps, train=False)
        te = kl_div(pi, pi_inferred)
        te_ls.append(te)
    te_arr = np.hstack(te_ls) + 0.001
    # print('te value range: {}\n{}'.format(np.min(te_arr, axis=1), np.max(te_arr, axis=1)))
    # te_arr = softmax(te_arr, temperature=0.01)
    return te_arr


def interv_action(action_sapce, state, acts, model, max_num, ids, eps):
    """
    Do(a_bar=a_i) and Do(a_bar=0) for each agent. That is, imagine what if the adjacent agent act with a_j in 1 vs 1
    scenario and imagine what if the adjacent agent do nothing in 1 vs 1 scenario. By comparing the two, we can get the
    treatment effect which represent the importance of a_i.
    """
    onehot_acts = list(map(lambda x: np.eye(action_sapce)[x], acts[ids]))  # Per row according to an agent's action
    te_ls = []
    for agent_idx in range(len(ids)):
        agent_idx_act = np.tile(onehot_acts[agent_idx], (max_num, 1))
        _, pi_imagine = model.act(state=state, prob=agent_idx_act[ids], eps=eps, acts=acts[ids])
        _, pi_do0 = model.act(state=state, prob=np.zeros_like(agent_idx_act[ids]), eps=eps, acts=acts[ids])
        te = kl_div(pi_do0, pi_imagine)
        te_ls.append(te)
    te_arr = np.hstack(te_ls)
    te_arr = softmax(te_arr, temperature=5)
    return te_arr


def random_weight(action_space, state, acts, model, max_num, ids, eps):
    """
    Return random weights
    """
    # te_arr = np.ones([len(ids), len(ids)])
    te_arr = np.random.rand(len(ids), len(ids))
    # te_arr = softmax(te_arr, temperature=5)
    return te_arr



def calculate_te(action_space, state, acts, model, max_num, ids, eps, infer_method='weight'):
    """
    Calculate treatment effects.
    """
    te_arr = None
    if infer_method == 'weight':
        te_arr = interv_weight(action_space, state, acts, model, max_num, ids, eps)
    elif infer_method == 'a_bar':
        te_arr = interv_action(action_space, state, acts, model, max_num, ids, eps)
    elif infer_method == 'random':
        te_arr = random_weight(action_space, state, acts, model, max_num, ids, eps)
    return te_arr
