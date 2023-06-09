from . import ac
from . import q_learning

AC = ac.ActorCritic
MFAC = ac.MEMFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ
AttMFQ = q_learning.AttentionMFQ


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps, moment_order=3):
    if algo_name == 'mfq' or algo_name == 'causal_mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'attention_mfq':
        model = AttMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env,moment_order=moment_order)
    elif algo_name == 'mfac_bin':
        model = MFAC(sess, human_name, handle, env, moment_order=moment_order)
    elif algo_name == 'mfac_leg':
        model = MFAC(sess, human_name, handle, env, moment_order=moment_order)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    return model
