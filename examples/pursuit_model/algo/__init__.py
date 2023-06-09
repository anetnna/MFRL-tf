from . import ac
from . import q_learning

AC = ac.ActorCritic
MFAC = ac.MFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ
AttMFQ = q_learning.AttentionMFQ
MEMFQ = q_learning.MEMFQ


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps):
    if algo_name == 'mfq' or algo_name == 'causal_mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=400000)
    elif algo_name == 'attention_mfq':
        model = AttMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'me_mfq':
        model = MEMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    return model
