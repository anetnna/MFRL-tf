import tensorflow.compat.v1 as tf
import magent
from examples.battle_model.algo import spawn_ai

tf.disable_v2_behavior()

import os, sys, getopt

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(dir_path, replace_from, replace_to, add_prefix, dry_run):
    with tf.Session() as sess:
        env = magent.GridWorld('battle', map_size=40)
        handles = env.get_handles()
        model = spawn_ai('mfq', sess, env, handles[0], 'mfq-me', 400)
        model_proxy = spawn_ai('mfq', sess, env, handles[1], 'mfq-oppo', 400)
        sess.run(tf.global_variables_initializer())

        model.load(dir_path, step=1999)
        l_vars, r_vars = model.vars, model_proxy.vars
        assert len(l_vars) == len(r_vars)
        [tf.assign(model_proxy.vars[i], model.vars[i]) for i in range(len(l_vars))]
        # Load the variable
        # var = tf.train.load_variable(checkpoint_dir, var_name)
        # Set the new name

        if not dry_run:
            # Save the variables
            # sess.run(tf.global_variables_initializer())
            if not os.path.exists(os.path.join('data/models/rename_mfq_map40-0')):
                os.mkdir(os.path.join('data/models/rename_mfq_map40-0'))
            model_proxy.save('data/models/rename_mfq_map40-0', step=1999)

def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])
