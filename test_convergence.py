import tensorflow as tf
from main import main
import numpy as np

max_experiments = 50

class InlineArgs:
    gpu = '0'
    run = 'train'
    conf = ''

def test_convergence():
    config_file = 'test_convergence'
    n_errors = 0
    n_ok = 0
    n_experiments = 0
    while n_experiments < max_experiments:
        config = config_file
        n_experiments += 1
        try:
            print('==============================================================================================')
            print('==============================================================================================')
            print('Experiment: ' + config)
            inline_args = InlineArgs()
            inline_args.conf = config
            tf.reset_default_graph()
            main(inline_args)
            n_ok += 1
        except:
            n_errors += 1
            print('Exception in experiment. Going to the next one')

        print('')
        print('==============================================================================================')
        print('Partial results:')
        print('n_ok: ' + str(n_ok))
        print('n_errors: ' + str(n_errors))
        print('n_experiments: ' + str(n_experiments))
        print('Percent OK: ' + str(float(n_ok) / n_experiments * 100.0))

    return


if __name__ == '__main__':
    test_convergence()





