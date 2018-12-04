import TrainEnv
import tensorflow as tf
import subprocess
import socket
import webbrowser
import tools
import os


def prepare_tensorboard(sess):
    merged = tf.summary.merge_all()
    base_dir = tools.get_base_dir()
    summary_writer = tf.summary.FileWriter(os.path.join(base_dir, 'tensorboard'), sess.graph)
    command = '/home/xian/venvs/ssd_tf/bin/tensorboard --logdir=' + os.path.join(base_dir, 'tensorboard')
    process = subprocess.Popen(["start", "cmd", "/k", command], shell=True)
    hostname = socket.gethostname()
    tensorboard_url = 'http://' + hostname + ':6006'
    webbrowser.open(tensorboard_url, new=1, autoraise=True)

    return merged, summary_writer, tensorboard_url


args = UpdatePredictConfiguration()

net = TrainEnv.TrainEnv(args, 'interactive')

sess = net.start_interactive_session(args)

# Tensorboard:
merged, summary_writer, tensorboard_url = prepare_tensorboard(sess)

