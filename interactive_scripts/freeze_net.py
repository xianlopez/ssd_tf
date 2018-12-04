import TrainEnv
from predict_config import UpdatePredictConfiguration
import tensorflow as tf
import os
import sys
import tools

# Read configuration from predict_config:
args = UpdatePredictConfiguration()

# Create network and start interactive session:
net = TrainEnv.TrainEnv(args, 'interactive')
sess = net.start_interactive_session(args)

# Save graph:
protobuf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'protobufs')
tf.train.write_graph(sess.graph_def, protobuf_dir, 'net_graph.pb', False)

# Frozen graph:
python_path = sys.executable
input_graph = os.path.join(tools.get_base_dir(), 'protobufs', 'net_graph.pb')
output_graph = os.path.join(tools.get_base_dir(), 'protobufs', 'frozen_' + args.model_name + '.pb')
command = python_path + \
          r' -m tensorflow.python.tools.freeze_graph' + \
          r' --input_graph=' + input_graph + \
          r' --input_checkpoint=' + args.weights_file + \
          r' --input_binary=true' + \
          r' --output_graph=' + output_graph + \
          r' --output_node_names=Softmax'
os.system(command)

