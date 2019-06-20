import tensorflow as tf
import tools
import logging
import ssd
import InteractiveDataReader
import DataReader
import os


class InteractiveEnv:

    def __init__(self, opts, ssd_config):
        self.th_conf = opts.th_conf
        self.use_nms = opts.use_nms
        self.th_nms = opts.th_nms
        self.gpu_memory_fraction = opts.gpu_memory_fraction

        self.ssd_config = ssd_config

        self.predictions = None
        self.inputs = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.reader = None
        self.network = None
        self.sess = None

        # Initialize network:
        self.generate_graph(opts.weights_file)

    def start_interactive_session(self):
        logging.info('Starting interactive session')
        self.sess = tf.Session(config=tools.get_config_proto(self.gpu_memory_fraction))
        self.restore_fn(self.sess)
        logging.info('Interactive session started')
        return self.sess

    def forward_batch(self, batch):
        batch_size = batch.shape[0]
        predictions = self.sess.run(fetches=self.predictions, feed_dict={self.is_training: False, self.inputs: batch})
        predictions = self.postprocess_grid_predictions(predictions, batch_size, self.th_conf, self.use_nms, self.th_nms)
        return predictions

    def generate_graph(self, weights_file):
        classnames_path = os.path.join(os.path.dirname(weights_file), 'classnames.txt')
        assert os.path.exists(classnames_path), 'Cannot find file with class names at ' + classnames_path
        self.classnames, self.nclasses = read_classnames(classnames_path)
        self.network = ssd.ssd_net(self.ssd_config, self.nclasses)
        self.define_inputs()
        self.build_model()
        vars_to_restore = tf.contrib.framework.get_variables_to_restore(include=[n.name for n in tf.global_variables()])  # TODO: is it necessary to specify the variables?
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(weights_file, vars_to_restore)

    def define_inputs(self):
        input_shape = self.network.get_input_shape()
        self.reader = InteractiveDataReader.InteractiveDataReader(input_shape[0])
        self.inputs = self.reader.build_inputs()

    def build_model(self):
        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
        net_output, self.predictions = self.network.build(self.inputs, self.ssd_config, self.is_training)

    def postprocess_grid_predictions(self, predictions, batch_size, th_conf, use_nms, th_nms):
        bndboxes_batched = []
        for i in range(batch_size):
            bndboxes_no_suppresion = self.network.decode_preds(predictions[i, ...], th_conf)
            if use_nms:
                bndboxes = []
                for cl in range(self.nclasses):
                    boxes_this_class = [x for x in bndboxes_no_suppresion if x.classid == cl]
                    pred_list = tools.non_maximum_suppression(boxes_this_class, th_nms)
                    bndboxes.extend(pred_list)
            else:
                bndboxes = bndboxes_no_suppresion
            bndboxes_batched.append(bndboxes)
        return bndboxes_batched


def read_classnames(classnames_path):
    with open(classnames_path, 'r') as fid:
        lines = fid.read().split('\n')
        classes_dict = {}
        max_id = -1
        for line in lines:
            if line != '':
                split = line.split(',')
                class_id = int(split[0])
                class_name = split[1]
                classes_dict[class_id] = class_name
                max_id = max(max_id, class_id)
        nclasses = max_id + 1
        classnames = []
        for i in range(nclasses):
            classnames.append(classes_dict[i])
    return classnames, nclasses

