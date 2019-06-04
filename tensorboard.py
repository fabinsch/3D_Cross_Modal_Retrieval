import io
import numpy as np
import tensorflow as tf

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()






        '''writer.add_embedding(
            out,
            metadata=label_batch.data,
            label_img=data_batch.data,
            global_step=n_iter)'''