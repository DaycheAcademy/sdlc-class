import os.path

from tensorflow.keras.callbacks import Callback
# from tensorflow.python.keras.callbacks import Callback


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        super(EpochCheckpoint, self).__init__()
        self.output_path = output_path
        self.every = every
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (self.start_at + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path,
                                  "epoch_{}.hdf5".format(self.start_at + 1)])
            # p = os.path.sep.join([self.output_path,
            #                       "epoch_{}.keras".format(self.start_at + 1)])
            self.model.save(p, overwrite=True)
            # self.model.save_weights(p, overwrite=True)
        self.start_at += 1



