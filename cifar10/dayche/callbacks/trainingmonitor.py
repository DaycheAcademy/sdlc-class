import json
import os
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.python.keras.callbacks import BaseLogger

# wrapper
class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_at=0):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
        self.history = dict()

    def on_train_begin(self, logs=None):
        self.history = dict()

        if self.json_path:
            if os.path.exists(self.json_path):
                self.history = json.load(open(self.json_path, 'r'))

                if self.start_at > 0:
                    for k in self.history.keys():
                        self.history[k] = self.history[k][:self.start_at]

    def on_epoch_end(self, epoch, logs=None):
        # get loss, accuracy and update them
        if logs is None:
            logs = dict()
        for k,v in logs.items():
            l = self.history.get(k, [])
            l.append(v)
            self.history[k] = l

        if self.json_path:
            with open(self.json_path, 'w') as f:
                f.write(json.dumps(self.history, indent=4))

        if len(self.history["loss"]) > 1:
            N = np.arange(0, len(self.history["loss"]))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.history["loss"], label="training_loss")
            plt.plot(N, self.history["val_loss"], label="validation_loss")
            plt.plot(N, self.history["accuracy"], label="training_accuracy")
            plt.plot(N, self.history["val_accuracy"], label="validation_accuracy")
            plt.title("Training loss and accuracy at Epoch {}"
                      .format(len(self.history["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower right")
            plt.savefig(self.fig_path)
            plt.close()









