# A Binary Multilayerperceptron Classifier. Currently Depends on a custom
# dataset class defined in higgs_dataset.py. Also it is assumend that there
#are no error in the shape of the dataset
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from sklearn.metrics import roc_auc_score, roc_curve
from draw_nn import PlotNN

class BinaryMLP:
    """A Binary Classifier using a Multilayerperceptron.

    Makes probability predictions on a set of features (A 1-dimensional
    numpy vector belonging either to the 'signal' or the 'background'.

    """

    def __init__(self, n_features, h_layers, savedir):
        """Initializes the Classifier.

        Arguments:
        ----------------
        nfeatures (int):
            The number of input features.
        hlayers (list):
            A list representing the hidden layers. Each entry gives the number
            of neurons in the equivalent layer.

        Attributes:
        ----------------
        name (str):
            Name of the model.
        savedir (str):
            Path to directory everything will be saved in.
        trained (bool):
            Flag wether model has been trained or not
        """
        self.n_features = n_features
        self.h_layers = h_layers
        self.n_labels = 1
        self.name = savedir.rsplit('/')[-1]
        self.savedir = savedir

        # check wether model file exists
        if os.path.exists(self.savedir + '/{}.ckpt'.format(self.name)):
            self.trained = True
        else:
            self.trained = False

        # create directory if needed
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def _get_parameters(self):
        """Creates tensorflow Variables in two lists.

        Returns:
        --------------
        weights (list):
            A dictionary with the tensorflow Variables for the weights.
        biases (list):
            A dictionary with the tensorflow Variables for the biases.
        """
        n_features = self.n_features
        h_layers = self.h_layers

        weights = [tf.Variable(
            tf.random_normal([n_features, h_layers[0]], stddev=tf.sqrt(2.0/n_features)),
            name = 'W_1')]
        biases = [tf.Variable(tf.zeros([h_layers[0]]),
                              name = 'B_1')]

        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(tf.random_normal(
                    [h_layers[i-1], h_layers[i]], stddev=tf.sqrt(2.0/h_layers[i-1])),
                name = 'W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.zeros([h_layers[i]]),
                                          name = 'B_{}'.format(i+1)))

        weights.append(tf.Variable(
                tf.random_normal([h_layers[-1], 1], stddev=tf.sqrt(2.0/h_layers[-1])),
                name = 'W_out'))
        biases.append(tf.Variable(tf.zeros([1]),
                      name = 'B_out'))

        return weights, biases

    def _model(self, data, W, B, keep_prob=1.0):
        """Model for the multi layer perceptron

        Arguments:
        --------------
        data (tf.placeholder):
            A tensorflow placeholder.
        W (list):
            A list with the tensorflow Variables for the weights.
        B (list):
            A list with the tensorflow Variables for the biases.

        Returns:
        ---------------
        out (tf.tensor)
            Prediction of the model.
        """
        layer = tf.nn.dropout(
            tf.nn.relu(tf.matmul(data, W[0]) + B[0]), keep_prob)

        if len(self.h_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(
                    tf.nn.relu(tf.matmul(layer, weight) + bias), keep_prob)

        out = tf.matmul(layer, W[-1]) + B[-1]

        return tf.nn.sigmoid(out)

    def train(self, train_data, val_data, epochs=10,
              batch_size=100, keep_prob=1.0, beta=0.0):
        """Trains the classifier

        Arguments:
        -------------
        train_data (custom dataset):
            Contains training data.
        val_data (custom dataset):
            Contains validation data.
        savedir (string):
            Path to directory to save Plots.
        epochs (int):
            Number of iterations over the whole trainig set.
        batch_size (int):
            Number of batches fed into on optimization step.
        keep_prob (float):
            Probability of a neuron to 'activate'.
        beta (float):
            L2 regularization coefficient. Defaul 0.0 = regularization off.

        """

        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.placeholder(tf.float32, [None, 1])

            x_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32),
                                 trainable=False, name='x_mean')
            x_std = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32),
                                trainable=False, name='x_std')

            x_scaled = tf.div(tf.sub(x, x_mean), x_std)
            
            weights, biases = self._get_parameters()

            #prediction
            y_ = self._model(x_scaled, weights, biases, keep_prob)
            yy_ = self._model(x_scaled, weights, biases)

            # loss function
            xentropy = -(tf.mul(y, tf.log(y_)) + tf.mul(1-y,tf.log(1-y_)))
            l2_reg = beta*self._l2_regularization(weights)
            loss = tf.reduce_mean(tf.mul(w,xentropy)) + l2_reg

            # optimizer
            train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

            # initialize the variables
            init = tf.initialize_all_variables()
            saver = tf.train.Saver(weights + biases)

        train_start = time.time()

        with tf.Session(graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
            sess.run(init)
            train_auc = []
            val_auc = []
            train_losses = []

            print(110*'-')
            print('Train model: {}'.format(self.model_loc))
            print(110*'-')
            print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch',
                'Training Loss', 'AUC Training Score', 'AUC Validation Score'))
            print(110*'-')

            for epoch in range(epochs):
                total_batches = int(train_data.n/batch_size)
                epoch_loss = 0
                
                for _ in range(total_batches):
                    train_x, train_y, train_w= train_data.next_batch(batch_size)

                    _, train_loss= sess.run([train_step, loss],
                                        {x: train_x, y: train_y, w: train_w})
                    epoch_loss += train_loss

                # set internal dataframe index to 0
                train.shuffle()
                
                train_losses.append(np.mean(epoch_loss))

                # monitor training
                train_pre = sess.run(yy_, {x : train_data.x})
                train_auc.append(roc_auc_score(train_data.y, train_pre))
                val_pre = sess.run(yy_, {x : val_data.x})
                val_auc.append(roc_auc_score(val_data.y, val_pre))
                
                print('{:^25} | {:^25.4e} | {:^25.4f} | {:^25.4f}'.format(
                    epoch+1, train_losses[-1], train_auc[-1], val_auc[-1]))
                
                saver.save(sess, self.model_loc)
                
            print(110*'-')
            train_end = time.time()

            self._validation(val_pre, val_data.y)
            self._plot_auc_dev(train_auc, val_auc, epochs)
            self._plot_loss(train_losses)

            self.trained = True

            self._write_parameters(epochs, batch_size, keep_prob, beta,
                                   (train_end - train_start)/60)
            print('Model saved in: \n{}'.format(self.savedir))

    def _l2_regularization(self, weights):
        """Calculate and adds the squared values of the weights. This is used
        for L2 Regularization.
        """
        weights = map(lambda x: tf.nn.l2_loss(x), weights)

        return sum(weights)

    def _write_parameters(self, epochs, batch_size, keep_prob, beta, time):
        """Writes network parameters in a .txt file
        """

        with open('{}/info.txt'.format(self.savedir), 'w') as f:
            f.write('Training Epochs: {}\n'.format(epochs))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Training Time: {} min\n'.format(time))


    def _validation(self, pred, labels):
        """Validation of the training process.
        Makes plots of ROC curves and displays the development of AUC score.

        Arguments:
        ----------------
        pred (np.array, shape(-1,)):
            Predictions for data put in the model.
        labels (np.array, shape(-1)):
            Lables of the validation dataset.
        epoch (int):
            Epoch which was used for validation.

        Returns:
        ----------
        auc_sore (float):
            Number between 0 and 1.0. Displays the model's quality.
        """

        # distribution
        y = np.hstack((pred, labels))
        sig = []
        bg = []
        for i in range(len(y)):
            if y[i, 1]==1:
                sig.append(y[i, 0])
            else:
                bg.append(y[i,0])

        bin_edges = np.linspace(0, 1, 50)
        plt.hist(sig, bins=bin_edges, color='black',histtype='step',
                 label='Signal', normed='True')
        plt.hist(bg, bins=bin_edges, color='red',histtype='step',
                 label='Background', normed='True', linestyle='--')

        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel('NN Output')
        plt.ylabel('norm. to unit area')
        plt_name = self.name + '_dist'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

        # roc curve
        pred = np.reshape(pred, -1)
        fpr, tpr, thresh = roc_curve(labels, pred)
        auc = roc_auc_score(labels, pred)
        #plot the roc_curve
        plt_name = self.name +  '_roc'
        plt.plot(tpr, np.ones(len(fpr)) - fpr, color='red',
         lw=1.,label='ROC Curve (area = %0.2f)' % auc)
        #make the plot nicer
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.xlabel('Signal Efficiency')
        plt.ylabel('Background Rejection')
        plt.legend(loc='best', frameon=False)
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def predict_prob(self, data):
        """Predict probability of a new feauture to belong to the signal.

        Arguments:
        ----------------
        data (custom data set):
            Data to classify.

        Returns:
        ----------------
        prob (np.array):
            Contains probabilities of a sample to belong to the signal.
        """


        if not self.trained:
            sys.exit('Model {} has not been trained yet'.format(self.name))

        predict_graph = tf.Graph()
        with predict_graph.as_default():
            weights, biases = self._get_parameters()
            x = tf.placeholder(tf.float32, [None, self.n_features])
            x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
            x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')

            x_scaled = tf.div(tf.sub(x, x_mean), x_std)
            
            y_prob = self._model(x_scaled, weights, biases)
            
            saver = tf.train.Saver()
        with tf.Session(graph = predict_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            prob = sess.run(y_prob, {x: data})

        return prob

    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        plt.plot(train_loss, label= 'Training Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        plt.legend(loc=0, frameon=False)
        plt.savefig(self.savedir + '/loss.pdf')
        plt.savefig(self.savedir + '/loss.png')
        plt.savefig(self.savedir + '/loss.eps')
        plt.clf()

    def _plot_auc_dev(self, train_auc, val_auc, nepochs):
        """Plot ROC-AUC-Score development
        """
        plt.plot(train_auc, color='red', label='Training')
        plt.plot(val_auc, color='black', label='Validation')
        # make plot nicer
        plt.xlabel('Epoch')
        plt.ylabel('ROC_AUC')
        plt.legend(loc='best', frameon=False)

        # save plot
        plt_name = self.name + '_auc_dev'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def plot_nn(self, branches):  
        weight_graph = tf.Graph()
        with weight_graph.as_default():
            weights, biases = self._get_parameters()
            saver = tf.train.Saver()
        with tf.Session(graph=weight_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            w = []
            b = []
            for weight, bias in zip(weights, biases):
                w.append(sess.run(weight))
                b.append(sess.run(bias))
                plot_nn = PlotNN(branches, w, b)
                plot_nn.render(self.savedir)
        for ww in w:
            print(25*'-')
            print(ww)
