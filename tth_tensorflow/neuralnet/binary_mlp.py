# Written by Max Welsch.
# A Binary Multilayerperceptron Classifier. Currently depends on a custom
# dataset class defined in higgs_dataset.py. Also it is assumed that there
# are no errors in the shape of the dataset
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from sklearn.metrics import roc_auc_score, roc_curve

class BinaryMLP:
    """A Binary Classifier using a Multilayerperceptron.

    Makes probability predictions on a set of features (A 1-dimensional
    numpy vector belonging either to the 'signal' or the 'background').

    """

    def __init__(self, n_features, h_layers, savedir):
        """Initializes the Classifier.

        Arguments:
        ----------------
        n_features (int):
            The number of input features.
        h_layers (list):
            A list representing the hidden layers. Each entry gives the number
            of neurons in the equivalent layer.

        Attributes:
        ----------------
        name (str):
            Name of the model.
        savedir (str):
            Path to directory everything will be saved to.
        trained (bool):
            Flag whether model has been trained or not
        """
        self.n_features = n_features
        self.h_layers = h_layers
        self.n_labels = 1
        self.name = savedir.rsplit('/')[-1]
        self.savedir = savedir

        # check whether model file exists
        if os.path.exists(self.savedir + '/{}.ckpt'.format(self.name)):
            self.trained = True
        else:
            self.trained = False

        # create directory if needed
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def _get_parameters(self):
        """Creates the tensorflow Variables in two lists.

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
        # if more than 1 hidden layer -> create additional weights and biases
        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(tf.random_normal(
                    [h_layers[i-1], h_layers[i]], stddev=tf.sqrt(2.0/h_layers[i-1])),
                name = 'W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.zeros([h_layers[i]]),
                                          name = 'B_{}'.format(i+1)))
        
        # connect the last hidden layer to the output layer
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

        # if more than 1 hidden layer -> generate output via multiple weight
        # matrices
        if len(self.h_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(
                    tf.nn.relu(tf.matmul(layer, weight) + bias), keep_prob)

        out = tf.matmul(layer, W[-1]) + B[-1]

        return tf.nn.sigmoid(out)

    def train(self, train_data, val_data, epochs=10,
              batch_size=100, keep_prob=0.9, beta=0.0):
        """Trains the classifier

        Arguments:
        -------------
        train_data (custom dataset):
            Contains training data.
        val_data (custom dataset):
            Contains validation data.
        savedir (string):
            Path to directory to save plots.
        epochs (int):
            Number of iterations over the whole trainig set.
        batch_size (int):
            Number of batches fed into one optimization step.
        keep_prob (float):
            Probability of a neuron to 'activate'.
        beta (float):
            L2 regularization coefficient. Default 0.0 = regularization off.

        """

        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.placeholder(tf.float32, [None, 1])

            weights, biases = self._get_parameters()

            # prediction
            y_ = self._model(x, weights, biases, keep_prob)
            yy_ = self._model(x, weights, biases)

            # loss function
            xentropy = -(tf.mul(y, tf.log(y_)) + tf.mul(1-y,tf.log(1-y_)))
            l2_reg = beta*self._l2_regularization(weights)
            loss = tf.reduce_mean(tf.mul(w,xentropy)) + l2_reg

            # optimizer
            train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

            # initialize the variables
            init = tf.initialize_variables(tf.all_variables(), name="nInit")
            # saver = tf.train.Saver(weights + biases)
            saver = tf.train.Saver()

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
                    # train in batches
                    train_x, train_y, train_w= train_data.next_batch(batch_size)
                    _, train_loss= sess.run([train_step, loss],
                                        {x: train_x, y: train_y, w: train_w})
                    epoch_loss += train_loss
                train_losses.append(np.mean(epoch_loss))

                # monitor training
                train_pre = sess.run(yy_, {x : train_data.x})
                train_auc.append(roc_auc_score(train_data.y, train_pre))
                val_pre = sess.run(yy_, {x : val_data.x})
                val_auc.append(roc_auc_score(val_data.y, val_pre))
                print('{:^25} | {:^25.4f} | {:^25.4f} | {:^25.4f}'.format(
                    epoch+1, train_losses[-1], train_auc[-1], val_auc[-1]))

                saver.save(sess, self.model_loc)
            print(110*'-')
            train_end = time.time()

            
            # Save weights and graph
            # weights_ = [weight.eval() for weight in weights]
            # biases_ = [bias.eval() for bias in biases]
            # np.savetxt(self.savedir + '/weights.txt', weights_, fmt='%1.4e', delimiter=';', newline='\n')
            # np.savetxt(self.savedir + '/biasas.txt', biases_, fmt='%1.4e',delimiter=';',newline='\n')
            # for variable in tf.trainable_variables():
            #     tensor = tf.constant(variable.eval())
            #     tf.assign(variable, tensor, name="nWeights")
            # tf.train.write_graph(sess.graph_def, 'graph/', 'my_graph.pb',
            #         as_text=False)

            
            self._validation(val_pre, val_data.y)
            self._plot_auc_dev(train_auc, val_auc, epochs)
            self._plot_loss(train_losses)

            self.trained = True

            self._write_parameters(epochs, batch_size, keep_prob, beta,
                                   (train_end - train_start)/60)
            print('Model saved in: \n{}'.format(self.savedir))

    def _l2_regularization(self, weights):
        """Calculates and adds the squared values of the weights. This is used
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
            Predictions for data put into the model.
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
        # concatenates the two arrays pred and labels
        y = np.hstack((pred, labels))
        sig = []
        bg = []
        # split data into signal and background
        for i in range(len(y)):
            if y[i, 1]==1:
                sig.append(y[i, 0])
            else:
                bg.append(y[i,0])

        # create histograms
        nbins = 25
        plt.hist(sig, bins=nbins, color='black',histtype='step',
                 label='Signal', normed='True')
        plt.hist(bg, bins=nbins, color='red',histtype='step',
                 label='Background', normed='True', linestyle='dashed')

        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel('Predicted Probability')
        plt.ylabel('arb. unit')
        plt.grid(True)
        plt_name = self.name + '_dist'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

        # roc curve
        pred = np.reshape(pred, -1)
        # fpr = false positive rate, true positive rate
        fpr, tpr, thresh = roc_curve(labels, pred)
        auc = roc_auc_score(labels, pred)
        # plot the roc_curve
        plt_name = self.name +  '_roc'
        plt.plot(tpr, np.ones(len(fpr)) - fpr, color='red',
         lw=1.,label='ROC Curve (area = %0.2f)' % auc)
        # make the plot nicer
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.grid()
        plt.xlabel('Signal Efficiency')
        plt.ylabel('Background Rejection')
        plt.title('ROC Curve')
        plt.legend(bbox_to_anchor=(1,1))
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()


    def predict_prob(self, data):
        """Predict probability of a new feature to belong to the signal.

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
            y_prob = self._model(x, weights, biases)
            # init = tf.initialize_all_variables()
            saver = tf.train.Saver()
        with tf.Session(graph = predict_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            prob = sess.run(y_prob, {x: data})

        return prob


    def get_weights(self, branches):
        weight_graph = tf.Graph()
        with weight_graph.as_default():
            weights, biases = self._get_parameters()
            saver = tf.train.Saver()
        with tf.Session(graph=weight_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            w = []
            for weight in weights:
                w.append(sess.run(weight))
            self._plot_weight(w[0], branches)


    def _plot_weight(self, weight, branches):
        x = np.sum(weight, axis=1)
        x_pos = np.arange(len(x))
        plt.bar(x_pos, x, align='center')
        plt.xticks(x_pos, branches, rotation=30, ha='right')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()


    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        plt.plot(train_loss, label= 'Training Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.savefig(self.savedir + '/loss.pdf')
        plt.savefig(self.savedir + '/loss.png')
        plt.savefig(self.savedir + '/loss.eps')
        plt.clf()


    def _plot_auc_dev(self, train_auc, val_auc, nepochs):
        """Plot ROC-AUC-Score development
        """
        plt.plot(train_auc, color='red', label='AUC Training')
        plt.plot(val_auc, color='black', label='AUC Validation')
        # make plot nicer
        plt.xlabel('Epoch')
        plt.ylabel('ROC_AUC')
        plt.title('ROC_AUC Development')
        plt.legend(loc='best')
        plt.grid(True)

        # save plot
        plt_name = self.name + '_auc_dev'


        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()


    # def _save(self, filename):
    #     with tf.Session(graph=train_graph) as sess: 
    #         for variable in tf.trainable_variables():
    #             tensor = tf.constant(variable.eval())
    #             tf.assign(variable, tensor, name="nWeights")
    #         tf.train.write_graph(self.sess.graph_def, 'graph/', filename,
    #             as_text=False)
