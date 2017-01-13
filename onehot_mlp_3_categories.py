# A one-hot output vector multi layer perceptron classifier. Currently depends on
# a custom dataset class defined in higgs_dataset.py. It is also assumed that
# there are no errors in the shape of the dataset.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import sys
import time
import pickle

# from sklearn.metrics import roc_auc_score, roc_curve

class OneHotMLP:
    """A one-hot output vector classifier using a multi layer perceptron.

    Makes probability predictions on a set of features (a 1-dimensional numpy
    vector belonging either to the 'signal' or the 'background').
    """


    def __init__(self, n_features, h_layers, out_size, savedir):
        """Initializes the Classifier.

        Arguments:
        ----------------
        n_features (int):
            The number of input features.
        h_layers (list):
            A list representing the hidden layers. Each entry gives the number
            of neurons in the equivalent layer, [30,40,20] would describe a
            network of three hidden layers, containing 30, 40 and 20 neurons.
        out_size (int):
            The size of the one-hot output vector.

        Attributes:
        ----------------
        name (str):
            Name of the model.
        savedir (str):
            Path to the directory everything will be saved to.
        trained (bool):
            Flag whether the model has been trained or not.
        """

        self.n_features = n_features
        self.h_layers = h_layers
        self.out_size = out_size
        self.name = savedir.rsplit('/')[-1]
        self.savedir = savedir

        # check whether the model file exists
        if os.path.exists(self.savedir + '/{}.ckpt'.format(self.name)):
            self.trained = True
        else:
            self.trained = False

        # create directory if necessary
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        
        self.cross_savedir = self.savedir + '/cross_checks'
        if not os.path.isdir(self.cross_savedir):
            os.makedirs(self.cross_savedir)


    def _get_parameters(self):
        """Creates the TensorFlow Variables in two lists. 

        Returns:
        ----------------
        weights (list):
            A dictionary with the TensorFlow Variables for the weights.
        biases (list):
            A dictionary with the TensorFlow Variables for the biases.
        """

        n_features = self.n_features
        h_layers = self.h_layers

        weights = [tf.Variable(tf.random_normal([n_features, h_layers[0]], stddev=tf.sqrt(2.0/n_features)), name = 'W_1')]
        # biases = [tf.Variable(tf.zeros([h_layers[0]]), name = 'B_1')]
        biases = [tf.Variable(tf.random_normal([h_layers[0]], stddev =
            tf.sqrt(2.0 / (h_layers[0]))), name = 'B_1')]


        # weights = [tf.Variable(tf.random_uniform([n_features, h_layers[0]],
        #     minval=0.0, maxval=1.0), name='W_1')]
        # biases = [tf.Variable(tf.random_uniform([h_layers[0]], minval = 0.0,
        #     maxval = 1.0), name = 'B_1')]


        # if more than 1 hidden layer -> create additional weights and biases
        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(tf.random_normal([h_layers[i-1],
                    h_layers[i]], stddev = tf.sqrt(2.0 / h_layers[i-1])), name =
                    'W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.zeros([h_layers[i]]), name =
                    'B_{}'.format(i+1)))
                # weights.append(tf.Variable(tf.random_uniform([h_layers[i-1],
                #     h_layers[i]], minval = 0.0, maxval = 1.0), name =
                #     'W_{}'.format(i+1)))
                # biases.append(tf.Variable(tf.random_uniform([h_layers[i]],
                #     minval = 0.0, maxval = 1.0), name = 'B_{}'.format(i+1)))

        # connect the last hidden layer to the output layer
        weights.append(tf.Variable(tf.random_normal([h_layers[-1], self.out_size],
            stddev = tf.sqrt(2.0/h_layers[-1])), name = 'W_out'))
        biases.append(tf.Variable(tf.zeros([self.out_size]), name = 'B_out'))
        # weights.append(tf.Variable(tf.random_uniform([h_layers[-1],
        #     self.out_size], minval = 0.0, maxval = 1.0), name = 'W_out'))
        # biases.append(tf.Variable(tf.random_uniform([self.out_size], minval =
        #     0.0, maxval = 1.0), name = 'B_out'))

        
        return weights, biases

    def _model(self, data, W, B, keep_prob=1.0):
        """Model for the multi layer perceptron
        
        Arguments:
        ----------------
        data (tf.placeholder):
            A TensorFlow placeholder.
        W (list):
            A list with the TensorFlow Variables for the weights.
        B (list):
            A list with the TensorFlow Variables for the biases.

        Returns:
        ----------------
        out (tf.tensor)
            Prediction of the model.
        """

        layer = tf.nn.dropout(tf.nn.relu(tf.matmul(data, W[0]) + B[0]),
                keep_prob)
        # if more the 1 hidden layer -> generate output via multiple weight
        # matrices 
        if len(self.h_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(tf.nn.relu(tf.matmul(layer, weight) +
                    bias), keep_prob)

        out = tf.matmul(layer, W[-1]) + B[-1]
        # return tf.nn.softplus(out)
        return tf.nn.sigmoid(out)
        # return tf.nn.relu(out)
        # return out

    def train(self, train_data, val_data, epochs = 10, batch_size = 100,
            learning_rate = 1e-3, keep_prob = 0.9, beta = 0.0, out_size=1):
        """Trains the classifier

        Arguments:
        ----------------
        train_data (custom dataset):
            Contains training data.
        val_data (custom dataset):
            Contains validation data.
        savedir (string):
            Path to the directory to save plots.
        epochs (int): 
            Number of iterations over the whole training set.
        batch_size (int):
            Number of batches fed into one optimization step.
        keep_prob (float):
            Probability of a neuron to 'fire'.
        beta (float):
            L2 regularization coefficient; default 0.0 = regularization off.
        """

        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y = tf.placeholder(tf.float32, [None, out_size])
            w = tf.placeholder(tf.float32, [None, 1])

            weights, biases = self._get_parameters()

            # prediction
            y_ = self._model(x, weights, biases, keep_prob)
            yy_ = self._model(x, weights, biases)
            # loss function
            # xentropy = - (tf.mul(y, tf.log(y_ + 1e-10)) + tf.mul(1-y, tf.log(1-y_ + 1e-10)))
            # xentropy = tf.reduce_sum(tf.mul( - y, tf.log(y_ + 1e-10)))
            xentropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)
            # l2_reg = 0.0
            l2_reg = beta * self._l2_regularization(weights)
            loss = tf.reduce_mean(tf.mul(w, xentropy)) + l2_reg
            # loss = tf.reduce_mean(np.sum(np.square(np.subtract(y,y_))))
            # optimizer
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


            # initialize all variables
            init = tf.initialize_all_variables()
            saver = tf.train.Saver(weights + biases)
        train_start = time.time()
        
        with tf.Session(graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
            sess.run(init)
            train_accuracy = []
            val_accuracy = []
            train_auc = []
            val_auc = []
            train_losses = []

            print(110*'-')
            print('Train model: {}'.format(self.model_loc))
            print(110*'_')
            print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training Loss', 
                'Training Accuracy', 'Validation Accuracy'))
            print(110*'-')

            cross_train_list = []
            cross_val_list = []
            for epoch in range(epochs):
                total_batches = int(train_data.n/batch_size)
                epoch_loss = 0
                for _ in range(total_batches):
                    # train in batches
                    train_x, train_y, train_w=train_data.next_batch(batch_size)
                    _, train_loss = sess.run([train_step, loss], {x:train_x, y:train_y, w:train_w})
                    epoch_loss += train_loss
                train_losses.append(np.mean(epoch_loss))
                train_data.shuffle()

                # monitor training
                train_pre = sess.run(yy_, {x:train_data.x})
                train_corr, train_mistag, train_cross = self._validate_epoch( 
                        train_pre, train_data.y, epoch)
                print('train: {}'.format((train_corr, train_mistag)))
                train_accuracy.append(train_corr / (train_corr + train_mistag))
                
                val_pre = sess.run(yy_, {x:val_data.x})
                #         val_pre, val_data.y, epoch)
                val_corr, val_mistag, val_cross = self._validate_epoch(val_pre,
                        val_data.y, epoch)
                print('validation: {}'.format((val_corr, val_mistag)))
                val_accuracy.append(val_corr / (val_corr + val_mistag))
                
                
                print('{:^25} | {:^25.4f} | {:^25.4f} | {:^25.4f}'.format(epoch + 1, 
                    train_losses[-1], train_accuracy[-1], val_accuracy[-1]))
                saver.save(sess, self.model_loc)
                cross_train_list.append(train_cross)
                cross_val_list.append(val_cross)
                self._plot_cross(train_cross, val_cross, epoch + 1)

                if ((epoch+1) % 10 == 0):
                    self._plot_accuracy(train_accuracy, val_accuracy, epochs)
                    self._plot_loss(train_losses)
                    self._write_list(cross_train_list, 'train_cross')
                    self._write_list(cross_val_list, 'val_cross')
                    self._write_list(train_losses, 'train_losses')
                    self._write_list(train_accuracy, 'train_accuracy')
                    self._write_list(val_accuracy, 'val_accuracy')

            print(110*'-')
            train_end=time.time()

            # self._validation(val_pre, val_data.y)
            # self._plot_auc_dev(train_auc, val_auc, epochs)
            self._plot_accuracy(train_accuracy, val_accuracy, epochs)
            self._plot_loss(train_losses)
            self.trained = True
            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    (train_end - train_start) / 60)
            self._write_list(cross_train_list, 'train_cross')
            self._write_list(cross_val_list, 'val_cross')
            self._write_list(train_losses, 'train_losses')
            self._write_list(train_accuracy, 'train_accuracy')
            self._write_list(val_accuracy, 'val_accuracy')

            print('Model saved in: \n{}'.format(self.savedir))


    def _l2_regularization(self, weights):
        """Calculates and adds the squared values of the weights. This is used
        for L2 regularization.
        """
        # Applies tf.nn.l2_loss to all elements of weights
        weights = map(lambda x: tf.nn.l2_loss(x), weights)
        return sum(weights)


    def _validate_epoch(self, pred, labels, epoch):
        """Evaluates the training process.

        Arguments:
        ----------------
        pred (np.array):
            Predictions made by the model for the data fed into it.
        labels (np.array):
            Labels of the validation dataset.

        Returns:
        ----------------

        """
        pred_onehot = self._onehot(pred, len(pred))
        # print (pred_onehot[0], labels[0])
        correct = 0
        mistag = 0

        arr_cross = np.zeros((self.out_size, self.out_size),dtype=np.int)
        for i in range(pred_onehot.shape[0]):
            # TODO
            equal = True
            # print(pred_onehot.shape[1])
            # if ((epoch + 1) % 10 == 0): 
            #     index_true = np.argmax(labels[i])
            #     index_pred = np.argmax(pred_onehot[i])
            #     arr_cross[index_true][index_pred] += 1
            index_true = np.argmax(labels[i])
            index_pred = np.argmax(pred_onehot[i])
            arr_cross[index_true][index_pred] += 1
            for j in range(pred_onehot.shape[1]):
                if (pred_onehot[i][j] != labels[i][j]):
                    equal = False
            if (equal == True):
                correct += 1
            else:
                mistag += 1
        return correct, mistag, arr_cross

    def _onehot(self, arr, length):
        # TODO
        for i in range(arr.shape[0]):
            arr2 = arr[i]
            ind = np.argmax(arr2)
            for j in range(arr2.shape[0]):
                if (j == ind):
                    arr2[j] = 1.0
                else:
                    arr2[j] = 0.0
            arr[i] = arr2
        return arr



    def _write_parameters(self, epochs, batch_size, keep_prob, beta, time):
        """Writes network parameters in a .txt. file
        """

        with open('{}/info.txt'.format(self.savedir),'w') as f:
            f.write('Training Epochs: {}\n'.format(epochs))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Training Time: {} min\n'.format(time))


    def predict_prob(self, data):
        """Predict probability of a new feature to belong to the signal or
        different background sources. 

        Arguments:
        ----------------
        data (custom data set):
            Data to classify.

        Returns:
        ----------------
        prob (np.array):
            Contains probabilities of a sample to belong to the signal or
            different background sources.
        """

        if not self.trained():
            sys.exit('Model {} has not been trained yet.'.format(self.name))

        predict_graph = tf.Graph()
        with predict_graph.as_default():
            weights, biases = self._get_parameters()
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y_prob = self._model(x, weights, biases)
            saver = tf.train.Saver()
        with tf.Session(graph = predict_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            prob = sess.run(y_prob, {x:data})

        return prob


    def get_weights(self, branches):
        weight_graph = tf.Graph()
        with weight_graph.as_default():
            weights, biases = self._get_parameters()
            saver = tf.train.Saver()
        with tf.Session(graph = weight_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            w = []
            for weight in weights:
                w.append(sess.run(weight))
            # TODO: plot all weights, not just first layer
            self._plot_weight(w[0], branches)


    def _plot_weights(self, weight, branches):
        x = np.sum(weight, axis = 1)
        x_pos = np.arange(len(x))
        plt.bar(x_pos,x,align='center')
        plt.xticks(x_pos, branches, rotation = 30, ha = 'right')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('weight0.pdf')
        plt.show()
        plt.clf()


    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        plt.plot(train_loss, label='Training Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(-2,2))
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.savefig(self.savedir + '/loss.pdf')
        plt.savefig(self.savedir + '/loss.png')
        plt.savefig(self.savedir + '/loss.eps')
        plt.clf()


    def _plot_accuracy(self, train_accuracy, val_accuracy, epochs):
        """Plot the training and validation accuracies.
        """
        plt.plot(train_accuracy, color = 'red', label='Training accuracy')
        plt.plot(val_accuracy, color = 'black', label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_accuracy'
        
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def _plot_auc_dev(self, train_auc, val_auc, nepochs):
        """Plot ROC-AUC-Score development
        """
        plt.plot(train_auc, color='red', label='AUC Training')
        plt.plot(val_auc, color='black', label='AUC Validation')
        plt.xlabel('Epoch')
        plt.ylabel('ROC_AUC')
        plt.title('ROC_AUC Development')
        plt.legend(loc = 'best')
        plt.grid(True)
        plt_name = self.name + '_auc_dev'

        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()


    def _plot_cross(self, arr_train, arr_val, epoch):
        # print("Drawing scatter plot 1.")
        # print(t_true)
        # print(t_pred)
        # plt.scatter(t_true, t_pred, s=6)
        # plt.hexbin(t_true, t_pred)
        # plt.axis(t_true.min(), t_true.max(), t_pred.min(), t_pred.max())
        arr_train_float = np.zeros((arr_train.shape[0], arr_train.shape[1]),
            dtype = np.float32)
        arr_val_float = np.zeros((arr_val.shape[0], arr_val.shape[1]), dtype =
                np.float32)
        for i in range(arr_train.shape[0]):
            row_sum = 0
            for j in range(arr_train.shape[1]):
                row_sum += arr_train[i][j]
            for j in range(arr_train.shape[1]):
                arr_train_float[i][j] = arr_train[i][j] / row_sum
        for i in range(arr_val.shape[0]):
            row_sum = 0
            for j in range(arr_val.shape[1]):
                row_sum += arr_val[i][j]
            for j in range(arr_val.shape[1]):
                arr_val_float[i][j] = arr_val[i][j] / row_sum
        print(arr_train)
        print('-----------------')
        print(arr_val)
        x = np.linspace(0, self.out_size, self.out_size + 1)
        y = np.linspace(0, self.out_size, self.out_size + 1)
        xn, yn = np.meshgrid(x,y)
        plt.pcolormesh(xn, yn, arr_train_float)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        labels = ['ttH', 'tt + bb', 'tt + light']
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.title("Heatmap: Training")
        plt.savefig(self.cross_savedir + '/{}_train.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.png'.format(epoch))
        plt.clf()
        # print("Drawing scatter plot 2.")
        # plt.scatter(v_true, v_pred, s=6)
        # plt.hexbin(v_true, v_pred)
        # plt.axis(v_true.min(), v_true.max(), v_pred.min(), v_pred.max())
        plt.pcolormesh(xn, yn, arr_val_float)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.title("Heatmap: Validation")
        plt.savefig(self.cross_savedir + '/{}_validation.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation.png'.format(epoch))
        plt.clf()
        # print("Done.")


    def _write_list(self, outlist, name):
        """Writes a list of arrays into a name.txt file."""
        path = self.cross_savedir + '/' + name + '.txt'

        with open(path, 'wb') as out:
            pickle.dump(outlist, out)
