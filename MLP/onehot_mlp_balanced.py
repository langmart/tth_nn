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
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

# from sklearn.metrics import roc_auc_score, roc_curve

class OneHotMLP:
    """A one-hot output vector classifier using a multi layer perceptron.

    Makes probability predictions on a set of features (a 1-dimensional numpy
    vector belonging either to the 'signal' or the 'background').
    """


    def __init__(self, n_features, h_layers, out_size, savedir, labels_text):
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
        self.labels_text = labels_text

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
        
        self.hists_savedir_train = self.cross_savedir + '/hists_train/'
        if not os.path.isdir(self.hists_savedir_train):
            os.makedirs(self.hists_savedir_train)
        self.hists_savedir_val = self.cross_savedir + '/hists_val/'
        if not os.path.isdir(self.hists_savedir_val):
            os.makedirs(self.hists_savedir_val)
        self.weights_savedir = self.cross_savedir + '/weights/'
        if not os.path.isdir(self.weights_savedir):
            os.makedirs(self.weights_savedir)
        self.mistag_savedir = self.cross_savedir + '/mistag/'
        if not os.path.isdir(self.mistag_savedir):
            os.makedirs(self.mistag_savedir)


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

        weights = [tf.Variable(tf.random_normal([n_features, h_layers[0]], 
            stddev=tf.sqrt(2.0/n_features)), name = 'W_1')]
        biases = [tf.Variable(tf.zeros([h_layers[0]]), name = 'B_1')]
        # biases = [tf.Variable(tf.random_normal([h_layers[0]], stddev =
        #     tf.sqrt(2.0 / (h_layers[0]))), name = 'B_1')]


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
                # biases.append(tf.Variable(tf.random_normal([h_layers[i]], stddev
                #     = tf.sqrt(2.0 / h_layers[i])), name = 'B_{}'.format(i+1)))
                # weights.append(tf.Variable(tf.random_uniform([h_layers[i-1],
                #     h_layers[i]], minval = 0.0, maxval = 1.0), name =
                #     'W_{}'.format(i+1)))
                # biases.append(tf.Variable(tf.random_uniform([h_layers[i]],
                #     minval = 0.0, maxval = 1.0), name = 'B_{}'.format(i+1)))

        # connect the last hidden layer to the output layer
        weights.append(tf.Variable(tf.random_normal([h_layers[-1], self.out_size],
            stddev = tf.sqrt(2.0/h_layers[-1])), name = 'W_out'))
        biases.append(tf.Variable(tf.zeros([self.out_size]), name = 'B_out'))
        # biases.append(tf.Variable(tf.random_normal([self.out_size], stddev
        #     = tf.sqrt(2.0 / self.out_size)), name = 'B_out'))
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
        # return tf.nn.sigmoid(out)
        # return tf.nn.relu(out)
        return out

    def train(self, train_data, val_data, optimizer='Adam', epochs = 10, batch_size = 100,
            learning_rate = 1e-3, keep_prob = 0.9, beta = 0.0, out_size=6,
            optimizer_options=[]):
        """Trains the classifier

        Arguments:
        ----------------
        train_data (custom dataset):
            Contains training data.
        val_data (custom dataset):
            Contains validation data.
        optimizer (string):
            Name of the optimizer to be built.
        epochs (int): 
            Number of iterations over the whole training set.
        batch_size (int):
            Number of batches fed into one optimization step.
        learning_rate (float):
            Optimizer learning rate.
        keep_prob (float):
            Probability of a neuron to 'fire'.
        beta (float):
            L2 regularization coefficient; default 0.0 = regularization off.
        out_size (int):
            Dimension of output vector, i.e. number of classes.
        optimizer_options (list):
            List of additional options for the optimizer; can have different
            data types for different optimizers.
        """

        self.optname = optimizer
        self.learning_rate = learning_rate
        self.optimizer_options = optimizer_options

        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y = tf.placeholder(tf.float32, [None, out_size])
            w = tf.placeholder(tf.float32, [None, 1])

            weights, biases = self._get_parameters()

            # prediction
            y_ = self._model(x, weights, biases, keep_prob)
            yy_ = tf.nn.softmax(self._model(x, weights, biases))
            # loss function
            # xentropy = - (tf.mul(y, tf.log(y_ + 1e-10)) + tf.mul(1-y, tf.log(1-y_ + 1e-10)))
            # xentropy = tf.reduce_sum(tf.mul( - y, tf.log(y_ + 1e-10)))
            xentropy = tf.nn.softmax_cross_entropy_with_logits(y_,y)
            # l2_reg = 0.0
            l2_reg = beta * self._l2_regularization(weights)
            # loss = tf.reduce_mean(tf.mul(w, xentropy)) + l2_reg
            loss = tf.reduce_mean(tf.reduce_sum(tf.mul(w, xentropy))) + l2_reg
            # loss = tf.reduce_mean(np.sum(np.square(np.subtract(y,y_))))
            # optimizer
            optimizer = self._build_optimizer()
            train_step = optimizer.minimize(loss)

            # initialize all variables
            init = tf.initialize_all_variables()
            saver = tf.train.Saver(weights + biases)
        train_start = time.time()
        
        # Non-static memory management; memory can be allocated on the fly.
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        
        with tf.Session(config=sess_config, graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
            sess.run(init)
            train_accuracy = []
            val_accuracy = []
            train_auc = []
            val_auc = []
            train_losses = []
            train_cats = []
            val_cats = []
            print('Now normalizing training data.')
            train_data.normalize()
            print('Done.')
            print('Now normalizing validation data.')
            val_data.normalize()
            print('Done.')


            print(110*'-')
            print('Train model: {}'.format(self.model_loc))
            print(110*'_')
            print('{:^25} | {:^25} | {:^25} | {:^25}'.format('Epoch', 'Training Loss', 
                'Training Accuracy', 'Validation Accuracy'))
            print(110*'-')

            cross_train_list = []
            cross_val_list = []
            for epoch in range(epochs):
                total_batches = int(train_data.n_events/batch_size)
                epoch_loss = 0
                for _ in range(total_batches):
                    # train in batches
                    train_x, train_y, train_w=train_data.next_batch(batch_size)
                    _, train_loss, weights_for_plot = sess.run([train_step,
                        loss, weights], {x:train_x, y:train_y, w:train_w})
                    epoch_loss += train_loss
                train_losses.append(np.mean(epoch_loss))
                train_data.shuffle()

                # monitor training
                train_pre = sess.run(yy_, {x:train_data.created_x})
                train_corr, train_mistag, train_cross, train_cat = self._validate_epoch( 
                        train_pre, train_data.created_y, epoch)
                print('train: {}'.format((train_corr, train_mistag)))
                train_accuracy.append(train_corr / (train_corr + train_mistag))
                
                val_pre = sess.run(yy_, {x:val_data.created_x})
                val_corr, val_mistag, val_cross, val_cat = self._validate_epoch(val_pre,
                        val_data.created_y, epoch)
                print('validation: {}'.format((val_corr, val_mistag)))
                val_accuracy.append(val_corr / (val_corr + val_mistag))
                
                
                print('{:^25} | {:^25.4f} | {:^25.4f} | {:^25.4f}'.format(epoch + 1, 
                    train_losses[-1], train_accuracy[-1], val_accuracy[-1]))
                saver.save(sess, self.model_loc)
                cross_train_list.append(train_cross)
                cross_val_list.append(val_cross)
                train_cats.append(train_cat)
                val_cats.append(val_cat)

                if (epoch % 20 == 0):
                    self._plot_loss(train_losses)
                    self._write_list(cross_train_list, 'train_cross')
                    self._write_list(cross_val_list, 'val_cross')
                    self._write_list(train_losses, 'train_losses')
                    self._write_list(train_accuracy, 'train_accuracy')
                    self._write_list(val_accuracy, 'val_accuracy')
                    self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                            val_cats, epochs)
                    self._plot_weight_matrices(weights, epoch)
                    self._plot_cross(train_cross, val_cross, epoch + 1)
                    self._plot_hists(train_pre, val_pre, epoch)
                    self._plot_cross_dev(cross_train_list, cross_val_list,
                            epoch+1)

            print(110*'-')
            train_end=time.time()

            self._plot_accuracy(train_accuracy, val_accuracy, train_cats,
                    val_cats, epochs)
            self._plot_loss(train_losses)
            self._write_parameters(epochs, batch_size, keep_prob, beta,
                    (train_end - train_start) / 60)
            self._plot_weight_matrices(weights, epoch)
            self._plot_cross(train_cross, val_cross, epoch + 1)
            self._plot_hists(train_pre, val_pre, epoch)
            self._plot_cross_dev(cross_train_list, cross_val_list, epoch+1)
            self._write_list(cross_train_list, 'train_cross')
            self._write_list(cross_val_list, 'val_cross')
            self._write_list(train_losses, 'train_losses')
            self._write_list(train_accuracy, 'train_accuracy')
            self._write_list(val_accuracy, 'val_accuracy')
            self.trained = True

            print('Model saved in: \n{}'.format(self.savedir))


    def _l2_regularization(self, weights):
        """Calculates and adds the squared values of the weights. This is used
        for L2 regularization.
        """
        # Applies tf.nn.l2_loss to all elements of weights
        # weights = map(lambda x: tf.nn.l2_loss(x), weights)
        # return sum(weights)
        losses = [tf.nn.l2_loss(w) for w in weights]
        return tf.add_n(losses)


    def _validate_epoch(self, pred, labels, epoch):
        """Evaluates the training process.

        Arguments:
        ----------------
        pred (np.array):
            Predictions made by the model for the data fed into it.
        labels (np.array):
            Labels of the validation dataset.
        epoch (int):
            Training epoch.
        Returns:
        ----------------

        """
        pred_onehot = self._onehot(pred, len(pred))
        # print (pred_onehot[0], labels[0])
        correct = 0
        mistag = 0

        arr_cross = np.zeros((self.out_size, self.out_size),dtype=np.int)
        for i in range(pred_onehot.shape[0]):
            equal = True
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
        cat_acc = np.zeros((self.out_size), dtype=np.float32)
        for i in range(self.out_size): 
            cat_acc[i] = arr_cross[i][i] / (np.sum(arr_cross, axis=1)[i])

        
        return correct, mistag, arr_cross, cat_acc


    def _build_optimizer(self):
        """Returns a TensorFlow Optimizer.
        """
        if (self.optname == 'Adam'):
            try:
                beta1 = self.optimizer_options[0]
            except IndexError:
                beta1 = 0.9
            try:
                beta2 = self.optimizer_options[1]
            except IndexError:
                beta2 = 0.999
            try:
                epsilon = self.optimizer_options[2]
            except IndexError:
                epsilon = 1e-8
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1,
                    beta2=beta2, epsilon=epsilon)
            print('Building Adam Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     beta1: {}'.format(beta1))
            print('     beta2: {}'.format(beta2))
            print('     epsilon: {}'.format(epsilon))
        elif (self.optname == 'GradDescent'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            print('Building Gradient Descent Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
        elif (self.optname == 'Adagrad'):
            try:
                initial_accumulator_value = self.optimizer_options[0]
            except IndexError:
                initial_accumulator_value = 0.1
            optimizer = tf.train.AdagradOptimizer(self.learning_rate,
                    initial_accumulator_value=initial_accumulator_value)
            print('Building Adagrad Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     initial_accumulator_value: {}'
                    .format(initial_accumulator_value))
        elif (self.optname == 'Adadelta'):
            try:
                rho = self.optimizer_options[0]
            except IndexError:
                rho = 0.95
            try:
                epsilon = self.optimizer_options[1]
            except IndexError:
                epsilon = 1e-8
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, rho=rho,
                epsilon=epsilon)
            print('Building Adadelta Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     rho: {}'.format(rho))
            print('     epsilon: {}'.format(epsilon))
        elif (self.optname == 'Momentum'):
            try:
                momentum = self.optimizer_options[0]
            except IndexError:
                momentum = 0.9
            try:
                use_nesterov = self.optimizer_options[1]
            except IndexError:
                use_nesterov = False
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                    momentum=momentum, use_nesterov=use_nesterov)
            print('Building Momentum Optimizer.')
            print('     learning_rate: {}'.format(self.learning_rate))
            print('     momentum: {}'.format(momentum))
            print('     use_nesterov: {}'.format(use_nesterov))
        else:
            print('No Optimizer with name {} has been implemented.'
                    .format(self.optname))
            sys.exit('Aborting.')
        return optimizer

    def _onehot(self, arr, length):
        # TODO
        dummy_array = arr
        for i in range(arr.shape[0]):
            arr2 = dummy_array[i]
            ind = np.argmax(arr2)
            for j in range(arr2.shape[0]):
                if (j == ind):
                    arr2[j] = 1.0
                else:
                    arr2[j] = 0.0
            dummy_array[i] = arr2
        return dummy_array


    def _write_parameters(self, epochs, batch_size, keep_prob, beta, time):
        """Writes network parameters in a .txt. file
        """

        with open('{}/info.txt'.format(self.savedir),'w') as f:
            f.write('Training Epochs: {}\n'.format(epochs))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Training Time: {} min\n'.format(time))
            f.write('Optimizer: {}\n'.format(self.optname))
            f.write('Learning rate {}\n'.format(self.learning_rate))
            if (self.optimizer_options):
                f.write('Optimizer options: {}'.format(self.optimizer_options))


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


    def _plot_accuracy(self, train_accuracy, val_accuracy, train_cats, val_cats, epochs):
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
        
        arr = np.zeros((self.out_size, len(train_cats)))
        for i in range(len(train_cats)):
            for j in range(self.out_size):
                arr[j][i] = train_cats[i][j]
        for j in range(self.out_size):
            plt.plot(arr[j], label = self.labels_text[j])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Categories: Training Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_categories_train'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()
        arr = np.zeros((self.out_size, len(val_cats)))
        for i in range(len(val_cats)):
            for j in range(self.out_size):
                arr[j][i] = val_cats[i][j]
        for j in range(self.out_size):
            plt.plot(arr[j], label = self.labels_text[j])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Categories: Validation Accuracy development')
        plt.legend(loc='best')
        plt.grid(True)
        plt_name = self.name + '_categories_val'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()


    def _plot_cross(self, arr_train, arr_val, epoch):
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
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        plt.title("Heatmap: Training")
        plt.savefig(self.cross_savedir + '/{}_train.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.png'.format(epoch))
        plt.clf()
        plt.pcolormesh(xn, yn, arr_train_float)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        plt.title("Heatmap: Training")
        plt.savefig(self.cross_savedir + '/{}_train.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train.png'.format(epoch))
        plt.clf()
        plt.pcolormesh(xn, yn, arr_val_float)
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        plt.title("Heatmap: Validation")
        plt.savefig(self.cross_savedir + '/{}_validation.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation.png'.format(epoch))
        plt.clf()
        
        # Draw again with LogNorm colors
        plt.pcolormesh(xn, yn, arr_train_float, norm=colors.LogNorm(vmin=1e-6,
            vmax=1.0))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        plt.title("Heatmap: Training")
        plt.savefig(self.cross_savedir + '/{}_train_colorlog.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train_colorlog.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_train_colorlog.png'.format(epoch))
        plt.clf()
        plt.pcolormesh(xn, yn, arr_val_float, norm=colors.LogNorm(vmin=1e-6,
            vmax=1.0))
        plt.colorbar()
        plt.xlim(0, self.out_size)
        plt.ylim(0, self.out_size)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self.labels_text)
        ax.set_yticklabels(self.labels_text)
        plt.title("Heatmap: Validation")
        plt.savefig(self.cross_savedir + '/{}_validation_colorlog.pdf'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation_colorlog.eps'.format(epoch))
        plt.savefig(self.cross_savedir + '/{}_validation_colorlog.png'.format(epoch))
        plt.clf()


    def _plot_weight_matrices(self, w, epoch):
        for i in range(len(w)):
            weight = w[i]
            np_weight = weight.eval()
            xr, yr = np_weight.shape
            x = range(xr + 1)
            y = range(yr + 1)
            xn, yn = np.meshgrid(y,x)
            plt.pcolormesh(xn, yn, np_weight)
            plt.colorbar()
            plt.xlim(0, np_weight.shape[1])
            plt.ylim(0, np_weight.shape[0])
            plt.xlabel('out')
            plt.ylabel('in')
            title = "Heatmap: weight[{}], epoch: {}".format(i+1, epoch+1)
            plt.title(title)
            plt.savefig(self.weights_savedir +
                    '/epoch{}_weight{}.pdf'.format(epoch+1, i+1))
            plt.clf()



    def _write_list(self, outlist, name):
        """Writes a list of arrays into a name.txt file."""
        path = self.cross_savedir + '/' + name + '.txt'

        with open(path, 'wb') as out:
            pickle.dump(outlist, out)


    def _plot_hists(self, arr_train, arr_val, epoch):
        """Plot histograms of probability distributions.

        Arguments:
        ----------------
        arr_train (array):
            Array of shape (n_events_train, out_size) containing probabilities
            for each event to belong to each category.
        arr_val (array):
            Array of shape (n_events_val, out_size) containing probabilities for
            each event to belong to each category.
        """

         
        for i in range(arr_train.shape[1]):
            n, bins, patches = plt.hist(arr_train[:,i], bins=100, normed=False)
            plt.savefig(self.hists_savedir_train + str(epoch+1) + '_' + str(i+1) + '.pdf')
            plt.savefig(self.hists_savedir_train + str(epoch+1) + '_' + str(i+1) + '.eps')
            plt.savefig(self.hists_savedir_train + str(epoch+1) + '_' + str(i+1) + '.png')
            plt.clf()
        for i in range(arr_val.shape[1]):
            n, bins, patches = plt.hist(arr_val[:,i], bins=100, normed=False)
            plt.savefig(self.hists_savedir_val + str(epoch+1) + '_' + str(i+1) + '.pdf')
            plt.savefig(self.hists_savedir_val + str(epoch+1) + '_' + str(i+1) + '.eps')
            plt.savefig(self.hists_savedir_val + str(epoch+1) + '_' + str(i+1) + '.png')
            plt.clf()


    def _plot_cross_dev(self, train_list, val_list, epoch):
        """Plot the development of misclassifications.

        Arguments:
        ----------------
        train_list (list):
            List containing a cross check array for each epoch.
        val_list (list):
            List containing a cross check array for each epoch.
        epoch (int):
            Epoch.
        """
        train_x_classified_as_y = np.zeros((epoch, self.out_size, self.out_size))
        val_x_classified_as_y = np.zeros((epoch, self.out_size, self.out_size))
        train_y_classified_as_x = np.zeros((epoch, self.out_size,
            self.out_size))
        val_y_classified_as_x = np.zeros((epoch, self.out_size, self.out_size))


        for list_index in range(len(train_list)):
            arr_train = train_list[list_index]
            arr_val = val_list[list_index]
            row_sum_train = np.sum(arr_train, axis=1)
            column_sum_train = np.sum(arr_train, axis=0)
            row_sum_val = np.sum(arr_val, axis=1)
            column_sum_val = np.sum(arr_val, axis=0)
            
            for i in range(arr_train.shape[0]):
                for j in range(arr_train.shape[1]):
                    if (row_sum_train[i] != 0):
                        train_x_classified_as_y[list_index,i,j] = arr_train[i][j] / row_sum_train[i]
                    else:
                        train_x_classified_as_y[list_index,i,j] = arr_train[i][j]
                    if (row_sum_val[i] != 0):
                        val_x_classified_as_y[list_index,i,j] = arr_val[i][j] / row_sum_val[i]
                    else:
                        val_x_classified_as_y[list_index,i,j] = arr_val[i][j]
                    if (column_sum_train[i] != 0):
                        train_y_classified_as_x[list_index,i,j] = arr_train[i][j] / column_sum_train[i]
                    else:
                        train_y_classified_as_x[list_index,i,j] = arr_train[i][j]
                    if (column_sum_val[i] != 0):
                        val_y_classified_as_x[list_index,i,j] = arr_val[i][j] / column_sum_val[i]
                    else:
                        val_y_classified_as_x[list_index,i,j] = arr_val[i][j]

        for i in range(train_x_classified_as_y.shape[1]):
            for j in range(train_x_classified_as_y.shape[2]):
                plt.plot(train_x_classified_as_y[:,i,j],
                        label=self.labels_text[j])
            plt.title(self.labels_text[i] + ' classified as')
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'train_x_{}_as.pdf'.format(i))
            plt.savefig(self.mistag_savedir + 'train_x_{}_as.png'.format(i))
            plt.savefig(self.mistag_savedir + 'train_x_{}_as.eps'.format(i))
            plt.clf()
        for i in range(val_x_classified_as_y.shape[1]):
            for j in range(val_x_classified_as_y.shape[2]):
                plt.plot(val_x_classified_as_y[:,i,j],
                        label=self.labels_text[j])
            plt.title(self.labels_text[i] + ' classified as')
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'val_x_{}_as.pdf'.format(i))
            plt.savefig(self.mistag_savedir + 'val_x_{}_as.png'.format(i))
            plt.savefig(self.mistag_savedir + 'val_x_{}_as.eps'.format(i))
            plt.clf()
        for i in range(train_y_classified_as_x.shape[2]):
            for j in range(train_y_classified_as_x.shape[1]):
                plt.plot(train_y_classified_as_x[:,i,j],
                        label=self.labels_text[j])
            plt.title('classified as ' + self.labels_text[i])
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'train_as_x_{}.pdf'.format(i))
            plt.savefig(self.mistag_savedir + 'train_as_x_{}.png'.format(i))
            plt.savefig(self.mistag_savedir + 'train_as_x_{}.eps'.format(i))
            plt.clf()
        for i in range(val_y_classified_as_x.shape[2]):
            for j in range(val_y_classified_as_x.shape[1]):
                plt.plot(train_y_classified_as_x[:,i,j],
                        label=self.labels_text[j])
            plt.title('classified as ' + self.labels_text[i])
            plt.xlabel('Epoch')
            plt.ylabel('Tagging rate')
            plt.xlim(0, epoch)
            plt.ylim(0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')
            plt.savefig(self.mistag_savedir + 'val_as_x_{}.pdf'.format(i))
            plt.savefig(self.mistag_savedir + 'val_as_x_{}.png'.format(i))
            plt.savefig(self.mistag_savedir + 'val_as_x_{}.eps'.format(i))
            plt.clf()

