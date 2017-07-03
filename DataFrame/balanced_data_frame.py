import numpy as np
import sys

class DataFrame:

    def __init__(self, array, out_size, sizes, normalization):
        """Initializes the DataFrame.

        Arguments: 
        ----------------
        array (numpy ndarray):
            Array containing labels, data, and weights.
        out_size (int):
            Dimension of the output (i.e. label).
        sizes (numpy array):
            Array containing the number of events for each category.
        normalization (string):
            String indicating whether to use Gaussian or min-max norm.
        """

        self.out_size = out_size
        self.normalization = normalization
        self.y = array[:, :self.out_size]
        self.x = array[:, self.out_size:-1]
        self.w = array[:, -1]
        self.sizes = sizes
        self.n = self.x.shape[0]
        self.nfeatures = self.x.shape[1]

        print('Found {} events.'.format(self.n))
        print('Length of each event: {}'.format(self.nfeatures))

        print('Now shuffling data...')
        self.shuffle()
        print('done.')
        # print(self.w[0:100])

    def normalize(self):
        """Normalizes the training data.
        """
        print('Now normalizing; this may take some minutes.')
        if (self.normalization == 'minmax'):
            print('Normalizing using Min-Max normalization.')
            x_max = np.amax(self.x, axis=0).astype(np.float32)
            x_min = np.amin(self.x, axis=0).astype(np.float32)
            self.x = 2.0*np.nan_to_num((self.x - x_min) / (x_max - x_min))-1.0


        elif (self.normalization == 'gaussian'):
            print('Normalizing using Gaussian normalization.')
            x_mean = np.mean(self.x, axis=0).astype(np.float32)
            x_std = np.std(self.x, axis=0).astype(np.float32)
            self.x = np.nan_to_num((self.x - x_mean) / x_std)

        else:
            sys.exit('Only minmax and gaussian normalization are available.')

    
    
    def _create_array(self):
        self.n_events = np.amin(self.sizes)
        # print(self.n_events)
        ttH_y = np.zeros((self.n_events, self.out_size))
        ttH_x = np.zeros((self.n_events, self.nfeatures))
        ttH_w = np.zeros((self.n_events, 1))
        ttH_count = 0
        ttbb_y = np.zeros((self.n_events, self.out_size))
        ttbb_x = np.zeros((self.n_events, self.nfeatures))
        ttbb_w = np.zeros((self.n_events, 1))
        ttbb_count = 0
        tt2b_y = np.zeros((self.n_events, self.out_size))
        tt2b_x = np.zeros((self.n_events, self.nfeatures))
        tt2b_w = np.zeros((self.n_events, 1))
        tt2b_count = 0
        ttb_y = np.zeros((self.n_events, self.out_size))
        ttb_x = np.zeros((self.n_events, self.nfeatures))
        ttb_w = np.zeros((self.n_events, 1))
        ttb_count = 0
        ttcc_y = np.zeros((self.n_events, self.out_size))
        ttcc_x = np.zeros((self.n_events, self.nfeatures))
        ttcc_w = np.zeros((self.n_events, 1))
        ttcc_count = 0
        ttlight_y = np.zeros((self.n_events, self.out_size))
        ttlight_x = np.zeros((self.n_events, self.nfeatures))
        ttlight_w = np.zeros((self.n_events, 1))
        ttlight_count = 0
        
        for i in range(self.n):
            if (self.y[i][0] == 1.0):
                if (ttH_count < self.n_events):
                    ttH_y[ttH_count] = self.y[i]
                    ttH_x[ttH_count] = self.x[i]
                    ttH_w[ttH_count] = self.w[i]
                    ttH_count += 1
            if (self.y[i][1] == 1.0):
                if (ttbb_count < self.n_events):
                    ttbb_y[ttbb_count] = self.y[i]
                    ttbb_x[ttbb_count] = self.x[i]
                    ttbb_w[ttbb_count] = self.w[i]
                    ttbb_count += 1
            if (self.y[i][2] == 1.0):
                if (tt2b_count < self.n_events):
                    tt2b_y[tt2b_count] = self.y[i]
                    tt2b_x[tt2b_count] = self.x[i]
                    tt2b_w[tt2b_count] = self.w[i]
                    tt2b_count += 1
            if (self.y[i][3] == 1.0):
                if (ttb_count < self.n_events):
                    ttb_y[ttb_count] = self.y[i]
                    ttb_x[ttb_count] = self.x[i]
                    ttb_w[ttb_count] = self.w[i]
                    ttb_count += 1
            if (self.y[i][4] == 1.0):
                if (ttcc_count < self.n_events):
                    ttcc_y[ttcc_count] = self.y[i]
                    ttcc_x[ttcc_count] = self.x[i]
                    ttcc_w[ttcc_count] = self.w[i]
                    ttcc_count += 1
            if (self.y[i][5] == 1.0):
                if (ttlight_count < self.n_events):
                    ttlight_y[ttlight_count] = self.y[i]
                    ttlight_x[ttlight_count] = self.x[i]
                    ttlight_w[ttlight_count] = self.w[i]
                    ttlight_count += 1

        ys = np.vstack((ttH_y, ttbb_y, tt2b_y, ttb_y, ttcc_y, ttlight_y))
        xs = np.vstack((ttH_x, ttbb_x, tt2b_x, ttb_x, ttcc_x, ttlight_x))
        ws = np.vstack((ttH_w, ttbb_w, tt2b_w, ttb_w, ttcc_w, ttlight_w))

        perm = np.random.permutation(xs.shape[0])
        xs = xs[perm]
        ys = ys[perm]
        ws = ws[perm]

        return xs, ys, ws
            

     
    def shuffle(self):
        """Shuffles the data.
        """

        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.w = self.w[perm]

        for i in range(self.n):
            if (self.x[i].shape[0] != self.nfeatures):
                print('Some data does not have the right shape.')
            if (self.y[i].shape[0] != self.out_size):
                print('Some labels do not have the right shape.')
            # if (self.w[i].shape[0] != 1):
            #     print('Some weights do not have the right shape.')
        
        self.next_id = 0
        self.created_x, self.created_y, self.created_w = self._create_array()


    def next_batch(self, batch_size):
        """Returns the next batch of events.

        Arguments:
        ----------------
        batch_size (int):
            Size of each batch.
        """

        if (self.next_id + batch_size >= self.n):
            self.shuffle()

        cur_id = self.next_id
        self.next_id += batch_size

        return (np.array(self.created_x[cur_id:cur_id + batch_size]),
                np.array(self.created_y[cur_id:cur_id + batch_size]),
                np.array(self.created_w[cur_id:cur_id + batch_size]))

