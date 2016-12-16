import numpy as np

class DataFrame:

    def __init__(self, array, out_size):
        """Initializes the DataFrame.

        Arguments: 
        ----------------
        array (numpy ndarray):
            Array containing labels, data, and weights.
        out_size (int):
            Dimension of the output (i.e. label).
        """

        self.y = array[:, :out_size]
        self.x = array[:, out_size:-1]
        self.w = array[:, -1:]

        self.n = self.x.shape[0]
        self.nfeatures = self.x.shape[1]
        
        print('Found {} events.'.format(self.n))
        print('Length of each event: {}'.format(self.nfeatures))

        print('Now shuffling data...')
        self.shuffle()
        print('done.')

    def shuffle(self):
        """Shuffles the data.
        """

        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.w = self.w[perm]

        self.next_id = 0

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

        return (self.x[cur_id:cur_id + batch_size], self.y[cur_id:cur_id +
            batch_size], self.w[cur_id:cur_id + batch_size])

