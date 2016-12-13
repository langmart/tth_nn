import numpy as np

class DataFrame:
    """TODO
    """

    def __init__(self, array):
        self.x = array[:, 1:-1]
        self.y = array[:, :1]
        self.w = array[:, -1:]
        self.n = self.x.shape[0]
        self.nfeatures = self.x.shape[1]
        self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.w = self.w[perm]

        self.next_id = 0

    def next_batch(self, batch_size):
        if self.next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self.next_id
        self.next_id += batch_size

        return (self.x[cur_id:cur_id+batch_size],
                self.y[cur_id:cur_id+batch_size],
                self.w[cur_id:cur_id+batch_size])
