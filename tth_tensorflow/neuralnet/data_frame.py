# Written by Max Welsch.
import numpy as np

class DataFrame:
    """TODO
    """

    def __init__(self, x, y, out_size, ratio_weight=False):
        # self.x = x
        # self.y = y
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.out_size = out_size

        bins, _ = np.histogram(y)
        self.nsig, self.nbg = bins[-1], bins[0]
        if ratio_weight:
            # w_sig = np.full((self.nsig, 1), 1.0)
            # w_bg = np.full((self.nbg, 1), float(self.nsig)/float(self.nbg))
            w_sig = np.full((self.nsig, out_size), 1.0)
            w_bg = np.full((self.nbg, out_size), float(self.nsig)/float(self.nbg))
            self.w = np.vstack((w_sig, w_bg))
        else:
            # self.w = np.full((self.x.shape[0], 1), 1.0)
            self.w = np.full((self.x.shape[0], out_size), 1.0)

        self.n = x.shape[0]
        self.nfeatures = x.shape[1]
        self.shuffle()

    def shuffle(self):
        # might mess up with the data???
        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]

        if self.w is not None:
            self.w = self.w[perm]

        self.next_id = 0

    def shuffle2(self):
        # is self.x a numpy array?
        perm = np.random.permutation(self.n)
        # self.x = [self.x[perm[i]] for i in range(len(self.x))]
        self.x = [self.x[perm[i]] for i in range(self.n)]
        self.y = [self.y[perm[i]] for i in range(self.n)]

        if self.w is not None:
            self.w = [self.w[perm[i]] for i in range(self.n)]

        self.next_id = 0

    def next_batch(self, batch_size):
        if self.next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self.next_id
        self.next_id += batch_size

        # if self.w is not None:
        return (self.x[cur_id:cur_id+batch_size],
                    self.y[cur_id:cur_id+batch_size],
                    self.w[cur_id:cur_id+batch_size])
        # else:
        #     return (self.x[cur_id:cur_id+batch_size],
        #             self.y[cur_id:cur_id+batch_size])
