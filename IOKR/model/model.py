# Implementation

class IOKR:

    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.Ky = None
        self.sy = None
        self.M = None
        self.verbose = 0
        self.linear = False

    def fit(self, X, Y, L, sx, sy):

        t0 = time.time()
        self.X_tr = X
        self.sx = sx
        Kx = rbf_kernel(X, X, gamma=1 / (2 * self.sx))
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * L * np.eye(n))
        self.Y_tr = Y
        if self.linear:
            self.Ky = Y.dot(Y.T)
        else:
            self.Ky = rbf_kernel(Y, Y, gamma=1 / (2 * sy))
        if self.verbose > 0:
            print(f'Fitting time: {time.time() - t0} in s')

    def predict(self, X_te):

        t0 = time.time()
        Kx = rbf_kernel(self.X_tr, X_te, gamma=1 / (2 * self.sx))
        scores = self.Ky.dot(self.M).dot(Kx)
        idx_pred = np.argmax(scores, axis=0)
        if self.verbose > 0:
            print(f'Decoding time: {time.time() - t0} in s')

        return self.Y_tr[idx_pred].copy()