import numpy as np
import libmr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class OpenMax(object):
    def __init__(self, n_clases, alpha= 5, tail_size=50, distance= "Euclidean", threshold=0.95):
        self.n_clases = n_clases

        self.alpha = np.array([(alpha - i)/alpha if alpha >= i else 0 for i in np.arange(1,n_clases+1)], dtype=np.float32)
        self.tail_size = tail_size
        self.distance = distance
        self._mu = []
        self._mr_params = []
        self.plot = None
        self.threshold = threshold

    def _softmax(self, x):
        """
        Calcula Softmax
        inputs:
            x: array de forma (n_samples, n_classes)
        returns:
            array con la probabilidad de cada clase (softmax)
        """
        try:
            assert len(x.shape) == 2
            ex = np.exp(x - x.max(-1, keepdims=True))
            return ex/ex.sum(axis=-1, keepdims=True)
        except AssertionError:
            print("entrada tiene dim {}, debe ser  == 2".format(len(x.shape)))

    def _get_w_score(self, activation, param, mu):
        distancia = np.squeeze(cdist(activation, mu, self.distance))
        return param.w_score_vector(distancia)

    def EVT_params(self, activations, get_dist=False):
        mr = libmr.MR() #Invoca un objeto metarecognition
        mu = activations.mean(0, keepdims= True)
        distancia = np.sort(np.squeeze(cdist(activations, mu, self.distance)))
        if len(distancia) < self.tail_size:
            mr.fit_high(distancia, len(distancia)-1)
        else:
            mr.fit_high(distancia, self.tail_size)
        if get_dist:
            return mu, mr, distancia
        return mu, mr

    def fit(self, logits, y):
        y_ = np.argmax(logits, -1)
        for c in range(self.n_clases):
            idx = np.squeeze(np.argwhere(y == c)) #indices de los
            if idx.shape == () or not isinstance(idx, np.ndarray):
                raise ValueError("Debe ser mayor que 1")
            correct = np.squeeze(idx[np.argwhere(y[idx] == y_[idx])])
            c_activ = logits[correct]
            mu, param = self.EVT_params(c_activ)
            self._mu += [mu]
            self._mr_params += [param]

    def fit_view(self, logits, y):
        y_ = np.argmax(logits, -1)
        fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(14,10))
        xs = np.linspace(0, 60, 500)
        for i in range(2):
            ax[-1,i].set_xlabel("Distance from Mean Activation")
            for j in range(5):
                c = 5*i + j
                idx = np.squeeze(np.argwhere(y == c)) #indices de los
                if idx.shape == () or not isinstance(idx, np.ndarray):
                    raise ValueError("Debe ser mayor que 1")
                correct = np.squeeze(idx[np.argwhere(y[idx] == y_[idx])])
                c_activ = logits[correct]
                mu, param, dists = self.EVT_params(c_activ, True)
                ax[j,i].hist(dists, bins=20, density=True, label="Class {}".format(c))
                #ax[j,i].plot(xs, param.w_score_vector(xs), c="orange", linewidth=3, label="Class {}".format(c))
                ax[j,0].set_ylabel("Density")
    
                self._mu += [mu]
                self._mr_params += [param]
                
        #fig.savefig("openmax.eps", bbox_inches="tight")
        self.plot = [fig, ax]
    
    def evaluate_view(self, logits, y):
        fig, ax = self.plot
        for i in range(2):
            for j in range(5):
                c = 5*i + j
                mu = self._mu[c]
                distancia = np.squeeze(cdist(logits, mu, self.distance))
                ax[j,i].hist(distancia, bins=20, density=True, label="Class A", color="orange", alpha=0.3)
                ax[j,i].legend()
        fig.savefig("openmax.eps", bbox_inches="tight")

    def evaluate(self, logits, y):
        w_score = np.array([self._get_w_score(logits, self._mr_params[c], self._mu[c]) for c in np.arange(self.n_clases)]).T #Obtenemos el ajuste
        sorted_alphas = self.alpha[np.flip(np.argsort(logits, -1), -1)]

        v_hat = logits * (1- sorted_alphas * w_score)
        v_hat0 = (logits - v_hat).sum(-1)
        v_hat = np.concatenate((v_hat, np.expand_dims(v_hat0, 1)), axis= 1)

        softmax = self._softmax(v_hat)
        return softmax
        #softmax = self._softmax(logits)
