import numpy as np

class RBM(object):
    """Random Boltzman Machine

    Parameters
    -----------
    nepochs
    mini
    l_rate
    M

    Attributes
    ----------
    w, a, b
    """

    def __init__(self, l_rate = 1, nepochs = 50, mini = 500, M = 3, beta = 1, random_state = None):
        self.l_rate = l_rate
        self.nepochs = nepochs
        self.mini = mini
        self.M = M
        self.beta = beta
        if random_state:
            np.random.seed(random_state)
    
    def fit(self, v):
        """Train the Network"""
        self.L = v.shape[1]
        N = v.shape[0]
        self._initialize_weights(self.L, self.M)
        GAP = v[0].max() - v[0].min()
        m = 0
        for epoch in range(1, 1+self.nepoch):
            for n in range(N):
                if m==0:
                    #initialize
                    v_data, v_model = np.zeros(self.L), np.zeros(self.L)
                    h_data, h_model = np.zeros(self.M), np.zeros(self.M)
                    vh_data, vh_model = np.zeros((self.L,self.M)), np.zeros((self.L, self.M))

                #positive CD phase
                h = self.activate(v[n], self.w, self.b, GAP)
                #negative CD phase
                vf = self.activate(h, self.w.T, self.a, GAP)
                # positive CD pahse nr 2
                hf = self.activate(vf, self.w, self.b, GAP)

                v_data += v[n]
                v_model += vf
                h_data += h
                h_model += hf
                vh_data += np.outer(v[n].T, h)
                vh_model += np.outer(vf.T, hf)

                m+=1

                if m == self.mini:
                    C = self.l_rate/self.mini
                    self._update_w(vh_data, vh_model, C)
                    self._update_a(v_data, v_model, C)
                    self._update_b(h_data, h_model, C)
                    m = 0
            #randomize order
            np.random.shuffle(v)
            self.l_rate = self.l_rate/(0.05*self.l_rate + 1)
        return self

    def score(self, v, seq):
        L = v.shape[1]
        N = v.shape[0]
        GAP = v[0].max() - v[0].min()
        v1 = np.full((N, L), v.min())
        for n in range(N):
            h = self.activate(v[n],self.w, self.b, GAP)
            v1[n] = self.activate(h, self.w.T, self.a, GAP)
        count = 0
        for n in range(N):
            for i in range(len(seq)):
                if (v1[n] == seq[i]).all():
                    count += 1
                    break
        return count/N

    def _initialize_weights(self, L, M):
        sigma = np.sqrt(4./float(L+M))
        self.w = sigma * (2*np.random.rand(L,M) - 1)
        self.a = sigma * (2*np.random.rand(L) - 1)
        self.b = np.zeros(M)
    
    def _update_w(self, vh_data, vh_model, C):
        dw = C*(vh_data-vh_model)
        self.w += dw

    def _update_a(self, v_data, v_model, C):
        da = C*(v_data-v_model)
        self.a += da

    def _update_b(self, h_data, h_model, C):
        db = C*(h_data-h_model)
        self.b += db

    def activate(self, v_in, wei, bias, DE):
        act = np.dot(v_in, wei) + bias
        prob = 1./(1.+np.exp(-self.beta*DE*act))
        n = len(act)
        v_out = np.full(n, v_in.min())
        v_out[np.random.random_sample(n) < prob] = 1
        return v_out

