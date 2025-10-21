import numpy as np

class Ising(object):
    """
    A class for 2D grid-like Ising model. Notice that the class only stores
    model state (the current assignment of random variables); you have to 
    compute the needed potential / conditional probability on your own.
    """
    def __init__(self, dim, Js, Jst):
        """
        Initialize a random instance of 2D grid-like Ising model.
        
        Parameters:
            dim (int): the dimension of the grid 
                       (i.e. the model has dim*dim RVs)
            Js (float): the unary parameter $J_s$ 
            Jst (float): the binary parameter $J_{st}$
        """
        self._dim = dim
        self._Js = Js
        self._Jst = Jst
        self.init_state()

    def init_state(self):
        """
        Initialize the state of the model randomly.
        """ 
        self._state = np.random.randint(0, 2, (self._dim, self._dim))
        self._state = 2 * self._state - 1
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def Js(self):
        return self._Js
    
    @property
    def Jst(self):
        return self._Jst
    
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if isinstance(state, list):
            assert (len(state) == self._dim)
            assert (len(state[0]) == self._dim)
        elif isinstance(state, np.ndarray):
            assert (state.shape == (self._dim, self._dim))
        else:
            raise TypeError("only support list and np.ndarray")
            
        for i in range(self._dim):
            for j in range(self._dim):
                assert (state[i][j] == 1 or state[i][j] == -1)
        self._state = np.array(state)

def conditional(state, i, j, Js, Jst):
    '''
    Calculate the conditional probability in part (a). 
    Inputs: 
    (1) state:a numpy array representing the states of the Ising model 
    (2) i, j: the indices of the target state
    (3) Js: a float number denoting weights of Ising models
    (4) Jst: a float number denoting weights of Ising models
    Outputs: a float number between 0 and 1 representing the conditional probability in part (a)
    TODO: Implement the function
    '''
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    m, n = state.shape
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    sum_neighbors = 0
    for di, dj in dirs:
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            sum_neighbors += state[ni][nj]
    return sigmoid(2 * Js + 2 * Jst * sum_neighbors)
    

def log_unnormalized_p(state, Js, Jst):
    """
    Calculate the un-normalized probalitity of Ising models with parameters Js and Jst, i.e. log(\hat{p}) in Eqn B.1.
    Input:
    (1) state:a numpy array representing the states of the Ising model 
    (2) Js: a float number denoting weights of Ising models
    (3) Jst: a float number denoting weights of Ising models
    Output: a float number representing the un-normalized log potential of the state 

    TODO: Implement the function 
    """
    
    m, n = state.shape
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    sum_singleton = 0
    sum_pairwise = 0
    for i in range(m):
        for j in range(n):
            sum_singleton += Js * state[i][j]
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    sum_pairwise += Jst * state[i][j] * state[ni][nj]
    return sum_singleton + 1/2 * sum_pairwise
