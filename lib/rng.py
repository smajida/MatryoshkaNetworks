from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream
from random import Random

seed = 42

py_rng = Random(seed)
np_rng = RandomState(seed)
t_rng = RandomStreams(seed)
cu_rng = RandStream(seed)

def set_seed(n):
    global seed, py_rng, np_rng, t_rng
    
    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)
    t_rng = RandomStreams(seed)
