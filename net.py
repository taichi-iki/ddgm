# coding: utf-8

from functools import reduce

import chainer
import chainer.functions as F
import chainer.links as L

class DDGMNet(chainer.Chain):
    '''Generative Model with Probabilistic Model with paramters for MNIST'''
    def __init__(self):
        self.n_experts = 128
        self.z_dim = 10
        self.data_dim = 28*28
        super(DDGMNet, self).__init__(
                # Generative Model
                gm_linear1=L.Linear(self.z_dim, 128),
                gm_linear2=L.Linear(128, 128),
                gm_linear3=L.Linear(128, self.data_dim),
                # Energy Model
                # add em_ prefix to components of the energy model for a trick about gradients
                em_linear1=L.Linear(self.data_dim, 128),
                em_linear2=L.Linear(128, 128),
                em_experts=L.Linear(128, self.n_experts),
                em_bias=L.Bias(shape=(self.data_dim,)),
            )
        self.add_param('em_ln_var',  tuple())
        self.em_ln_var.data = self.xp.zeros(self.em_ln_var.data.shape, dtype='float32')
        self.em_params = []
        for k, v in self.namedparams():
            if k.startswith('/em_'): self.em_params.append(v)
    
    def scale_em_grads(self, factor):
        for p in self.em_params:
            p.grad *= factor
    
    def energy(self, x, train):
        y = x
        y = F.relu(self.em_linear1(y))
        y = F.sigmoid(self.em_linear2(y))
        y = self.em_experts(y)
        e_experts = F.sum(F.softplus(y)) #F.sum(F.relu(y) + F.log(1 + F.exp(-abs(y)))))
        e_global = F.sum(x * self.em_bias(x*F.broadcast_to(F.exp(-self.em_ln_var), x.data.shape)))
        E = (e_global - e_experts)/(y.data.shape[0])
        return E
    
    def layer_entropy(self, x):
        mb_size = x.data.shape[0]
        avr_each_act = F.broadcast_to(F.sum(x, axis=0)/mb_size, x.data.shape)
        var_elem = F.sum((x - avr_each_act)**2, axis=0)/mb_size
        var_elem = F.where(var_elem.data != 0, var_elem, self.xp.ones(var_elem.data.shape, dtype='float32'))
        h = 0.5*F.sum(F.log(var_elem))
        return h
    
    def generate(self, z=None, mb_len=1):
        if z is None:
            z = chainer.variable.Variable(self.xp.asarray(self.xp.random.uniform(
                -1.0, 1.0, size=(mb_len, self.z_dim)), dtype='float32'), volatile=False)
        y = z
        H = 0.0
        y = F.tanh(self.gm_linear1(y))
        H = self.layer_entropy(y) + H
        y = F.tanh(self.gm_linear2(y))
        H = self.layer_entropy(y) + H
        y = F.sigmoid(self.gm_linear3(y))
        return y, H
 
