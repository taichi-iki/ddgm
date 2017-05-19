# coding: utf-8

import numpy as np
import pickle
from PIL import Image

import chainer
from chainer import cuda
from chainer import optimizers
from chainer.optimizer import WeightDecay
from chainer import serializers

import net

def main():
    # Parameters
    epoch_max = 200
    minibatch_max = 64
    lambda_em = 1.0
    gpu_id = 0
    save_interval = 50
    model_name = 'model'
    dataset_path = 'mnist.pklb'
    
    # Preparation of dataset
    # Change this logic for your dataset
    with open(dataset_path, 'rb') as f:
        mnist = pickle.Unpickler(f).load()
    ds_train = mnist['image']['train']
    ds_train = [np.asarray(x.reshape(-1), dtype='float32')/255.0 for x in ds_train]
    print('the number of dataset', len(ds_train))
    
    # Setup of our model
    model = net.DDGMNet()
    if gpu_id >=0:
        model.to_gpu(gpu_id)
    opt = optimizers.Adam(alpha=1e-3)
    opt.setup(model)
    opt.add_hook(WeightDecay(rate=1e-6))
    
    # Learning loop
    for epoch_id in range(0, epoch_max):
        print('start epoch %d'%(epoch_id))
        
        # Evaluation
        fake, _ = model.generate(mb_len=10)
        fake = cuda.to_cpu(fake.data)
        img_array = (np.hstack([x.reshape((28,28)) for x in fake])*255).astype('uint8')
        Image.fromarray(img_array).save('%s_test_%d.png'%(model_name, epoch_id))
        print('generated gen_test_%d.png'%(epoch_id))
        
        sum_e_real = 0.0
        sum_e_fake = 0.0
        sum_entropy = 0.0
        sum_trial = 0
        order_dataset = np.random.permutation(len(ds_train))
        for i in range(0, len(order_dataset), minibatch_max):
            # Energy calculation
            mb = [model.xp.asarray(ds_train[j]) for j in order_dataset[i:i+minibatch_max]]
            mb_len = len(mb)
            mb = chainer.variable.Variable(model.xp.vstack(mb), volatile=False)
            e_real = model.energy(mb, True)
            fake, entropy = model.generate(mb_len=mb_len)
            e_fake = model.energy(fake, True)
            
            # Gradient calculation with backprop
            model.zerograds()
            e_fake.backward()
            # changing the sign of gradinets related with the energy model
            model.scale_em_grads(-lambda_em)
            (lambda_em * e_real - entropy).backward()
            
            # go
            opt.update()
            
            # Outputing information
            print('%d r:%f f:%f e:%f'%(i, float(e_real.data), float(e_fake.data), float(entropy.data)))
            sum_e_real += float(e_real.data)*mb_len
            sum_e_fake += float(e_fake.data)*mb_len
            sum_entropy += float(entropy.data)*mb_len
        
        if epoch_id % save_interval == 0:
            serializers.save_npz('%s_%d'%(model_name, epoch_id), model)
        avr_e_real = sum_e_real / len(ds_train)
        avr_e_fake = sum_e_fake / len(ds_train)
        avr_entropy = sum_entropy / len(ds_train)
        with open('%s_log.txt'%(model_name), 'a') as f:
            f.writelines(['%d\t%f\t%f\t%f\n'%(epoch_id, avr_e_real, avr_e_fake, avr_entropy)])
    
    serializers.save_npz('%s_%d'%(model_name, epoch_id), model)
    print('end')

if __name__ == '__main__':
    main()
