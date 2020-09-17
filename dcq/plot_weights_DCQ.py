import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


""" Binary AlexNet """
#ours = torch.load('examples/classifier_compression/logs/2019.08.17-050710/checkpoint.pth.tar')
#fp = torch.load('examples/classifier_compression/alexnet_bn.pth.tar')
#q_trained = torch.load('examples/classifier_compression/trained_models/alexnet_2bits_8Aug.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.23-132835/checkpoint.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-213440/checkpoint.pth.tar') # 5 epochs - all 1 bit - all trainable
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-220621/checkpoint.pth.tar') # 10 epochs - all 1 bit - all trainable

""" Binary AlexNet """
fp = torch.load('examples/classifier_compression/alexnet_bn.pth.tar')

#ours = torch.load('examples/classifier_compression/logs/2019.08.17-050710/checkpoint.pth.tar')
ours = torch.load('examples/classifier_compression/logs/2019.08.27-210120/best.pth.tar') # 20 epochs - 1, 1, fp - 0.001 LR

#q_trained = torch.load('examples/classifier_compression/trained_models/alexnet_2bits_8Aug.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.23-132835/checkpoint.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-213440/checkpoint.pth.tar') # 5 epochs - all 1 bit - all trainable
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-220621/checkpoint.pth.tar') # 10 epochs - all 1 bit - all trainable
q_trained = torch.load('examples/classifier_compression/logs/2019.08.27-185516/best.pth.tar') # 20 epochs - all 1 bit - all trainable - 0.002 LR


#fp = torch.load('partial_BP_models_fp/svhn.pth.tar')
#q_trained = torch.load('partial_BP_models/cifar_wrpn_all3.pth.tar')
#ours = torch.load('partial_BP_models_binary/svhn_1bit_stage4_dorefa_fixed.pth.tar')

# [features.module.0.weight, features.module.2.weight, features.module.3.weight]

def quantizei_tread(data, n_bit):
    step = (2**(n_bit-1)-0.5)
    mul = np.multiply(data, step)
    mul_round = np.round(mul)
    data_q = (mul_round.astype(np.float32)) / step
    print(set(data_q))
    return data_q

def quantize(data, n_bit):
    r = (2**(n_bit-1)-0.5)
    mul = np.multiply(data, r)
    mul_floor = np.floor(mul)
    data_q = (mul_floor.astype(np.float32)) / r
    data_q = data_q + 0.5/r
    print(set(data_q.ravel()))
    return data_q

#kernel_fp = fp['state_dict'][layer].data.cpu().numpy().ravel()
#kernel_fp_q = quantize(kernel_fp, 1)
#kernel_q_trained = q_trained['state_dict'][layer].data.cpu().numpy().ravel()
#kernel_ours = ours['state_dict'][layer].data.cpu().numpy().ravel()

f, axes = plt.subplots(4, 5, figsize=(8, 8), sharex=False)
#f, axes = plt.subplots(1, 3, figsize=(6, 2), sharex=False)
axes = axes.ravel()

i=0
#layer_name = 'module.features.3.weight'
layer_name = 'features.module.2.weight'
# 256, 48, 5, 5
for ax in axes:
   """ full precision """
   #kernel_fp = fp['state_dict'][layer_name].data[i,0,:,:].cpu().numpy()
   kernel_fp = fp['state_dict'][layer_name].data[15,i,:,:].cpu().numpy()
   kernel_fp_q = quantize(kernel_fp, 1)

   kernel_q_trained = q_trained['state_dict'][layer_name].data[15,i,:,:].cpu().numpy()

   #kernel_ours = ours['state_dict'][layer_name].data[i,0,:,:].cpu().numpy()
   kernel_ours = ours['state_dict'][layer_name].data[15,i,:,:].cpu().numpy()

   ax.imshow(kernel_fp ,cmap='bwr', interpolation='nearest', vmin=-0.07, vmax=0.07)
   #ax.imshow(kernel_fp_q ,cmap='binary', interpolation='nearest')
   #ax.imshow(kernel_ours ,cmap='binary', interpolation='nearest')
   #ax.imshow(kernel_q_trained ,cmap='binary', interpolation='nearest')
   i+=1
   #im = axes.imshow(kernel, axes=[i], cmap='binary', interpolation='nearest')

plt.savefig('figs/fp.png')

