import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

network_name = 'alexnet'
fp = torch.load('examples/classifier_compression/alexnet_bn.pth.tar')

q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-213440/checkpoint.pth.tar') # 5 epochs - all 1 bit - all trainable
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-220621/checkpoint.pth.tar') # 10 epochs - all 1 bit - all trainable

ours = torch.load('examples/classifier_compression/binary_alexnet_kernels/MSE_0.005/best.pth.tar')

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

#f, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=False)
#f, axes = plt.subplots(1, 3, figsize=(6, 2), sharex=False)
#axes = axes.ravel()

l = ['features.module.0.weight']
layer_name = l[2]
#  statistical analysis
kernel_fp = fp['state_dict'][layer_name].data.cpu().numpy()
kernel_fp_q = quantize(kernel_fp, 1)
kernel_q_trained = q_trained['state_dict'][layer_name].data.cpu().numpy()
kernel_ours = ours['state_dict'][layer_name].data.cpu().numpy()

fp = kernel_fp.ravel()
q = kernel_fp_q.ravel()

ours = kernel_ours.ravel()
q_trained = kernel_q_trained.ravel()

idx = []
w_eff = []
for i in range(0, len(q)):
   if q[i] != q_trained[i]:
   #if q[i] != ours[i]:
      idx.append(i)
      w_eff.append(fp[i])
print(len(q))
print(len(idx))
print(min(fp), max(fp))
print(min(w_eff), max(w_eff))


sns.set(palette="bright", color_codes=True)
#sns.distplot( layers[0] , ax=axes[0], color=color, bins=100, kde=False, axlabel='epoch#')
_ = plt.hist(fp, bins=100, label='Original')
_ = plt.hist(w_eff, alpha=0.8,  bins=80, edgecolor='none', label='Modified')
plt.ylabel('counts')
plt.xlabel('weight value')

#plt.title("Analysis")
plt.xlabel("Value")
plt.ylabel("Count")
#plt.legend()
plt.savefig('figs/May_'+network_name+'_conv2_hist_conventional.png')
#plt.savefig('figs/May_'+network_name+'_conv2_hist_ours.png')
