import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

#  statistical analysis
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

""" Binary AlexNet """
#ours = torch.load('examples/classifier_compression/logs/2019.08.17-050710/checkpoint.pth.tar')
ours = torch.load('examples/classifier_compression/logs/2019.08.27-210120/best.pth.tar') # 20 epochs - 1, 1, fp - 0.001 LR

fp = torch.load('examples/classifier_compression/alexnet_bn.pth.tar')
#q_trained = torch.load('examples/classifier_compression/trained_models/alexnet_2bits_8Aug.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.23-132835/checkpoint.pth.tar')
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-213440/checkpoint.pth.tar') # 5 epochs - all 1 bit - all trainable
#q_trained = torch.load('examples/classifier_compression/logs/2019.08.26-220621/checkpoint.pth.tar') # 10 epochs - all 1 bit - all trainable 
q_trained = torch.load('examples/classifier_compression/logs/2019.08.27-185516/checkpoint.pth.tar') # 20 epochs - all 1 bit - all trainable - 0.002 LR 



layer = 'features.module.2.weight'

kernel_fp = fp['state_dict'][layer].data.cpu().numpy().ravel()
kernel_fp_q = quantize(kernel_fp, 1)
kernel_q_trained = q_trained['state_dict'][layer].data.cpu().numpy().ravel()
kernel_ours = ours['state_dict'][layer].data.cpu().numpy().ravel()

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

#plt.hist(fp, bins=100, histtype='stepfilled', normed=False, color='b', label='full precision')
#plt.hist(w_eff, bins=3, histtype='stepfilled', normed=False, color='r', alpha=0.25, label='modified')

sns.set(palette="bright", color_codes=True)
#sns.distplot( layers[0] , ax=axes[0], color=color, bins=100, kde=False, axlabel='epoch#')
xmax = 0.2
xmin = -xmax
#_ = plt.hist(fp, bins=100, label='Original', range=[xmin,xmax])
_ = plt.hist(fp, bins=500, edgecolor='none', label='Original')
_ = plt.hist(w_eff, alpha=0.8,  bins=200, edgecolor='none', label='Modified')
plt.xlim(-0.15, 0.15)
plt.ylabel('Count')
plt.xlabel('Value')
#plt.title("Analysis")
#plt.legend()
plt.savefig('figs/August_AlexNet_hist_conv2.png')
