import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


m = torch.load('examples/classifier_compression/logs/2019.08.17-020503/best.pth.tar')
# (m['state_dict']['module.conv1.float_weight'].data.cpu().numpy().ravel())

fig, ax = plt.subplots(1, 4)
f, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=False)
axes = axes.ravel()
i = 0
for ax in axes:
   #print(m['state_dict']['module.conv1.float_weight'].data.cpu().numpy().shape)
   #kernel = m['state_dict']['module.conv1.weight'].data[i,0,:,:].cpu().numpy()
   step = 1
   """ full precision """
   #feature_map = m['state_dict']['module.conv2.float_weight'].data[:,:,0,i].cpu().numpy()
   #ref_act = np.load('examples/classifier_compression/q_act_0.npy') 
   #ref_act = np.load('examples/classifier_compression/ref_act_0.npy') 
   ref_act = np.load('examples/classifier_compression/kldiv_ref_act_2031.npy') 
   #ref_act = np.load('examples/classifier_compression/ref_act.npy') 
   #ref_act = np.load('examples/classifier_compression/ref_act_'+str(step)+'.npy') 

   #feature_map = ref_act[:,:,2,i]
   feature_map = ref_act[100,i,:,:]
   #feature_map = ref_act[i,10,:,:]
   #L = len(kernel_fp.ravel())

   #ax.imshow(feature_map ,cmap='binary', interpolation='nearest')
   #ax.imshow(feature_map ,cmap='magma')
   ax.imshow(feature_map ,cmap='viridis')
   #ax.imshow(feature_map ,cmap='gray', interpolation='nearest')
   i+=1
   #im = axes.imshow(kernel, axes=[i], cmap='binary', interpolation='nearest')

plt.savefig('feature_maps/kldiv_ref.png')
#plt.savefig('feature_maps/alexnet_binary_losses_L1_1.png')

