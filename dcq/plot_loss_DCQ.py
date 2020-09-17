import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

step = 20
m = torch.load('examples/classifier_compression/logs/2019.08.17-020503/best.pth.tar')
# (m['state_dict']['module.conv1.float_weight'].data.cpu().numpy().ravel())

fig, ax = plt.subplots(1, 4)
f, axes = plt.subplots(8, 8, figsize=(25, 25), sharex=False)
axes = axes.ravel()
i = 0
for ax in axes:

   #ref_act = np.load('examples/classifier_compression/fmaps/ref_act_'+str(i*10)+'.npy') 
   #ref_act = np.load('examples/classifier_compression/fmaps0/ref_act_'+str(i*10)+'.npy') 
   #q_act = np.load('examples/classifier_compression/fmaps/q_act_'+str(i*10)+'_35.npy') 
   #q_act = np.load('examples/classifier_compression/fmaps0/q_act_'+str(i*10)+'_31.npy') 


   ref_act = np.load('examples/classifier_compression/fmaps0/test3_ref_act_'+str(step)+'.npy') 
   q_act = np.load('examples/classifier_compression/fmaps0/test3_q_act_'+str(step)+'_31.npy') 

   ref = ref_act[100,i,:,:]
   q = q_act[100,i,:,:]

   #loss = np.abs(np.subtract(ref, q)) # L1 
   #loss = np.subtract(ref, q)**2 # MSE
   #loss = np.log(np.cosh(np.subtract(ref, q))) # L1 

   delta = 0.4
   loss1 = np.abs(np.subtract(ref, q)) # L1 
   loss2 = np.subtract(ref, q)**2 # MSE
   loss1[loss1 < delta] = 0
   loss2[loss2 > delta] = 0
   loss = np.add(loss1, loss2) 

   #ax.imshow(feature_map, cmap='binary', interpolation='nearest')
   pos = ax.imshow(loss, cmap='Reds', vmin=0, vmax = 1)
   #pos = ax.imshow(q, cmap='viridis', vmin=0, vmax=2)
   fig.colorbar(pos, ax=ax)
   #pos.clim(0, 2)
   #ax.colorbar()
   #ax.imshow(feature_map, cmap='viridis')
   #ax.imshow(feature_map, cmap='hsv')
   #ax.imshow(feature_map, cmap='cubehelix')
   #ax.imshow(feature_map, cmap='gray', interpolation='nearest')
   i+=1
   #im = axes.imshow(kernel, axes=[i], cmap='binary', interpolation='nearest')

#plt.colorbar()
plt.savefig('feature_maps/Huber_loss_'+str(step)+'.png')
