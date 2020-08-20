#!/usr/bin/env python
# coding: utf-8

# In[ ]:


infolder, outfolder = find_folder()
trigs=list(range(101, 151))+list(range(201, 251))
#trigs=list(range(101, 151))
event_ids={str(x):x for x in trigs}
im_times=(-0.1,1)
filt=(0.03,200)
subs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
np.random.seed(10)
n_perm = 20  # number of permutations
n_pseudo = 11  # number of pseudo-trials
results = di.compute_combined(folder, subs, filt, im_times, event_ids)
    
    

get_ipython().run_line_magic('run', 'general_tools.ipynb')
import discr_inverted as di
infolder, outfolder = find_folder()
trigs=list(range(101, 151))+list(range(201, 251))
event_ids={str(x):x for x in trigs}
im_times=(-0.1,1)
filt=(0.03,200)
subs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
subs = [18]
np.random.seed(10)
n_perm = 20  # number of permutations
n_pseudo = 11  # number of pseudo-trials
results = di.compute_time(infolder, subs, filt, im_times, event_ids)


