#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Author: Dan Nemrodov <dan.nemrodov@utoronto.ca>
# Lab: Adrian Nestor Lab, U of T
# Date: August 2020




import numpy as np
import numpy.matlib as nm
import pandas as pd
from skimage import color
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
from scipy import stats
import statsmodels.stats.multitest as smt
from PIL import Image
from pathlib import Path, PurePath


class Reconstruction:
    
    """
    A class used to create reconstructions of images

    ...

    Attributes
    ----------
    rc_df : pandas DataFrame
        confusion matrix with index and columns as identity names of stimuli to be reconstructed
    tr_df : pandas DataFrame
        confusion matrix with index and columns as identity names of the training stimuli
    ims_df : pandas DataFrame
        images of training stimuli as columns, including zeros of the background
    dims_n : int
        the maximum number of dimenstions to use for reconstruction
    im_size : tuple
        the size of the images
    rc_names : list
        list of the names to be reconstructed
    tr_names : list
        list of the trainig names
    ims_n : int
        the number of the training stimuli
    bck : bool
        boolean vectro with False for background pixels and True for image pxels
    loo_dict : dict
        dictionary containing faces spaces with projections for each stimuli to be reconstructed
    lab_df : pandas DataFrame
        DataFrame with columns as images in labciel 
    rcd_df : pandas DataFrame
        DataFrame with columns as reconstructed stimuli in labciel color space
    rcd_rgb_df : pandas DataFrame
        DataFrame with columns as reconstructed stimuli in rgb color space
    info : dict
        dictionary with auxillary information regarding the reconstruction procedure
    training_face_space : pandas DataFrame
        MDS - derived stimuli space with dimensions as columns
    perm_n : int
        number of permutations if dimensions are deterimined by permutations
    selcd_dims : dict
        dictionary with dimensions used for reconstruction for each stimulus
    minpixels : int
        the mininmum number of significant pixels required for a dimension to be used for reconstruction
    result : pandas DataFrame
        columns with 1 for smaller euclidean distance and 0 for longer exuclidean distance
    summary : pandas DataFrame
        accuracy for each reconstructed stimuli


    Methods
    -------
    project_face_space()
         Create a leave-one-out space for each reconstructed face
    ims2lab()
        converts pandas dataframe with images into a dataframe in lab ciel space
    lab2ims()
        converts pandas dataframe with images as labciel into a dataframe with images in rgb space
    face_reconst()
        reconstucts faces 
    show_reoncs()
        print out reconstructed images one after another either in rgb (default) or lab color space
    convert2ims()
        converts stripped stimuli vector into 3-D image with background in rgb (default) or lab space
    distance_test()
        performs euclidean distance test for reconstruction by comparing each reconstructed image with 
        origin and all other images (default) or each original image with it's reconstruction and all 
        other reconstructions
    show_image()
        shows a specific reconsructed image called by name
    select_dims()
        computes informative dimensions for reconsruction using permutations
    add_column_to_summary()
        adds another column to summary dataframe with additional information regarding the stimuli
    save_images()
        saves all reconsructed images as jpg files
    
    
    """

    def __init__(self, rc_conf_df, tr_conf_df, ims_df, dims_n, image_size, bck = None, loo_dict = dict()):
        
        '''Constructor. Important: the names must match between confusibility matrices

        Parameters
        ----------
        rc_conf_df : pandas DataFrame
            DataFrame with confusibility matrix and names as index and columns representing the targets
        tr_conf_df : pandas DataFrame
            DataFrame with confusibility matrix and names as index and columns representing the training set
        ims_df : pandas DataFrame
            DataFrame with images as columns, including background pixels. Column names - stimuli names
        dims_n : int
            Maximum number of dimensions in stimulus space
        image_size : tuple
            tuple with three elements representing the size of each image dimension
        Optional:
        ----------
        loo_dict : dict
            precomputed stimulus space with projected coordinates for each stimulus to be reconsructed  
            
        >>> recon = Reconstruction(rc_conf_df=dist, tr_conf_df=dist, ims_df=stims, dims_n=20, image_size=(98,75,3))
        >>> recon.project_face_space(kind = 'custom').ims2lab().face_reconst(blow='both').lab2ims().distance_test(kind = 'other')
        >>> result = recon_io.add_column_to_summary(descr, 'names')
            
            '''
        self.rc_df = rc_conf_df.sort_index(axis=0).sort_index(axis=1)
        self.tr_df = tr_conf_df.sort_index(axis=0).sort_index(axis=1)
        self.ims_df = ims_df.sort_index(axis=1)
        self.dims_n = dims_n
        self.im_size = image_size
        self.rc_names = list(self.rc_df.columns)
        self.tr_names = list(self.tr_df.columns)
        self.ims_n = len(self.tr_names)
        if bck is None:
            self.bck = np.mean(self.ims_df,axis=1).astype('bool')
        else:
            self.bck = pd.Series(bck)
        self.loo_dict = loo_dict
        self.lab_df = []
        self.rcd_df = []
        self.rcd_rgb_df = []
        self.info = dict()
        self.training_face_space = pd.DataFrame(self._cmdscale(self.tr_df)[0],index = self.tr_names)
        self.perm_n = None
        self.selcd_dims = {key:nm.repmat(True, 1, self.dims_n).ravel() for key in self.rc_names}
        self.minpixels = None
        self.result = None
        self.summary = None
        
    def project_face_space(self, kind = 'custom', **kwargs):
        ''' Create a leave-one-out space for each target stimulus      
        '''
        if kind == 'custom':
            # creating a target face space with all target faces
            target_all_space = self._cmdscale(self.rc_df)[0] # creating face space for all recon faces
            target_all_space_df = pd.DataFrame(target_all_space[:,:self.dims_n], index=list(self.rc_names))
            
            for name in self.rc_names: # looping over all target faces
                
                # computing training and target conf df without loo name
                train_conf_df = self.tr_df.drop(columns = name, index = name).copy()

                
                # names without loo name
                train_names = train_conf_df.index
                
                # computing face spaces without loo
                target_space = target_all_space_df.drop(index=name)
                target_names = target_space.index
                train_space = pd.DataFrame(self._cmdscale(train_conf_df)[0][:,:self.dims_n], index = train_names)
                train_target_space = train_space.loc[target_names,:].values 

                # computing procrustes projection using default code ?
                R, s = orthogonal_procrustes(train_target_space, target_space) #?
                train_space.loc[name] = np.dot(target_all_space_df.loc[name,:],R.T)*s
                
                # computing procrustes projection using Matlab imported code
#                 d, Z, tform = self.procrustes(target_space.values, train_target_space)
#                 train_space.loc[name] = tform['scale'] * np.dot(target_all_space_df.loc[name,:], tform['rotation']) + tform['translation']
                
                # sorting and assigning to the dictionary
                train_space = train_space.sort_index(axis = 0) 
                self.loo_dict[name] = train_space
        else:
            mds_scale = MDS(n_components=self.dims_n, dissimilarity = 'precomputed', n_init=10, max_iter=1000)
            all_rc = mds_scale.fit_transform(self.rc_df.values)
            all_rc = pd.DataFrame(all_rc[:,:self.dims_n], index=list(self.rc_names))
            for name in self.rc_names: # looping over all recon faces
                temp_rc_names = set(self.rc_names).difference({name}) # excluding leave-one-out face name
                temp_tr = self.tr_df.drop(columns = name, index = name) # excluding loo row and column in training data
                temp_tr_names = temp_tr.index # names left for training data
                temp_tr = pd.DataFrame(mds_scale.fit_transform(temp_tr), index=temp_tr_names) # mds on training data
                temp_rc_tr = temp_tr.loc[temp_rc_names,:].values # matching faces from  training face space to recon
                temp_rc = self.rc_df.loc[temp_rc_names,temp_rc_names] # choosing recon confusibility matrix
                temp_rc = mds_scale.fit_transform(temp_rc) # running mds on recon data
                R, s = orthogonal_procrustes(temp_rc_tr, temp_rc[:,:self.dims_n])
                temp_tr.loc[name] = np.dot(all_rc.loc[name,:],R)*s
                temp_tr = temp_tr.sort_index(axis = 0) 
                self.loo_dict[name] = temp_tr
        return self
    
    def ims2lab(self):
        self.lab_df = self.ims_df.copy()
        for name in self.lab_df.columns:
            temp = self.lab_df[name].values
            temp = np.reshape(temp, self.im_size, order='F')
            temp = np.uint8(temp)
            temp = color.rgb2lab(temp)
            temp = temp.ravel(order = 'F')
            self.lab_df[name] = temp
        return self
    
    def lab2ims(self):
        self.rcd_rgb_df = self.rcd_df.copy()
        for name in self.rcd_rgb_df.columns:
            temp = self.rcd_rgb_df[name].values
            temp = np.reshape(temp, self.im_size, order = 'F')
            temp = 255.*np.clip(color.lab2rgb(temp),0,1)
            temp = temp.ravel(order = 'F')
            self.rcd_rgb_df[name] = np.uint8(temp)
        return self

    def face_reconst(self, blow=None):  
        recons_dict = dict()
        for i, name in enumerate(self.rc_names):
            
            # preparing images: removing backgrounnd, removing name row, and converting to numpy array
            ims = self.lab_df[self.bck].drop(columns=name).values# removing the background (pixels X ims)
            
            # separating coefficients for the target and training faces            
            loadings_current = self.loo_dict[name].drop(index=name).values # removing the name to be reconstructed from table 1
            cfs = self.loo_dict[name].loc[name,:] # coefficents of the face to be reconstructed
            
            # computing origin face
            origin_face = self._comp_origin(loadings_current, ims) 
            
            CI_mat = []
            for dimension in range(self.dims_n):
                if self.selcd_dims[name][dimension]:
                    CI = self._compute_classification_image(loadings_current, ims, dimension)
                    CI_mat.append(cfs[dimension]*CI/2)
                else:
                    pass
            cnt = np.sum(np.array(CI_mat),axis=0)
            
            out, coefs, stim_mean, stim_cont, out_mean = self._assemble_recon(cnt, origin_face, stims=ims, blow=blow)
            rc = np.zeros((self.lab_df.shape[0]))
            rc[self.bck] = out
            recons_dict[name] = rc
            self.info[name] = {'origin':origin_face, 'cnt':cnt, 'coef':coefs, 'CIs':CI_mat,
                                   'stimuli_mean':stim_mean, 'stimuli_contrast':stim_cont, 'out_mean':out_mean,
                                  'loadings':cfs}          
        self.rcd_df = pd.DataFrame(recons_dict)
        return self
    
    def _comp_origin(self, loadings, ims):
        dist = np.sqrt(np.sum(np.square(loadings), axis=1))
        dist = dist*(1/np.sum(dist, axis=0))
        return np.sum(ims*dist, axis=1)

    def _compute_classification_image(self, loadings, ims, dim):
        ind_pos = loadings[:,dim]>0
        ims_pos = ims[:,ind_pos]
        ims_neg = ims[:,np.logical_not(ind_pos)]
        loadings_pos = loadings[ind_pos, dim]
        loadings_neg = loadings[np.logical_not(ind_pos),dim]*-1
        prot_pos = np.sum(ims_pos*loadings_pos ,axis = 1)
        prot_neg = np.sum(ims_neg*loadings_neg, axis = 1)
        CI = prot_pos - prot_neg
        return CI
    
    def _assemble_recon(self, cnt, origin, stims, blow=0):
        stims = np.reshape(stims,(stims.shape[0]//3,3,-1), order='F')
        stims_mean = np.mean(stims, axis=0)
        stims_contrast = np.std(stims, axis=0)
        av_stims_mean = np.mean(stims_mean, axis=1)
        av_stims_contrast = np.mean(stims_contrast,axis=1)
        if not blow:
            out = cnt + origin
            coefs = None
            out_mean = np.mean(out,axis=0)
        elif blow.lower() == 'both':
            origin = np.reshape(origin,(origin.shape[0]//3,-1), order='F')
            cnt = np.reshape(cnt,(cnt.shape[0]//3,-1),order='F')
            coefs = [self._compContrastCoef(origin[:,i], cnt[:,i], av_stims_contrast[i]) for i in range(3)]
            out = [origin[:,i]+coefs[i]*cnt[:,i] for i in range(3)]
            out = [out[i] + av_stims_mean[i]-np.mean(out[i]) for i in range(3)]
            out = np.array(out).T
            out_mean = np.mean(out,axis=0)
            out = np.reshape(out,(out.shape[0]*3,), order='F')
        elif blow.lower() == 'mean':
            coefs = None
            origin = np.reshape(origin,(origin.shape[0]//3,-1), order='F')
            cnt = np.reshape(cnt,(cnt.shape[0]//3,-1),order='F')
            out = [origin[:,i]+cnt[:,i] for i in range(3)]
            out = [out[i] + av_stims_mean[i]-np.mean(out[i]) for i in range(3)]
            out = np.array(out).T
            out_mean = np.mean(out,axis=0)
            out = np.reshape(out,(out.shape[0]*3,), order='F')
        return (out, coefs, av_stims_mean, av_stims_contrast, out_mean)
        
    def _compContrastCoef(self, x, y, target):
        cont_x = np.std(x)
        cont_y = np.std(y)
        cov_xy = np.cov(x,y)[0,1]
        b = 2*cov_xy
        a = cont_y**2
        c = cont_x**2 - target**2
        return max(np.roots([a, b, c]))
    
    def show_recons(self, data='rgb'):
        for name in self.rc_names:
            if data is 'rgb':
                image = np.uint8(self.rcd_rgb_df[name].values)
            elif data is 'lab':
                image = self.rcd_df[name].values
            else:
                image = data[name].values
            image = np.reshape(image, self.im_size, order='F')
            plt.imshow(image)
            plt.title(name)
            plt.show()
            
    def convert2ims(self, data, type = 'rgb'):
        out = np.zeros((self.ims_df.shape[0]))
        out[self.bck] = data
        out = np.reshape(out, self.im_size, order = 'F')
        if type=='rgb':
            out = color.lab2rgb(out)
        return out
    
    def distance_test(self, kind = 'one_reconstructed_vs_all'):
        self.result = dict()
        if kind=='one_reconstructed_vs_all':
            for name in self.rcd_df.columns:
                same = distance.euclidean(self.rcd_df[name], self.lab_df[name])
                others = [distance.euclidean(self.rcd_df[name], self.lab_df[i]) for i in self.lab_df.drop(columns=name).columns]
                self.result[name] = same<np.array(others)
        else:
            for name in self.lab_df.columns:
                same = distance.euclidean(self.rcd_df[name], self.lab_df[name])
                #import pdb; pdb.set_trace()
                others = [distance.euclidean(self.lab_df[name], self.rcd_df[i]) for i in self.rcd_df.drop(columns=name).columns]
                self.result[name] = same<np.array(others)
        print(f'The overall accuracy is {np.mean(pd.DataFrame(self.result).values)*100:.2f}%')
        self.result = pd.DataFrame(self.result)
        self.summary = pd.DataFrame({'values':np.mean(self.result).values, 'names':self.result.columns}).sort_values(by=['names'])
        self.rc_df.sort_index()
        assert list(self.summary['names'])==list(self.rc_df.index)
        self.summary['acc'] = np.sum(self.rc_df.values, axis=1)/(self.rc_df.shape[0]-1)
        return self.summary
    
    def custom_distance_test(self, target_df, train_df, kind = 'one_reconstructed_vs_all'):
        result = dict()
        if kind=='one_reconstructed_vs_all':
            for name in target_df.columns:
                same = distance.euclidean(target_df[name], train_df[name])
                others = [distance.euclidean(target_df[name], train_df[i]) for i in train_df.drop(columns=name).columns]
                result[name] = same<np.array(others)
        else:
            for name in train_df.columns:
                same = distance.euclidean(target_df[name], train_df[name])
                others = [distance.euclidean(train_df[name], target_df[i]) for i in target_df.drop(columns=name).columns]
                result[name] = same<np.array(others)
        print(f'The overall accuracy is {np.mean(pd.DataFrame(result).values)*100:.2f}%')
        result = pd.DataFrame(result)
        summary = pd.DataFrame({'values':np.mean(result).values, 'names':result.columns}).sort_values(by=['names'])
        return summary
    
    def show_image(self, df, name, size):
        image = df[name].values
        image = np.reshape(image, size, order='F')
        plt.imshow(image)
        plt.title(name)
        plt.show()
        
    def select_dims(self, perm_n=1000, q_val=0.1, min_pixels=10):
        ims = recon1.lab_df[self.bck].values
        self.perm_n = perm_n
        self.min_pixels = min_pixels
        for i, name in enumerate(self.rc_names):
            self._update_progress(i/len(self.rc_names))
            data = stats.zscore(self.loo_dict[name])
            p_dim = []
            for dim in np.arange(1,self.dims_n):
                CI_perm = []
                for perm in range(perm_n):
                    if perm == 0:
                        load_temp = data[:,dim]
                    else:
                        load_temp = np.random.permutation(data[:,dim])
                    CI_perm.append(self._compute_classification_image(np.expand_dims(load_temp, axis=1), ims, 0))
                p_dim.append(pd.DataFrame(np.array(CI_perm)).rank(pct=True).values[0,:])
            p_dims = [smt.multipletests(x, alpha = q_val, method='fdr_bh')[0] for x in p_dim]
            p_dims = np.sum(np.array(p_dims), axis=1)>min_pixels
            self.selcd_dims[name] = np.insert(p_dims, 0 , True).ravel()
        self._update_progress(1)
        return self
    
    def add_column_to_summary(self, new_df, name):
        self.summary = pd.merge(self.summary, new_df, how='outer', on=[name]).sort_values(by = 'names')
        return self.summary
    
    def save_images(self, folder):
        for name in self.rc_names:
            image = self.rcd_rgb_df[name].values
            name = name+'.jpg'
            pth = PurePath(folder,name)
            image = np.reshape(image, self.im_size, order='F')
            plt.imsave(pth, image)
            
    def export_args_to_csv(self, folder):
        # checkking consistency
        assert (self.rc_df.columns == self.ims_df.columns).all()
        assert (self.rc_df.columns == self.rc_df.index).all()
        
        # exporting confusibility matrix
        pth = PurePath(folder,'conf.csv')
        self.rc_df.to_csv(pth)
        
        # exporting images
        for name in self.ims_df.columns:
            image = np.uint8(self.ims_df[name].values)
            name = name+'.jpg'
            pth = PurePath(folder,name)
            image = np.reshape(image, self.im_size, order='F')
            plt.imsave(pth, image)
        

    def _cmdscale(self, D):
        """                                                                                       
        Classical multidimensional scaling (MDS)                                                  

        Parameters                                                                                
        ----------                                                                                
        D : (n, n) array                                                                          
            Symmetric distance matrix.                                                            

        Returns                                                                                   
        -------                                                                                   
        Y : (n, p) array                                                                          
            Configuration matrix. Each column represents a dimension. Only the                    
            p dimensions corresponding to positive eigenvalues of B are returned.                 
            Note that each dimension is only determined up to an overall sign,                    
            corresponding to a reflection.                                                        

        e : (n,) array                                                                            
            Eigenvalues of B.                                                                     

        """
        # Number of points                                                                        
        n = len(D)

        # Centering matrix                                                                        
        H = np.eye(n) - np.ones((n, n))/n

        # YY^T                                                                                    
        B = -H.dot(D**2).dot(H)/2

        # Diagonalize                                                                             
        evals, evecs = np.linalg.eigh(B)

        # Sort by eigenvalue in descending order                                                  
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]

        # Compute the coordinates using positive-eigenvalued components only                      
        w, = np.where(evals > 0)
        L  = np.diag(np.sqrt(evals[w]))
        V  = evecs[:,w]
        Y  = V.dot(L)

        return Y, evals
    
    def _update_progress(self, progress):
        ''' Creates a text progress bar and prints out iteration identity
        :param: progress - total number of iterations
        '''


        import time, sys
        from IPython.display import clear_output

        bar_length = 20
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        block = int(round(bar_length * progress))
        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
        print(text)

    

