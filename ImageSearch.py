'''
ImageSearch uses annoy to retrieve nearest neighbors and then does exact computation
'''

from annoy import AnnoyIndex
import h5py
import numpy as np
from scipy.spatial.distance import cdist


def build_index_annoy(h5fname , dset,out='data.ann',trees = 128,lazy=True):
    #establish connection to HDF5 file
    h5f = h5py.File(h5fname,'r')
    if lazy:
        X = h5f[dset]
    else:
        X = h5f[dset][:]

    #get dimension
    f = X.shape[1]

    #initialize annoy
    t = AnnoyIndex(f,'angular')

    #iterate over features, add to annoy
    for i,v in enumerate(X):
        t.add_item(i, v)

    #build and save index
    t.build(trees)
    t.save(out)

class ImageSearchAnnoy:
    '''
    load an Annoy index for approximate nearest neighbor computation
    Annoy's angular distance uses dist(u,v) = 2(1-cos(u,v))
    '''
    def __init__(self,dimensions,annf='ILSVRC2015.ann',imageListPath = '/home/scratch/benediktb/RegionOfInterest/ILSVRC2015_filelist.txt'):
        #load filenames
        with open(imageListPath,'r') as f:
            #self.line_to_file = {i:line.split('/')[-1].rstrip() for i,line in enumerate(f)}
            self.line_to_file = {i:line.rstrip() for i,line in enumerate(f)}
        self.A = AnnoyIndex(dimensions,'angular')
        self.A.load(annf)

    def run_query(self,query,n=100,accuracy_factor = 2):
        nearest,scores = self.A.get_nns_by_vector(query, n, search_k=n*int(accuracy_factor)*128, include_distances=True)
        return zip((self.line_to_file[i] for i in nearest),scores)


class ImageSearchAnnoyCombo:
    '''
    load an Annoy index for approximate nearest neighbor computation
    Annoy's angular distance uses dist(u,v) = 2(1-cos(u,v))
    '''
    def __init__(self,h5fname = 'X_ILSVRC2015.hdf5',annf='ILSVRC2015.ann',imageListPath = '/home/scratch/benediktb/RegionOfInterest/ILSVRC2015_filelist.txt',dset = 'fc6fc7'):
        #load h5 data
        h5f = h5py.File(h5fname,'r')
        self.X = h5f[dset]
        #load filenames
        with open(imageListPath,'r') as f:
            self.line_to_file = {i:line.rstrip() for i,line in enumerate(f)}
        self.A = AnnoyIndex(self.X.shape[1],'angular')
        self.A.load(annf)

    def run_query_approx(self,query,n=100,accuracy_factor = 5):
        nearest,scores = self.A.get_nns_by_vector(query, n, search_k=n*int(accuracy_factor)*128, include_distances=True)
        return zip((self.line_to_file[i] for i in nearest),scores)

    def run_query_exact(self,query,n=1000,nsmall=100):
        #retrieve approximate nearest neighbors using Annoy, then do exact ranking by loading from h5 into memory
        #use Annoy
        if n < nsmall:
            n = nsmall
        indexes = self.A.get_nns_by_vector(query, n, search_k=-1, include_distances=False)
        indexes_sorted = sorted(indexes)
        #use scipy cdist (or normalize first and do dot product for faster computation)
        #getting X by index from disc is very slow. 
        distance = (cdist(self.X[indexes_sorted], query.reshape((1,query.shape[0])), 'cosine'))[:,0]
        ind = np.argpartition(distance, nsmall)[:nsmall]#partial sort, indices for top n,
        s_ind = np.argsort(distance[ind])#sort 
        nearest = ind[s_ind]
        scoresorted = distance[ind][s_ind]
        return zip((self.line_to_file[indexes_sorted[i]] for i in nearest),scoresorted)



class ImageSearchMemory:
    '''
    Load entire dataset into memory, do exact similarity computation
    '''
    def __init__(self,h5fname = 'Xfc6fc7.h5',imageListPath = '/home/scratch/benediktb/RegionOfInterest/flickr_filelist.txt',dset = 'data'):
        #load h5 data, normalize with l2
        h5f = h5py.File(h5fname,'r')
        self.X = h5f[dset][:]
        norms = np.sqrt(np.einsum('ij,ij->i', self.X , self.X))
        norms[norms == 0.0] = 1.0
        self.X /= norms[:, np.newaxis]
        #load filenames
        with open(imageListPath,'r') as f:
            self.line_to_file = {i:line.rstrip() for i,line in enumerate(f)}

    def run_query(self,query,n=100):
        #retrieve approximate nearest neighbors using Annoy, then do exact ranking by loading from h5 into memory
        similarity = np.dot(self.X,np.divide(query,np.linalg.norm(query)))
        ind = np.argpartition(-similarity, n)[:n]#partial sort, indices for top n,
        s_ind = np.argsort(similarity[ind])[::-1]#sort 
        nearest = ind[s_ind]
        scoresorted = similarity[ind][s_ind]
        return zip((self.line_to_file[i] for i in nearest),scoresorted)

