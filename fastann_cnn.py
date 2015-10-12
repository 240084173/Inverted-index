import ctypes
import numpy as np
import fastann
import os,sys
import tables
from pprint import pprint as pp
from collections import defaultdict
import pickle as p

libfastann = ctypes.CDLL('libfastann.so')

featurefilename = "/home/stbook/hqy/image/features/BaojiTraffic1000/"
clsth5filename = "./clst.h5"
clstbinfilename = "clst.bin"
K = 5000
D = 4096
postfix = "fc6.binary"

ivresult = defaultdict(list)

def h5tobin():
    clst_obj = tables.open_file(clsth5filename,'r')
    center = clst_obj.root.clusters
    center[:,:].tofile(clstbinfilename)
    print center[:,:]
    clst_obj.close()

def read(filename):
    clst = np.fromfile(filename,dtype="int64")
    size = clst.size/5
    clst = clst.reshape(size,5)
    #print clst.shape
    return clst[:,-1]

def quantity():
    center = np.fromfile(clstbinfilename,dtype="float32")
    center = center.reshape((K,D))
    #center.dtype = "float64"
    print center.dtype
    print "kdtree build start"
    nno = fastann.build_kdtree(center, 8, 768)
    print "kdtree build finish"
    for filename in os.listdir(featurefilename):
        if filename.endswith(postfix):
            name = filename.split("_")
            nametxt = name[0] + '_box.txt'
            fc6feat = np.fromfile(featurefilename+filename,dtype='float32')
            fc6feat = fc6feat.reshape((fc6feat.shape[0]/4096,4096))
            argmins,mins = nno.search_nn(fc6feat)
            argmins = argmins.reshape((argmins.shape[0],1))
            index = np.loadtxt(featurefilename+nametxt,dtype = "int32")
            clst= np.append(index ,argmins,axis = 1)
            #clst.dtype="int32"
            clst.tofile(featurefilename+name[0]+'_clst.bin')
            print featurefilename+name[0]+'_clst.txt'
            #np. savetxt(featurefilename+name[0]+'_clst.txt',clst,fmt='%i',delimiter=',')
            #print argmins
        

def inverted_index(filename):
    clst = read(filename)
    #bins = np.arange(0,K+1)
    #(hist1,binedge)=np.histogram(clst,bins,(1,K+1))
    hist = np.bincount(clst)
    #hist1 = np.append(np.arange(hist.size),hist,axis=1)
    #print hist1
    nonzeros = np.nonzero(hist)[0]
    #print nonzeros
    for index in nonzeros:
        ivresult[index].append((filename.split('/')[-1].split('.')[0],hist[index]))
    #result = [[filename.split('/')[-1].split('.')[0]],hist]
    #pp(ivresult)
    #for bin in clst:
    #    if iv.has_key(bin):
    #        iv[bin].append(filename)
    #    else:
    #        iv[bin] = []
    #        iv[bin].append(filename)
    #pp(iv)
             
    
    

if __name__ == "__main__":
    #h5tobin()
    #quantity()
    #read(featurefilename+'00176_clst.bin')
    #inverted_index(featurefilename+'00017_clst.bin')
    os.system('find '+featurefilename+'>binfilepath1.txt -path \'*_clst.bin\'')
    images = np.loadtxt('binfilepath1.txt',dtype = str, delimiter = ',')
    #print images.size
    map(inverted_index,images)
    
    for k,v in ivresult.iteritems():
        print "dict[%s]:%s" %(k,v)
    #output = open('1000bj.pkl','wb')
    #p.dump(ivresult,output)
    #output.close()
    #r = open('1000bj.pkl','rb')
    
    #data1 = p.load(r)
    #pp(data1[2734])
    
    #pp(ivresult[2734])    
    
