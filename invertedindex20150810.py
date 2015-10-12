#!/usr/bin/env python

import os, sys,struct
import ctypes
import numpy as np
import time
from multiprocessing import Pool, Manager
from xml.dom import minidom , Node
import pickle, pprint
from itertools import  groupby 
from operator import itemgetter

clstFile = "./clstDir2/"#"/home/stbook/hqy/TRECVID/clstDir/"
dirid = ""
#471526

manager = Manager()
dictsift = manager.dict()
#listsift = [[] for i in range(1000000)]
#print listsift
#dictsift = {}
bins = np.arange(0,1000001)
print bins.shape
#inverted_index_table = np.zeros((1000000,47))



    
def buildInvertedIndex(binfilename):
        binfilename = binfilename.strip()
        pic = binfilename.split('/')
        pic = pic[len(pic)-1].split('.')[0]
	argmin = np.loadtxt(binfilename,delimiter=',', usecols=(3,), unpack=True)
	###histogram
        (hist, binedge) = np.histogram(argmin,bins,(1,1000001))
        index = np.hstack((binedge.reshape((1000001,1))[:1000000,:],hist.reshape((1000000,1))))
        nonzeros = np.array(index[index[:,1]>0])
        
        for indx,count in nonzeros:
                if indx in dictsift:
                        x = dictsift[indx]
                        x.extend([[dirid,pic,count]])
                        dictsift[indx] = x
                else:
                        dictsift[indx] = [[dirid,pic,count]]
        
        print binfilename
        #
        
         

if __name__ == "__main__":

        camera_count = np.loadtxt('shot.camera.frame.reference.txt', usecols=(0,),skiprows=1)
        print camera_count.shape
        #hist, binedge = np.histogram(camera_count,np.arange(0,245))
        #print hist.shape
        #print hist
        start = time.time()
        shotid = np.arange(0,244).reshape((244,1))
        
        #v=np.empty([244,1],dtype=str)
        v=np.empty([0,1],dtype=str)
        #### xml parse ####
        doc = minidom.parse("eastenders.collection.xml")
        doc.toxml('UTF-8')
        for child in doc.getElementsByTagName("filename"):
                #print child.firstChild.data
                v=np.vstack((v,child.firstChild.data.encode('utf-8','ignore')[:-5]))
                #np.append(a,child.firstChild.data.encode('utf-8','ignore'))
        #print v
        
        info = np.hstack((shotid,v))
        
        #dictionaryMerge = {}
        for filename in os.listdir(clstFile):
                #print filename
                dirid = info[np.where(info[:,1]==filename)][0,0]
                os.system("ls "+clstFile+filename+"/*.txt > temp.txt")
                images = np.loadtxt('temp.txt',dtype=str,delimiter=',')
                pool = Pool()
                value = pool.map(buildInvertedIndex,images)
                pool.close()
                pool.join()
                #print dictsift
                output = open(filename+'.pkl', 'wb')
                a = time.time()
                pickle.dump(dictsift.copy(), output,2)
                b = time.time()
                print b-a
                output.close()
                dictsift.clear()
                #print dictsift
           
       
        print dictsift
        
        for filename in os.listdir(clstFile):
                pkl_file = open(filename+'.pkl', 'rb')
                data1 = pickle.load(pkl_file)
                print data1
                #pprint.pprint(data1)
                pkl_file.close()
        
        end = time.time()
        print "%f seconds" %(end - start)
