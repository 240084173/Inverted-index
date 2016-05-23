#!/usr/bin/env python

import json
import numpy as np
import h5py
from multiprocessing import Pool, Manager
import cPickle as pickle
import time
from collections import Counter
#hqy
cluster_num=1000000
inverted_ix={}
shot_2_index={}
for i in range(cluster_num+1):
    inverted_ix[i] = {}


#{u'images_ix': [0], u'video_index': u'0', u'images_num': 1, u'name': u'shot0_1', u'shot_index': u'1'}

def buildInvertedIndex(shot_id_index,sidx):
    if shot_id_index['images_num']!= 0:
        bow_path = './b/bow_'+shot_id_index['video_index']+'.h5'
        f = h5py.File(bow_path,'r')
        start = shot_id_index['images_ix'][0]
        end = shot_id_index['images_ix'][-1]
        point_vq = f['data'][start:end+1,:]
        row = point_vq.shape[0]
        col = point_vq.shape[1]
        temp = point_vq.reshape((1,row * col))
        cur_bins,counts = np.unique(temp,return_counts=True)
        for idx,vq_bin in enumerate(cur_bins):
            inverted_ix[vq_bin][sidx] = counts[idx]



if __name__ == "__main__":

        filename = './shot_frame_brief.json'
        f = file(filename)
        source = f.read()
        shot_index = json.JSONDecoder().decode(source)

        print time.strftime('%Y-%m-%d %H:%M:%S')
        for idx,shot_id in enumerate(shot_index):
            shot_2_index[idx] = shot_id['name']
            buildInvertedIndex(shot_id,idx)
            if idx%10000==0:
                print idx
                print time.strftime('%Y-%m-%d %H:%M:%S')

        print time.strftime('%Y-%m-%d %H:%M:%S')
        index_file = 'dict2.pk'
        f1 = file(index_file, 'wb')
        pickle.dump(inverted_ix, f1, True)
        pickle.dump(shot_2_index, f1, True)
        f1.close()
        print time.strftime('%Y-%m-%d %H:%M:%S')

