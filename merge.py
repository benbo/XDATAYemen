import datetime
import h5py
import numpy as np

f1 = h5py.File('all_images.pool5_fc6-7.part1.h5', 'r')
f2 = h5py.File('all_images.pool5_fc6-7.part2.h5', 'r')
f3 = h5py.File('all_images.pool5_fc6-7.part3.h5', 'r')

d1 = f1['feats']
d2 = f2['feats']
d3 = f3['feats']

n1 = d1.shape[0]
n2 = d2.shape[0]
n3 = d3.shape[0]
dim = d1.shape[1]

with h5py.File('all_images.pool5.h5') as fout1, \
     h5py.File('all_images.fc6-7.h5') as fout2:
    out1 = fout1.create_dataset(
            'feats', dtype=np.float32,
            shape=(n1 + n2 + n3, dim - 8192)) # , chunksize=(1, dim-8192))
    out2 = fout2.create_dataset(
            'feats', dtype=np.float32,
            shape=(n1 + n2 + n3, 8192)) # , chunksize=(1, 8192))

    print("Writing d1 part 1: {:%H:%M}".format(datetime.datetime.now()))
    out1[:n1, :] = d1[:, :-8192]
    print("Writing d1 part 2: {:%H:%M}".format(datetime.datetime.now()))
    out2[:n1, :] = d1[:, -8192:]

    print("Writing d2: {:%H:%M}".format(datetime.datetime.now()))
    out1[n1:n1 + n2, :] = d2[:, :-8192]
    print("Writing d2 part 2: {:%H:%M}".format(datetime.datetime.now()))
    out2[n1:n1 + n2, :] = d2[:, -8192:]

    print("Writing d3: {:%H:%M}".format(datetime.datetime.now()))
    out1[-n3:, :] = d3[:, :-8192]
    print("Writing d3 part 2: {:%H:%M}".format(datetime.datetime.now()))
    out2[-n3:, :] = d3[:, -8192:]

    print("done: {:%H:%M}".format(datetime.datetime.now()))
