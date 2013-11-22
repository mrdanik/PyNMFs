# -*- coding: utf8 -*-
'''
NMF test script

.................................................................
Danila Doroshin
http://www.linkedin.com/in/daniladoroshin

2013
'''
import os
import numpy as np
import h5py
from scipy.sparse import lil_matrix
from PyNMFs import clear_dir, optsNMF, PyNMFs

def FromTxt(h5Path, name, txtPath, Transpose = False, dtype = 'float'):
    if dtype == 'int':
        lines =  [[int(var) for var in line.split()] for line in open(txtPath, 'r').readlines()]
    elif dtype == 'float':
        lines =  [[float(var) for var in line.split()] for line in open(txtPath, 'r').readlines()]
    elif dtype == 'double':
        lines =  [[double(var) for var in line.split()] for line in open(txtPath, 'r').readlines()]
    else:
        print 'Error! Unknown data type!'
        return
    
    rows = len(lines)
    cols = len(lines[0])
    matr = np.array(lines)
    
    base = h5py.File( h5Path, 'w' )
    if Transpose:
        h5Set = base.create_dataset( name, data=matr.transpose(), compression='lzf')
    else:
        h5Set = base.create_dataset( name, data=matr, compression='lzf')
    base.close()


def ToTxt(h5Path, name, txtPath):
    base = h5py.File( h5Path, 'r' )
    h5Set = base[name]
    data = np.empty(h5Set.shape, h5Set.dtype)
    h5Set.read_direct(data)
    base.close()
    np.savetxt(txtPath, data, fmt='%.18e', delimiter='  ', newline='\n')


def RandMatH5(rows, cols, h5Path, name):
    print 'Generating rand matrix. File %s...'%(h5Path)
    base = h5py.File( h5Path, 'w' )
    h5Set = base.create_dataset( name, data=np.abs(np.random.standard_normal((rows, cols)), dtype='f') )
    base.close()
    print 'Matrix in file %s is generated'%(h5Path)


def test_nmf():
    #count of parallel processes
    prcnt = 5
    
    ###################### preparing data ######################
    #read Y matrix from txt file and write it in to hdf5 file
    FromTxt('TestY.hdf5', 'Y', 'TestY.txt', Transpose = True, dtype = 'int')
    #read hdf5 file
    Yh5Base = h5py.File( 'TestY.hdf5', 'r' )
    Yh5 = Yh5Base['Y']
    #convert to sparse lil matrix
    lilMat = lil_matrix(Yh5.__array__())
    rN, cN = lilMat.shape
    
    #get nonzero elements and indexes from lilMat
    rows = []
    cols = []
    values = []
    for k,row in enumerate(lilMat.rows.tolist()):
        for col in row:
            rows.append(k)
            cols.append(col)
    for vl in lilMat.data:
        for v in vl:
            values.append(v)
    
    #lilMat will be splitted in to 'prcnt' matrices that contain rows 
    #and in to 'prcnt' matrices that contain cols
    shape_list_r = [len(range(rN)[k::prcnt]) for k in range(prcnt)]
    shape_list_c = [len(range(cN)[k::prcnt]) for k in range(prcnt)]
    user_lists = [[] for k in xrange(prcnt)]
    song_lists = [[] for k in xrange(prcnt)]
    item_lists = [[] for k in xrange(prcnt)]
    
    #rows
    YMats_row = []
    
    for row,col,val in zip(rows,cols,values):
        pr = row%prcnt
        row_r = row//prcnt
    
        user_lists[pr].append(col)
        song_lists[pr].append(row_r)
        item_lists[pr].append(val)
    
    for k in xrange(prcnt):
        lilM = lil_matrix( (shape_list_r[k], cN), dtype=np.dtype('>i4') )
        for (s,u,itm) in zip(song_lists[k], user_lists[k], item_lists[k]):
            lilM[s,u] = itm
        YMats_row.append(lilM.tocsr())
    
    user_lists = [[] for k in xrange(prcnt)]
    song_lists = [[] for k in xrange(prcnt)]
    item_lists = [[] for k in xrange(prcnt)]
    
    #cols
    YMats_col = []
    for row,col,val in zip(rows,cols,values):
        pr = col%prcnt
        col_r = col//prcnt
        
        song_lists[pr].append(row)
        user_lists[pr].append(col_r)
        item_lists[pr].append(val)
    
    
    for k in xrange(prcnt):
        lilM = lil_matrix( (shape_list_c[k], rN), dtype=np.dtype('>i4') )
        for (s,u,itm) in zip(song_lists[k], user_lists[k], item_lists[k]):
            lilM[u,s] = itm
        YMats_col.append(lilM.tocsr())
    
    ###################### initialize D,X and params ######################
    Dh5Path = 'D.hdf5'
    Xh5Path = 'X.hdf5'
    rows = rN
    cols = cN
    dim = 10 #D dimension = clasters count
    workdir = 'workdir'
    if not os.path.exists(workdir):
            os.makedirs(workdir)
    else:
        clear_dir(workdir)
    
    #NMF options
    opts = optsNMF()
    opts.m = dim
    opts.beta = 1
    
    #initialize D, X
    RandMatH5(rows, dim, Dh5Path, 'D')
    RandMatH5(dim, cols, Xh5Path, 'X')
    
    ###################### processing ######################
    pyNMFs = PyNMFs()
    pyNMFs.NMF(opts, YMats_row, YMats_col, Dh5Path, Xh5Path, workdir)
    
    ###################### save results ######################
    ToTxt(Dh5Path, 'D', 'D.txt')
    ToTxt(Xh5Path, 'X', 'X.txt')


if __name__ == '__main__':
    test_nmf()