'''
PyNMFs - Nonegative Matrix Factorization (NMF) for Sparse matrices
Y - input sparse matrix
Algorithm estimates matrices D,X such as
                                
            Y = D*X             
                                
Parallel represenation is used
D and X are dense matrices. D, X are stored in hdf5 files

Initially Y will be splitted by N row matrices and N col matrices
N is a number of parallel processes

Auto parallelization is on if NumPy MKL is used.
Set next command to the command prompt (for Windows) before launching python scripts
"set OMP_NUM_THREADS=K"
where K is the max number of processes for auto parallelization.
K = 1 is recommended

.................................................................
Danila Doroshin
http://www.linkedin.com/in/daniladoroshin

2013-2014
'''
# -*- coding: utf8 -*-
import os
from os.path import join, basename, splitext
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from multiprocessing import Process, Queue
from itertools import islice
import multiprocessing as mp
import numpy as np
import h5py
import time


DEBUG_TIME = True


class optsNMF():
    def __init__(self):
        self.m = 1                  #number of clusters
        self.conv_value = 1e-3      #tolerance for stopping criteria
        self.max_iter = 1000        #
        self.lmbd = 1               #regularization param
        self.beta = 2               #
        self.eps = 1e-5


class PyNMFs():
    def NMF(self, opts, Y_lil, (prcountRow, prcountCol), Dh5Path, Xh5Path):
        self.opts = opts
        self.prcountRow = prcountRow
        self.prcountCol = prcountCol
        self.Dh5Path = Dh5Path
        self.Xh5Path = Xh5Path
        self.shape_rows, self.shape_cols = Y_lil.shape
        self.shape_dim = self.getDShape()[1]
        #split Y for parallel processing
        self.Yr_Mlist, self.Yc_Mlist = self.SplitY(Y_lil, self.prcountRow, self.prcountCol)
        
        
        # creating shared memmory
        self.XmpArr = mp.RawArray('f', self.shape_dim * self.shape_cols)
        self.X = np.ndarray((self.shape_dim, self.shape_cols), buffer=self.XmpArr, dtype='f')
        
        self.DmpArr = mp.RawArray('f', self.shape_rows * self.shape_dim)
        self.D = np.ndarray((self.shape_rows, self.shape_dim), buffer=self.DmpArr, dtype='f')
        
        # reading D and X from hdf5
        Dh5Base = h5py.File( Dh5Path, 'r' )
        Dh5 = Dh5Base['D']
        Dh5.read_direct(self.D)
        Dh5Base.close()
        
        Xh5Base = h5py.File( Xh5Path, 'r' )
        Xh5 = Xh5Base['X']
        Xh5.read_direct(self.X)
        Xh5Base.close()
        
        # creating shared memmory for update matrixes
        self.XmpArrUpdateList = []
        self.XUpdateList = []
        for c_list in [range(self.shape_cols)[k::self.prcountCol] for k in xrange(self.prcountCol)]:
            cols_len = len(c_list)
            self.XmpArrUpdateList.append( mp.RawArray('f', self.shape_dim * cols_len) )
            self.XUpdateList.append( np.ndarray((self.shape_dim, cols_len), buffer=self.XmpArrUpdateList[-1], dtype='f') )
        
        self.DmpArrUpdateList = []
        self.DUpdateList = []
        for r_list in [range(self.shape_rows)[k::self.prcountRow] for k in range(self.prcountRow)]:
            rows_len = len(r_list)
            self.DmpArrUpdateList.append( mp.RawArray('f', rows_len * self.shape_dim ) )
            self.DUpdateList.append( np.ndarray(( rows_len, self.shape_dim ), buffer=self.DmpArrUpdateList[-1], dtype='f') )
        
        
        
        if DEBUG_TIME: avtm = 0
        iter = 0
        errs = []
        while iter < self.opts.max_iter:
            print '\n################# iter = %s #################'%(iter)
            if DEBUG_TIME: tb = time.time()
            
            self.XUpdate()
            self.DUpdate()
            
            if DEBUG_TIME: tb_beta = time.time()
            errs.append(self.BetaDivergence())
            if DEBUG_TIME: print '\nTime of BetaDivergence parallel = %s\n'%(time.time()-tb_beta)
            
            if errs[-1] == np.nan:
                print('NaN!')
                break
            
            #sparsy = sum(sum(lambda_factor.*X));
            delta = np.inf #inf
            if iter > 1:
                delta = (errs[iter-1]-errs[iter])/errs[iter-1];
            
            
            print '\niter = %d, error = %f, delta = %f\n'%(iter, errs[iter], delta);
            if delta < opts.conv_value:
                break
            iter += 1
            
            if DEBUG_TIME:
                te = time.time()
                avtm = ( avtm*(iter-1) + (te - tb) )/iter
                print 'Average time = %s\n'%(avtm)
                print 'Time of current iteration = %s\n'%(te - tb)
    
    
    def SplitY(self, lilY, prcntRow, prcntCol):
        rN, cN = lilY.shape
        
        #get nonzero elements and indexes from lilY
        rows = []
        cols = []
        values = []
        for k,row in enumerate(lilY.rows.tolist()):
            for col in row:
                rows.append(k)
                cols.append(col)
        for vl in lilY.data:
            for v in vl:
                values.append(v)
        
        #lilY will be splitted in to 'prcnt' matrices that contain rows 
        #and in to 'prcnt' matrices that contain cols
        shape_list_r = [len(range(rN)[k::prcntRow]) for k in range(prcntRow)]
        shape_list_c = [len(range(cN)[k::prcntCol]) for k in range(prcntCol)]
        user_lists = [[] for k in xrange(prcntRow)]
        song_lists = [[] for k in xrange(prcntRow)]
        item_lists = [[] for k in xrange(prcntRow)]
        
        #rows
        YMats_row = []
        
        for row,col,val in zip(rows,cols,values):
            pr = row%prcntRow
            row_r = row//prcntRow
        
            user_lists[pr].append(col)
            song_lists[pr].append(row_r)
            item_lists[pr].append(val)
        
        for k in xrange(prcntRow):
            lilM = lil_matrix( (shape_list_r[k], cN), dtype=np.dtype('>i4') )
            for (s,u,itm) in zip(song_lists[k], user_lists[k], item_lists[k]):
                lilM[s,u] = itm
            YMats_row.append(lilM.tocsr())
        
        user_lists = [[] for k in xrange(prcntCol)]
        song_lists = [[] for k in xrange(prcntCol)]
        item_lists = [[] for k in xrange(prcntCol)]
        
        #cols
        YMats_col = []
        for row,col,val in zip(rows,cols,values):
            pr = col%prcntCol
            col_r = col//prcntCol
            
            song_lists[pr].append(row)
            user_lists[pr].append(col_r)
            item_lists[pr].append(val)
        
        
        for k in xrange(prcntCol):
            lilM = lil_matrix( (shape_list_c[k], rN), dtype=np.dtype('>i4') )
            for (s,u,itm) in zip(song_lists[k], user_lists[k], item_lists[k]):
                lilM[u,s] = itm
            YMats_col.append(lilM.tocsr())
        
        return YMats_row, YMats_col
    
    
    def getDShape(self):
        # get dim and rows
        Dh5Base = h5py.File( self.Dh5Path, 'r' )
        rows, dim = Dh5Base['D'].shape
        Dh5Base.close()
        return [rows, dim]
    
    
    def XUpdate(self):
        print 'Updating matrix X...'
        
        # get cols and rows
        rows = self.shape_rows
        cols = self.shape_cols
        dim = self.shape_dim
        
        prcnt = self.prcountCol
        cols_list = [range(cols)[k::prcnt] for k in xrange(prcnt)]
        #print 'rows = %s, cols = %s, dim = %s'%(rows, cols, dim)
        
        #parallel processing
        if DEBUG_TIME: tb = time.time()
        print 'Starting %s parallel processes...'%(prcnt)
        params = (self.opts, (rows, cols, dim))
        if prcnt > 1:
            pr_list = []
            for k in xrange(prcnt):
                p = Process(target=PgradNgradX, args=(self.Yc_Mlist[k], self.XmpArr, self.DmpArr, params, cols_list[k], k, self.XmpArrUpdateList[k]))
                p.start()
                pr_list.append(p)
            
            #waight for processes
            for k in xrange(prcnt):
                pr_list[k].join()
        else:
            PgradNgradX(self.Yc_Mlist[0], self.XmpArr, self.DmpArr, params, cols_list[0], 0, self.XmpArrUpdateList[0])
            
        print 'All parallel processes have been complieted.'
        if DEBUG_TIME: print '\nTime of XUpdate parallel = %s\n'%(time.time()-tb)
        
        #update X
        if DEBUG_TIME: tb = time.time()
        print 'Merge of update matrixes for X...'
        for k in xrange(prcnt):
            self.X[:,cols_list[k]] = self.XUpdateList[k][:,]
        
        Xh5Base = h5py.File( self.Xh5Path, 'w' )
        Xh5 = Xh5Base.create_dataset( 'X', data=self.X )
        Xh5Base.close()
        print 'Sucsessfull update of X'
        if DEBUG_TIME: print '\nTime of XUpdate merge = %s\n'%(time.time()-tb) 
    
    
    def DUpdate(self):
        print 'Updating matrix D...'
        
        # get cols and rows
        rows = self.shape_rows
        cols = self.shape_cols
        dim = self.shape_dim
        
        prcnt = self.prcountRow
        rows_list = [range(rows)[k::prcnt] for k in range(prcnt)]
        #print 'rows = %s, cols = %s, dim = %s'%(rows, cols, dim)
        
        #parallel processing
        if DEBUG_TIME: tb = time.time()
        print 'Starting %s parallel processes...'%(prcnt)
        params = (self.opts, (rows, cols, dim))
        if prcnt > 1:
            pr_list = []
            for k in xrange(prcnt):
                p = Process(target=PgradNgradD, args=(self.Yr_Mlist[k], self.XmpArr, self.DmpArr, params, rows_list[k], k, self.DmpArrUpdateList[k]))
                p.start()
                pr_list.append(p)
            
            #waight for processes
            for k in xrange(prcnt):
                pr_list[k].join()
        else:
            PgradNgradD(self.Yr_Mlist[0], self.XmpArr, self.DmpArr, params, rows_list[0], 0, self.DmpArrUpdateList[0])
        
        print 'All parallel processes have been complieted.'
        if DEBUG_TIME: print '\nTime of DUpdate parallel = %s\n'%(time.time()-tb) 
        
        #update D
        if DEBUG_TIME: tb = time.time()
        print 'Get information from work dir and update D...'
        for k in xrange(prcnt):
            self.D[rows_list[k],] = self.DUpdateList[k][:,]
        
        
        Dh5Base = h5py.File( self.Dh5Path, 'w' )
        Dh5 = Dh5Base.create_dataset( 'D', data=self.D)
        Dh5Base.close()
        print 'Sucsessfull update of D'
        if DEBUG_TIME: print '\nTime of DUpdate merge = %s\n'%(time.time()-tb)
    
    
    def BetaDivergence(self):
        b_queue = Queue()
        rows = self.shape_rows
        cols = self.shape_cols
        dim = self.shape_dim
        mShapes = (rows, cols, dim)
        
        prcnt = self.prcountCol
        cols_list = [range(cols)[k::prcnt] for k in xrange(prcnt)]
        if prcnt > 1:
            pr_list = []
            for k in xrange(prcnt):
                p = Process(target=BetaDivergenceParallel, args=(self.Yc_Mlist[k], self.XmpArr, self.DmpArr, mShapes, self.opts.beta, self.opts.eps, cols_list[k], k, b_queue))
                p.start()
                pr_list.append(p)
        else:
            BetaDivergenceParallel(self.Yc_Mlist[k], self.XmpArr, self.DmpArr, mShapes, self.opts.beta, self.opts.eps, cols_list[k], k, b_queue)
        
        # get results from the queue
        res = 0.0; k = 0
        while k < prcnt:
            res += b_queue.get(True)
            k += 1
        
        if prcnt > 1:
            #waight for processes
            for k in xrange(prcnt):
                pr_list[k].join()
        
        return res

'''
Multiprocessing functions
'''
def PgradNgradX(Y_cols_csr, XmpArr, DmpArr, params, cols_list, ind, XmpArrUpdate):
    ################## MatLab code ##################
    '''
    %update X
    DX = D*X;
    if opts.beta<2
        DX(DX<eps) = eps;
    end
    pgradX = D'*(DX.^(opts.beta-1));
    ngradX = D'*((DX.^(opts.beta-2)).*Y);
    X = X .* ngradX ./ (pgradX + lambda_factor);
    '''
    #################################################
    opts, mShapes = params
    (rows, cols, dim) = mShapes
    
    # mp_arr and arr share the same memory
    # make it two-dimensional # D and arr share the same memory
    D = np.ndarray((rows, dim), buffer=DmpArr, dtype='f')
    X_tr = np.ndarray((dim, cols), buffer=XmpArr, dtype='f').transpose()
    
    cols_len = len(cols_list)
    
    updateXh5_array = np.ndarray((dim, cols_len), buffer=XmpArrUpdate, dtype='f')
    
    #inXh5 = iter(X_tr[cols_list,...])
    inXh5 = iter(X_tr)
    if cols_len>1:
        step = cols_list[1]-cols_list[0]
        inXh5slice = islice(inXh5, cols_list[0], cols_list[-1]+step, step)
    else:
        inXh5slice = islice(inXh5, cols_list[0], cols_list[0]+1, 1)
    
    for k,Xh5col in enumerate(inXh5slice):
        #if (k*100)/cols_len < ((k+1)*100)/cols_len: print 'Process N%s  %s'%(ind, ((k+1)*100)/cols_len) + '%'
        Ycol = Y_cols_csr.getrow(k)
        Xh5col = np.reshape(Xh5col, (dim,1))
        _y = np.reshape( np.dot(D, Xh5col),  (1,rows) )
        
        if opts.beta<2:
            _y[_y<opts.eps] = opts.eps
        
        pgradX = np.dot( (_y)**(opts.beta-1), D ).transpose()
        ngradX = ( (csr_matrix( Ycol.multiply( (_y)**(opts.beta-2) ).astype(np.float32) ).dot(D)).transpose() ).astype(np.float32)
        updateXh5_array[:,k] = Xh5col[:,0] * ngradX[:,0] / (pgradX[:,0] + opts.lmbd)
    
    


def PgradNgradD(Y_rows_csr, XmpArr, DmpArr, params, rows_list, ind, DmpArrUpdate):
    ################## MatLab code ##################
    '''
    %update D
    DX = D*X;
    if opts.beta<2
        DX(DX<eps) = eps;
    end
    pgradD = (DX.^(opts.beta-1))*X';
    pgradD(pgradD<eps)=eps;
    ngradD = ((DX.^(opts.beta-2)).*Y)*X';
    D = D .* ngradD ./ pgradD;
    '''
    #################################################
    opts, mShapes = params
    (rows, cols, dim) = mShapes
    
    # mp_arr and arr share the same memory
    # make it two-dimensional # D and arr share the same memory
    D = np.ndarray((rows, dim), buffer=DmpArr, dtype='f')
    X = np.ndarray((dim, cols), buffer=XmpArr, dtype='f')
    X_tr = X.transpose()
    
    
    rows_len = len(rows_list)
    
    updateDh5_array = np.ndarray((rows_len, dim), buffer=DmpArrUpdate, dtype='f')
    
    #inDh5 = iter(D[rows_list,...])
    inDh5 = iter(D)
    if rows_len>1:
        step = rows_list[1]-rows_list[0]
        inDh5slice = islice(inDh5, rows_list[0], rows_list[-1]+step, step)
    else:
        inDh5slice = islice(inDh5, rows_list[0], rows_list[0]+1, 1)
    
    for k,Dh5row in enumerate(inDh5slice):
        #if (k*100)/rows_len < ((k+1)*100)/rows_len: print 'Process N%s  %s'%(ind, ((k+1)*100)/rows_len) + '%'
        Yrow = Y_rows_csr.getrow(k)
        _y = np.reshape( np.dot(Dh5row,X), (1,cols) )
        if opts.beta<2:
            _y[_y<opts.eps] = opts.eps
        
        
        pgradD = np.dot( ( (_y)**(opts.beta-1) ), X_tr )
        pgradD[pgradD<opts.eps] = opts.eps
        
        ngradD = ( Yrow.multiply( (_y)**(opts.beta-2) ).astype(np.float32) ).dot(X_tr)
        ngradD = np.squeeze(np.asarray(ngradD))
        updateDh5_array[k,] = Dh5row * ngradD / pgradD[0,]
    


def BetaDivergenceParallel(Y_cols_csr, XmpArr, DmpArr, mShapes, beta, eps, cols_list, ind, rp_queue):
    
    # mp_arr and arr share the same memory
    # make it two-dimensional # D and arr share the same memory
    (rows, cols, dim) = mShapes
    D = np.ndarray((rows, dim), buffer=DmpArr, dtype='f')
    X_tr = np.ndarray((dim, cols), buffer=XmpArr, dtype='f').transpose()
    
    inXh5 = iter(X_tr[cols_list,...])
    
    
    r = 0.0
    if beta == 0:
        for k,Xh5col in enumerate(inXh5):
            _y = np.reshape( np.dot(D,Xh5col), (1,rows))
            _y[_y<eps] = eps
            _y = 1./_y
            Ycol = Y_cols_csr.getrow(k)
            
            div = Ycol.multiply(_y)
            r += np.sum (div - np.log(div+eps) - 1)
    elif beta == 1:
        for k,Xh5col in enumerate(inXh5):
            _y = np.reshape( np.dot(D,Xh5col), (1,rows))
            _y[_y<eps] = eps
            _my = 1./_y
            Ycol = Y_cols_csr.getrow(k)
            
            r += np.sum( Ycol.multiply(np.log(Ycol.multiply(_my)+eps)) - Ycol.todense() + _y )
    elif beta == 2:
        for k,Xh5col in enumerate(inXh5):
            _y = np.reshape( np.dot(D,Xh5col), (1,rows))
            _y[_y<eps] = eps
            Ycol = Y_cols_csr.getrow(k)
            
            r += 0.5 * np.sum( (Ycol.todense() - _y)**2 )
    
    rp_queue.put(r)

