import os
import struct
from .base import BaseANN
import h5py
import numpy as np
from \
  sklearn import preprocessing
import numpy


class NSSG(BaseANN):
  def __init__(self, metric, param):
    metric = str(metric)
    self.name = 'SSG(%s)' % (metric)
    self._metric = metric
    self._paramE = param['paramE']
    self._paramS = param['paramS']
    self._paramQ = param['paramQ']
  def fit(self, X):
    if X.dtype != numpy.float32:
     X = X.astype(numpy.float32)
    print('SSG: start indexing...')
    dim = len(X[0])
    print('SSG: # of data=' + str(len(X)))
    print('SSG: dimensionality=' + str(dim))
    index_dir = 'indexes'
    if not os.path.exists(index_dir):
      os.makedirs(index_dir)
      aIndex = os.path.join(index_dir, 'SSG' )
      if self._metric == 'E':
        X_normalized = preprocessing.normalize(X, norm='l2')
        fvecs_dir = 'fvecs'
        if not os.path.exists(fvecs_dir):
          os.makedirs(fvecs_dir)
        fvecs = os.path.join(fvecs_dir, 'base.fvecs')
        with open(fvecs, 'wb') as fp:
          for y in X_normalized:
            d = struct.pack('I', y.size)
            fp.write(d)
            for x in y:
              a = struct.pack('f', x)
              fp.write(a)
      else:
        fvecs_dir = 'fvecs'
        if not os.path.exists(fvecs_dir):
          os.makedirs(fvecs_dir)
        fvecs = os.path.join(fvecs_dir, 'base.fvecs')
        with open(fvecs, 'wb') as fp:
          for y in X:
            d = struct.pack('I', y.size)
            fp.write(d)
            for x in y:
              a = struct.pack('f', x)
              fp.write(a)
      parmEfanna = self._paramE
      parmSSG = self._paramS
      graph_dir = 'graph'
      SG = os.path.join(aIndex, 'grp')
      if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
      KNNG = os.path.join(graph_dir, 'KNNG-' + str(parmEfanna[0]) + '-' + str(parmEfanna[1]) + '-' + str(
        parmEfanna[2]) + '-' + str(parmEfanna[3]) + '-' + str(parmEfanna[4]) + '.graph')
      cmds = '/home/app/SSG/ssg-knng ' + str(fvecs) + ' ' + str(KNNG) + ' ' + str(
        parmEfanna[0]) + ' ' + str(parmEfanna[1]) + ' ' + str(parmEfanna[2]) + ' ' + str(
        parmEfanna[3]) + ' ' + str(
        parmEfanna[4]) + \
             '&& /home/app/SSG/build/tests/test_ssg_index ' + str(fvecs) + ' ' + str(KNNG) + ' ' + str(
        parmSSG[0]) + ' ' + str(parmSSG[1]) + ' ' + str(parmSSG[2]) + ' ' + str(SG)
      os.system(cmds)
  def set_query_arguments(self, L):
    self._L = L
  def query(self, v, n):
    if v.dtype != numpy.float32:
      v = v.astype(numpy.float32)
    index_dir = 'indexes'
    aIndex = os.path.join(index_dir, 'SSG')
    parmQue = self._paramQ
    fvecs = os.path.join('fvecs', 'base.fvecs')
    query = self._L
    SG = os.path.join(aIndex, 'grp')
    RE = os.path.join(aIndex, 'result')
    cmds = '/home/app/SSG/build/tests/test_ssg_optimized_search ' + str(fvecs) + ' ' + str(query) + ' ' + str(
      SG) + ' ' + str(
      v) + ' ' + str(n) + ' ' + str(
      RE)+ ' ' + str(parmQue[2])
    data = np.loadtxt(RE)
    with h5py.File("data.hdf5", "w") as f:
      dset = f.create_dataset("data", data.shape, dtype=data.dtype)
    with h5py.File("data.hdf5", "a") as f:
      dset = f["data"]
      dset[...] = data
    with h5py.File("data.hdf5", "r") as f:
      dset = f["data"]
      data = dset[:]
    result = data
    return result
