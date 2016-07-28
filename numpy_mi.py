#!/usr/bin/python
# Code by Peter Kasson, 2014

"""Python port of mutual information calculation."""

import gflags
import numpy
import sys

def entropy(arr):
  return numpy.sum([-a * numpy.log(a) if a else 0 for a in arr])

def mi_fast(arr1, arr2, nbins=32):
  """
  Performs MI calculation between two equal length arrays
  :param arr1: Array 1 to perform MI
  :param arr2: Array 2 to perform MI
  :param nbins: number of bins to use for histogram
  :return: returns MI value
  """
  xy_c = numpy.histogram2d(arr1, arr2, nbins)[0]
  x_c = numpy.sum(xy_c, 1)
  y_c = numpy.sum(xy_c, 0)
  nvals = len(arr1)
  Hxy = entropy(numpy.ravel(xy_c)) / nvals + numpy.log(nvals)
  Hx = entropy(numpy.ravel(x_c)) / nvals + numpy.log(nvals)
  Hy = entropy(numpy.ravel(y_c)) / nvals + numpy.log(nvals)
  return Hx + Hy - Hxy

def runMI_all(infilename, outfilename, nbins=32):
  """
  Runs mutual information on all columns of provided MI matrix.
  :param infilename: Numpy file containing matrix to calculate MI
  :param outfilename: Outfile to save MI matrix as npy file
  :param nbins: Number of bins to use for histogramming and digitization
  """
  dat = numpy.load(infilename)
  ncols = dat.shape[1]
  mimat = numpy.zeros([ncols, ncols])
  for i in range(ncols):
    for j in range(i, ncols):
        mimat[i, j] = mi_fast(dat[:, i], dat[:, j], nbins)
        mimat[j, i] = mimat[i, j]
  numpy.save(outfilename, mimat)



if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  gflags.DEFINE_string('infile', '', 'input matrix as npy file for which to perform MI')
  gflags.DEFINE_string('outfile', 'mi', 'output file base name')
  gflags.DEFINE_integer('numbins', 32,
                        'Number of bins')
  argv = FLAGS(sys.argv)
  runMI_all(FLAGS.infile, FLAGS.outfile, FLAGS.numbins)
