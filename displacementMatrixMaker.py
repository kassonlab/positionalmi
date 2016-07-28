"""
Calculates displacment matrix from a series of gromacs trajectories for subsequent
mutual information analysis
"""
import gflags
import sys
import glob
from msmbuilder import Trajectory #Requires msmbuilder 2.8 (http://msmbuilder.org/legacy/)l
import numpy as np
import os
import uuid
import cPickle as pickle

def displacement_process_launcher(clusteropts):
  """
  Launches the displacement threads
  :param clusteropts: parameters for displacement for each thread
  outputdir: output directory
  sourcedir: source directory containing the xtcs
  sourcepdb: pdb structure to calculate displacement
  index: gromacs index file for displacement atoms expects:
    1. Atoms in system
  alignongroup: If True, will look for an index file that consists of:
    1. Atoms in the system
    2. Atoms to align on
  :return: 1. Produces a series of numpy files consisting of a displacement matrix for each trajectory along with
  Fileorder.p which is a list of the order of trajectories in correspondence with the produced displacement
  2. Will also produce a displacement matrix of all matrices combined
  """
  infiles = glob.glob(clusteropts['sourcedir'])
  infiles.sort()
  pickle.dump( infiles, open( os.path.join(clusteropts['outputdir'],"Fileorder.p"), "wb"))
  displacements = [0] * len(infiles)
  i = 0
  for fname in infiles:
    displacement_process_thread(fname, clusteropts, i, displacements)
    i = + 1
  result = None
  for displacement in displacements:
      if result is None:
          result = displacement
      else:
          result = np.concatenate((result, displacement), axis=0)
  np.save(os.path.join(clusteropts['outputdir'],'all_displacments.np'), result)

def calculate_aligned_displacement(traj, tmp, index_file, align, gromacsbinpath='/usr/local/gromacs/bin'):
  """
  Takes a trajectory, does a rigid-body fit, returns aligned displacement
  :param traj: The trajectory file to calculate alignment
  :param tmp: Unique code to avoid file collisions of temporary files
  :param index_file: Index file to perform alignment and output
  :param align: Flag where if True, alignment is performed on second index entry
  :param gromacsbinpath: Location of gromacs install
  :return: returns a displacement numpy matrix
  """
  t2 = Trajectory.Trajectory(traj)
  (natoms, ndim) = np.shape(t2['XYZList'][0])
  t2['XYZList'] = np.zeros((1, natoms, ndim))
  t2['XYZList'][0] = traj['XYZList'][0]
  t2.SaveToPDB('displacetmp_%s.pdb' % tmp)
  traj.SaveToXTC('displacetmp_%s.xtc' % tmp)
  cmdstr = ('echo 0 0| %s -s displacetmp_%s.pdb -f displacetmp_%s.xtc '
            '-o displacealigned_%s.xtc -fit rot+trans' %
            (os.path.join(gromacsbinpath,'trjconv'), tmp, tmp, tmp))
  if align:
    # Whole structure alignment modified to alignment atoms if align is chosen
    cmdstr = ('echo 1 %s| %s -s displacetmp_%s.pdb -f displacetmp_%s.xtc '
              '-o displacealigned_%s.xtc -fit rot+trans -n %s' %
              (str(2), os.path.join(gromacsbinpath,'trjconv'), tmp, tmp, tmp, index_file))
  os.system(cmdstr)
  T2 = Trajectory.Trajectory.LoadFromXTC('displacealigned_%s.xtc' % tmp,
                                         PDBFilename='displacetmp_%s.pdb' % tmp)
  #clean up
  os.system('rm displacetmp_%s*; rm displacealigned_%s.xtc' % (tmp, tmp))
  displacement = np.zeros(T2['XYZList'].shape[:2])
  for i in range(T2['XYZList'].shape[0]):
    displacement[i] = np.sqrt(np.sum((T2['XYZList'][i, :, :]
                                      - T2['XYZList'][0, :, :]) ** 2, 1))
  return displacement

def displacement_process_thread(fname, clusteropts, i, displacements, gromacsbinpath='/usr/local/gromacs/bin'):
  """
  Calculates and writes a npy displacement matrix from a trajectory
  :param fname: File name of the trajectory on which to calculate displacement
  :param clusteropts: Options for displacement (see displacementProcessLauncher)
  :param i: The number to append at the end of the resulting file (used with Fileorder.p to track trajectories)
  :param displacements: Shared list to report the name of the resulting displacement matrix
  :param gromacsbinpath: Location of gromacs install
  :return: A npy file containing the displacement matrix of frames x atoms
  """
  tmp = uuid.uuid4()
  usePDB = clusteropts['sourcepdb']
  useXTC = fname
  erase = False
  if clusteropts['index'] is not None and clusteropts['index'] is not '':
      erase = True
      usePDB = 'sourcepdb_%s.pdb' % tmp
      useXTC = fname+'_%s.xtc' % tmp
      cmdstr = ('echo 0 0| %s -s %s -f %s '
              '-o %s -n %s' % (os.path.join(gromacsbinpath,'trjconv'), 
                               clusteropts['sourcepdb'], clusteropts['sourcepdb'], usePDB, clusteropts['index']))
      os.system(cmdstr)
      cmdstr = ('echo 0 0| %s -s %s -f %s '
              '-o %s -n %s' 
              % (os.path.join(gromacsbinpath,'trjconv'), 
                 clusteropts['sourcepdb'], fname, useXTC, clusteropts['index']))
      os.system(cmdstr)     
  traj = Trajectory.Trajectory.LoadFromXTC(XTCFilenameList=useXTC,PDBFilename=usePDB)
  if erase is True:
      os.system('rm %s' % usePDB)
      os.system('rm %s' % useXTC)
  displacement = calculate_aligned_displacement(traj, tmp, clusteropts['index'], clusteropts['alignongroup'])
  np.save(os.path.join(clusteropts['outputdir'],fname.split('/')[-1]+'_displacment.np'), displacement)
  displacements[i] = displacement





if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  gflags.DEFINE_string('sourcedir', 'foo',
    'Directory containing input xtc ')
  gflags.DEFINE_string('sourcepdb', 'topol.tpr',
    'PDB file to use as template for initial position')
  gflags.DEFINE_string('outputdir', 'foo',
    'Base directory for output')
  gflags.DEFINE_string('index', '',
    'Gromacs ndx file of atoms to calculate displacement on')
  gflags.DEFINE_boolean('alignongroup', False, 
                        'Align on group. If Selected then index should consist of two groups: index of all atoms and'
                        'index to align on')
  argv = FLAGS(sys.argv)
  clusteropts = dict()
  clusteropts['outputdir'] = FLAGS.outputdir
  clusteropts['sourcedir'] = FLAGS.sourcedir
  clusteropts['sourcepdb'] = FLAGS.sourcepdb
  clusteropts['index'] = FLAGS.index
  clusteropts['alignongroup'] = FLAGS.alignongroup
  displacement_process_launcher(clusteropts)
