#!/usr/bin/env python2.7
#########################################################################################################
# This is a script for computing the number of voronoi edges shared by different lipids in a ternary
# mixture of DPPC, DIPC, and CHOL in the MARTINI coarse-grained model.
#
# This script uses the (x,y) coordinates of lipid head groups to perform Voronoi tessellation
# The outputs from this script are:
# (1) counts of voronoi edges between lipids in each leaflet
# (2) tertiary (lipid1 + lipid2 + lipid3) mixing entropy and binary (lipid1 + lipid2) mixing entropy, ignoring lipid3 (CHOL)
# (3) the ratio of #CHOL-DPPC / #CHOL-DIPC Voronoi edges
#
# The selection of lipids can be changed by changing the definitions of lipid[1-3] and sel[1-3]
# The system topology and trajectory can be set by changing the definitions of top and traj
#
# Usage is: ternary_lipid_voronoi_edges.py <side>
# where <side> is "up" or "down", analyzing either the upper or lower leaflet of a MARTINI DPPC:DIPC:CHOL#
# membrane using induces of the membrane as if the system was built using the insane.py tool
#########################################################################################################

import string
import struct
import sys
import numpy as np
from MDAnalysis import *
import MDAnalysis
import MDAnalysis.lib.distances
import time
import numpy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import math
import MDAnalysis.lib.NeighborSearch as NS
from scipy.spatial import Voronoi, voronoi_plot_2d
import multiprocessing as mp


print 'Initiating Voroni Tesselation'

top = '../confout.gro'
traj = '../run.xtc'
side = sys.argv[1] # "up" for upper leaflet "down" for lower leaflet

u = MDAnalysis.Universe(top,traj)

#Lipid Residue names
lipid1 ='DPPC'
lipid2 ='DIPC'
lipid3 ='CHOL'

# Atom selections
sel1 = 'resname %s and name PO4'%lipid1
sel2 = 'resname %s and name PO4'%lipid2
sel3 = 'resname %s and name ROH'%lipid3

# Identify number of residues in each lipid and extract only he top residues (for now)

#Frames to be calculated
#end_f   = 11001 #8001
end_f = u.trajectory.n_frames
print end_f
start_f = 0
skip    = 10

# Number of processors to use in multiprocessing
nprocs = 16

frames = np.arange(start_f, end_f)[::skip]
n_frames = len(frames)

######################################################################################

# Make empty arrays to hold contact counts
ens_Lpd1_Lpd1 = np.zeros(n_frames)
ens_Lpd2_Lpd2 = np.zeros(n_frames)
ens_Lpd3_Lpd3 = np.zeros(n_frames)
ens_Lpd1_Lpd2 = np.zeros(n_frames)
ens_Lpd1_Lpd3 = np.zeros(n_frames)
ens_Lpd2_Lpd3 = np.zeros(n_frames)

ens_sum_bonds = np.zeros(n_frames)
ens_avg_bonds = np.zeros(n_frames)

ens_mix_entropy = np.zeros(n_frames)

######################################################################################
def voronoi_tessel(ts):
    # set the time step
    print 'Frame %i in %i'%(ts, end_f)
    u = MDAnalysis.Universe(top,traj)
    u.trajectory[ts-1] # What the hell is MDAnalysis doing...? This changes the frame to frame "ts"

# Select atoms within this particular frame
    lpd1_atoms = u.select_atoms(sel1)
    lpd2_atoms = u.select_atoms(sel2)
    lpd3_atoms = u.select_atoms(sel3)
    
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms

    # atoms in the upper leaflet as defined by insane.py or the CHARMM-GUI membrane builders
    # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
    if side == 'up':
        lpd1i = lpd1_atoms[:((num_lpd1)/2)]
        lpd2i = lpd2_atoms[:((num_lpd2)/2)]

        lipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms)

        lpd3i = ns_lipids.search(lipids,15.0)
    elif side == 'down':
        lpd1i = lpd1_atoms[((num_lpd1)/2):]
        lpd2i = lpd2_atoms[((num_lpd2)/2):]

        lipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms)

        lpd3i = ns_lipids.search(lipids,15.0)
    lpd_atms = lpd1i + lpd2i + lpd3i


#Extracting the coordinates
    Pxyz = lpd_atms.coordinates()
    Pxy = []
    for l in range(0,len(Pxyz)) :
        Pxy.append([Pxyz[l][0],Pxyz[l][1]])
#Extracting xy coordinates and residue names

    atm_list = []
    for a in range(0, len(Pxyz)):
        atm_list.append([Pxyz[a][0],Pxyz[a][1],lpd_atms[a].resname])

#Introducing PBC
    x_box = u.dimensions[0]
    y_box = u.dimensions[1]

    xplus   = []
    xminus  = []
    xyplus  = []
    xyminus = []

    for atm in range(0 ,len(atm_list)):
        xplus.append([atm_list[atm][0]+x_box,atm_list[atm][1],atm_list[atm][2]])
        xminus.append([atm_list[atm][0]-x_box,atm_list[atm][1],atm_list[atm][2]])

    atm_list_px = atm_list + xplus + xminus

    for atm in range(0 ,len(atm_list_px)):
        xyplus.append([atm_list_px[atm][0],atm_list_px[atm][1]+y_box,atm_list_px[atm][2]])
        xyminus.append([atm_list_px[atm][0],atm_list_px[atm][1]-y_box,atm_list_px[atm][2]])

    atm_list_p = atm_list_px + xyplus + xyminus


    atm_xy = []
    for i in range(0,len(atm_list_p)) :
        atm_xy.append([atm_list_p[i][0],atm_list_p[i][1]])


    vor = Voronoi(atm_xy)
    vor_s = Voronoi(Pxy)
    vertices       = vor.vertices

    ridge_points = vor.ridge_points
    Lpd1_Lpd1_I = 0
    Lpd2_Lpd2_I = 0
    Lpd3_Lpd3_I = 0
    Lpd1_Lpd2_I = 0
    Lpd1_Lpd3_I = 0
    Lpd2_Lpd3_I = 0

    Lpd1_Lpd1_E = 0
    Lpd2_Lpd2_E = 0
    Lpd3_Lpd3_E = 0
    Lpd1_Lpd2_E = 0
    Lpd1_Lpd3_E = 0
    Lpd2_Lpd3_E = 0

    r_length  = len(ridge_points)


  
    for k in range (0,r_length) :
        ridge_k = ridge_points[k]
        Li = atm_list_p[int(ridge_k[0])]
        Lj = atm_list_p[int(ridge_k[1])]

#Lipids INSIDE the box 

        if 0 < Li[0] < x_box and 0 < Li[1] < y_box and 0 < Lj[0] < x_box and 0 < Lj[1] < y_box :
        
            if Li[2] == lipid1 and Lj[2] == lipid1:
                Lpd1_Lpd1_I = Lpd1_Lpd1_I + 1
                
            if Li[2] == lipid2 and Lj[2] == lipid2:
                Lpd2_Lpd2_I = Lpd2_Lpd2_I + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid3:
                Lpd3_Lpd3_I = Lpd3_Lpd3_I + 1
                
            if Li[2] == lipid1 and Lj[2] == lipid2:
                Lpd1_Lpd2_I  = Lpd1_Lpd2_I + 1
                
            if Li[2] == lipid2 and Lj[2] == lipid1:
                Lpd1_Lpd2_I = Lpd1_Lpd2_I + 1

            if Li[2] == lipid1 and Lj[2] == lipid3:
                Lpd1_Lpd3_I  = Lpd1_Lpd3_I + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid1:
                Lpd1_Lpd3_I = Lpd1_Lpd3_I + 1

            if Li[2] == lipid2 and Lj[2] == lipid3:
                Lpd2_Lpd3_I  = Lpd2_Lpd3_I + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid2:
                Lpd2_Lpd3_I = Lpd2_Lpd3_I + 1                
                
#Lipids at the EDGE of the box                
                
        if 0 <= Li[0] < x_box and 0 <= Li[1] < y_box or 0 <= Lj[0] < x_box and 0 <= Lj[1] < y_box :

            if Li[2] == lipid1 and Lj[2] == lipid1:
                Lpd1_Lpd1_E = Lpd1_Lpd1_E + 1
                
            if Li[2] == lipid2 and Lj[2] == lipid2:
                Lpd2_Lpd2_E = Lpd2_Lpd2_E + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid3:
                Lpd3_Lpd3_E = Lpd3_Lpd3_E + 1
                
            if Li[2] == lipid1 and Lj[2] == lipid2:
                Lpd1_Lpd2_E  = Lpd1_Lpd2_E + 1
                
            if Li[2] == lipid2 and Lj[2] == lipid1:
                Lpd1_Lpd2_E = Lpd1_Lpd2_E + 1

            if Li[2] == lipid1 and Lj[2] == lipid3:
                Lpd1_Lpd3_E  = Lpd1_Lpd3_E + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid1:
                Lpd1_Lpd3_E = Lpd1_Lpd3_E + 1

            if Li[2] == lipid2 and Lj[2] == lipid3:
                Lpd2_Lpd3_E  = Lpd2_Lpd3_E + 1
                
            if Li[2] == lipid3 and Lj[2] == lipid2:
                Lpd2_Lpd3_E = Lpd2_Lpd3_E + 1    

                
#Total = LipidsInside + (Lipids including EDGES - Lipids Inside)/2 -----> Correction for over counting the lipids in periodic images
    Lpd1_Lpd1 = Lpd1_Lpd1_I + (Lpd1_Lpd1_E - Lpd1_Lpd1_I)/2
    Lpd2_Lpd2 = Lpd2_Lpd2_I + (Lpd2_Lpd2_E - Lpd2_Lpd2_I)/2
    Lpd3_Lpd3 = Lpd3_Lpd3_I + (Lpd3_Lpd3_E - Lpd3_Lpd3_I)/2
    Lpd1_Lpd2 = Lpd1_Lpd2_I + (Lpd1_Lpd2_E - Lpd1_Lpd2_I)/2
    Lpd1_Lpd3 = Lpd1_Lpd3_I + (Lpd1_Lpd3_E - Lpd1_Lpd3_I)/2
    Lpd2_Lpd3 = Lpd2_Lpd3_I + (Lpd2_Lpd3_E - Lpd2_Lpd3_I)/2

    sum_bonds = Lpd1_Lpd1 + Lpd2_Lpd2 + Lpd3_Lpd3 + Lpd1_Lpd2 + Lpd1_Lpd3 + Lpd2_Lpd3

#Considering only Similar Lipid (SL) and Dissimilar Lipid (DL) Bonds
    SL = Lpd1_Lpd1 + Lpd2_Lpd2 + Lpd3_Lpd3
    DL = Lpd1_Lpd2 + Lpd1_Lpd3 + Lpd2_Lpd3

    #Calculating Fractions    
    X_SL = float(SL)/float(sum_bonds)  #Similar Lipid
    X_DL  = float(DL)/float(sum_bonds) #Dissimilar Lipid

#Mixing Entropy
    mix_entropy = -(X_SL * np.log2(X_SL)) +( X_DL * np.log2(X_DL))

#Calculating Averages
    sum_bonds = Lpd1_Lpd1 + Lpd1_Lpd2 + Lpd2_Lpd2
    avg_bonds = float(sum_bonds)/float(len(atm_list))

#Plotting Voroni Diagrams

#   plt.figure()
#   vor_rgns = vor.regions
#   l_vor_rgns = len(vor_rgns)

#   vor_points = vor.point_region
#   l_vor_points = len(vor_points)
    
#   plt.clf()
#   voronoi_plot_2d(vor_s)

#   for p in range(0, l_vor_points):
#       rgn = vor_rgns[vor_points[p]]
#       L = atm_list_p[p]
#       if not -1 in rgn and 0 <= L[0] < x_box and 0 <= L[1] < y_box :
# #           print p , rgn
#           if L[2] == lipid1:
#              polygon = [vor.vertices[i] for i in rgn]
#              plt.fill(*zip(*polygon), color='blue')
#           if L[2] == lipid2:
#              polygon = [vor.vertices[i] for i in rgn]
#              plt.fill(*zip(*polygon), color='red')
#           if L[2] == lipid3:
#               polygon = [vor.vertices[i] for i in rgn]
#               plt.fill(*zip(*polygon), color='grey')
#   plt.savefig('img' + str('%03d' %vrn_frm) + '.png')
    return Lpd1_Lpd1, Lpd2_Lpd2, Lpd3_Lpd3, Lpd1_Lpd2, Lpd1_Lpd3, Lpd2_Lpd3, sum_bonds, avg_bonds, mix_entropy

pool = mp.Pool(processes=nprocs)
print 'Initiating multiprocessing with %i processors'%nprocs
results = pool.map(voronoi_tessel, frames)

ens_Lpd1_Lpd1_np = []
ens_Lpd2_Lpd2_np = []
ens_Lpd3_Lpd3_np = []
ens_Lpd1_Lpd2_np = []
ens_Lpd1_Lpd3_np = []
ens_Lpd2_Lpd3_np = []
ens_sum_bonds_np = []
ens_avg_bonds_np = []
ens_mix_entropy_np = []
for i in range(n_frames):
    ens_Lpd1_Lpd1_np.append(results[i][0])
    ens_Lpd2_Lpd2_np.append(results[i][1])
    ens_Lpd3_Lpd3_np.append(results[i][2])
    ens_Lpd1_Lpd2_np.append(results[i][3])
    ens_Lpd1_Lpd3_np.append(results[i][4])
    ens_Lpd2_Lpd3_np.append(results[i][5])
    ens_sum_bonds_np.append(results[i][6])
    ens_avg_bonds_np.append(results[i][7])
    ens_mix_entropy_np.append(results[i][8])

ens_Lpd1_Lpd1_np = np.asarray(ens_Lpd1_Lpd1_np)
ens_Lpd2_Lpd2_np = np.asarray(ens_Lpd2_Lpd2_np)
ens_Lpd3_Lpd3_np = np.asarray(ens_Lpd3_Lpd3_np)
ens_Lpd1_Lpd2_np = np.asarray(ens_Lpd1_Lpd2_np)
ens_Lpd1_Lpd3_np = np.asarray(ens_Lpd1_Lpd3_np)
ens_Lpd2_Lpd3_np = np.asarray(ens_Lpd2_Lpd3_np)
ens_sum_bonds_np = np.asarray(ens_sum_bonds_np)
ens_avg_bonds_np = np.asarray(ens_avg_bonds_np)
ens_mix_entropy_np = np.asarray(ens_mix_entropy_np)

# Define output file names
if side == "up":
    Lpd1_Lpd1_fn   = 'upper_Lpd1_Lpd1.dat'
    Lpd2_Lpd2_fn   = 'upper_Lpd2_Lpd2.dat'
    Lpd3_Lpd3_fn   = 'upper_Lpd3_Lpd3.dat'
    Lpd1_Lpd2_fn   = 'upper_Lpd1_Lpd2.dat'
    Lpd1_Lpd3_fn   = 'upper_Lpd1_Lpd3.dat'
    Lpd2_Lpd3_fn   = 'upper_Lpd2_Lpd3.dat'
    sum_bonds_fn   = 'upper_sum_bonds.dat'
    avg_bonds_fn   = 'upper_avg_bonds.dat'
    mix_entropy_fn = 'upper_mix_entropy.dat'

elif side == "down":
    Lpd1_Lpd1_fn   = 'lower_Lpd1_Lpd1.dat'
    Lpd2_Lpd2_fn   = 'lower_Lpd2_Lpd2.dat'
    Lpd3_Lpd3_fn   = 'lower_Lpd3_Lpd3.dat'
    Lpd1_Lpd2_fn   = 'lower_Lpd1_Lpd2.dat'
    Lpd1_Lpd3_fn   = 'lower_Lpd1_Lpd3.dat'
    Lpd2_Lpd3_fn   = 'lower_Lpd2_Lpd3.dat'
    sum_bonds_fn   = 'lower_sum_bonds.dat'
    avg_bonds_fn   = 'lower_avg_bonds.dat'
    mix_entropy_fn = 'lower_mix_entropy.dat'

#Writing Outputs
np.savetxt(Lpd1_Lpd1_fn,ens_Lpd1_Lpd1_np)
np.savetxt(Lpd2_Lpd2_fn,ens_Lpd2_Lpd2_np)
np.savetxt(Lpd3_Lpd3_fn,ens_Lpd3_Lpd3_np)
np.savetxt(Lpd1_Lpd2_fn,ens_Lpd1_Lpd2_np)
np.savetxt(Lpd1_Lpd3_fn,ens_Lpd1_Lpd3_np)
np.savetxt(Lpd2_Lpd3_fn,ens_Lpd2_Lpd3_np)
np.savetxt(sum_bonds_fn,ens_sum_bonds_np)
np.savetxt(avg_bonds_fn,ens_avg_bonds_np)
np.savetxt(mix_entropy_fn,ens_mix_entropy_np)
print 'Calculation Complete'

#Computing binary mixing entropy and cholesterol fractions
if side == "up":
    Lpd1_Lpd1 = np.loadtxt('upper_Lpd1_Lpd1.dat')
    Lpd1_Lpd2 = np.loadtxt('upper_Lpd1_Lpd2.dat')
    Lpd1_Lpd3 = np.loadtxt('upper_Lpd1_Lpd3.dat')
    
    Lpd2_Lpd2 = np.loadtxt('upper_Lpd2_Lpd2.dat')
    Lpd2_Lpd3 = np.loadtxt('upper_Lpd2_Lpd3.dat')
    
    Lpd3_Lpd3 = np.loadtxt('upper_Lpd3_Lpd3.dat')
    
    SL = Lpd1_Lpd1 + Lpd2_Lpd2
    DL = Lpd1_Lpd2
    
    X_SL = np.true_divide(SL, SL + DL)  #Similar Lipid
    X_DL = np.true_divide(DL, SL + DL) #Dissimilar Lipid
    
    mix_entropy = -((X_SL * np.log2(X_SL)) +( X_DL * np.log2(X_DL)))
    chol_fraction = Lpd1_Lpd3 / Lpd2_Lpd3
    
    np.savetxt('upper_binary_mix_entropy.dat', mix_entropy)
    np.savetxt('upper_chol_fraction.dat', chol_fraction)

if side == "down":
    Lpd1_Lpd1 = np.loadtxt('lower_Lpd1_Lpd1.dat')
    Lpd1_Lpd2 = np.loadtxt('lower_Lpd1_Lpd2.dat')
    Lpd1_Lpd3 = np.loadtxt('lower_Lpd1_Lpd3.dat')
    
    Lpd2_Lpd2 = np.loadtxt('lower_Lpd2_Lpd2.dat')
    Lpd2_Lpd3 = np.loadtxt('lower_Lpd2_Lpd3.dat')
    
    Lpd3_Lpd3 = np.loadtxt('lower_Lpd3_Lpd3.dat')
    
    SL = Lpd1_Lpd1 + Lpd2_Lpd2
    DL = Lpd1_Lpd2
    
    X_SL = np.true_divide(SL, SL + DL) #Similar Lipid
    X_DL = np.true_divide(DL, SL + DL) #Dissimilar Lipid
    
    mix_entropy = -((X_SL * np.log2(X_SL)) +( X_DL * np.log2(X_DL)))
    chol_fraction = Lpd1_Lpd3 / Lpd2_Lpd3
    
    np.savetxt('lower_binary_mix_entropy.dat', mix_entropy)
    np.savetxt('lower_chol_fraction.dat', chol_fraction)