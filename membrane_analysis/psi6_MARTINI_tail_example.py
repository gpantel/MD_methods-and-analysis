#########################################################################################################
# This script is for the analysis of Psi6 bond-orientational order in ternary DPPC:DIPC:CHOL lipid 
# bilayers. It employs a plane-fitting method to allow for a more accurate measurement of Psi6
# 
# The script uses MDAnalysis to extract the coordinates for analysis. It could be re-purposed for
# analyzing 2D bond-orientational order in any 2D or quasi-2D system if the coordinates can be
# interpreted by MDAnalysis.
# 
# The method get_side_coordinates_and_box is where edits to the particle selection may be made.
#
# Current usage is: python psi6_MARTINI_tail_example.py <side>
# where <side> is "up" or "down", analyzing either the upper or lower leaflet of a MARTINI DPPC:DIPC:CHOL# membrane using induces of the membrane as if the system was built using the insane.py tool
#########################################################################################################

import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
import sys
import multiprocessing as mp
from time import time

# input
nprocs = 16
top  = 'confout.gro'
traj = 'run.xtc'
side = sys.argv[1] # "up" for upper leaflet "down" for lower leaflet
skip = 10

u = MDAnalysis.Universe(top,traj)
n_frames = u.trajectory.n_frames
frames = np.arange(n_frames)[::skip]


# set reference vector
vref = np.array([[1,0,0],[0,0,0]])

def get_side_coordinates_and_box(frame):
    u = MDAnalysis.Universe(top,traj)
    u.trajectory[frame]

    x, y, z = u.trajectory.ts.triclinic_dimensions[0][0], u.trajectory.ts.triclinic_dimensions[1][1], u.trajectory.ts.triclinic_dimensions[2][2]
    box = np.array([x, y, z])

    ### Determining side of the bilayer CHOL belongs to in this frame
    #Lipid Residue names
    lipid1 ='DPPC'
    lipid2 ='DIPC'
    lipid3 ='CHOL'
    
    lpd1_atoms = u.select_atoms('resname %s and name PO4'%lipid1)
    lpd2_atoms = u.select_atoms('resname %s and name PO4'%lipid2)
    lpd3_atoms = u.select_atoms('resname %s and name ROH'%lipid3)
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms
    # atoms in the upper leaflet as defined by insane.py or the CHARMM-GUI membrane builders
    # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
    # this must be done because CHOL rapidly flip-flops between leaflets in the MARTINI model
    # so we must assign CHOL to each leaflet at every time step, and in large systems
    # with substantial membrane undulations, a simple cut-off in the z-axis just will not cut it
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
    
    # ID center of geometry coordinates for cholesterol on indicated bilayer side
    lpd3_coords = np.zeros((len(lpd3i.resnums),3))
    for i in np.arange(len(lpd3i.resnums)):
        resnum = lpd3i.resnums[i]
        group = u.select_atoms('resnum %i and (name R1 or name R2 or name R3 or name R4 or name R5)'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_coords[i] = group_cog
    
    # ID coordinates for lipids on indicated bilayer side, renaming variables
    lpd1_atoms = u.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
    lpd2_atoms = u.select_atoms('resname %s and (name D2A or name D2B)'%lipid2)
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms
    
    # select lipid tail atoms beloging to the selected bilayer side
    if side == 'up':
        lpd1i = lpd1_atoms[:((num_lpd1)/2)]
        lpd2i = lpd2_atoms[:((num_lpd2)/2)]
    
    elif side == 'down':
        lpd1i = lpd1_atoms[((num_lpd1)/2):]
        lpd2i = lpd2_atoms[((num_lpd2)/2):]
    
    # assign lpd1 and lpd2 coordinates, completing the assignment of all coordinates from which psi6 will be computed
    lpd1_coords = lpd1i.coordinates()
    lpd2_coords = lpd2i.coordinates()

    lpd_coords = np.vstack((lpd1_coords,lpd2_coords,lpd3_coords))
    lpd_coords = lpd_coords.astype('float32')
    return lpd_coords, box

def standard_fit(X):
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    return C, N

# The projection of these points onto the best-fit plane
def projection(x, C, N):
    rows, cols = x.shape
    NN = np.tile(N, (rows, 1))
    D = np.dot(x-C, N)
    DD = np.tile(D, (cols, 1)).T
    return x - DD * NN

def dist(vec):
    distance = np.sqrt(np.power(vec[0],2) + np.power(vec[1],2) + np.power(vec[2],2))
    return distance

def angles_normvec_psi6(coords, atom, box):
    distarr = MDAnalysis.lib.distances.distance_array(coords,coords,box=box)

    nn_inds = np.argsort(distarr[atom])[0:7] # the first index is the reference atom
    centered_coords = np.zeros((7,3))
    center_coord = coords[atom]
    for i in np.arange(7):
        centered_coords[i] = coords[nn_inds[i]] - center_coord

    C, N = standard_fit(centered_coords)

    projected_coords = projection(centered_coords,C,N)
    projected_vref = projection(vref,C,N)[0]

    centered_coords = projected_coords - projected_coords[0]
    centered_vref = projected_vref - projected_coords[0]
    centered_N = N - projected_coords[0]

    angles = np.zeros(6)
    for neighbor in np.arange(1,7):
        # compute the angle against the reference vector
        norm = dist(centered_vref)*dist(centered_coords[neighbor])
        angle = np.arccos(np.dot(centered_vref,centered_coords[neighbor])/norm)
        if np.isnan(angle) == True:
            angle = 0.0
        # check whether angle belongs to 1st and 3rd or 2nd and 4th circle quadrants using 
        # a little trick with the normal vector
        if np.dot(centered_N,np.cross(centered_vref,centered_coords[neighbor])) < 0.0:
            angle = (np.pi*2) - angle
        angles[neighbor-1] = angle

    psi6 = np.mean( np.cos(angles*6) + (1j*np.sin(angles*6)))

    return psi6, angles

def get_psi6(frame):
    start = time()
    print('Finding psi6, normvec in frame %i of %i'%(frame, n_frames))
    coords, box = get_side_coordinates_and_box(frame)
    n_atoms = coords.shape[0]
    psi6s = np.zeros(n_atoms,dtype=complex)
    angles = np.zeros((n_atoms,6))
    for atom in np.arange(n_atoms):
        psi6s[atom], angles[atom] = angles_normvec_psi6(coords, atom, box)
    print('Finished after', time() - start, 'seconds')
    return psi6s, angles

pool = mp.Pool(processes=nprocs)
print 'Initiating multiprocessing with %i processors'%nprocs
results = pool.map(get_psi6, frames)

atom_angles = []
atom_psi6s  = []
for i in range(len(results)):
    atom_psi6s.append(results[i][0])
    atom_angles.append(results[i][1])

# write out the complex vector computed for psi6 and also
# write out both the angles to each neighbor of each particle
if side == 'up':
    np.save('psi6s_upper_tail.npy', atom_psi6s)
    np.save('angles_upper_tail.npy', atom_angles)
elif side == 'down':
    np.save('psi6s_lower_tail.npy', atom_psi6s)
    np.save('angles_lower_tail.npy', atom_angles)
