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
nprocs = 6
top  = 'DPPC29-DUPC29-CHOL42.gro'
traj = 'DPPC29-DUPC29-CHOL42.xtc'
side = sys.argv[1] # "up" for upper leaflet "down" for lower leaflet
stride = 1

u = MDAnalysis.Universe(top,traj)
n_frames = u.trajectory.n_frames
frames = np.arange(n_frames)[::stride]

# set reference vector
vref = np.array([[1,0,0],[0,0,0]])

def wrap_positions(v, refv, box): # v and refv same number of entries
    for i in range(len(v)):
        vwrapped = np.copy(v)
        if v[i] - refv[i] < 0:
            if np.abs(v[i] - refv[i]) >= (box[i]/2): vwrapped += box[i]
        if v[i] - refv[i] >= 0:
            if np.abs(v[i] - refv[i]) >= (box[i]/2): vwrapped -= box[i]
    return vwrapped

def get_side_coordinates_and_box(frame):
    u = MDAnalysis.Universe(top,traj)
    u.trajectory[frame]

    #x, y, z = u.trajectory.ts.triclinic_dimensions[0][0], u.trajectory.ts.triclinic_dimensions[1][1], u.trajectory.ts.triclinic_dimensions[2][2]
    #box = np.array([x, y, z])

    ### Determining side of the bilayer CHOL belongs to in this frame    
    lipid_atoms = u.select_atoms('resname DPPC DIPC and name PO4')
    chol_atoms = u.select_atoms('resname CHOL and name ROH')
    # atoms in the upper leaflet as defined by insane.py, the CHARMM-GUI membrane builders (last I checked), or MolPainter
    # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
    # this must be done because CHOL rapidly flip-flops between leaflets in the MARTINI model
    # so we must assign CHOL to each leaflet at every time step, and in large systems
    # with substantial membrane undulations, a simple cut-off in the z-axis just will not cut it
    if side == 'up':
        lpdi = lipid_atoms[:int((lipid_atoms.n_atoms)/2)]
        lipids = lpdi
        ns_lipids = NS.AtomNeighborSearch(chol_atoms)
        choli = ns_lipids.search(lipids,15.0)
    elif side == 'down':
        lpdi = lipid_atoms[int((lipid_atoms.n_atoms)/2):]
        lipids = lpdi
        ns_lipids = NS.AtomNeighborSearch(chol_atoms)
        choli = ns_lipids.search(lipids,15.0)

    # ID center of geometry coordinates for cholesterol on indicated bilayer side
    chol_coords = np.zeros((len(choli.resnums),3))
    for i in np.arange(len(choli.resnums)):
        resnum = choli.resnums[i]
        group = u.select_atoms('resnum %i and (name R1 R2 R3 R4 R5)'%resnum)
        group_cog = group.center_of_geometry()
        chol_coords[i] = group_cog
    
    # ID coordinates for lipids on indicated bilayer side, renaming variables
    lipid_atoms = u.select_atoms('(resname DPPC and (name C2A or name C2B)) or (resname DUPC and (name D2A or name D2B))')
    
    # select lipid tail atoms beloging to the selected bilayer side
    if side == 'up':
        lpdi = lipid_atoms[:int((lipid_atoms.n_atoms)/2)]
    elif side == 'down':
        lpdi = lipid_atoms[int((lipid_atoms.n_atoms)/2):]
    lipid_coords = lpdi.positions
    all_coords = np.vstack((lipid_coords,chol_coords))
    all_coords = all_coords.astype('float32')
    return all_coords, u.atoms.dimensions #box

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
        # center coords while respecting PBC...
        coord = wrap_positions(coords[nn_inds[i]], coords[nn_inds[0]], box)
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
    print('Finding psi6, normvec in frame %i of %i'%(frame+1, n_frames))
    coords, box = get_side_coordinates_and_box(frame)
    n_atoms = coords.shape[0]
    psi6s = np.zeros(n_atoms,dtype=complex)
    angles = np.zeros((n_atoms,6))
    for atom in np.arange(n_atoms):
        psi6s[atom], angles[atom] = angles_normvec_psi6(coords, atom, box)
    print('Finished after', time() - start, 'seconds')
    return psi6s, angles

if __name__ == "__main__":   

    pool = mp.Pool(processes=nprocs)
    print('Initiating multiprocessing with %i processors'%nprocs)
    results = pool.map(get_psi6, frames)

    atom_angles = []
    atom_psi6s  = []
    for i in range(len(results)):
        atom_psi6s.append(results[i][0])
        atom_angles.append(results[i][1])

    # write out the complex vector computed for psi6 and also
    # write out both the angles to each neighbor of each particle
    # this is a list-of-lists, as the number of cholesterol in upper and lower leaflets can change due to flipping
    if side == 'up':
        np.save('psi6s_upper_tail.npy', atom_psi6s, allow_pickle=True)
        np.save('angles_upper_tail.npy', atom_angles, allow_pickle=True)
    elif side == 'down':
        np.save('psi6s_lower_tail.npy', atom_psi6s, allow_pickle=True)
        np.save('angles_lower_tail.npy', atom_angles, allow_pickle=True)
