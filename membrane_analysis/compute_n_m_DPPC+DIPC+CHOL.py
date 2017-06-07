#########################################################################################################
# This is a script for extracting average, per-frame order parameters dependent on the
# number of intra- and inter-leaflet lipids involved in the formation of a cluster within a
# ternary DPPC:DIPC:CHOL lipid bilayer
#
# The selection of lipids can be modified by changing the relevant atom selections and groups within
# the "compute_p2s" method and the "analyze_clusters" method
#########################################################################################################

# import libraries
from collections import Counter
import scipy.cluster
import numpy as np
import sys
import multiprocessing as mp
import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import time

verbose = False

# input files and parameters
print("Loading inputs...")
gro = "confout.gro"
xtc = "run.xtc"
skip = 10 # interval to sample frames from the input trajectory/ies
u = MDAnalysis.Universe(gro,xtc)
N_cutoff = 5.8 # cutoff for intra-leaflet cluster identifitcation (in angstroms for MDAnalysis)
M_cutoff = 7.0 # cutoff for inter-leaflet cluster identification (in angstroms for MDAnalysis)
nprocs = 16 # number of stupidly parallel processors to use
frames = np.arange(u.trajectory.n_frames)[::skip][0:11000/skip] # frames from which clusters will be extracted

# return periodic images
def get_periodic_xyz(xyz,c):
    # Make images of surrounding xy plane cells for 8 mirror images around the real cell
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

    images = []
    # First, add the real cell
    images.append(np.array([x  ,y  ,z]).T) # this is the real cell
    ### MID
    # top row of mid
    images.append(np.array([x+c,y+c,z  ]).T)
    images.append(np.array([x  ,y+c,z  ]).T)
    images.append(np.array([x-c,y+c,z  ]).T)
    # mid row of mid
    images.append(np.array([x+c,y  ,z]).T)
    #images.append(np.array([x, ,y  ,z]).T) # This would be the real cell!
    images.append(np.array([x-c,y  ,z]).T)
    # bot row of mid
    images.append(np.array([x+c,y-c,z  ]).T)
    images.append(np.array([x  ,y-c,z  ]).T)
    images.append(np.array([x-c,y-c,z  ]).T)
    images = np.array(images)
    return images

######## P2 START ########
### This will only analyze DPPC, though you can easily change this selection to whatever groups you like
def compute_p2s(group):
    lpd = group.select_atoms('resname DPPC')

    vref = np.array([0,0,1]) # reference vector -- along the z-axis
    resids = np.unique(lpd.resids)
    num_atoms = lpd.n_atoms
    num_lpd = lpd.n_residues

    p2s = []
    for res in resids:
        GL1 = lpd.select_atoms('resid %i and name GL1'%res)
        GL2 = lpd.select_atoms('resid %i and name GL2'%res)
        C4A = lpd.select_atoms('resid %i and name C4A'%res)
        C4B = lpd.select_atoms('resid %i and name C4B'%res)
        GL1_crd = GL1.coordinates()[0]
        GL2_crd = GL2.coordinates()[0]
        C4A_crd = C4A.coordinates()[0]
        C4B_crd = C4B.coordinates()[0]

        # tail A
        GL1_crd = GL1_crd - C4A_crd
        norm = dist(GL1_crd)*dist(vref)
        angleA = np.true_divide(np.dot(vref,GL1_crd),norm)
        if angleA == np.nan:
            angleA = 0.

        p2A = 0.5*((3*np.square(angleA))-1)
        print('chain A angle, p2:', np.rad2deg(angleA), p2A)

        # tail B
        GL2_crd = GL2_crd - C4B_crd
        norm = dist(GL2_crd)*dist(vref)
        angleB = np.true_divide(np.dot(vref,GL2_crd),norm)
        if angleB == np.nan:
            angleB = 0.

        p2B = 0.5*((3*np.square(angleA))-1)
        print('chain B angle, p2:', np.rad2deg(angleB), p2B)

        p2s.append(p2A)
        p2s.append(p2B)

    p2s = np.array(p2s)
    return np.mean(p2s)
######## P2 END ########

######## PSI6 START ########
def standard_fit(x):
    # Center the selected coordinates about (0, 0, 0)
    m = np.average(x, axis=0)
    mx = x - m
    # Find the singular value decomposition of the coordinates
    U, S, V = np.linalg.svd(mx)
    # Find eigenvector correspodning to the smallest eigenvalue
    # This will be used to project the coordinates into a plane of best fit
    N = V[-1]
    return m, N

def projection(x, m, N):
    rows, cols = x.shape
    NN = np.tile(N, (rows, 1))
    # We project the coordinates into the plane of best fit
    D = np.dot(x-m, N)
    DD = np.tile(D, (cols, 1)).T
    return x - DD * NN

def dist(vec):
    distance = np.sqrt(np.power(vec[0],2) + np.power(vec[1],2) + np.power(vec[2],2))
    return distance

def angles_normvec_psi6(coords, atom, box):
    distarr = MDAnalysis.lib.distances.distance_array(coords,coords,box=box)

    vref = np.array([[0,0,1],[0,0,0]]) # reference vector -- along the z-axis
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

    return psi6#, angles

def get_psi6(coords, atomindices, box):
    n_atoms = len(atomindices)
    psi6s = np.zeros(n_atoms,dtype=complex)
    #angles = np.zeros((n_atoms,6))
    psi6s = []
    for atom in atomindices:
        #psi6s[atom], angles[atom] = angles_normvec_psi6(coords, atom, box)
        psi6s.append(angles_normvec_psi6(coords, atom, box))
    psi6s = np.array(psi6s)
    return psi6s
######## PSI6 END ########

def extract_clusters(xyz, c, resids):
    pbc_xyz = np.concatenate(get_periodic_xyz(xyz,c)) # aim_atom in 9 periodic images
    # single-linkage clusteirng
    # assigments {[0,x1][1,x2]...} 0,1,2,... refers to index of aim_atom (not the index in original pdb), x1,x2,... refers to which cluster this atom belongs to
    assignments  = scipy.cluster.hierarchy.fclusterdata(pbc_xyz,N_cutoff,criterion='distance',metric='euclidean',method='single')
    # index of cluster in order from largest to smallest, starting from 1
    ranked_clusters = np.argsort(Counter(assignments).values())[::-1]+1

    accepted_groups = [] # groups of the real clusters
    #accepted_resids = [] # resids of the real clusters
    used_inds = [] # list of list of indices that have already been confirmed as clusters with real components
    for cluster in ranked_clusters:
        # relative index of aim_atom in THIS cluster
        ind_all = [i for i, x in enumerate(assignments) if x == cluster]
        # find the indices of atoms as they would appear in the real cell
        ind_mod = np.mod(ind_all,len(xyz)).tolist()

        ind_ori = sorted(ind_mod) # we'll use these to get the residue ids the selection belongs to -- a solution that only works for singel molecule selections of aim_mol
        if any((True for x in ind_ori if x in used_inds)) == False: #check if a larget cluster already contains this atom
            used_inds = used_inds + ind_ori
            mol_groups = []
            #resids = []
            for index in ind_ori:
                resid = resids[index]
                #resids.append(resid)
                mol_group = u.select_atoms('resid %i'%resid)
                mol_groups.append(mol_group)
            mol_groups = np.sum(mol_groups) # contatenate the groups together -- this can be used for our analysis!
            accepted_groups.append(mol_groups)
            #accepted_resids.append(resids)
            #c_size = mol_groups.n_residues # size of the cluster
            if verbose == True:
                print("frame %i in %i, cluster %i in %i TRUE"%(frame,u.trajectory.n_frames,cluster,len(ranked_clusters)))

        else:
            if verbose == True:
                print("frame %i in %i, cluster %i in %i FALSE"%(frame,u.trajectory.n_frames,cluster,len(ranked_clusters)))
    return accepted_groups#, accepted_resids

# iterate frames of aim_atom in trajectory
results = []
def analyze_clusters(frame):
    u = MDAnalysis.Universe(gro,xtc)
    print('Frame %i of %i......'%(frame,u.trajectory.n_frames))
    start = time.time()
    u.trajectory[frame]

    ######## Discriminate between leaflets in this frame to construct leaflet-separated groups
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
    lpd1i_up = lpd1_atoms[:((num_lpd1)/2)]
    lpd2i_up = lpd2_atoms[:((num_lpd2)/2)]
    lipids_up = lpd1i_up + lpd2i_up
    ns_lipids_up = NS.AtomNeighborSearch(lpd3_atoms)
    lpd3i_up = ns_lipids_up.search(lipids_up,15.0)
    lpd3i_up_resids = np.unique(lpd3i_up.resids)

    lpd1i_down = lpd1_atoms[((num_lpd1)/2):]
    lpd2i_down = lpd2_atoms[((num_lpd2)/2):]
    lipids_down = lpd1i_down + lpd2i_down
    ns_lipids_down = NS.AtomNeighborSearch(lpd3_atoms)
    lpd3i_down = ns_lipids_down.search(lipids_down,15.0)
    lpd3i_down_resids = np.unique(lpd3i_down.resids)

    # ID center of geometry coordinates for cholesterol on indicated bilayer side
    lpd3_up_coords = np.zeros((len(lpd3i_up.resnums),3))
    for i in np.arange(len(lpd3i_up.resnums)):
        resnum = lpd3i_up.resnums[i]
        group = u.select_atoms('resnum %i and (name R1 or name R2 or name R3 or name R4 or name R5)'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_up_coords[i] = group_cog

    lpd3_down_coords = np.zeros((len(lpd3i_down.resnums),3))
    for i in np.arange(len(lpd3i_down.resnums)):
        resnum = lpd3i_down.resnums[i]
        group = u.select_atoms('resnum %i and (name R1 or name R2 or name R3 or name R4 or name R5)'%resnum)
        group_cog = group.center_of_geometry()
        lpd3_down_coords[i] = group_cog

    # ID coordinates for lipids on indicated bilayer side, renaming variables
    lpd1_atoms = u.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
    lpd2_atoms = u.select_atoms('resname %s and (name D2A or name D2B)'%lipid2)
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms

    # select lipid tail atoms beloging to the selected bilayer side
    lpd1i_up = lpd1_atoms[:((num_lpd1)/2)]
    lpd2i_up = lpd2_atoms[:((num_lpd2)/2)]

    lpd1i_down = lpd1_atoms[((num_lpd1)/2):]
    lpd2i_down = lpd2_atoms[((num_lpd2)/2):]

    # assign lpd1 coordinates, completing the assignment of all coordinates from which psi6 will be computed
    lpd1_up_coords = lpd1i_up.coordinates()
    lpd2_up_coords = lpd2i_up.coordinates()

    lpd1_down_coords = lpd1i_down.coordinates()
    lpd2_down_coords = lpd2i_down.coordinates()

    lpd_up_coords = np.vstack((lpd1_up_coords,lpd3_up_coords))
    lpd_up_coords = lpd_up_coords.astype('float32')

    lpd_up_full_coords = np.vstack((lpd1_up_coords,lpd3_up_coords,lpd2_up_coords)) # un-evaluated indices will go last
    lpd_up_full_coords = lpd_up_full_coords.astype('float32')

    lpd_down_coords = np.vstack((lpd1_down_coords,lpd3_down_coords))
    lpd_down_coords = lpd_down_coords.astype('float32')

    lpd_down_full_coords = np.vstack((lpd1_down_coords,lpd3_down_coords,lpd2_down_coords)) # un-evaluated indices will go last
    lpd_down_full_coords = lpd_down_full_coords.astype('float32')

    # Select tail atoms for M inter-leaflet cluster analysis
    lpd1_tail_atoms = u.select_atoms('resname %s and (name C4A or name C4B)'%lipid1)
    num_tail_lpd1 = lpd1_tail_atoms.n_atoms
    lpd1i_tail_up = lpd1_tail_atoms[:((num_tail_lpd1)/2)]
    lpd1i_tail_down = lpd1_tail_atoms[((num_tail_lpd1)/2):]

    lpd3i_tail_up = []
    for resid in lpd3i_up_resids:
        sel = u.select_atoms('resid %i and name C2'%resid)
        lpd3i_tail_up.append(sel)
    lpd3i_tail_up = np.sum(lpd3i_tail_up)

    lpd3i_tail_down = []
    for resid in lpd3i_down_resids:
        sel = u.select_atoms('resid %i and name C2'%resid)
        lpd3i_tail_down.append(sel)
    lpd3i_tail_down = np.sum(lpd3i_tail_down)

    tail_up = lpd1i_tail_up + lpd3i_tail_up
    tail_down = lpd1i_tail_down + lpd3i_tail_down
    ########################################################################################  

    resids_up = np.concatenate([lpd1i_up.resids, np.unique(lpd3i_up.resids)])
    resids_down = np.concatenate([lpd1i_down.resids, np.unique(lpd3i_down.resids)])

    c = u.trajectory.ts.triclinic_dimensions[0][0] # box length
    box = u.trajectory.ts.triclinic_dimensions

    up_cluster_groups   = extract_clusters(lpd_up_coords, c, resids_up)
    down_cluster_groups = extract_clusters(lpd_down_coords, c, resids_down)

    N_absPsi6s = []
    N_p2s = []
    N_c_sizes = []
    #Compute structural parameters for n -- intra-leaflet cluster parameters on both sides
    haystack = lpd1i_up.indices
    for group in up_cluster_groups:
        if group.select_atoms('resname DPPC').n_residues != 0: # only evaluate DPPC-containing clusters
            # do structural analysis and save the cluster size
            p2 = compute_p2s(group) # mean P2 for the cluster
            c_size = group.select_atoms('name C4A or name C4B or name C2').n_atoms # the number of carbon chains in the cluster
            N_p2s.append(p2)
            N_c_sizes.append(c_size)
            # do psi6 ONLY for dppc molecules
            packed = group.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
            needles = packed.indices
            st = set(needles)
            atomindices = [i for i, e in enumerate(haystack) if e in st]
            absPsi6 = np.mean(np.abs(get_psi6(lpd_up_full_coords, atomindices, box)))
            N_absPsi6s.append(absPsi6)

    print('finished upper leaflet intra-leaflet clustering, n!')
    haystack = lpd1i_down.indices
    for group in down_cluster_groups:
        if group.select_atoms('resname DPPC').n_residues != 0: # only evaluate DPPC-containing clusters
            # do structural analysis and save the cluster size
            p2 = compute_p2s(group) # mean P2 for the cluster
            c_size = group.select_atoms('name C4A or name C4B or name C2').n_atoms # the number of carbon chains in the cluster
            N_p2s.append(p2)
            N_c_sizes.append(c_size)
            # do psi6 ONLY for dppc molecules
            packed = group.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
            needles = packed.indices
            st = set(needles)
            atomindices = [i for i, e in enumerate(haystack) if e in st]
            absPsi6 = np.mean(np.abs(get_psi6(lpd_down_full_coords, atomindices, box)))
            N_absPsi6s.append(absPsi6)

    print('finished lower leaflet intra-leaflet clustering, n!')
    M_absPsi6s = []
    M_p2s = []
    M_c_sizes = []
    # Compute structural parameters for m -- inter-leaflet cluster parameters on both sides
    haystack = lpd1i_down.indices
    for group in up_cluster_groups:
        if group.select_atoms('resname DPPC').n_residues != 0: # only evaluate DPPC-containing clusters
            # find the residue ids of the inter-leaflet DPPC molecules within M_cutoff ang of molecules in the cluster            
            ns_inter_lipids = NS.AtomNeighborSearch(lpd1i_tail_down) # using inter-leaflet DPPC molecules....
            inter_lipids = ns_inter_lipids.search(group, M_cutoff) # find inter-leaflet DPPC molecules
            # find the residue ids of the inter-leaflet DPPC and CHOL molecules within M_cutoff ang of molecules in the cluster            
            ns_inter_atoms = NS.AtomNeighborSearch(tail_down) # using inter-leaflet DPPC and CHOL molecules....
            inter_atoms = ns_inter_atoms.search(group, M_cutoff) # find inter-leaflet DPPC and CHOL molecules
            if inter_atoms != []:
                c_size = inter_atoms.n_atoms
                M_c_sizes.append(c_size)
            else:
                M_c_sizes.append(0)

            if inter_lipids != []:
                inter_lipids_resids = np.unique(inter_lipids.resids)
                inter_lipid_groups = []
                for resid in inter_lipids_resids:
                    sel = u.select_atoms('resid %i'%resid)
                    inter_lipid_groups.append(sel)
                inter_group = np.sum(inter_lipid_groups)
                p2 = compute_p2s(inter_group)
                M_p2s.append(p2)
                # do psi6 ONLY for dppc molecules
                packed = inter_group.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
                needles = packed.indices
                st = set(needles)
                atomindices = [i for i, e in enumerate(haystack) if e in st]
                absPsi6 = np.mean(np.abs(get_psi6(lpd_down_full_coords, atomindices, box)))
                M_absPsi6s.append(absPsi6)
            else:
                M_p2s.append(0)
                M_absPsi6s.append(0)

    print('finished upper leaflet inter-leaflet clustering, m!')
    haystack = lpd1i_up.indices
    for group in down_cluster_groups:
        if group.select_atoms('resname DPPC').n_residues != 0: # only evaluate DPPC-containing clusters
            # find the residue ids of the inter-leaflet DPPC molecules within 10 ang of molecules in the cluster            
            ns_inter_lipids = NS.AtomNeighborSearch(lpd1i_tail_up) # using inter-leaflet DPPC molecules....
            inter_lipids = ns_inter_lipids.search(group, M_cutoff) # find inter-leaflet DPPC molecules within 10 ang of the cluster
            # find the residue ids of the inter-leaflet DPPC and CHOL molecules within M_cutoff ang of molecules in the cluster            
            ns_inter_atoms = NS.AtomNeighborSearch(tail_up) # using inter-leaflet DPPC and CHOL molecules....
            inter_atoms = ns_inter_atoms.search(group, M_cutoff) # find inter-leaflet DPPC and CHOL molecules
            if inter_atoms != []:
                c_size = inter_atoms.n_atoms
                M_c_sizes.append(c_size)
            else:
                M_c_sizes.append(0)

            if inter_lipids != []:
                inter_lipids_resids = np.unique(inter_lipids.resids)
                inter_lipid_groups = []
                for resid in inter_lipids_resids:
                    sel = u.select_atoms('resid %i'%resid)
                    inter_lipid_groups.append(sel)
                inter_group = np.sum(inter_lipid_groups)
                p2 = compute_p2s(inter_group)
                M_p2s.append(p2)
                # do psi6 ONLY for dppc molecules
                packed = inter_group.select_atoms('resname %s and (name C2A or name C2B)'%lipid1)
                needles = packed.indices
                st = set(needles)
                atomindices = [i for i, e in enumerate(haystack) if e in st]
                absPsi6 = np.mean(np.abs(get_psi6(lpd_up_full_coords, atomindices, box)))
                M_absPsi6s.append(absPsi6)
            else:
                M_p2s.append(0)
                M_absPsi6s.append(0)

    print('finished lower leaflet inter-leaflet clustering, m!')
    N_c_sizes  = np.array(N_c_sizes)
    N_p2s      = np.array(N_p2s)
    N_absPsi6s = np.array(N_absPsi6s)
    M_c_sizes  = np.array(M_c_sizes)
    M_p2s      = np.array(M_p2s)
    M_absPsi6s = np.array(M_absPsi6s)

    #results.append( [N_c_sizes, M_c_sizes, N_p2s, M_p2s, N_absPsi6s, M_absPsi6s] )
    print('This whole process ran for', time.time()-start)
    return N_c_sizes, M_c_sizes, N_p2s, M_p2s, N_absPsi6s, M_absPsi6s

pool = mp.Pool(processes=nprocs)
print('Initiating multiprocessing with %i processors'%nprocs)
results = pool.map(analyze_clusters, frames)

N_c_sizes  = []
N_c_p2s    = []
N_absPsi6s = []
M_c_sizes  = []
M_c_p2s    = []
M_absPsi6s = []
for i in range(len(results)):
    N_c_sizes.append(results[i][0])
    M_c_sizes.append(results[i][1])

    N_c_p2s.append(results[i][2])
    M_c_p2s.append(results[i][3])

    N_absPsi6s.append(results[i][4])
    M_absPsi6s.append(results[i][5])

N_c_sizes  = np.array(N_c_sizes)
M_c_sizes  = np.array(M_c_sizes)
N_c_p2s    = np.array(N_c_p2s)
M_c_p2s    = np.array(M_c_p2s)
N_absPsi6s = np.array(N_absPsi6s)
M_absPsi6s = np.array(M_absPsi6s)

results = np.array([N_c_sizes, M_c_sizes, N_c_p2s, M_c_p2s, N_absPsi6s, M_absPsi6s]).T
np.save('n_m_%.1f_%.1f.npy'%(N_cutoff,M_cutoff),results)
