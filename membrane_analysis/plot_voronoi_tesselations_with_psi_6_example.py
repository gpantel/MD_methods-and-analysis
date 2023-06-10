import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import multiprocessing as mp
import sys, os
import numpy as np
import MDAnalysis
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi#, voronoi_plot_2d
import MDAnalysis.lib.NeighborSearch as NS

if not os.path.exists(os.getcwd()+'/psi_movie'):
    os.makedirs(os.getcwd()+'/psi_movie')

#Indexs for atom selection are according to the psf file
top = 'DPPC29-DIPC29-CHOL42.gro'
traj = 'DPPC29-DIPC29-CHOL42.xtc'
stride = 1
nprocs = 6

#Loading trajectory
u = MDAnalysis.Universe(top, traj)

u.trajectory[-1]
xmin = -10
ymin = xmin
#xmax = u.trajectory.ts.dimensions[0] + 15
xmax=300
ymax = xmax

s = np.around(300./(xmax - xmin)) 

frames = np.arange(u.trajectory.n_frames)[::stride]
frame_indices = np.arange(len(frames))

side = 'up'
psis = np.load('psi6s_upper_tail.npy', allow_pickle=True)[::stride]

if side == 'up':
    sidename = 'upper'
if side == 'down':
    sidename = 'lower'

def dist(vec):
    distance = np.sqrt(np.power(vec[0],2) + np.power(vec[1],2))
    return distance

P1 = np.array([1,0]) # Fully real unit vector in the complex plane
o_o = np.array([0,0]) # the origin
P2 = o_o
#for frame_index in frame_indices:
def plot_voronoi(frame_index):
    frame = frames[frame_index]
    print('Frame %i of %i'%(frame, frames[-1]))
    # do not redraw if enabled
    if 1==2: #len(glob('psi_movie/%s_img'%(sidename)+ str('%05d.png'%frame) )) == 1:
        #continue
        return

    else:
        u = MDAnalysis.Universe(top, traj)
        u.trajectory[frame]

        psi_frame = psis[frame_index]

########################## Atom selection start ##########################
        #Lipid Residue names
        lipid_atoms = u.select_atoms('resname DPPC DIPC and name PO4')
        chol_atoms = u.select_atoms('resname CHOL and name ROH')
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
        lipid_atoms = u.select_atoms('(resname DPPC and (name C2A or name C2B)) or (resname DIPC and (name D2A or name D2B))')
        
        # select lipid tail atoms beloging to the selected bilayer side
        if side == 'up':
            lpdi = lipid_atoms[:int((lipid_atoms.n_atoms)/2)]
        elif side == 'down':
            lpdi = lipid_atoms[int((lipid_atoms.n_atoms)/2):]
        lipid_coords = lpdi.positions
        all_coords = np.vstack((lipid_coords,chol_coords))
        all_coords = all_coords.astype('float32')
########################## Atom selection end ##########################

        angles = []
        for j in range(all_coords.shape[0]):
            a_o = np.array([np.real(psi_frame[j]),np.imag(psi_frame[j])]) # the atom's orientation
            P3 = a_o
            P12 = P2-P1
            P13 = P3-P1
            P23 = P3-P2
            P12_len = dist(P2-P1)
            P13_len = dist(P3-P1)
            P23_len = dist(P3-P2)
            norm = P12_len*P23_len
            angle = np.arccos(np.dot(P12,P23)/norm)
            angles.append(angle)
        psi = (np.rad2deg(np.array(angles))%30)/(30.)
        #print('min, max', np.amin(psi), np.amax(psi))

        Pxyz = all_coords
        Pxy = []
        for l in range(0,len(Pxyz)) :
            Pxy.append([Pxyz[l][0],Pxyz[l][1]])
#Extracting xy coordinates and residue names
        atm_list = []
        for a in range(0, len(Pxyz)):
            atm_list.append([Pxyz[a][0],Pxyz[a][1],psi[a]]) # list of psi 6 values corresponding to the atm_list with a unique identifier

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
        ridge_vertices = vor.ridge_vertices
        vertices       = vor.vertices
        ridge_points = vor.ridge_points
        r_length  = len(ridge_points)

        #Plotting Voroni Diagrams
        vor_rgns   = vor.regions
        vor_points = vor.point_region

        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 16}

        matplotlib.rc('font', **font)

        fig, ax = plt.subplots(figsize=(7,6))
        pl.clf()

#        voronoi_plot_2d(vor_s)  #plot the lines on the tesselation
        for p in range(0, len(vor_points)):
            rgn = vor_rgns[vor_points[p]]
            L = atm_list_p[p]
            if not -1 in rgn and 0 <= L[0] < x_box and 0 <= L[1] < y_box :
                polygon = [vor.vertices[i] for i in rgn]
                plt.fill(*zip(*polygon), color=cm.jet(L[2]))
        plt.ylabel(r'y ($\AA$)', fontsize=24)
        plt.xlabel(r'x ($\AA$)', fontsize=24)

        cax = ax.imshow(np.array([[1,1],[1,1]]), cmap=cm.jet, vmin=0, vmax=30)
        cbar = plt.colorbar(cax)
        cbar.set_label(r'$\Psi_6^k$', size=26)
        plt.scatter(all_coords[:,0], all_coords[:,1], c='k', zorder=1000000, s=s)

        # z order should be a very large number -- it makes the scatter points
        # appear on top of the filled in voronoi tessels
        plt.title('Frame %i'%frame)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks([0, 50, 100, 150, 200, 250, 300],['0', '50', '100', '150', '200', '250' , '300'])
        plt.yticks([0, 50, 100, 150, 200, 250, 300],['0', '50', '100', '150', '200', '250' , '300'])
        plt.tight_layout()
        if side == 'up':
            plt.savefig('psi_movie/'+'upper_img' + str('%05d'%frame) + '.png')
        if side == 'down':
            plt.savefig('psi_movie/'+'lower_img' + str('%05d'%frame) + '.png')
    return

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = mp.Pool(processes=nprocs)
    print('Initiating multiprocessing with %i processors'%nprocs)
    results = pool.map(plot_voronoi, frame_indices)
