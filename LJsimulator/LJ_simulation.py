from __future__ import print_function
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as u
from reducedstatedatareporter import ReducedStateDataReporter
import numpy as np
import os, sys
import parmed as pmd
import json
from sys import platform

# This JSON file loads up the user input parameters
configdic = json.load(open(sys.argv[1]))
# need to know the path to packmol
path_to_packmol = '/home/gpantel/packmol-18.169/packmol'

#################### This section is for user input. Users do not need to touch anything outside of this section
### Parameters that set file names
pdb_prefix = configdic['pdb_prefix'] # prefix to initial coordinate PDB file, final coordinate PDB file
dcd_prefix = configdic['dcd_prefix'] # name of the DCD file excluding the extension ".dcd"
data_name  = configdic['data_name'] # name of the data file including the desired extension name (e.g. ".dat")

### Parameters that determine system size, density, ratios of particles (system composition)
N = configdic['N'] # Number of particles to employ in the system
# list of particles ratios in order of type
particle_ratios = configdic['particle_ratios'] #e.g. [1, 2, 1] would be system of 25% type 1, 50% type 2, 25% type 3.
density_r = configdic['density_r'] # system density in reduced units
dimensions = configdic['dimensions'] # 2 or 3 dimensions are acceptable.

### Parameters that determine interaction energies, particle sizes, and particle masses
# LJ epsilon cross-terms are written in a 2D numpy array of size MxM where M is the number of particles
# epsilons are in reduced units, kJ/mol in OpenMM (i.e. 1.0 = epsilon, 2.0 = 2epislon)
epsilonAR_r = np.array(configdic['epsilonAR_r'], dtype="float64")
sigmas_r = configdic['sigmas_r'] # LJ sigmas in reduced units in order of particle type, nanometers in OpenMM
masses_r = configdic['masses_r'] # masses in reduced units in order of particle type, amu in OpenMM

### Parameters that determine number of simulation steps and data writing intervals
numsteps = configdic['numsteps'] # number of simulation steps to take in reduced unit time step
data_interval = configdic['data_interval'] # interval in steps between which system thermodynamic data is written
coordinate_interval = configdic['coordinate_interval'] # interval in steps between which system coordinates are written to a DCD-formatted trajectory

### Parameters for the flat well restraint
restraint_widthscale = configdic['restraint_widthscale']
forcewall = configdic['forcewall']
enable_restraint = configdic['enable_restraint']

### Set the simulation platform, toggle minimization, and set the initial condition
platform_type = configdic['platform_type']
minimization = configdic['minimization'] # Set this to "True" if you want to perform minimization
initial_condition = configdic['initial_condition'] # mixed or stripe

### Optional parameters
# restart files
if 'rstin_prefix' in configdic.keys():
    rstin_prefix = configdic['rstin_prefix']
if 'rstout_prefix' in configdic.keys():
    rstout_prefix = configdic['rstout_prefix']
# NetCDF files containing positions, forces, and velocities
if 'nc_prefix' in configdic.keys() and 'frcvel_interval' in configdic.keys():
    nc_prefix = configdic['nc_prefix'] # name of the NetCDF file excluding the extension ".nc"
    frcvel_interval = configdic['frcvel_interval'] # interval in steps between which forces and velcities are written to a NetCDF file

### Set the system temperature with two MUTUALLY EXCLUSIVE options...
if 'T_r' in configdic.keys() and 'kbT_chi' not in configdic.keys():
    T_r = configdic['T_r'] # temperature in reduced units T_r = kbT/epsilon
elif 'kbT_chi' in configdic.keys() and 'T_r' not in configdic.keys():
    kbT_chi = configdic['kbT_chi'] # specific type of reduced temperature for binary phase separations
else:
    print('You must define either T_r or kbT_chi in the input')
kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA

#################### Everything insensitive to user input used to set up simulations should be below here
#os.system('export OPENMM_CPU_THREADS=%i'%numthreads)
M = len(particle_ratios) # number of particle types in the system
particle_ratios = np.true_divide(np.array(particle_ratios),np.sum(np.array(particle_ratios)))
particle_numbers = np.around(N*particle_ratios).astype('int')
N = np.sum(particle_numbers)
masses_r_avg = np.sum(masses_r*particle_ratios)
sigmas_r = np.array(sigmas_r)
sigmas_r_avg = np.sum(sigmas_r*particle_ratios)
# this is a factor to scale down the box edges and tolerance used by packmol necessary in system preparation
if dimensions == 2:
    packingscale = 0.95
if dimensions == 3:
    packingscale = 0.9
sigmas_r_avg_packmol = sigmas_r_avg*10
epsilons_r_avg = np.mean(np.dot(epsilonAR_r,particle_ratios))
timestep_r = np.sqrt(np.true_divide(np.sqrt(masses_r_avg)*np.square(sigmas_r_avg_packmol)\
             ,epsilons_r_avg))*u.femtoseconds
cutoff_r = 2.5*np.sum(sigmas_r*particle_ratios) # cutoff for LJ interactions in reduced units
# equivalent gamma to using timestep=2 fs, gamma=1 ps^-1.
gamma_r = 1000.0*0.5*timestep_r/u.picoseconds # timestep needs to be scaled up by 1000x fs^-1 scale as it is interpreted in ps^-1

if 'T_r' in configdic.keys() and 'kbT_chi' not in configdic.keys():
    T = np.true_divide(epsilons_r_avg*T_r, 0.008314462175)*u.kelvin # the temperature needed to achieve the reduced temperature T_r
elif 'kbT_chi' in configdic.keys() and 'T_r' not in configdic.keys():
    chi = -((epsilonAR_r[1][0] + epsilonAR_r[0][1])/2.) - ((epsilonAR_r[0][0] + epsilonAR_r[1][1])/2.) 
    T_r = kbT_chi*np.true_divide(chi,kB)

T = np.true_divide(epsilons_r_avg*T_r, 0.008314462175)*u.kelvin # the temperature needed to achieve the reduced temperature T_r

# generate sigmaAR as an MxM matrix using Lorentz-Berthelot combination rule (arithmetic mean)
sigmaAR_r = np.zeros((M,M), dtype="float64")
for i in range(M):
    for j in range(i,M):
        sigmaAR_r[i][j] = np.true_divide(sigmas_r[i]+sigmas_r[j],2.0)
        sigmaAR_r[j][i] = sigmaAR_r[i][j]

epsilonLST_r = (epsilonAR_r).ravel().tolist()
sigmaLST_r   = (sigmaAR_r).ravel().tolist()


############ Building initial system configuration
print('Constructing system using PACKMOL')
if dimensions != 2 and dimensions != 3:
    print("dimensions may only be set to 2 or 3.")
    sys.exit()
# these reduced units are interpreted as nm in OpenMM
dimension_product_r_packmol = np.true_divide(N*(np.power(sigmas_r_avg_packmol,dimensions)),density_r) # system volume or area determined in reduced units
box_edge_r = np.true_divide(np.power(dimension_product_r_packmol, np.true_divide(1.0,dimensions)),10) # box edge length in reduced units
box_edge_r_packmol = box_edge_r*10*packingscale
### Write packmol input files if the PDB file does not already exist
packmol_file = open('packmol_input.inp', 'w')
packmol_file.write('tolerance %f \n'%(sigmas_r_avg_packmol*packingscale))
packmol_file.write('filetype pdb \n')
if platform != 'windows':
    # PDB files will not have the box dimensions in windows. A little inconvenience.
    packmol_file.write('add_box_sides = 1.0 \n')
packmol_file.write('output %s \n'%('%s_initial.pdb'%pdb_prefix))

first_x_edge = 0.0
for alpha in range(1,M+1):
    next_x_edge = first_x_edge + (particle_ratios*box_edge_r_packmol)[alpha-1]
    # Write a PDB file for the particle for use by packmol
    #name them P1, P2, P3.... PM
    packmol_pdb_file = open('P%i.pdb'%alpha, 'w')
    packmol_pdb_file.write('ATOM      1  P  P%s    1       0.000   0.000  00.00                       Ar'%(str(alpha) + (3-len(str(alpha)))*' '))
    packmol_pdb_file.close()

    # Then write in the packmol input file instructions for that particle
    packmol_file.write('structure P%i.pdb \n'%alpha)
    packmol_file.write('  number %i \n'%particle_numbers[alpha-1])
    if dimensions == 2:
        if initial_condition == 'mixed':
            packmol_file.write('  inside box 0. 0. %f %f %f %f \n'%(np.true_divide(box_edge_r_packmol,2.0),\
                            box_edge_r_packmol, box_edge_r_packmol, np.true_divide(box_edge_r_packmol,2.0)))
        elif initial_condition == 'stripe':
            packmol_file.write('  inside box %f 0. %f %f %f %f \n'%( first_x_edge, np.true_divide(box_edge_r_packmol,2.0),\
                                                  next_x_edge, box_edge_r_packmol, np.true_divide(box_edge_r_packmol,2.0)))
            print('initial_condition must be mixed or stripe')

    if dimensions == 3:
        if initial_condition == 'mixed':
            packmol_file.write('  inside box 0. 0. 0. %f %f %f \n'%(box_edge_r_packmol, box_edge_r_packmol, box_edge_r_packmol))
        elif initial_condition == 'stripe':
            packmol_file.write('  inside box %f 0. 0. %f %f %f \n'%( first_x_edge,\
                                                  next_x_edge, box_edge_r_packmol, box_edge_r_packmol))
        else:
            print('initial_condition must be mixed or stripe')
    first_x_edge = next_x_edge
    packmol_file.write('end structure \n')
packmol_file.close()
###
os.system('%s < packmol_input.inp'%path_to_packmol)

### Clean up packmol files
os.system('rm P*.pdb')
os.system('rm packmol_input.inp')
# temporarily change CRYST1 to a remark to avoid confusing app.PDBFile
if platform == "linux" or platform == "linux2":
    os.system('sed -i "s/CRYST1/REMARK/g" %s_initial.pdb'%pdb_prefix)
elif platform == "darwin":
    os.system('sed -i "" "s/CRYST1/REMARK/g" %s_initial.pdb'%pdb_prefix)
pdb = app.PDBFile('%s_initial.pdb'%pdb_prefix)
positions = pdb.positions
# change CRYST1 back from a remark
if platform == "linux" or platform == "linux2":
    os.system('sed -i "6s/REMARK/CRYST1/g" %s_initial.pdb'%pdb_prefix)
elif platform == "darwin":
    os.system('sed -i "" "6s/REMARK/CRYST1/g" %s_initial.pdb'%pdb_prefix)

### Write a TCL script to wrap & visualize the system in VMD after simulation
wrapper_file = open('%s.tcl'%pdb_prefix, 'w')
wrapper_file.write('mol modcolor 0 0 ResName \n')
if dimensions == 2:
    wrapper_file.write('mol modstyle 0 0 Points %f \n'%(sigmas_r_avg_packmol*0.6))
if dimensions == 3:
    wrapper_file.write('mol modstyle 0 0 Points %f \n'%(sigmas_r_avg_packmol*2.0))
wrapper_file.write('display projection Orthographic')
wrapper_file.close()


############ Building OpenMM simulation
print('Constructing OpenMM simulation')
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(box_edge_r, 0, 0), mm.Vec3(0, box_edge_r, 0), mm.Vec3(0, 0, box_edge_r))
customNonbondedForce = mm.CustomNonbondedForce('4*eps*((sig/r)^12-(sig/r)^6); eps=epsilon(type1, type2); sig=sigma(type1, type2)')
customNonbondedForce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
customNonbondedForce.setCutoffDistance(min(box_edge_r*0.49*u.nanometers, cutoff_r*u.nanometers))
customNonbondedForce.addTabulatedFunction('epsilon', mm.Discrete2DFunction(M, M, epsilonLST_r))
customNonbondedForce.addTabulatedFunction('sigma', mm.Discrete2DFunction(M, M, sigmaLST_r))
customNonbondedForce.addPerParticleParameter('type')

# set the initial particle parameters
particle_index = 0
for alpha in range(M):
    for i in range(particle_index, particle_numbers[alpha] + particle_index):
        system.addParticle(masses_r[alpha]*u.amu)
        customNonbondedForce.addParticle([alpha])
        particle_index += 1
system.addForce(customNonbondedForce)



if enable_restraint == True:
    width=((1./M)*box_edge_r)*0.5*restraint_widthscale
    flatres = mm.CustomExternalForce('forcewall * (px^2); \
                                   px = max(0, delta); \
                                   delta = r - droff; \
                                   r = abs(periodicdistance(x, y, z, x0, y, z));')
    flatres.addGlobalParameter('forcewall', forcewall)
    flatres.addGlobalParameter('droff', width)
    flatres.addPerParticleParameter('x0')
    particle_index = 0

    first_x_edge = 0.0
    for alpha in range(M):
        next_x_edge = first_x_edge + (particle_ratios*box_edge_r)[alpha]
        x0 = next_x_edge - ((next_x_edge - first_x_edge)/2.)
        for i in range(particle_index, particle_numbers[alpha] + particle_index):
            flatres.addParticle(particle_index, [x0])
            particle_index += 1
        first_x_edge = next_x_edge
    system.addForce(flatres)

##########################################################################################################################

def CustomLangevinIntegrator(temperature=298.0*u.kelvin, collision_rate=91.0/u.picoseconds, timestep=1.0*u.femtoseconds):
    # Compute constants.
    kT = kB * temperature
    gamma = collision_rate


    # Create a new custom integrator.
    integrator = mm.CustomIntegrator(timestep)

    #
    # If dimensions == 2, set up a dummy variable to remove z-axial velocities
    #
    if dimensions == 2:
        integrator.addPerDofVariable("dumv", 1.0)
        integrator.setPerDofVariableByName("dumv", [mm.Vec3(x=1.0, y=1.0, z=0.0)])
    #
    # Integrator initialization.
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("T", temperature) # temperature
    integrator.addGlobalVariable("b", np.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();
    
    #
    # Metropolized symplectic step.
    #
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    if dimensions == 2: # To get a 2D system, make z-velocities zero when moving x
        integrator.addComputePerDof("x", "x + v*dumv*dt")
    else:
        integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    if dimensions == 2: # Remove the resulting z-velocities to get the correct Kinetic Energy
        integrator.addComputePerDof("v", "v*dumv")

    return integrator

integrator = CustomLangevinIntegrator(T, gamma_r, timestep_r)
if platform_type == 'CUDA':
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
elif platform_type == 'OpenCL':
    platform = mm.Platform.getPlatformByName('OpenCL')
    properties = {'OpenCLPrecision': 'mixed'}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
elif platform_type == 'CPU':
    platform = mm.Platform.getPlatformByName('CPU')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
else:
    print("platform_type can be either CUDA, OpenCL, or CPU")
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(T)

# load restart if asked
if 'rstin_prefix' in configdic.keys():
    with open(rstin_prefix+'.rst', 'r') as f:
        simulation.context.setState(XmlSerializer.deserialize(f.read()))

############ Running OpenMM simulation
if minimization == True:
    print('Performing SD minimization')
    simulation.minimizeEnergy(tolerance=0.1)
    print('Minimized energy:', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(pdb.topology, positions, open('%s_minimized.pdb'%pdb_prefix , 'w'))

print('Initiating MD simulation for %i steps'%numsteps)
simulation.reporters.append(app.DCDReporter('%s.dcd'%dcd_prefix, coordinate_interval))
if 'nc_prefix' in configdic.keys() and 'frcvel_interval' in configdic.keys():
    simulation.reporters.append(pmd.openmm.NetCDFReporter('%s.nc'%nc_prefix, frcvel_interval, crds=True, vels=True, frcs=True))
simulation.reporters.append(ReducedStateDataReporter(sys.stdout, data_interval, dimensions, step=True, 
    potentialEnergy=True, kineticEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=numsteps, separator='\t'))
simulation.reporters.append(ReducedStateDataReporter(data_name, data_interval, dimensions, step=True, 
    potentialEnergy=True, kineticEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=numsteps, separator='\t'))
simulation.step(numsteps)

# write restart files
if 'rstout_prefix' in configdic.keys():
    state = simulation.context.getState( getPositions=True, getVelocities=True )
    with open(rstout_prefix+'.rst', 'w') as f:
        f.write(mm.XmlSerializer.serialize(state))
