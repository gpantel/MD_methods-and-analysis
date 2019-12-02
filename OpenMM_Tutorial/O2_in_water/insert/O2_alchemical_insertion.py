import simtk.openmm as mm
import simtk.openmm.app as app
from simtk import unit
# we will use openmmtools for building a TIP3P water box
# this can be installed via conda:
# conda install -c omnia openmmtools
import openmmtools

# have openmmtools construct a fully-parameterized cubic 1728-water box
waterbox = openmmtools.testsystems.WaterBox(box_edge=26.2*unit.angstrom, cutoff=12.0*unit.angstrom,\
   model='tip3p', switch_width=2.0*unit.angstrom, constrained=True, dispersion_correction=True, nonbondedMethod=app.PME)

# waterbox contains a "system" object that is fully ready for MD simulation, but we're going to
# forget about this for the moment, and totally parameterize the system from the ground-up in OpenMM
# we are using the coordinates and parameters
system = waterbox.system

# let's add the 1st and 2nd atom of O2 ("OA" and "OB") to the system
# we will insert it to (-0.0508, 0, 0), (0.0508, 0, 0)
import numpy as np
system.addParticle(15.999*unit.amu)
system.addParticle(15.999*unit.amu)
# the two particles we've added have indices 1728 and 1729.

positions_list = (waterbox.positions/unit.nanometer).tolist()
positions_list.append([-0.0508, 0.0, 0.0])
positions_list.append([0.0508, 0.0, 0.0])
positions = np.array(positions_list)*unit.nanometer 
# and now let's transpose the position again s.t. it sits within (0,0,0),(Lx,Ly,Lz)
positions[:,0] -= np.amin(positions[:,0])
positions[:,1] -= np.amin(positions[:,1])
positions[:,2] -= np.amin(positions[:,2])

# add a constraint between the 1st and 2nd atom of O2
system.addConstraint(1728, 1729, 0.1016*unit.nanometer)

# add O2 to the system topology
topology = waterbox.topology
topology.addChain('O2') # this is the 2nd Chain in the topology
O2chain = list(topology.chains())[-1] # select the 2nd chain object
topology.addResidue('O2', O2chain) # add a residue called 'O2' to the Chain 'O2'
O2residue = list(topology.residues())[-1]
topology.addAtom('OA', app.Element.getBySymbol('O'), O2residue) 
topology.addAtom('OB', app.Element.getBySymbol('O'), O2residue) 

# we're going to pull the parameters for water out of the waterbox system to form the alchemical part of the LJ interaction
# get the list of all Force objects in the sytem
forces = { force.__class__.__name__ : force for force in system.getForces() }
# pull out the NonbondedForce. We're going to use all of the parameter values in this.
waterbox_nbforce = forces['NonbondedForce']

# we will create a CustomNonbondedForce to represent all alchemical LJ interactions
# via this force, water-water interactions are zero, water-O2 interactions are scaled by lambda
# we will set the initial value of lambda to 0 -- no interaction between O2 and water
lambda_value = 0.0
alchemical_nbforce = mm.CustomNonbondedForce("""4*epsilon*l12*(1/((alphaLJ*(1-l12) + (r/sigma)^6)^2) - 1/(alphaLJ*(1-l12) + (r/sigma)^6));
                                sigma=0.5*(sigma1+sigma2);
                                epsilon=sqrt(epsilon1*epsilon2);
                                alphaLJ=0.5;
                                l12=1-(1-lambda)*step(useLambda1+useLambda2-0.5)""")
alchemical_nbforce.addPerParticleParameter("sigma") # parameter #1
alchemical_nbforce.addPerParticleParameter("epsilon") # parameter #2
alchemical_nbforce.addPerParticleParameter("useLambda") # parameter #3. 1==Alchemical, 0==Not Alchemical
alchemical_nbforce.addGlobalParameter("lambda", lambda_value) # set the initial lambda to 0 (O2 turned off)
alchemical_nbforce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
alchemical_nbforce.setCutoffDistance(12.0*unit.angstrom) 
for particle_index in range(system.getNumParticles()):
    if particle_index in [1728, 1729]:
        # Add nonbonded LJ parameters for our O2
        sigma = 0.313*unit.nanometer
        epsilon = 0.4029*unit.kilojoule/unit.mole # unit.kilojoules_per_mole is exactly the same
        alchemical_nbforce.addParticle([sigma, epsilon, 1]) # lambda starts at 0, we will increase it to 1
    elif particle_index not in [1728, 1729]:
        # Add nonbonded LJ parameters for a water
        sigma = waterbox_nbforce.getParticleParameters(particle_index)[1]
        epsilon = waterbox_nbforce.getParticleParameters(particle_index)[2]
        alchemical_nbforce.addParticle([sigma, epsilon, 0]) # lambda is ALWAYS 0
system.addForce(alchemical_nbforce)
# use a switching function to smoothly truncate forces to zero from 10-12 angstrom
alchemical_nbforce.setUseSwitchingFunction(use=True)
alchemical_nbforce.setSwitchingDistance(2.0*unit.angstrom)


# and now we're going to build out a "normal", non-alchemical NonbondedForce.
# Even though O2 will not be part of this force, any NonbondedForce object
# must contain parameters for EVERY atom in the system, even if the parameters are zero
nbforce = mm.NonbondedForce()
nbforce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
nbforce.setCutoffDistance(12.0*unit.angstrom)
nbforce.setUseSwitchingFunction(use=True)
nbforce.setSwitchingDistance(2.0*unit.angstrom)
# use the long-range dispersion correction for isotropic fluids in NPT
# Michael R. Shirts, David L. Mobley, John D. Chodera, and Vijay S. Pande.
# Accurate and efficient corrections for missing dispersion interactions in molecular simulations.
# Journal of Physical Chemistry B, 111:13052–13063, 2007.
nbforce.setUseDispersionCorrection(True)
for particle_index in range(system.getNumParticles()):
    # set LJ parameters of each paricle
    if particle_index in [1728, 1729]:
        # O2 has 0 -- the only way it interats with water is via alchemical_nbforce
        charge = 0.0*unit.coulomb 
        sigma = 0.0*unit.nanometer
        epsilon = 0.0*unit.kilojoule/unit.mole
        nbforce.addParticle(charge, sigma, epsilon)
    elif particle_index not in [1728, 1729]:
        # Add nonbonded LJ parameters for a water
        charge = waterbox_nbforce.getParticleParameters(particle_index)[0]
        sigma = waterbox_nbforce.getParticleParameters(particle_index)[1]
        epsilon = waterbox_nbforce.getParticleParameters(particle_index)[2]
        nbforce.addParticle(charge, sigma, epsilon)
system.addForce(nbforce)

# now let's delete the original normal NonbondedForce of the system (waterbox_nbforce)
system.removeForce(2)

print('Constructing integrator')
integrator = mm.LangevinIntegrator(310*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)

print('Constructing and adding Barostat to system')
barostat = mm.MonteCarloBarostat(1.0*unit.bar, 310*unit.kelvin, 25)
system.addForce(barostat)

print('Selecting MD platform')
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0"

print('Constructing simulation context')
simulation = app.Simulation(topology, system, integrator, platform, properties)

print('Setting initial condition')
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(310*unit.kelvin)

print('Minimizing with lambda=0')
simulation.minimizeEnergy()
minpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(topology, minpositions, open('O2_alchemical_insertion_min.pdb', 'w'))

print('Setting reporters')

mdsteps   = 500*11 #   1 ps per lambda condition at 2 fs timestep
dcdperiod =  500   #   1 ps at 2 fs timestep
logperiod =   50   # 0.1 ps at 2 fs timestep
from sys import stdout # we'll use this to print output to the terminal during simulation
simulation.reporters.append(app.StateDataReporter(stdout, logperiod, step=True, 
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.StateDataReporter('O2_alchemical_insertion.log', logperiod, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.DCDReporter('O2_alchemical_insertion.dcd', dcdperiod))

# Now let's do the alchemical insertion... every 1 ps we'll increase lambda by 0.1
# We'll be able to indirectly observe the alchemical insertion via Total Energy over time
for i in range(10):
    print('Simulating for 1 ps at lambda=%f'%lambda_value)
    simulation.step(500)
    lambda_value += 0.1
    simulation.context.setParameter('lambda', lambda_value)
print('Simulating for 1 ps at lambda=%f'%lambda_value)
simulation.step(500)

lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(topology, lastpositions, open('O2_alchemical_insertion_last.pdb', 'w'))

# Now we will save a serialization of this simulation into OpenMM's native XML format
# We can re-initialize the system later for further simulations without all of the bothersome set-up by loading these files!
# We'll write exactly the same XML files Folding@home uses to transfer simulation data for restarts to/from users
state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, getParameters=True, enforcePeriodicBox=True)

# system.xml contains all of the force field parameters
with open('O2_system.xml', 'w') as f:
     system_xml = mm.XmlSerializer.serialize(system)
     f.write(system_xml)
# integrator.xml contains the congiruation for the integrator
with open('O2_integrator.xml', 'w') as f:
     integrator_xml = mm.XmlSerializer.serialize(integrator)
     f.write(integrator_xml)

# state.xml contains positions, velocities, forces, the barostat
with open('O2_state.xml', 'w') as f:
     f.write(mm.XmlSerializer.serialize(state))

# there is also a binray "Checkpoint" file that can resume the state of the random number generator....
# but restarts using a "Checkpoint" file only work on the same hardware+software combination.
# this is part of why OpenMM's premier Thermostat is Langevin.
simulation.saveCheckpoint('O2_state.chk')
