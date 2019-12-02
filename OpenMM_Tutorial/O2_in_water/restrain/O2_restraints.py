import simtk.openmm as mm
import simtk.openmm.app as app
from simtk import unit

# load up the system, integrator, and state. Easy!
system = mm.XmlSerializer.deserialize(open('../insert/O2_system.xml').read())
integrator = mm.XmlSerializer.deserialize(open('../insert/O2_integrator.xml').read())
state = mm.XmlSerializer.deserialize(open('../insert/O2_state.xml').read())

# we'll just take the topology from here...
pdb = app.PDBFile('../insert/O2_alchemical_insertion_last.pdb')
topology = pdb.topology

# let's specify our simulation platform again
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0"

# and we could reconstruct the simulation no problem, if we wanted to
#simulation = app.Simulation(topology, system, integrator, platform, properties)
#simulation.context.setState(state)
# but we won't do that yet. Let's make a restraint to pull O2 to the center of the system

# let's make the linear restraint, setting x0, y0, z0 to the center of the system
centerforce = mm.CustomExternalForce("k*(abs(x-x0)+abs(y-y0)+abs(z-z0))")
centerforce.addGlobalParameter("k", 5.0*unit.kilojoule/unit.angstrom/unit.mole)
centerforce.addPerParticleParameter("x0")
centerforce.addPerParticleParameter("y0")
centerforce.addPerParticleParameter("z0")
import numpy as np
xmean = np.mean(np.array(state.getPositions()/unit.nanometer)[:,0])*unit.nanometer
ymean = np.mean(np.array(state.getPositions()/unit.nanometer)[:,1])*unit.nanometer
zmean = np.mean(np.array(state.getPositions()/unit.nanometer)[:,2])*unit.nanometer
centerforce.addParticle(1728, mm.Vec3(xmean, ymean, zmean))
centerforce.addParticle(1729, mm.Vec3(xmean, ymean, zmean))
system.addForce(centerforce)

# ok now let's do some simulation using this restraint
simulation = app.Simulation(topology, system, integrator, platform, properties)
simulation.context.setState(state)

# set up reporters so we can see what's going on...
mdsteps   = 55000   # 110 ps total simulation
dcdperiod =    50   # 0.1 ps at 2 fs timestep
logperiod =    50   # 0.1 ps at 2 fs timestep
from sys import stdout # we'll use this to print output to the terminal during simulation
simulation.reporters.append(app.StateDataReporter(stdout, logperiod, step=True, 
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.StateDataReporter('O2_restraints.log', logperiod, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.DCDReporter('O2_restraints.dcd', dcdperiod))

# 5000 steps should be more than enough to pull O2 to the center
simulation.step(5000)
# ok now let's remove centerforce from the system
simulation.system.removeForce(5)

# and now let's make a new force restraining O2 to a 6-angstrom flat-well harmonic potential
flatzforce = mm.CustomExternalForce('k * (pz^2); \
                               pz = max(0, delta); \
                               delta = r - width; \
                               r = abs(periodicdistance(x, y, z, x, y, z0));')
flatzforce.addGlobalParameter('k', 1.0*unit.kilojoule/unit.angstrom**2/unit.mole)
flatzforce.addGlobalParameter('width', 0.3*unit.nanometer)
flatzforce.addPerParticleParameter('z0')
flatzforce.addParticle(1728, [zmean])
flatzforce.addParticle(1729, [zmean])
simulation.system.addForce(flatzforce)

# we're going to "reinitialize" the simulation context to totally remove "centerforce"
# if we don't do this, the last force of "centerforce" lingers in the system
# we also need to do this to now include "flatzforce" in the system
positions = simulation.context.getState(getPositions=True).getPositions()
velocities = simulation.context.getState(getVelocities=True).getVelocities()
simulation.context.reinitialize()
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)

# Let's simulate O2 in this flat well for 100 ps
simulation.step(50000)
