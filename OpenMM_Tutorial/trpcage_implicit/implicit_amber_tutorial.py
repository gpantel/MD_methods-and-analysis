import simtk.openmm as mm # contains functions MD work
import simtk.openmm.app as app # contains functions for i/o
from simtk import unit # controls unique object types for physical units

# app.AmberPrmtopFile parses the system topology and force field parameters, constructing an object.
# We'll store the constructed object as "forcefield"
# there are other topology file parsers in app: GromacsTopFile, CharmmPsfFile+CharmmParameterSet
# these parsers actually quite work well (except for restraints and constraints)
print('Parsing system topology')
forcefield = app.AmberPrmtopFile('ff99sb_jacs2002.prmtop')

# app.PDBfile parses the system coordinates, constructing an object.
# We'll store the constructed object as "coord"
# there are several other parses in app: AmberInpCrd, CharmmCrdFile, GromacsGroFile
print('Parsing system coordinates')
coord = app.PDBFile('linear_trpcage.pdb')

# forcefield.createSystem constructs an object containing the complete force field parameters of the system
# We can actually modify "system" after construction and even between MD steps
print('Constructing sytem')
system = forcefield.createSystem(implicitSolvent=app.OBC2, soluteDielectric=1.0, solventDielectric=78.5)

# the integrator object is only the thermostat+MD integrator
# There is only a Langevin and Andersen thermostat in OpenMM in the pre-compiled distribution
# However, you can define custom integrators (more on this later)
print('Constructing integrator')
integrator = mm.LangevinIntegrator(325*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)

# mm.Platform selects the MD engine to use. There are engines for CPU, CUDA, and OpenCL
# CUDA is the most-optimized MD engine
print('Selecting MD platform')
platform = mm.Platform.getPlatformByName('CUDA')

# We can set the precision scheme to use for MD.
# "single": Nearly all work is done with single precision.
# "mixed": Forces are in single precision. Integration steps are in double precision.
# "double": All work is done with double precision. GPUs are bad at this. Can't really use it.
# "mixed" is generally fast and stable. I always use "mixed".
properties = {'CudaPrecision': 'mixed'}

# We should also specify which GPU we want to use for the simulation
properties["DeviceIndex"] = "0"

# app.Simulation constructs an "context" object to unify and control:
# system topology, parameters, integrator, MD platform
# simulation controls the context
print('Constructing simulation context')
simulation = app.Simulation(forcefield.topology, system, integrator, platform, properties)

# now we need set up the simulation's context with an initial conditition
print('Setting initial condition')
simulation.context.setPositions(coord.positions)
simulation.context.setVelocitiesToTemperature(325*unit.kelvin)

# let's decide here now long we want to run the simulation and the file writing period
mdsteps    =  2500000  #  5  ns at 2 fs timestep
dcdperiod  =    50000  #  10 ps at 2 fs timestep
logperiod  =    50000  #  10 ps at 2 fs timestep

# now we will set up "reporters" to write out thermodynamic data and coordinates
# you can choose which system variables you want to have output from app.StateDataReporter
print('Setting reporters')
from sys import stdout # we'll use this to print output to the terminal during simulation
simulation.reporters.append(app.StateDataReporter(stdout, logperiod, step=True, 
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, progress=True, 
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.StateDataReporter('trpcage_implicit.log', logperiod, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.DCDReporter('trpcage_implicit.dcd', dcdperiod))

# now let's minimize the system to convergence (steepest descent)
print('Minimizing')
simulation.minimizeEnergy()

# let's save the minimized structure as a PDB by
# (1) exctracting the system positions from the context
# (2) Using app.PDBfile to write a PDB of the system with these positions
minpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(forcefield.topology, minpositions, open('trpcage_implicit_min.pdb', 'w'))

# and now we run the simulation for "mdsteps" number of steps
print('Running MD')
simulation.step(mdsteps)

# just for this tutorial, let's also save a PDB of the last frame of the simulation
lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(forcefield.topology, lastpositions, open('trpcage_implicit_last.pdb', 'w'))
