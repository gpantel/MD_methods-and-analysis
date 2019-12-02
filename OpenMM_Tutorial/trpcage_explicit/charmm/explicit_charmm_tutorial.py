import simtk.openmm as mm
import simtk.openmm.app as app
from simtk import unit

# input topology, psf, and force field files generated from CHARMM-GUI Solution Builder
print('Parsing system topology')
topology = app.CharmmPsfFile('step3_charmm2omm.psf')
parameters = app.CharmmParameterSet('top_all36_prot.rtf',\
            'par_all36_prot.prm' ,'toppar_water_ions.str')

print('Parsing system coordinates')
# for using PBC in OpenMM, we need to make sure that
# the origin of the sytem is at (0,0,0)
# and that the extremum of the system is at (Lx, Ly, Lz)
coord = app.PDBFile('step3_pbcsetup.pdb')
# translate the coordinates, we'll use numpy here.
import numpy as np
xyz = np.array(coord.positions/unit.nanometer)
xyz[:,0] -= np.amin(xyz[:,0])
xyz[:,1] -= np.amin(xyz[:,1])
xyz[:,2] -= np.amin(xyz[:,2])
coord.positions = xyz*unit.nanometer

print('Constructing sytem')
# set periodic box vectors
topology.setBox(7.5*unit.nanometer, 7.5*unit.nanometer, 7.5*unit.nanometer)
# use PME for long-range electrostatics, cutoff for short-range interactions
# constrain H-bonds with RATTLE, constrain water with SETTLE
system = topology.createSystem(parameters, nonbondedMethod=app.PME, 
    nonbondedCutoff=1.2*unit.nanometers, constraints=app.HBonds, rigidWater=True, 
    ewaldErrorTolerance=0.0005)

print('Constructing integrator')
integrator = mm.LangevinIntegrator(325*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)

print('Constructing and adding Barostat to system')
barostat = mm.MonteCarloBarostat(1.0*unit.bar, 5.0*unit.kelvin, 25)
system.addForce(barostat)

print('Selecting MD platform')
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0"

print('Constructing simulation context')
simulation = app.Simulation(topology.topology, system, integrator, platform, properties)

print('Setting initial condition')
simulation.context.setPositions(coord.positions)
# we'll set the initial temperature to 5 K, before we do simulated annealing
simulation.context.setVelocitiesToTemperature(5*unit.kelvin)

mdsteps   = 65000 # 100 ps at 2 fs timestep
dcdperiod =   500 # 100 fs at 2 fs timestep
logperiod =   500 # 100 fs at 2 fs timestep

print('Setting reporters')
from sys import stdout # we'll use this to print output to the terminal during simulation
simulation.reporters.append(app.StateDataReporter(stdout, logperiod, step=True, 
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.StateDataReporter('trpcage_explicit.log', logperiod, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True, volume=True, density=True,
    remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(app.DCDReporter('trpcage_explicit.dcd', dcdperiod))

print('Minimizing')
simulation.minimizeEnergy()
minpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(topology.topology, minpositions, open('trpcage_explicit_min.pdb', 'w'))

print('Running Simulated Annealing MD')
# every 1000 steps raise the temperature by 5 K, ending at 325 K
T = 5
for i in range(65):
    simulation.step( int(mdsteps/65) )
    integrator.setTemperature( (T+(i*T))*unit.kelvin )
    barostat.setDefaultTemperature( (T+(i*T))*unit.kelvin )

lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(topology.topology, lastpositions, open('trpcage_explicit_last.pdb', 'w'))
