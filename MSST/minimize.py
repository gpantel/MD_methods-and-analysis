from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
pdb = app.PDBFile('1L2Y.pdb')
forcefield = app.ForceField('amber99sbnmr.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, 
     constraints=None, rigidWater=False)
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 
    2.0*unit.femtoseconds)
platform = mm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
print('Minimizing...')
simulation.minimizeEnergy(tolerance=0.0000001*unit.kilojoules_per_mole)
final_positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(pdb.topology, final_positions, open('1L2Y_min.pdb', 'w'))
