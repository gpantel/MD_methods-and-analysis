#############################################################################################
# This script is a hack of Peter Eastmann's Simulated Tempering imlementation in OpenMM to
# implement Mass Scaling Simulated Tempering
# The implementation this script is derivative of is all described in the following comments
# section and the class "MSST"
# - George A. Pantelopulos
#############################################################################################

from __future__ import print_function

"""
simulatedtempering.py: Implements simulated tempering

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2015 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__author__ = "Peter Eastman and George A. Pantelopulos"
__version__ = "1.0"

import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import math
import random
import os, sys
import mdtraj as md
import parmed
import numpy as np
from sys import stdout

try:
    import bz2
    have_bz2 = True
except: have_bz2 = False

try:
    import gzip
    have_gzip = True
except: have_gzip = False

class MSST(object):
    """MSST implements the mass-scaling simulated tempering method.
    See Nagai, T., Pantelopulos, G. A., Takahashi, T. and Straub, J. Comput. Chem. 37, 2017-2028 (2016).
    
    It runs a simulation while allowing the temperature to vary and rescaling masses.  At high temperatures, 
    it can more easily cross energy barriers to explore a wider area of conformation space.  At low temperatures, it can
    thoroughly explore each local region. Masses are rescaled such that longer time steps may be used at high temperature.
    For details on simulated tempering, see Marinari, E. and Parisi, G., Europhys. Lett. 19(6). pp. 451-458 (1992).
    
    The set of temperatures to sample can be specified in two ways.  First, you can explicitly provide a list
    of temperatures by using the "temperatures" argument.  Alternatively, you can specify the minimum and
    maximum temperatures, and the total number of temperatures to use.  The temperatures are chosen spaced
    exponentially between the two extremes. Additionally, sets of atoms are selected to linearly scale up with
    temperature, down with temperature, or not at all, setting the maximum and minimum scaling factor for each
    of these sets. The initial masses of all particles (Tmin_masses) should be provided in a 1-D array. For example,
 
    st = MSST(simulation, numTemperatures=21, minTemperature=273*unit.kelvin, maxTemperature=819*unit.kelvin, \
              mass_upscale_atom_indices=scale_up_atom_indices, mass_downscale_atom_indices=scale_down_atom_indices, \
              other_atom_indices=other_atom_indices, Tmin_masses=Tmin_masses)
    
    After creating the MSST object, call step() on it to run the simulation.
    
    Transitions between temperatures are performed at regular intervals, as specified by the "tempChangeInterval"
    argument.  For each transition, a new temperature is selected using the independence sampling method, as
    described in Chodera, J. and Shirts, M., J. Chem. Phys. 135, 194110 (2011).
    
    Simulated tempering requires a "weight factor" for each temperature.  Ideally, these should be chosen so
    the simulation spends equal time at every temperature.  You can specify the list of weights to use with the
    optional "weights" argument.  If this is omitted, weights are selected automatically using the Wang-Landau
    algorithm as described in Wang, F. and Landau, D. P., Phys. Rev. Lett. 86(10), pp. 2050-2053 (2001).
    
    To properly analyze the results of the simulation, it is important to know the temperature and weight factors
    at every point in time.  The MSST object functions as a reporter, writing this information
    to a file or stdout at regular intervals (which should match the interval at which you save frames from the
    simulation).  You can specify the output file and reporting interval with the "reportFile" and "reportInterval"
    arguments.
    """

    def __init__(self, simulation, temperatures=None, numTemperatures=None, minTemperature=None, maxTemperature=None, \
        initTemperature=0, weights=None, alphamax=3.0, alphamin=0.25, tempChangeInterval=25, reportInterval=1000, \
        mass_upscale_atom_indices=None, mass_downscale_atom_indices=None, other_atom_indices=None, Tmin_masses=None, reportFile=stdout):
        """Create a new MSST.
        
        Parameters:
         - simulation (Simulation) The Simulation defining the System, Context, and Integrator to use
         - temperatures (list=None) The list of temperatures to use for tempering, in increasing order.
         - numTemperatures (int=None) The number of temperatures to use for tempering.  If temperatures is not None,
           this is ignored.
         - minTemperature (temperature=None) The minimum temperature to use for tempering.  If temperatures is not None,
           this is ignored.
         - maxTemperature (temperature=None) The maximum temperature to use for tempering.  If temperatures is not None,
           this is ignored.
         - initTemperature (temperature index=0) The index of the initial temperature for the system.
         - weights (list=None) The weight factor for each temperature.  If none, weights are determined on-the-fly using the Wang-Landau algorithm.
         - alphamin (float=0.25) The minimum alpha for downscaling alphas.
         - alphamax (float=3.0) The maximum alpha for upscaling alphas.
         - tempChangeInterval (int=25) The interval (in time steps) at which to attempt transitions between temperatures
         - reportInterval (int=1000) The interval (in time steps) at which to write information to the report file
         - mass_upscale_atom_indices (1d numpy array=None) The indices of the atoms for which masses will be scaled up
         - mass_downscale_atom_indices (1d numpy array=None) The indices of the atoms for which masses will be scaled up
         - other_atom_indices (1d numpy array=None) The indices of the atoms for which masses will not be scaled
         - Tmin_masses (1dx3 numpy array=None) The masses of all atoms in the system prior to any scaling.
         - reportFile (string or file=stdout) The file to write reporting information to, specified as a file name or file object
        """
        self.simulation = simulation
        if temperatures is None:
            if unit.is_quantity(minTemperature):
                minTemperature = minTemperature.value_in_unit(unit.kelvin)
            if unit.is_quantity(maxTemperature):
                maxTemperature = maxTemperature.value_in_unit(unit.kelvin)
            self.temperatures = [minTemperature*((float(maxTemperature)/minTemperature)**(i/float(numTemperatures-1))) for i in range(numTemperatures)]*unit.kelvin
        else:
            numTemperatures = len(temperatures)
            self.temperatures = [(t.value_in_unit(unit.kelvin) if unit.is_quantity(t) else t)*unit.kelvin for t in temperatures]
            if any(self.temperatures[i] >= self.temperatures[i+1] for i in range(numTemperatures-1)):
                raise ValueError('The temperatures must be in strictly increasing order')

        self.other_atom_indices = other_atom_indices
        self.mass_upscale_atom_indices = mass_upscale_atom_indices
        self.mass_downscale_atom_indices = mass_downscale_atom_indices
        self.Tmin_masses = Tmin_masses

        ### Linearlly-spaced alphas based on alphamin, aphamax, and temperature input if not supplied to MSST
        Tmin, Tmax = self.temperatures[0], self.temperatures[-1]
        def downalpha(Tmax, Tmin, alphamin, temp):
            alpha = 1 - (((1 - alphamin)*(temp-Tmin))/(Tmax-Tmin))
            return alpha

        def upalpha(Tmax, Tmin, alphamax, temp):
            alpha = 1 + (((alphamax - 1)*(temp-Tmin))/(Tmax-Tmin))
            return alpha

        # define alphas for scaling selected group masses up and down
        upalphas   = []
        downalphas = []
        for temp in self.temperatures:
            upalphas.append(upalpha(self.temperatures[-1],self.temperatures[0],alphamax,temp))
            downalphas.append(downalpha(self.temperatures[-1],self.temperatures[0],alphamin,temp))
        self.upalphas = upalphas
        self.downalphas = downalphas

        self.tempChangeInterval = tempChangeInterval
        self.reportInterval = reportInterval
        self.inverseTemperatures = [1.0/(unit.MOLAR_GAS_CONSTANT_R*t) for t in self.temperatures]
        self.transitionCounts = 0.0 # number of successful transitions
        self.transitionRatio  = 0.0 # ratio of successful transitions per total number of transitions

        # If necessary, open the file we will write reports to.
        self._openedFile = isinstance(reportFile, str)
        if self._openedFile:
            # Detect the desired compression scheme from the filename extension
            # and open all files unbuffered
            if reportFile.endswith('.gz'):
                if not have_gzip:
                    raise RuntimeError("Cannot write .gz file because Python could not import gzip library")
                self._out = gzip.GzipFile(fileobj=open(reportFile, 'wb', 0))
            elif reportFile.endswith('.bz2'):
                if not have_bz2:
                    raise RuntimeError("Cannot write .bz2 file because Python could not import bz2 library")
                self._out = bz2.BZ2File(reportFile, 'w', 0)
            else:
                self._out = open(reportFile, 'w')
        else:
            self._out = reportFile
        
        # Initialize the weights.
        if weights is None:
            self._weights = [0.0]*numTemperatures
            self._updateWeights = True
            self._weightUpdateFactor = 1.0
            self._histogram = [0]*numTemperatures
            self._hasMadeTransition = False
        else:
            self._weights = weights
            self._updateWeights = False

        # Select the initial temperature.
        self.currentTemperature = initTemperature
        self.simulation.integrator.setTemperature(self.temperatures[self.currentTemperature])
        
        # We set the mass in accordance with initial temperature and check the alpha values
        print("employed upalphas is ", upalphas)
        print("employed downalphas is", downalphas)
        print("scaling_up index", self.mass_upscale_atom_indices)
        print("scaling_down index", self.mass_downscale_atom_indices)
        if self.mass_upscale_atom_indices != None:
            mass_upscale_atom_indices_masses = self.upalphas[self.currentTemperature]*self.Tmin_masses
            for k in range(len(self.mass_upscale_atom_indices)): # set upscale atom masses to their current masses
                self.simulation.system.setParticleMass(self.mass_upscale_atom_indices[k], mass_upscale_atom_indices_masses[self.mass_upscale_atom_indices[k]]*unit.dalton)

        if self.mass_downscale_atom_indices != None:
            mass_downscale_atom_indices_masses = self.downalphas[self.currentTemperature]*self.Tmin_masses
            for k in range(len(self.mass_downscale_atom_indices)): # set upscale atom masses to their current masses
                self.simulation.system.setParticleMass(self.mass_downscale_atom_indices[k], mass_downscale_atom_indices_masses[self.mass_downscale_atom_indices[k]]*unit.dalton)

        masses = np.zeros(simulation.topology.getNumAtoms()) # make empty array
        for i in range(simulation.topology.getNumAtoms()): # set indices of array to particle masses in order of side chain atom index
            masses[i] = simulation.system.getParticleMass(i)/unit.dalton
        print("masses are", masses, file=sys.stderr)
        init_positions = simulation.context.getState(getPositions=True).getPositions()
        self.simulation.context.reinitialize()
        simulation.context.setPositions(init_positions)
        simulation.context.setVelocitiesToTemperature(Tmin*unit.kelvin)

        # Add a reporter to the simulation which will handle the updates and reports.
        class STReporter(object):
            def __init__(self, st):
                self.st = st

            def describeNextReport(self, simulation):
                st = self.st
                steps1 = st.tempChangeInterval - simulation.currentStep%st.tempChangeInterval
                steps2 = st.reportInterval - simulation.currentStep%st.reportInterval
                steps = min(steps1, steps2)
                isUpdateAttempt = (steps1 == steps)
                return (steps, False, isUpdateAttempt, False, isUpdateAttempt)

            def report(self, simulation, state):
                st = self.st
                if simulation.currentStep%st.tempChangeInterval == 0:
                    st._attemptTemperatureChange(state)
                if simulation.currentStep%st.reportInterval == 0:
                    st._writeReport()
        
        simulation.reporters.append(STReporter(self))
        
        # Write out the header line.
        headers = ['Steps', 'Temperature (K)', 'Transition Success Ratio']
        for t in self.temperatures:
            headers.append('%gK Weight' % t.value_in_unit(unit.kelvin))
        print('#"%s"' % ('"\t"').join(headers), file=self._out)

    def __del__(self):
        if self._openedFile:
            self._out.close()
    
    @property
    def weights(self):
        return [x-self._weights[0] for x in self._weights]

    def step(self, steps):
        """Advance the simulation by integrating a specified number of time steps."""
        self.simulation.step(steps)
    
    def _attemptTemperatureChange(self, state):
        """Attempt to move to a different temperature."""
        i = self.currentTemperature
        
        probability = [np.exp((self._weights[j]-self._weights[i])-(self.inverseTemperatures[j]-self.inverseTemperatures[i])*state.getPotentialEnergy()) for j in range(len(self._weights))]
        scale = 1.0/sum(probability)
        probability = [p*scale for p in probability]
        r = random.random()
        print(r, probability[np.argsort(probability)[-1]])
        if r < probability[np.argsort(probability)[-1]]:
            j = np.argsort(probability)[-1]
            if j != self.currentTemperature:
                print(j)
                # Select this temperature.
                self._hasMadeTransition = True
                self.currentTemperature = j
                self.simulation.integrator.setTemperature(self.temperatures[j])
                # set up scaling factors for moving from state i to state j
                st_scale = math.sqrt(self.temperatures[j]/self.temperatures[i])
                up_msst_scale   = (math.sqrt(self.upalphas[i]/self.upalphas[j]))*(math.sqrt(self.temperatures[j]/self.temperatures[i]))
                down_msst_scale = (math.sqrt(self.downalphas[i]/self.downalphas[j]))*(math.sqrt(self.temperatures[j]/self.temperatures[i]))
                # extract current velocities
                current_velocities = state.getVelocities().value_in_unit(unit.nanometers/unit.picoseconds)
                # extract current Positions
                positions = simulation.context.getState(getPositions=True).getPositions()
                # set up empty array to set velocities inside
                velocities = np.zeros((len(state.getVelocities().value_in_unit(unit.nanometers/unit.picoseconds)),3))
                # scale velocities with temp for non-selected atoms
                if self.other_atom_indices != None:
                    for k in range(len(self.other_atom_indices)):
                        velocities[self.other_atom_indices[k]] = st_scale*(current_velocities[self.other_atom_indices[k]])
                # set set masses and scale velocities with mass and temp for selected atom indices in target state j
                if self.mass_upscale_atom_indices != None:
                    mass_upscale_atom_indices_masses = self.upalphas[j]*self.Tmin_masses
                    for k in range(len(self.mass_upscale_atom_indices)): # set upscale atom masses to their current masses
                        self.simulation.system.setParticleMass(self.mass_upscale_atom_indices[k], mass_upscale_atom_indices_masses[self.mass_upscale_atom_indices[k]]*unit.dalton)
                        velocities[self.mass_upscale_atom_indices[k]] = up_msst_scale*(current_velocities[self.mass_upscale_atom_indices[k]])
                if self.mass_downscale_atom_indices != None:
                    mass_downscale_atom_indices_masses = self.downalphas[j]*self.Tmin_masses
                    for k in range(len(self.mass_downscale_atom_indices)): # set upscale atom masses to their current masses
                        self.simulation.system.setParticleMass(self.mass_downscale_atom_indices[k], mass_downscale_atom_indices_masses[self.mass_downscale_atom_indices[k]]*unit.dalton)
                        velocities[self.mass_downscale_atom_indices[k]] = down_msst_scale*(current_velocities[self.mass_downscale_atom_indices[k]])
                # reinitialize the context to truly set the new masses. @TODO There are ways around this if we don't use any restraints!
                self.simulation.context.reinitialize()
                self.simulation.context.setPositions(positions)
                self.simulation.context.setVelocities(velocities)
                self.transitionCounts += 1.0
                self.transitionRatio = np.true_divide(self.transitionCounts,((self.simulation.currentStep/self.tempChangeInterval)))
            if self._updateWeights:
                # Update the weight factors.
                self._weights[j] -= self._weightUpdateFactor
                self._histogram[j] += 1
                minCounts = min(self._histogram)
                if minCounts > 20 and minCounts >= 0.2*sum(self._histogram)/len(self._histogram):
                    # Reduce the weight update factor and reset the histogram.
                    self._weightUpdateFactor *= 0.5
                    self._histogram = [0]*len(self.temperatures)
                    self._weights = [x-self._weights[0] for x in self._weights]
                elif not self._hasMadeTransition and probability[self.currentTemperature] > 0.99:
                    # Rapidly increase the weight update factor at the start of the simulation to find
                    # a reasonable starting value.
                    self._weightUpdateFactor *= 2.0
                    self._histogram = [0]*len(self.temperatures)
            return
        r -= probability[j]

    def _writeReport(self):
        """Write out a line to the report."""
        temperature = self.temperatures[self.currentTemperature].value_in_unit(unit.kelvin)
        values = [temperature]+self.weights
        print(('%d\t' % self.simulation.currentStep) + ('%.3f\t' % np.around(temperature,3)) + ('%.3f\t' % np.around(self.transitionRatio,3)) + '\t'.join('%g' % v for v in values), file=self._out)


# Function for finding alpha (mass scaling factor) per temperature
def current_alpha(Tmax, Tmin, alphamin, temp):
    alpha = 1 - (((1 - alphamin)*(temp-Tmin))/(Tmax-Tmin))
    return alpha

###### THIS IS WHERE ALL OF THE SIMULATION PARAMETERS ARE SET ######
rep = int(sys.argv[1])
plat = 'CPU' # This may be set to 'CPU', 'CUDA', or 'OpenCL' - use which ever you wish
# minimum, maximum, and number of exponentially-spaced temperatures to use
Tmin              = 273
Tmax              = 819
nTemps            = 21
Tinit             = 0 # temperature index (from 0 to nTemps-1)
weights           = None
# simulation timestep
timestep          = 1.0*unit.femtoseconds # time step
# total simulation time
totaltime         = 0.10*unit.nanoseconds # total simulation time
total_steps       = int(np.around(totaltime/timestep)) # round total steps
# how often to save coordinates, simulation data
savetime          = 0.10*unit.picoseconds
save_steps        = int(np.around(savetime/timestep)) # round data writing interval
# how often to save a checkpoint file
checktime         = 5.0*unit.nanoseconds # checkpoint writing frequency
check_steps       = int(np.around(checktime/timestep)) # round checkpoint writing interval
# how often to attempt to change temperatures & masses
exchange_interval = 100
# set scaling factor for atom set to be scaled up - or down - to
alphamax          = 3.0
alphamin          = 1./3.
# numerical precision to use for GPU platform to attempt to run on
precision         = 'mixed' # options are 'single', 'mixed', and 'double'
# number of threads for CPU platform to attempt to run on, rather than the maximum possible
nthreads          = 8
# set up file names
trajname          = 'traj_%i.nc'%rep # contains coordinates data at savesteps interval
outname           = 'traj_%i.out'%rep # contains simulation data at savesteps interval
stname            = 'traj_msst_%i.out'%rep # contains information on current temperature and sampling weights
checkname         = 'traj_%i.chk'%rep # contains checkpoint data at checksteps interval
finalname         = 'final_%i.pdb'%rep # will be the structura at the end of the simulation

# put in simulation parameters here
print('Loading simulation topology, forcefield, and initial coordinates')
prmtop = app.AmberPrmtopFile('equil.prmtop')
inpcrd = app.PDBFile('min1.pdb')

print('Extracting protein sidechain atom indices from PDB file')
pdb = md.load('min1.pdb')
# atom indices to scale up in mass
#scale_up_ind = pdb.topology.select('symbol == H')
scale_up_ind = pdb.topology.select('all')
# atom indices to scale down in mass
scale_down_ind = [] #pdb.topology.select('not backbone and not symbol == H')
# all reamining atom indices
all_ind = pdb.topology.select('all')
other_ind = np.sort(list(set(all_ind) - set(scale_up_ind) - set(scale_down_ind)))

print('Constructing system...')
# Very standard options for AMBER GBSAII implicit solvation
system = prmtop.createSystem(implicitSolvent=app.OBC2, soluteDielectric=1.0, solventDielectric=78.5)

integrator = mm.LangevinIntegrator(Tmin*unit.kelvin, 1.0/unit.picosecond, timestep)

if plat == 'CPU':
    platform = mm.Platform.getPlatformByName(plat)
    os.system('export OPENMM_CPU_THREADS=%i'%nthreads) # set environment variable to control number of threads
    simulation = app.Simulation(prmtop.topology, system, integrator, platform)
elif plat == ('CUDA' or 'OpenCL'):
    platform = mm.Platform.getPlatformByName(plat)
    properties = {'CudaPrecision': precision}
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)

simulation.context.setPositions(inpcrd.positions)
simulation.context.setVelocitiesToTemperature(Tmin*unit.kelvin)

### Make an empty array to hold original masses to be input to MSST
Tmin_masses = np.zeros(pdb.n_atoms) # make empty array
for i in range(pdb.n_atoms): # set indices of array to particle masses in order of side chain atom index
    Tmin_masses[i] = simulation.system.getParticleMass(i)/unit.dalton
print("original masses are", Tmin_masses, file=sys.stderr)

print('Setting up reporters...')
simulation.reporters.append(app.StateDataReporter(outname, save_steps, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
simulation.reporters.append(app.StateDataReporter(stdout, save_steps, step=True,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
simulation.reporters.append(app.CheckpointReporter(checkname, check_steps))
#simulation.reporters.append(parmed.openmm.NetCDFReporter(trajname, save_steps, crds=True, vels=True))
simulation.reporters.append(parmed.openmm.NetCDFReporter(trajname, save_steps, crds=True, vels=True, frcs=True))

st = MSST(simulation, numTemperatures=nTemps, reportInterval=save_steps, weights=weights, tempChangeInterval=exchange_interval,\
                        initTemperature=Tinit, minTemperature=Tmin*unit.kelvin, maxTemperature=Tmax*unit.kelvin, mass_upscale_atom_indices=scale_up_ind, \
                        mass_downscale_atom_indices=scale_down_ind, other_atom_indices=other_ind, Tmin_masses=Tmin_masses, reportFile=stname)


print('Running sim tempering for %s steps'%(total_steps))
st.step(total_steps)

final_positions = simulation.context.getState(getPositions=True).getPositions()
final_masses = np.zeros(pdb.n_atoms) # make empty array
for i in range(pdb.n_atoms): # set indices of array to particle masses in order of side chain atom index
    final_masses[i] = simulation.system.getParticleMass(i)/unit.dalton
print("final masses are", final_masses, file=sys.stderr)
app.PDBFile.writeFile(prmtop.topology, final_positions, open(finalname, 'w'))
