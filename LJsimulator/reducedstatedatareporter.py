"""
reducedstatedatareporter.py: Outputs data about a simulation in reduced units

Hacked by George Pantelopulos

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Robert McGibbon

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
from __future__ import absolute_import
from __future__ import print_function
__author__ = "Peter Eastman"
__version__ = "1.0"

try:
    import bz2
    have_bz2 = True
except: have_bz2 = False

try:
    import gzip
    have_gzip = True
except: have_gzip = False

import simtk.openmm as mm
import simtk.unit as unit
import math
import time

class ReducedStateDataReporter(object):
    """StateDataReporter outputs information about a simulation, such as energy and temperature, to a file.

    To use it, create a StateDataReporter, then add it to the Simulation's list of reporters.  The set of
    data to write is configurable using boolean flags passed to the constructor.  By default the data is
    written in comma-separated-value (CSV) format, but you can specify a different separator to use.
    """

    def __init__(self, file, reportInterval, dimensions, T_r, step=False, time=False, potentialEnergy=False, kineticEnergy=False, totalEnergy=False, temperature=False, 
                 progress=False, remainingTime=False, speed=False, elapsedTime=False, separator=',', totalSteps=None):
        """Create a StateDataReporter.

        Parameters
        ----------
        file : string or file
            The file to write to, specified as a file name or file object
        reportInterval : int
            The interval (in time steps) at which to write frames
        dimensions : int
            The number of dimensions in the system
        T_r : float
            The reduced temperature
        step : bool=False
            Whether to write the current step index to the file
        time : bool=False
            Whether to write the current time to the file
        potentialEnergy : bool=False
            Whether to write the potential energy to the file
        kineticEnergy : bool=False
            Whether to write the kinetic energy to the file
        totalEnergy : bool=False
            Whether to write the total energy to the file
        temperature : bool=False
            Whether to write the instantaneous temperature to the file
        progress : bool=False
            Whether to write current progress (percent completion) to the file.
            If this is True, you must also specify totalSteps.
        remainingTime : bool=False
            Whether to write an estimate of the remaining clock time until
            completion to the file.  If this is True, you must also specify
            totalSteps.
        speed : bool=False
            Whether to write an estimate of the simulation speed in ns/day to
            the file
        elapsedTime : bool=False
            Whether to write the elapsed time of the simulation in seconds to
            the file.
        separator : string=','
            The separator to use between columns in the file
        totalSteps : int=None
            The total number of steps that will be included in the simulation.
            This is required if either progress or remainingTime is set to True,
            and defines how many steps will indicate 100% completion.
        """
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        if (progress or remainingTime) and totalSteps is None:
            raise ValueError('Reporting progress or remaining time requires total steps to be specified')
        if self._openedFile:
            # Detect the desired compression scheme from the filename extension
            # and open all files unbuffered
            if file.endswith('.gz'):
                if not have_gzip:
                    raise RuntimeError("Cannot write .gz file because Python could not import gzip library")
                self._out = gzip.GzipFile(fileobj=open(file, 'wb', 0))
            elif file.endswith('.bz2'):
                if not have_bz2:
                    raise RuntimeError("Cannot write .bz2 file because Python could not import bz2 library")
                self._out = bz2.BZ2File(file, 'w', 0)
            else:
                self._out = open(file, 'w')
        else:
            self._out = file
        self._step = step
        self._time = time
        self._dimensions = dimensions
        self._T_r = T_r
        self._potentialEnergy = potentialEnergy
        self._kineticEnergy = kineticEnergy
        self._totalEnergy = totalEnergy
        self._temperature = temperature
        self._progress = progress
        self._remainingTime = remainingTime
        self._speed = speed
        self._elapsedTime = elapsedTime
        self._separator = separator
        self._totalSteps = totalSteps
        self._hasInitialized = False
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needEnergy = potentialEnergy or kineticEnergy or totalEnergy or temperature

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, self._needsPositions, self._needsVelocities, self._needsForces, self._needEnergy)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(simulation)
            headers = self._constructHeaders()
            print('#"%s"' % ('"'+self._separator+'"').join(headers), file=self._out)
            self._out.flush()
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Query for the values
        values = self._constructReportValues(simulation, state)

        # Write the values.
        print(self._separator.join(str(v) for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _constructReportValues(self, simulation, state):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        """
        system = simulation.system
        T = simulation.integrator.getGlobalVariableByName('T')
        values = []
        clockTime = time.time()
        if self._progress:
            values.append('%.1f%%' % (100.0*simulation.currentStep/self._totalSteps))
        if self._step:
            values.append( simulation.currentStep )
        if self._time:
            values.append(state.getTime().value_in_unit(unit.femtoseconds))
        if self._potentialEnergy:
            values.append( state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / system.getNumParticles() )
        if self._kineticEnergy:
            values.append( state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) / system.getNumParticles() )
        if self._totalEnergy:
            values.append( (state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole) / system.getNumParticles() )
        if self._temperature and self._dimensions == 2:
            values.append( ( (2*state.getKineticEnergy()/(self._dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin) / (T*(2./3.)) )*self._T_r )
        if self._temperature and self._dimensions == 3:
            values.append( ( (2*state.getKineticEnergy()/(self._dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin) / T )*self._T_r )
        if self._speed:
            elapsedHours = (clockTime-self._initialClockTime)/3600.0
            elapsedSteps = simulation.currentStep-self._initialSteps
            if elapsedHours > 0.0:
                values.append('%.3g' % (elapsedSteps/elapsedHours))
            else:
                values.append('--')
        if self._elapsedTime:
            values.append(time.time() - self._initialClockTime)
        if self._remainingTime:
            elapsedSeconds = clockTime-self._initialClockTime
            elapsedSteps = simulation.currentStep-self._initialSteps
            if elapsedSteps == 0:
                value = '--'
            else:
                estimatedTotalSeconds = (self._totalSteps-self._initialSteps)*elapsedSeconds/elapsedSteps
                remainingSeconds = int(estimatedTotalSeconds-elapsedSeconds)
                remainingDays = remainingSeconds//86400
                remainingSeconds -= remainingDays*86400
                remainingHours = remainingSeconds//3600
                remainingSeconds -= remainingHours*3600
                remainingMinutes = remainingSeconds//60
                remainingSeconds -= remainingMinutes*60
                if remainingDays > 0:
                    value = "%d:%d:%02d:%02d" % (remainingDays, remainingHours, remainingMinutes, remainingSeconds)
                elif remainingHours > 0:
                    value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
                elif remainingMinutes > 0:
                    value = "%d:%02d" % (remainingMinutes, remainingSeconds)
                else:
                    value = "0:%02d" % remainingSeconds
            values.append(value)
        return values

    def _initializeConstants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (Simulation) The simulation to generate a report for
        """
        system = simulation.system
        if self._temperature:
            # Compute the number of degrees of freedom.
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0*unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3
            self._dof = dof

    def _constructHeaders(self):
        """Construct the headers for the CSV output

        Returns: a list of strings giving the title of each observable being reported on.
        """
        headers = []
        if self._progress:
            headers.append('Progress (%)')
        if self._step:
            headers.append('Step')
        if self._time:
            headers.append('Time (step)')
        if self._potentialEnergy:
            headers.append('Potential Energy (epsilon/N)')
        if self._kineticEnergy:
            headers.append('Kinetic Energy (epsilon/N)')
        if self._totalEnergy:
            headers.append('Total Energy (epsilon/N)')
        if self._temperature:
            headers.append('Temperature (kbT/epsilon)')
        if self._speed:
            headers.append('Speed (steps/hour)')
        if self._elapsedTime:
            headers.append('Elapsed Time (s)')
        if self._remainingTime:
            headers.append('Time Remaining')
        return headers

    def _checkForErrors(self, simulation, state):
        """Check for errors in the current state of the simulation

        Parameters
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """
        if self._needEnergy:
            energy = (state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole)
            if math.isnan(energy):
                raise ValueError('Energy is NaN')
            if math.isinf(energy):
                raise ValueError('Energy is infinite')

    def __del__(self):
        if self._openedFile:
            self._out.close()
