#%% addViscous.py
"""

Viscous correction code to augment Cart3D solutions.

Alexander Ward
April 2017

"""

#http://www.dtic.mil/dtic/tr/fulltext/u2/a045367.pdf

from math import exp, log, asin, sin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq
import time as time

#%% Inputs
""" 
Here are the majority of the user defined parameters. They are grouped into:
    Simulation      - Parameters on the simulation itself
    Freestream      - Freestream gas properties
    Transition      - Parameters controlling how you want to treat transition
    Vehicle         - Reference area and length, wall temperature
    Post Processing - Properties to save/report, slices to take
"""


""" -- SIMULATION -- """
#------------
# Cart3D solution filename {str}
cart3DFilename = 'Components.i.triq'
#------------
# Streamline filename {str}
streamlineFilename = '4.csv'
#------------
# Number of Tecplot generated streamlines {int}
nStreamlines = 68
#------------
# maximum number of steps in the the Tecplot streamlines {int}
maxSteps = 19999
#------------
# If you don't have streamlines available, set to 0 to guess the development 
# length based on flow direction (positive x) NOTE: If the vehicle starts in 
# negative x, you must set an offset to translate the nose to x = 0.
streamlineFlag = 0      # {bool}
LEoffset = 0.           # {float}
#------------
# If you want lots of info to print out, set this to true, {bool}
verbose = 0
#------------
# If you just want to debug, tell the code to only iterate over part of the 
# .triq file. Make sure to set the flag as well.
sampleAmount = 5000     # {int}
sampleFlag = 1           # {bool}


""" -- FREESTREAM -- """
#------------
# Freestream temperature, [K], {float}
T_inf = 62.157
#------------
# Freestream pressure, [Pa], {float}
p_inf = 3163.4
#------------
# Freestream density, [kg/m^3], {float}
rho_inf = 0.177268215
#------------
# Freestream Mach number, {float}
M_inf = 4.5099
#------------
# Ratio of specific heats, [Pa] (constant), {float}
g_inf = 1.4
#------------
# Angle of attack
alpha = 0. * np.pi/180
#------------
# Sideslip angle
beta = 0. * np.pi/180
#------------

""" -- TRANSITION -- """
#------------
# Set the point of transition (Re or streamline based length) OR set to False
# to use correlation.
criticalLocation = 0
#------------
# Set to 'laminar' or 'turbulent' if you want to assume and ignore transition
regime = 'turbulent'
#------------
# Roughness induced transition location (streamline coordinate)
roughnessCoordinate = 0
#------------
# Set true to have no transitional region (i.e. fully turbulent when tripped).
immediateTransitionFlag = 1
#------------

""" -- VEHICLE -- """
#------------
# Wall temperature of the vehicle surface [K], {float}
T_wall = 316.2
#------------
# Reference area for aerodynamic force coefficients
S_ref = 3.
#------------
# Reference length for aerodynamic moment coefficients (measured from nose)
L_ref = 1.5
#------------


""" -- POST PROCESSING -- """
#------------
# If you DON'T want to save a .csv file of local properties, set flag to 0
writeDataFlag = 1       # {bool}
#------------
# Additional properties to save to a .csv file (called flowData.csv) {List}
# You will always get ['x', 'y', 'z', 'cf', 'ch', 'Velocity', 'T_aw']
# Choose from ['M', 'rho', 'p', 'cp', 'T_e', 'U', 'V', 'W', 'BL Regime', 
#              'Dev Length', 'Re']
additionalProperties = ['Dev Length', 'Re', 'BL Regime']   # {list of str}
#------------
# Write data to this filename
outputFilename = 'flowData'    # {str}
#------------
# Save data to VTK format for nice viewing in ParaView
vtkFlag = 0
#------------
# Save data to csv
csvFlag = 0
#------------



###############################################################################
""" You probably (hopefully) won't have to change anything below this line. """
###############################################################################
#%% The code

class Fluid(object):
    """ Contains all the constant fluid properties
    """
    def __init__(self):
        # Sutherland law reference conditions and constants
        self.mu_0 = 1.716e-5
        self.T_0 = 273
        self.sutherland_S = 111
        
        # Gas constants
        self.R_universal = 8.314510
        self.R_air = 287.058
        self.molarMass_air = 28.964923942499997
        # Freestream conditions
        self.T_inf = T_inf
        self.p_inf = p_inf
        self.g_inf = g_inf
        
        self.a_inf = (self.g_inf * self.R_air * self.T_inf) **0.5

        # Constant specific heat value {float}
        self.constantCp = 1.015

    def calculate_cp(self, T):
        """
        This function calculates the specific heat capacity and ratio fo specific heats 
        at a temeprature T [K]. It uses the polynomial curve fit taken from NASA's CEA code
        McBride, B. Zehe, M. Gordon, S. (2002)
        "NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species"
    
        molar mass air = 28.964923942499997
    
        Specific heat capacities at 298.15 K:
        Cp{N2, O2, Ar, CO2} = 29.124, 29.378, 20.786, 37.135
        
        """
        verbose = 0
        
        if T < 200.:
            T = 200.
            if verbose:
                print "Problem, Temp < 200 K, I'll set it to 200 K"
        elif T > 6000.:
            T = 6000.
            if 1:
                print "Problem, Temp > 6000 K, I'll set it to 6000 K"
            
        if T < 1000.:
            N2 = [2.210371497E+04, -3.818461820E+02, 6.082738360E+00, -8.530914410E-03, 1.384646189E-05, -9.625793620E-09, 2.519705809E-12]
            O2 = [-3.425563420E+04, 4.847000970E+02, 1.119010961E+00, 4.293889240E-03, -6.836300520E-07,-2.023372700E-09, 1.039040018E-12]
            Ar = [0., 0., 2.5, 0., 0., 0., 0.]
            CO2 = [4.943650540E+04, -6.264116010E+02, 5.301725240E+00, 2.503813816E-03, -2.127308728E-07, -7.689988780E-10, 2.849677801E-13]
        else:
            N2 = [5.877124060E+05, -2.239249073E+03, 6.066949220E+00, -6.139685500E-04, 1.491806679E-07, -1.923105485E-11, 1.061954386E-15]
            O2 = [-1.037939022E+06, 2.344830282E+03, 1.819732036E+00, 1.267847582E-03,-2.188067988E-07, 2.053719572E-11, -8.193467050E-16]
            Ar = [2.010538475E+01, -5.992661070E-02, 2.500069401E+00, -3.992141160E-08, 1.205272140E-11, -1.819015576E-15, 1.078576636E-19]
            CO2 = [1.176962419E+05, -1.788791477E+03, 8.291523190E+00, -9.223156780E-05, 4.863676880E-09, -1.891053312E-12, 6.330036590E-16]
            
        coefficients = 8.314510 * np.array([N2, O2, Ar, CO2])
        
        temperatureVector = np.array([T**-2, T**-1, 1., T, T**2, T**3, T**4])
        
        cp_species = np.dot(coefficients, temperatureVector) / np.array([28.01340, 31.99880, 39.94800, 44.00950])
        
        cp_air = np.sum(cp_species * np.array([.78084, .20947, .009365, .000319]))
            
        gamma = cp_air*self.molarMass_air/(cp_air*self.molarMass_air - self.R_universal)
    
        return cp_air
        
#%%     
class Streamline(Fluid):
    """ This class takes care of the streamlines. It imports them, calculates 
    the running length and then sorts them to efficiently calculate the local 
    running length in each cell.
    """
    def __init__(self):
        """ """
        
    def importer(self, filename, FieldFlag=0):
        """ This function imports the streamline data by iterating over the 
        Tecplot generated file. Since we are iterating over it anyway, we 
        calculate the length of the streamline for every point as well, saving 
        it in self.streamlineLengths.
        
        No flow properties along the streamline are saved - only the length, 
        and x, y, z coordinates. The flow properties for each area element are
        imported form the actual solution (.triq file).
        """
        if not streamlineFlag:
            print "No streamlines - we'll use a coordinate based running length"
            return (None, None, None)
        
        print 'Importing streamline data...'
        
        self.filename = filename
        
        with open(self.filename, 'r') as dataFile:
            data = dataFile.read().split('Streamtrace')
        
        # Iterate through and construct the streamline dataframes
        Streamline.streamlines = 1000.*np.ones((maxSteps, nStreamlines, 3))
        Streamline.streamlineLengths = 1000.*np.ones([maxSteps, nStreamlines])
        
        count = 0 # if FieldFlag else 1
        
        length = 0 # Initialise maxLength counter to be small
        
        streamlineStepLengths = []
        streamlineTotalLengths = []
        
        for zone in data:
            # This is iterating over the file, selecting the streamlines
            #print 'Zone', count
            zone = zone.strip().split('\n')
            
            streamlineStepLength = len(zone)
            streamlineStepLengths.append(int(streamlineStepLength))
            
#            if count == 0: 
#                # Delete the surface label
#                del zone[1]; del zone[0]
#                if FieldFlag == False:
#                   # We don't want to read and store the velocity field data
#                   count += 1; continue
            L = 0.
            rowCount = 0
            coords = []
            streamlineLength = []
            
            for row in zone:
                # This is iterating over the individual streamlines
                row = row.split(',')
                
                if rowCount == 0:
                    x, y, z = (float(row[0]), float(row[1]), float(row[2]))
                    
                else:
                    xNew, yNew, zNew = (float(row[0]), float(row[1]), float(row[2]))
                    L += ((x-xNew)**2 + (y-yNew)**2 + (z-zNew)**2)**0.5
                    x, y, z = xNew, yNew, zNew
                    
                    if xNew < x:
                        print "Warning: a streamline may be recirculating"
                        names = ['Streamline', 'x', 'y', 'z', 'Length']
                        properties = [count, xNew, yNew, zNew, L]
                        for c1, c2 in zip(names, properties):
                            print "%-10s %s" % (c1, c2)
                    
                coords.append((x, y, z))
                streamlineLength.append(L)
                rowCount += 1
                
                
#            if count == 0 and FieldFlag:
#                # We're constructing the surface
#                    self.field = zone
#            else:
#                # We're constructing a streamline
#                #print np.shape(coords)
#                streamlines[:len(zone), 3*(count):3*count+3] = np.array(coords)
#                streamlineLengths[:len(zone), count] = np.array(streamlineLength)
        
            Streamline.streamlines[:streamlineStepLength, count:count+1, :] = np.array(coords).reshape(streamlineStepLength, 1, 3)
            Streamline.streamlineLengths[:streamlineStepLength, count:count+1] = np.array(streamlineLength).reshape(streamlineStepLength, 1)   
            streamlineTotalLengths.append(streamlineLength[-1])
            
            count += 1
        
        Streamline.maxStreamlineSteps = max(streamlineStepLengths)
        Streamline.streamlineLengths = Streamline.streamlineLengths[:Streamline.maxStreamlineSteps,:,]
        
        sortedLengthIndices = np.argsort(streamlineStepLengths)[::-1]
        
        # Sort the streamlines into order of increasing length
        Streamline.streamlines = Streamline.streamlines[:, sortedLengthIndices, :]
        sortedStepLengths = np.array(streamlineStepLengths)[sortedLengthIndices]
        Streamline.maxStepLength = sortedStepLengths[0]        
        
    
        # Tolerance on positioning the streamlines
        tol = 0.001
        # The first (longest and hopefully ~first in x)
        Streamline.firstStagnationPoint = Streamline.streamlines[:, 0, 0]
        
        for n in xrange(nStreamlines-1):
            # Iterate through the streamlines and adjust the starting position.
            
            # Skip the first (longest) streamline - we're adjusting all the others 
            # relative to it.
            n += 1
        
            # Get length of current streamline
            length = sortedStepLengths[n]
            # Get current streamline
            streamline = Streamline.streamlines[:length, n:n+1, :]
            # Get the running length vector
            streamlineLength =  Streamline.streamlineLengths[:length, n:n+1]
            # Get the starting x position of the current streamline
            xStartCurrent = streamline[0, 0, 0]
            
            try:
                # Try to find a position based on a small tolerance
                newRow = max((np.where(abs(Streamline.firstStagnationPoint - xStartCurrent) < tol)[0][0], 0))
            
            except:
                # If not increase the tolerance a lot to ensure we find a spot.
                if verbose:
                    print "Streamline adjustment failed (tol =", tol, "), increasing tolerance by 10."
                newRow = max((np.where(abs(Streamline.firstStagnationPoint - xStartCurrent) < tol*10)[0][0], 0))
                
            # move the streamline to new starting location
            if newRow + length >= Streamline.maxStepLength:
                # We are moving the streamline further back
                # Physically this means this streamline finishes behind the longest
                # Perhaps a swept wing where the trailing edge is behind the fuselage
                print "Warning: Attempted to push streamline outside the streamlines array!"
                Streamline.streamlines[newRow:newRow+length, n:n+1, :] = streamline
                Streamline.streamlineLengths[newRow:newRow+length, n:n+1] = streamlineLength
                
            else:        
                Streamline.streamlines[newRow:newRow+length, n:n+1, :] = streamline
                Streamline.streamlineLengths[newRow:newRow+length, n:n+1] = streamlineLength
            
            
            # Overwrite old area
            Streamline.streamlines[:newRow, n:n+1, :] = 1000.*np.ones((newRow, 1, 3))

        # Adjust streamlines to the actual maximum streamline step length not 
        # the maximum possible tecplot streamline steps
        Streamline.streamlines = Streamline.streamlines[:Streamline.maxStepLength, :, :]
        
        # Finished importing and calculating lengths
        print 'Streamline import and length calculation complete.', count-1, 'streamlines imported with max', self.maxStreamlineSteps, 'steps.', '\n'
        if verbose:
            print 'Maximum calculated streamline length is', max(Streamline.streamlineLengths), 'units.'

        # Check to ensure the longest streamline is weirdly longer than the next
        perCentLonger = (sortedStepLengths[0] - sortedStepLengths[1])/sortedStepLengths[1]
        if perCentLonger >= 10.:
            print "Warning: The longest streamline is", perCentLonger, "% longer than the next longest"
            
        """
        Include some checks here. Is max length >> length of vehicle? 
        Any streamline in the opposite direction to the flow (recirculation)
        Any streamline >> longer than all the others/neighbours?
        """
        return Streamline.streamlines, streamlineTotalLengths, Streamline.maxStepLength

#%%    
class Data(Streamline):
    def __init__(self, filename):
        """
        """
        Streamline.__init__(self)
        self.filename = filename
        self.flowData = pd.DataFrame()
        
        # Set the defaults for transition
        self.naturalTransitionFlag = 0
        self.ReynoldsTransitionFlag = 0
        self.coordinateTransitionFlag = 0
        self.roughnessInducedFlag = 0
        self.laminarOnlyFlag = 0
        self.turbulentOnlyFlag = 0
        self.immediateTransitionFlag = 0

        self.coneCorrectionFlag = 1
        
        # Work out what we want to do with transition.
        if criticalLocation == 0:
            self.naturalTransitionFlag = 1
        elif criticalLocation > 1000.:
            self.ReynoldsTransitionFlag = 1
            if verbose:
                print "Using a Reynolds number based transition criterion."
        elif criticalLocation < 1000.:
            self.coordinateTransitionFlag = 1
            if verbose:
                print "Using a Streamline coordinate length based transition criterion."
        elif roughnessCoordinate != 0:
            self.roughnessInducedFlag = 1
            if verbose:
                print "Using a roughness location based transition criterion."
        if regime == 'laminar':
            self.laminarOnlyFlag = 1
            if verbose:
                print "Laminar only simulation."
        elif regime == 'turbulent':
            self.turbulentOnlyFlag = 1
            if verbose:
                print "Turbulent only simulation."
        if immediateTransitionFlag:
            self.immediateTransitionFlag = 1
            if verbose:
                print "Immediate transition - no transitional flow region."
        else:
            print "Sorry, not sure what you want to do about transition."        
    
    def triqImporter(self, FieldFlag=1):
        """ Imports the area data from the Tecplot field data.
        """
        lineNumber = 0
        print 'Importing flow data...'
        
        with open(self.filename, 'r') as dataFile:
            for line in dataFile:               
                if lineNumber == 0:
                    # We're on the first line
                    #print line
                    self.nVertices, self.nTriangles, nScalars = (int(x) for x in line.split())
                    lineNumber += 1
            
            # Read in the vertex information
            self.vertices = pd.read_csv(self.filename, delim_whitespace=1, 
                                   names=['x', 'y', 'z'],
                                   skiprows=1, nrows=self.nVertices, memory_map=1)
 
            # Read in the triangle information
            self.triangles = pd.read_csv(self.filename, delim_whitespace=1, 
                                    names=['v1', 'v2', 'v3'],
                                    skiprows=self.nVertices+1, nrows=self.nTriangles, 
                                    memory_map=1)
            
            
            if sampleFlag:
                self.triangles = self.triangles.sample(sampleAmount).transpose()
            else:
                self.triangles = self.triangles.transpose()
            
            
            # Read in the flow information
            temp = pd.read_csv(self.filename, delim_whitespace=1, 
                               names=["rho","U","V","W","P"],
                               skiprows=self.nVertices+2*self.nTriangles+1, nrows=2*self.nVertices, 
                               memory_map=1)
            
            self.flow = temp.iloc[1::2, :].reset_index()
            
            if sampleFlag:
                self.nTriangles = sampleAmount
            
            print "Field import complete", self.nTriangles, 'elements,', self.nVertices, 'vertices.', '\n'
            return self.vertices, self.triangles, self.flow
        

    def getProperties(self):
        """
        Calculate all the flow properties of the triangles and add it all to the 
        flowData dataframe. This function just applies the master function to
        each row of the triangles DataFrame (list of vertices).
        It calculates the centroid and averages the data to the centroid.
        """
        print 'Running main code now...'
#        print self.nTriangles, 'elements,', self.nVertices, 'vertices.', '\n'
        
        self.count = 1
        self.startTime = time.time()
        self.percentComplete = 0
        
        # Set up counters to keep track of problematic cells.
        self.badMachCount = 0
        self.badCfCount = 0
        self.badTempCount = 0
        self.badVelCount = 0
        
        # Watch for very cold wall - Switch to Spalding & Chi
        self.spaldingFlag = 0
        # Watch for high Mach numbers - Switch to Coles
        self.colesFlag = 0
        
        # Run the main calculation
        self.flowData = pd.DataFrame(self.triangles.apply(self.master, axis=0))

        timeElapsed =  time.time() - self.startTime
        m, s = divmod(timeElapsed, 60); h, m = divmod(m, 60)
        
        print 'Viscous correction code complete.', '\n'
        print 'Time elapsed', "%d:%02d:%02d" % (h, m, s)
        print 'Average time per loop', timeElapsed/self.count
        
        if self.spaldingFlag:
            print "Warning: T_aw/T_w < 0.2 was encountered - Spalding & Chi method was employed but be careful of results."
        if self.colesFlag:
            print "Warning: M > 10 was encountered, the van Driest is known to be inaccurate - Coles' method (1964) might be better here."
        
        print "Bad cell counts (Total %d):" % self.nTriangles
        names = ['Mach', 'cf', 'T', 'Vel']
        if sampleFlag:
            amounts = [(self.badMachCount/float(sampleAmount) * 100., self.badMachCount),
                       (self.badCfCount/float(sampleAmount) * 100., self.badCfCount),
                       (self.badTempCount/float(sampleAmount) * 100., self.badTempCount),
                       (self.badVelCount/float(sampleAmount) * 100., self.badVelCount)]
        else:
            amounts = [(self.badMachCount/float(self.nTriangles) * 100., self.badMachCount),
                       (self.badCfCount/float(self.nTriangles) * 100., self.badCfCount),
                       (self.badTempCount/float(self.nTriangles) * 100., self.badTempCount),
                       (self.badVelCount/float(self.nTriangles) * 100., self.badVelCount)]
        for c1, c2 in zip(names, amounts):
            print "%-10s %s" % (c1, c2)
        print '\n'
                    
        return self.flowData.transpose()

    def master(self, row):
        """ This function iterates over the list of triangle vertices. To only 
        iterate over a potentially very long list, all calculations are done 
        at once - looping only once. Unfortuantely to avoid the overhead 
        associated with calling functions in python, most calculations are 
        done inside master() - this makes it long and difficult to read.
        
        The properties calculated include:
            A - The area of the triangle calculated with the cross product of 
                the vectors.
            n - The normal of the triangle, again from the cross product. By 
                convention normal is point OUTWARDS, into the flow.
            Cx, Cy, Cz - The coordinates of the centroid of each triangle.
            Re - The local Reynolds number calculated form an interpolated 
                 guess at the local running length based on the two closest 
                 streamlines (either side of the point).
            Cf - The local skin friction coefficient. Check associated docs.
            Ch - The local heat transfer coefficient (Stanton number). Check 
                 associated docs.
            
        The following properties are taken from the Cart3D solution file and
        (currently) linearly interpolated to the centroid. Note Cart3D 
        normalises its data against the freestream value and gamma (=1.4).
            rho - density [kg/m^3]
            U - x velocity [m/s]
            V - y velocity [m/s]
            W - z velocity [m/s]
            p - pressure [Pa]
            
        
        Currently takes a simple average of the properties - should implement 
        a weighted average based on distance from centroid to vertex when areas 
        get bigger. Depending on computational cost, set up a tolerance.
        """
        
        """
        if row some multiple of total number of triangles:
            print out a status update and estimate of time
        """
        #print "count", self.count
        if verbose:
            reportingInterval = 1
        else:
            reportingInterval = 5
            
        timeElapsed = time.time() - self.startTime                
        
        if self.count%(reportingInterval * self.nTriangles/100) == 0 or self.count == 1000:
            m, s = divmod(timeElapsed, 60); h, m = divmod(m, 60)
            print self.percentComplete, '% of elements completed so far. Wall clock time', "%d:%02d:%02d" % (h, m, s)
            printFlag = 0
            if self.percentComplete > 0:
                timeRemaining = timeElapsed *(100 - self.percentComplete)/self.percentComplete
                mRemaining, sRemaining = divmod(timeRemaining, 60)
                hRemaining, mRemaining = divmod(mRemaining, 60)                           
                print "Approximately", "%d:%02d:%02d" % (hRemaining, mRemaining, sRemaining), "remaining."
                printFlag = 1
            if self.count == 1000 and not printFlag:
                timeRemaining = timeElapsed/1000. * self.nTriangles
                mRemaining, sRemaining = divmod(timeRemaining, 60)
                hRemaining, mRemaining = divmod(mRemaining, 60)
                print "Rough initial estimate:", "%d:%02d:%02d" % (hRemaining, mRemaining, sRemaining), "remaining."
            self.percentComplete += reportingInterval; #self.count += 1
            

        
        # These are the vertices of the specific triangle - they correspond to 
        # indices in the vertices AND flow DataFrames
        # Note STL is indexed from 1, hence we need to minus one to get the 
        # dataframe index.
        v1i, v2i, v3i = row[0] - 1, row[1] - 1, row[2] - 1
        if v1i > self.nVertices or v2i > self.nVertices or v3i > self.nVertices:
            print 'Vertex indexing has died - max > than number of vertices.'
        
        # These are the (x, y, z) indices of each vertex
        v1 = np.array(self.vertices.iloc[v1i])
        v2 = np.array(self.vertices.iloc[v2i])
        v3 = np.array(self.vertices.iloc[v3i])

        # Form two vectors forming the triangle
        v1v2 = v2 - v1
        v1v3 = v3 - v1
        
        # Calculate area and normal from cross product given the above vectors.
        area = 0.5 * np.linalg.norm(np.cross(v1v2, v1v3))
        normal = tuple(np.cross(v1v2, v1v3)/area)
        
        #  Calculate the centroid coodinates.
        centroidx = np.mean([v1[0], v2[0], v3[0]])
        centroidy = np.mean([v1[1], v2[1], v3[1]])
        centroidz = np.mean([v1[2], v2[2], v3[2]])
        centroid = (centroidx, centroidy, centroidz)

        # Calculate the mean surface flow properties at the centroid of each triangle.
        # Order: Cp,Rho,U,V,W,Pressure
        properties = np.mean([self.flow.iloc[v1i], self.flow.iloc[v2i], self.flow.iloc[v3i]], axis=0)
        self.rho, U, V, W, self.p = properties[1], properties[2], properties[3], properties[4], properties[5]

        # Undo the normalisation Cart3D uses for some currently unknown reason
        self.rho *= rho_inf; U *= Fluid.a_inf; V *= Fluid.a_inf; W *= Fluid.a_inf; self.p *= rho_inf*Fluid.a_inf**2.
#        print 'rho', self.rho, 'U', U, 'V', V, 'W', W, 'p', self.p
        
        # Need to catch the problematic data Cart3D sometimes produces - 
        # generally degenerencies in small cut cells. Known issue.
        if self.p < 1e-1:
            self.p = 1e-1
            if verbose:
                print "Warning: Pressure < 1e-1 at", v1, v2, v3, "setting to 1e-1 Pa."
        if self.rho < 1e-6:
            self.rho = 1e-6
            if verbose:
                print "Warning: Density < 1e-6 at", v1, v2, v3, "setting to 1e-6 kg/m^3."
        
        # Calculate local velocity vector
        self.velocityMagnitude = (U**2. + V**2. + W**2.)**0.5
        velocityDirection = np.array([U, V, W], dtype='float64') / self.velocityMagnitude
        #print 'velocity', velocityMagnitude
        
        if self.velocityMagnitude > 1.5*M_inf*Fluid.a_inf:
            self.badVelCount += 1
            if verbose:
                print "Warning: velocity > 1.5x freestrem at", v1, v2, v3

        # Calculate the temperature based on ideal gas law
        self.T = self.p / (self.rho * Fluid.R_air)
        #print 'T', self.T
        
        if self.T > 800.:
            #print "Warning: High edge temperature, constant Cp assumption might be in trouble - consider variable Cp."
            self.badTempCount += 1
            
        
        # Calculate local Mach number
        try:
            self.M = self.velocityMagnitude/((g_inf*Fluid.R_air*self.T)**0.5)
            if self.M > 1.5 * M_inf:
#                print "Warning high Mach number,", self.M, "Temperature is", self.T
                self.badMachCount += 1
#                print "x coordinate is", centroid[0], "Are you in the wake?"
                self.M = M_inf
            if self.M > 10.:
                self.colesFlag = 1
        except:
            print 'Check local sound speed at', v1, v2, v3
        
        # Calculate local dynamic viscosity using Keye's law if T < 95 Kelvin
        if self.T < 95.:
            self.mu = (1.488 * 10**-6.) * self.T**0.5 / (1. + 122.1*(10.**(-5/self.T))/self.T)
            self.mu_wall = (1.488 * 10**-6.) * T_wall**0.5 / (1. + 122.1*(10.**(-5/T_wall))/T_wall)
        else:
            self.mu = Fluid.mu_0 * (self.T/Fluid.T_0)**(3./2) * ((Fluid.T_0 + Fluid.sutherland_S) / (self.T + Fluid.sutherland_S))
            self.mu_wall = Fluid.mu_0 * (T_wall/Fluid.T_0)**(3./2) * ((Fluid.T_0 + Fluid.sutherland_S) / (T_wall + Fluid.sutherland_S))
        #print 'mu/mu_wall', self.mu/self.mu_wall

        # Calculate the local streamline based running length
        self.calculate_runningLength(centroid)
        
        # Calculate the local Reynolds number
        self.localReynolds = self.rho * self.velocityMagnitude * self.localRunningLength / self.mu
        #print 'localReynolds', self.localReynolds
        
        # We always assume a laminar boundary layer
        self.laminar = 1; self.transitional = 0; self.turbulent = 0

        if self.laminar:
           self.calculate_laminarCf()
                        
        if self.transitional:
            self.calculate_transitionalCf()

        if self.turbulent:
            self.calculate_turbulentCf()
        
        # The above computed skin friction coefficients should be corrected for
        # thickness with form factors. Unique factors are implmented here 
        # for wings and bodies
                
        wallShearMagnitude = self.cf * 0.5 * self.rho * self.velocityMagnitude**2.
        wallShearVector = wallShearMagnitude * velocityDirection
        viscousForce = wallShearVector*area
        
        # Calculate Reynold's analogy factor
        Pr_T = 0.86; Pr_L = 0.71
        bracket = 1./(5.*0.4) * (1. - Pr_T) * ((np.pi**2.)/6. + 1.5*(1. - Pr_T)) + (Pr_L/Pr_T - 1.) + log(1. + (5./6.)*(Pr_L/Pr_T - 1.))
        s = Pr_T * (1. + 5.*(self.cf/2.)**0.5 * bracket)
        
        # Calculate Stanton number from modified Reynold's analogy
        ch = (1./s) * self.cf/2.
        
        # Calculate heat transfer coefficient
        h = ch*self.rho*self.velocityMagnitude*Fluid.calculate_cp(self.T)
        
        # Calculate heat transfer into the wall
        bracket = self.T * (1. + self.r * (g_inf - 1.)/2. * self.M**2.) - T_wall
        q_wall = ch * self.rho*self.velocityMagnitude * Fluid.calculate_cp(self.T) * bracket
        
        if verbose:
            print 'Local properties...'
            names = ['area', 'centroidx', 'centroidy', 'centroidz', 'normal', 'rho', 'U', 'V', 'W', 'p', 'cf', 'Ff']
            properties = [area, centroidx, centroidy, centroidz, normal, self.rho, U, V, W, self.p, self.cf, viscousForce]
            for c1, c2 in zip(names, properties):
                print "%-10s %s" % (c1, c2)
            print '\n'
        
        # Increment the element counter
        self.count += 1
    
        return pd.Series({'A': area, 
                          'x': centroid[0],
                          'y': centroid[1],
                          'z': centroid[2],
                          'n': normal,
                          'rho': self.rho,
                          'U': U,
                          'V': V,
                          'W': W,
                          'Velocity': self.velocityMagnitude,
                          'M': self.M,
                          'p': self.p,
                          'cf': self.cf,
                          'ch': ch,
                          #'cp': self.cp,
                          'BL Regime': self.BLregime,
                          'Dev Length': self.localRunningLength,
                          'Re': self.localReynolds,
                          'Ff': viscousForce,
                          'T_e': self.T,
                          'T_aw': self.T_adiabaticWall,
                          'q_wall': q_wall})

    def calculate_runningLength(self, centroid, searchBracket=200):
        """ This function calculates the local running length given 
        a location.
        
        If there is streamline data available - it will use that, otherwise 
        it just uses a cartesian based running length (i.e. x coordinate).
        
        """
        # Firstly check we actually want the streamline running length
        if not streamlineFlag:
            # MAKE SURE THIS ISN'T NEGATIVE
            self.localRunningLength = centroid[0] + LEoffset
            if self.localRunningLength <= 0.005:
                # Need to include blunted leading edge effects here but for the moment
                # we'll just set it to 0.005
                self.localRunningLength = 0.005
            return
        
        # Populate a large array repeating the current location
#        self.Location = np.tile(centroid, (self.maxStreamlineSteps, nStreamlines))
        self.Location = np.broadcast_to(centroid, (searchBracket, nStreamlines, 3))
        currentX = centroid[0]
        
#        print 'Current centroid', centroid
        
        # Tolerance on finding the position on the streamlines
        tol = 0.001
        
        try:
            # Try to find a position based on a small tolerance
            rowPosition = max((np.where(abs(Streamline.firstStagnationPoint - currentX) < tol)[0][0], 0))
        
        except:
            # If not increase the tolerance a lot to ensure we find a spot.
            if verbose:
                print "Row position adjustment failed (tol =", tol, "), increasing tolerance by 10."
            rowPosition = max((np.where(abs(Streamline.firstStagnationPoint - currentX) < tol*10)[0][0], 0))
        
        if rowPosition <= searchBracket/2:
            # We are at the top of the array
            self.streamlineSection = self.streamlines[:searchBracket, :, :]
        
        elif rowPosition >= Streamline.maxStepLength - searchBracket/2:
            # We are at the bottom of the array
            self.streamlineSection = self.streamlines[searchBracket:, :, :]
        else:
            # We are in the middle
            self.streamlineSection = self.streamlines[rowPosition-searchBracket/2:rowPosition+searchBracket/2, :, :]
        
#        print "Streamline section goes between", self.streamlineSection[0, 0, 0], self.streamlineSection[-1, 0, 0]
        
        # delta x, delta y, delta z from location to every point
        self.deltas = self.Location - self.streamlineSection
        
        # Square the distances
        self.deltas = np.square(self.deltas)
        
        # Separate dx, dy and dz to sum together
        dx = self.deltas[:, :, 0]
        dy = self.deltas[:, :, 1]
        dz = self.deltas[:, :, 2]
        
        # Take the square root to find the Euclidean distance
        self.distances = np.sqrt(dx + dy + dz)
        
        """
        POTENTIAL SPEED IMPROVEMENT
        # possibly need to have:
        # temp = np.asfortranarray(self.distances)
        # streamlineMinimumsIndices = temp.argmin(axis=0)
        """
        
        """
        NEED TO INCLUDE GRAD CHECK HERE TO ENSURE STREAMLINES ARE ON THE CORRECT SIDE OF THE OBJECT
        """        
        
        # Indices of two closest streamlines (column indices)
        # Column index of two closest streamline points
        neighbouringStreamlineIndices = self.distances.min(axis=0).argsort(kind='mergesort')[:2] # index
#        print 'neighbouringStreamlineIndices', neighbouringStreamlineIndices
        # Indices of the step number to the minimum distances
        # Row index of two closest streamline points
        neighbouringStreamlineStepIndices = self.distances.argsort(axis=0, kind='mergesort')[0, neighbouringStreamlineIndices]
#        print 'neighbouringStreamlineStepIndices', neighbouringStreamlineStepIndices

#        # Indices of the two closest streamline points
#        neighbouringStreamlines_indices = np.array([neighbouringStreamlineStepIndices, neighbouringStreamlineIndices])
#        print 'neighbouringStreamlines_indices', neighbouringStreamlines_indices
        
        # Distances to two closest streamlines
        neighbouringStreamlines_distances =  self.distances[neighbouringStreamlineStepIndices, neighbouringStreamlineIndices] # value
#        print "neighbouringStreamline_distances", neighbouringStreamlines_distances

        if np.max(abs(neighbouringStreamlines_distances)) > 1.:
            print "WARNING: Closest streamline seems to be far away at", np.max(neighbouringStreamlines_distances), "m."
            print 'Current centroid', centroid
                    
        # Running length at the two neighbouring streamline points
        # Need to correct the indexing because we only look at a window above
        neighbouringStreamlineStepIndices = neighbouringStreamlineIndices + rowPosition
#        neighbouringStreamlines_indices = np.array([neighbouringStreamlineStepIndices, neighbouringStreamlineIndices])
        neighbouringStreamlines_lengths = Streamline.streamlineLengths[neighbouringStreamlineStepIndices, neighbouringStreamlineIndices]
#        print 'neighbouringStreamlines_lengths', neighbouringStreamlines_lengths

        # Linearly interpolate between two neighbouring streamlines
        self.localRunningLength = float(neighbouringStreamlines_lengths[0] + neighbouringStreamlines_distances[0]*np.diff(neighbouringStreamlines_lengths)/np.sum(neighbouringStreamlines_distances))
#        print 'localRunningLength', self.localRunningLength

        if self.localRunningLength <= 0.005:
            # Need to include blunted leading edge effects here but for the moment
            # we'll just set it to 0.005
            self.localRunningLength = 0.005

    def calculate_laminarCf(self, checkFlag=1):
        # Check to ensure flow isn't transitional
        if checkFlag:
            if not self.laminarOnlyFlag:
                # Not doing a laminar only analysis
                if self.turbulentOnlyFlag:
                    # Running turbulent only analysis
                    self.laminar = 0; self.transitional = 0; self.turbulent = 1
                    return
                elif self.naturalTransitionFlag:
                    # Natural transition criterion
                    self.Re_critical = 10.**(6.421 * exp((1.209e-4) * self.M**2.641))
                    """
                    NEED TO INCLUDE THE WING SWEEP STUFF HERE
                    Re_critical = Re_critical*(0.787 * cos(wingLEsweep)**4.346 - 0.7221*exp(-0.0991*wingLEsweep) + 0.9464)
                    """
                    if self.localReynolds >= self.Re_critical:
                        # The flow is transitional, break out of the laminar analysis
                        self.laminar = 0; self.transitional = 1; self.turbulent = 0
                        return
                    
                elif self.roughnessInducedFlag:
                    # Roughness induced transition condition
                    pass
                    
                elif self.ReynoldsTransitionFlag:
                    # Critical Reynolds criterion
                    if criticalLocation >= self.localReynolds:
                        self.laminar = 0; self.transitional = 1; self.turbulent = 0
                        return
                elif self.coordinateTransitionFlag:
                    if criticalLocation >= self.localRunningLength:
                        self.laminar = 0; self.transitional = 1; self.turbulent = 0
                        return

        # The above transition checks all showed that it was laminar flow, 
        # continue laminar analysis:

        # Calculate the laminar skin friction coefficient
        # Set recovery factor
        self.r = 0.85 # van Driest says 0.85 to 0.89 for lam to turbs          
    
        # Calculate the adiabatic wall temperature
        self.T_adiabaticWall = (1. + self.r*((g_inf - 1)/2.) * self.M**2.) * self.T
#        T_awOnT = self.T_adiabaticWall/self.T
        
        # Reference temperature
#        T_reference = self.T*(0.45 * 0.55 * T_awOnT + 0.16*self.r*(g_inf - 1)/2. * self.M**2.)
        T_reference = self.T*(1. + 0.032*self.M**2. + 0.58 * (T_wall/self.T - 1.))
        
        # Reference density
        rho_reference = self.p/(Fluid.R_air * T_reference)
        
        # Reference viscosity
        mu_reference = 1.458e-6 * ((T_reference)**1.5) / (T_reference + 110.4)
        
        # Reference Reynolds
#        Re_reference = self.M * (g_inf*Fluid.R_air*T_reference)**0.5 * rho_reference * self.localRunningLength / mu_reference
        Re_reference = self.velocityMagnitude * rho_reference * self.localRunningLength / mu_reference
        
        try:
            cf = 0.664 / (Re_reference)**0.5
        except:
            print 'Calculation of laminar flow Cf failed'
            cf = 0.
 
        if self.coneCorrectionFlag:
            # Flow is 3D, apply cone rule correction
            cf *= 1.73

        self.cf = cf
        
        # This is to show lam (0 = BLregime) vs transitional (1 < BLregime < 0)
        # vs turb flow (BLregime = 0)
        self.BLregime = 0
        
        return self.cf
        
    
    def calculate_transitionalCf(self):
        # Set recovery factor
        self.r = 0.87 # van Driest says 0.85 to 0.89 for lam to turbs
            
#        self.criticalRunningLength_start = (6.421*self.mu*exp(1.209e-4 * self.M**2.641)) / (self.rho * self.velocityMagnitude)
#        criticalRunningLength_end = self.criticalRunningLength_start * (1. + self.Re_critical**(-0.2))

        # Check we aren't turbulent
        if self.immediateTransitionFlag:
            # Ignoring transitional region
            self.laminar = 0; self.transitional = 0; self.turbulent = 1
            return
        
#        elif self.localRunningLength >= criticalRunningLength_end:
#             Flow is now fully turbulent
#            self.laminar = 0; self.transitional = 0; self.turbulent = 1
#            return
            
        else:
            # The above checks all showed we are still in a transitional region
            
            cf_laminar = self.calculate_laminarCf(checkFlag=0)
            try:
                cf_turbulent = self.calculate_turbulentCf(r=0.87)
            except:
                print "Calculation of transitional flow turbulent Cf failed"
                names = ['Local Re', 'mu', 'mu_wall', 'T_aw', 'T_edge', 'P', 'rho']
                properties = [self.localReynolds, self.mu, self.mu_wall, self.T_adiabaticWall, self.T, self.p, self.rho]
                for c1, c2 in zip(names, properties):
                    print "%-10s %s" % (c1, c2)
                print '\n'

            # Set up the variables to vary between laminar and turbulent skin friction coefficients.
            exponent = -3. *(exp(log(2)/(5.*self.criticalRunningLength_start) * self.Re_critical**(-0.2)*(self.localRunningLength - self.criticalRunningLength_start)) - 1.)**2.
            epsilon = 1 - exp(exponent)
            
            try:
                cf = (1-epsilon)*cf_laminar + epsilon*cf_turbulent  
            except:
                print "Calculation of transitional flow Cf failed"
                
            if self.coneCorrectionFlag:
                # Flow is 3D, apply cone rule correction
                cf *= (1-epsilon)*1.15 + epsilon*1.73

        self.cf = cf
        
        # This is to plot lam (BLregime = 0) vs transitional (0 < BLregime < 1)
        # vs turb flow (BLregime = 1)
        self.BLregime = 0.5
        
        return self.cf

        
    def calculate_turbulentCf(self, r=0.89):
        #print "Turbulent flow"
        # Calculate the turbulent skin fricton coefficient
        # van Driest says r = 0.85 to 0.89 for lam to turbs
        self.r = r
        
        self.T_adiabaticWall = (1. + self.r*((g_inf - 1.)/2.) * self.M**2.) * self.T
        # Quick wall temp check
        if T_wall/self.T_adiabaticWall < 0.2:
            self.spaldingFlag = 1
            cf = self.calculate_turbulentCf_spaldingChi()
        else:
            # Set up the variables/coefficients for the Van Driest estimate
            aSquared = self.r * (g_inf - 1.)/2. * self.M**2. * self.T/T_wall
            b = self.T_adiabaticWall/T_wall - 1.
            denominator = (b**2. + 4.*aSquared)**0.5
            
            A = self.clean_A(aSquared, b, denominator)
            B = self.clean_B(aSquared, b, denominator)
            
            # Solve the implicit equation for skin friction            
            cf_func = lambda cf: 4.15*log(self.localReynolds*cf*self.mu/self.mu_wall, 10) + 1.7 - ((np.arcsin(A) + np.arcsin(B)) / ((cf*(self.T_adiabaticWall - self.T)/self.T)**0.5))
            
            
            try:
                cf = brentq(cf_func, 1e-15, 0.1)
                self.calculate_turbulentCf_spaldingChi()
            except:
                if verbose:
                    print "Calculation of turbulent Cf failed, Flow properties at culprit cell below."
                    print "Am I in the Wake? Running length is", self.localRunningLength, "Set cf to zero."
                    names = ['Local Re', 'length', 'mu', 'mu_wall', 'T_aw', 'T_edge', 'T_wall', 'p', 'rho', 'velocity', 'Mach']
                    properties = [float(self.localReynolds), float(self.localRunningLength), self.mu, self.mu_wall, self.T_adiabaticWall, self.T, T_wall, self.p, self.rho, self.velocityMagnitude, self.M]
                    for c1, c2 in zip(names, properties):
                        print "%-10s %s" % (c1, c2)
                    print '\n'
                cf = 0.
    
                # USE THE SMART MEADER CORRELATION IF VAN DRIEST FAILS
                # Reference temperature
    #            T_reference = self.T*(1. + 0.032*self.M**2. + 0.58 * (T_wall/self.T - 1.))
    #            
    #            # Reference density
    #            rho_reference = self.p/(Fluid.R_air * T_reference)
    #            
    #            # Reference viscosity
    #            mu_reference = 1.458e-6 * ((T_reference)**1.5) / (T_reference + 110.4)
    #            
    #            # Reference Reynolds
    #            Re_reference = self.velocityMagnitude * rho_reference * self.localRunningLength / mu_reference
    #            
    #            cf = 0.02296/(Re_reference**0.139) * (rho_reference/self.rho)**0.861 * (mu_reference/self.mu)**0.139
                
                self.badCfCount += 1
            
            if self.coneCorrectionFlag:
                # Flow is 3D, apply cone rule correction
                cf *= 1.15
        
        # End cf (van driest or Spalding & Chi) estimate
        self.cf = cf
        
        # This is to plot lam (BLregime = 0) vs transitional (0 < BLregime < 1)
        # vs turb flow (BLregime = 1)
        self.BLregime = 1
        
        return self.cf
        
    
    def clean_A(self, a, b, denominator):
        """
        This function is required to avoid math domain errors in an arcsin 
        calculation in the Van Driest calculation.
        """
        A = ( 2.*a - b ) / denominator
        if A < -1.:
            return -1.
        elif A > 1.:
            return 1.
        else:
            return A
        
        
    def clean_B(self, a, b, denominator):
        """
        This function is required to avoid math domain errors in an arcsin 
        calculation in the Van Driest calculation.
        """
        B = ( b ) / denominator
        if B < -1.:
            return -1.
        elif B > 1.:
            return 1.
        else:
            return B
        
        
    def calculate_turbulentCf_spaldingChi(self, r=0.89):
        # Calculate the turbulent skin fricton coefficient using the Spalding Chi method
        # This is more accurate than Van driest for T_wall/self.T_adiabaticWall < 0.2
        # van Driest says r = 0.85 to 0.89 for lam to turbs
        self.r = 0.89

        # Set up the variables/coefficients for the estimate
        # Various wall temperature ratios
        TawOnT = self.T_adiabaticWall/self.T
        TwOnT = T_wall/self.T
        
        denominator = ( (TawOnT + TwOnT)**2. - 4.*TwOnT )**0.5
        
        alpha = (TawOnT + TwOnT - 2.) / denominator
        beta = (TawOnT - TwOnT) / denominator
        
        F_c = (TawOnT - 1.) / (np.arcsin(alpha) + np.arcsin(beta))**2.
        
        # Solve the implicit equation for the incompressible skin friction
        LHS = self.localReynolds / (F_c*(TawOnT**0.772 * TwOnT**-1.474))    
        K = 0.4
        E = 12.
        kappa = lambda cf: K * (2./cf)**0.5
        
#        bracket = (2. + (2. - kappa)**2.)*exp(kappa) - 6. - 2.*kappa - (1./12)*kappa**4. - (1./20)*kappa**5. - (1./60)*kappa**6. - (1./256)*kappa**7.
        bracket = lambda cf: (2. + (2. - kappa(cf))**2.)*exp(kappa(cf)) - 6. - 2.*kappa(cf) - (1./12)*kappa(cf)**4. - (1./20)*kappa(cf)**5. - (1./60)*kappa(cf)**6. - (1./256)*kappa(cf)**7.
        
        cf_inc_func = lambda cf: (1./12)*(2./cf)**2. + (1./(E*K**3.)) * bracket(cf) - LHS
        
        
        try:
            cf_inc = brentq(cf_inc_func, 5e-6, 0.1)
            cf = (1./F_c) * cf_inc
            
        except:
#            print "Calculation of turbulent Cf failed, Flow properties at culprit cell below."
#            print "Am I in the Wake? Running length is", self.localRunningLength, "Set cf to zero."
#            names = ['Local Re', 'length', 'mu', 'mu_wall', 'T_aw', 'T_edge', 'T_wall', 'p', 'rho', 'velocity', 'Mach']
#            properties = [float(self.localReynolds), float(self.localRunningLength), self.mu, self.mu_wall, self.T_adiabaticWall, self.T, T_wall, self.p, self.rho, self.velocityMagnitude, self.M]
#            for c1, c2 in zip(names, properties):
#                print "%-10s %s" % (c1, c2)
#            print '\n'
            cf = 0.
            self.badCfCount += 1
        
        if self.coneCorrectionFlag:
            # Flow is 3D, apply cone rule correction
            cf *= 1.15
        
        return cf
        
#%%
class postProcessor(Data):
    
    def __init__(self, Field, flowData):
        self.flowData = flowData
        self.propertiesToSave = ['cf', 'ch', 'Velocity', 'T_aw'] + additionalProperties
    
    def viscousForceCoefficients(self):
        """ This function will calculate and return the viscous force 
        coefficients. The forces are calculated and stored here.
        
        Sign convention
            x  -  positive toward tail (flow in the direction of positive x)
            y  -  positive upwards
            z  -  positive left spanwise sitting in cockpit facing forwards
            
        """
        # Visous forces in body axes
        viscousForces_body = sum(self.flowData.loc['Ff'])
        
        # Transform to wind axes
        
        viscousForces = viscousForces_body
        
        # Calculate velocity
        u_inf = M_inf*Fluid.a_inf
        
        cl_viscous = viscousForces[0]/(0.5*S_ref*rho_inf*u_inf**2.)
        cd_viscous = viscousForces[1]/(0.5*S_ref*rho_inf*u_inf**2.)
        
        return cl_viscous, cd_viscous
    
        
    def viscousMomentCoefficients(self):
        """ Similar to the above this function will calculate the viscous 
        pitching moment coefficients.
        Sign convention
            Directions same as above
            Positive rotations defined by RH rule
        """
        
        cm_viscous = 5
        
        return cm_viscous


    def saveAs_CSV(self, outputFilename=outputFilename, properties=['x', 'y', 'z', 'cf', 'ch', 'Velocity', 'T_aw']):
        """ This function will write the flow data to file so we can view it 
        in Paraview.
        
        """
        
        outputFilename += '.csv'
        
        if additionalProperties != []:
            for i in additionalProperties:
                properties.append(i)
        
#        self.flowData = self.flowData.round(decimals=5)
        
        self.flowData.to_csv(outputFilename, sep=',', columns=properties, index=0, index_label=0, float_format='%.3f')
            
        print "output file saved as", outputFilename
        
    def saveSlice_CSV(self, outputFilename=outputFilename, xSlice=[], ySlice=[], zSlice=[]):
        """ Take a slice and save it to csv """
        outputFilename += '_slice.csv'
        
#        # This defines how 'narrow' slice we want. Why am I writing this if ParaView will do it fark
#        tol = 1e-2
#        
#        # Pre allocate empty DF here?
#        slicedData = pd.DataFrame()
#        
#        if not xSlice:
#            # We have some slices along x to make
#            for point in xSlice:
#                # we want to slice at all of these points
#                > xSlice[point] - tol
#            self.flowData.transpose().loc[(self.flowData.transpose()["x"] > 0.599 & self.flowData.transpose()["x"] < 0.601 &  self.flowData.transpose()["z"] == 0), "cf"]
#        elif not ySlice:
#            # Slices along y to take
#        elif not zSlice:
#            # And slices aong z
        
        flowData = self.flowData.apply(pd.to_numeric, errors='ignore')
        
        slicedData_indices = (flowData["z"] > -0.01) & (flowData["z"] < 0.01)
        
        slicedData = flowData.loc[slicedData_indices]
        
        slicedData.to_csv(outputFilename, sep=',', index=0, index_label=0)
        
        print "Slices saved in", outputFilename


    def saveAs_VTK(self, outputFilename):
        """
        Write the flow data as a VTK unstructured grid - STILL NOT SURE WHY???
        """
        outputFilename += '.vtu'
        vtuFile = open(outputFilename, "w")
        
        NumberOfPoints = Field.nVertices
        NumberOfTriangles = Field.nTriangles
        
        # Write the header
        vtuFile.write("<VTKFile type=\"UnstructuredGrid\" byte_order=\"BigEndian\">\n")
        vtuFile.write("<UnstructuredGrid>")
        vtuFile.write("<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" %
                     (NumberOfPoints, NumberOfTriangles))
        
        # Write the point coordinates
        vtuFile.write("<Points>\n")
        vtuFile.write(" <DataArray type=\"Float32\" NumberOfComponents=\"3\"")
        vtuFile.write(" format=\"ascii\">\n")
        for index in range(NumberOfPoints-500000):
            x, y, z = Field.vertices.iloc[index]
            vtuFile.write(" %e %e %e\n" % (x, y, z))        
        vtuFile.write(" </DataArray>\n")
        vtuFile.write("</Points>\n")

        vtuFile.write("<Cells>\n")    
        
        # Write the connectivity
        vtuFile.write(" <DataArray type=\"Int32\" Name=\"connectivity\"")
        vtuFile.write(" format=\"ascii\">\n")
        temp = Field.triangles.transpose()
        for index in range(NumberOfTriangles):
            v1, v2, v3 = temp.iloc[index]
            vtuFile.write(" %d %d %d\n" % (v1, v2, v3))
        vtuFile.write(" </DataArray>\n")
        
        # Write the offsets
#        vtuFile.write(" <DataArray type=\"Int32\" Name=\"offsets\"")
#        vtuFile.write(" format=\"ascii\">\n")
#        # Since all of the point-lists are concatenated, these offsets into the connectivity
#        # array specify the end of each cell.
#        for point in range(NumberOfTriangles):
#            if two_D:
#                conn_offset = 4*(1+i+j*nic)
#            else:
#                conn_offset = 8*(1+i+j*nic+k*(nic*njc))
#            vtuFile.write(" %d\n" % conn_offset)
#        vtuFile.write(" </DataArray>\n")     
        
        # Write the types
        vtuFile.write(" <DataArray type=\"UInt8\" Name=\"types\"")
        vtuFile.write(" format=\"ascii\">\n")
        VTKtype = 5 # VTK_TRIANGLE
        for point in range(NumberOfTriangles):
                    vtuFile.write(" %d\n" % VTKtype)
        vtuFile.write(" </DataArray>\n")
        vtuFile.write("</Cells>\n")
    
        # Write the flow variables
        vtuFile.write("<CellData>\n")
        # Write variables from the dictionary.
        for variable in self.propertiesToSave:
            vtuFile.write(" <DataArray Name=\"%s\" type=\"Float32\" NumberOfComponents=\"1\"" % (variable))
            vtuFile.write(" format=\"ascii\">\n")
            for index in range(NumberOfTriangles):
                vtuFile.write(" %e\n" % Field.flowData.transpose()[variable].iloc[index])
            vtuFile.write(" </DataArray>\n")

        # Write the velocity vector - have to do this separately because it's a vector
        vtuFile.write(" <DataArray Name=\"Velocity vector\" type=\"Float32\" NumberOfComponents=\"3\"")
        vtuFile.write(" format=\"ascii\">\n")
        for index in NumberOfTriangles:
            U, V, W = (Field.flowData.transpose()['U'].iloc[index], 
                       Field.flowData.transpose()['V'].iloc[index], 
                       Field.flowData.transpose()['W'].iloc[index])
            vtuFile.write(" %e %e %e\n" % (U, V, W))
        vtuFile.write(" </DataArray>\n")
        
        # Write footers and close file
        vtuFile.write("</CellData>\n")
        vtuFile.write("</Piece>\n")
        vtuFile.write("</UnstructuredGrid>\n")
        vtuFile.write("</VTKFile>\n")
        vtuFile.close()
        
        return

#%%  -----  Run the program
if __name__ == '__main__':
    print time.strftime("%H:%M:%S"), 'Starting....'
    # Run Tecplot in batch mode to generate and save the streamline data
#    try:
#        call(['tec360', '-b', 'Components.i.plt', 'retrieveStreamlines.mcr'])
#    except:
#        print 'Import of Tecplot streamline data failed'

    # Initialise Fluid class - sets up basic fluid and freestream properties
    
    Fluid = Fluid()
    
    # Initialise streamline class
    Streamlines = Streamline()
    streamlineCoordinates, streamlineLengths, maxSteplength = Streamlines.importer(streamlineFilename)
    StreamlinesDict = Streamline.__dict__
    
    # Initialise a data class, this contains all the Field (Cart3D) data.
    Field = Data(cart3DFilename)
    
    # Import Cart3D data
    vertices, triangles, flow  = Field.triqImporter()
    
    # Run the actual code - calculate viscous forces
    flowData = Field.getProperties()
    flowData = flowData.round(decimals=5)
    
    
    post = postProcessor(Field, flowData)
    
    if csvFlag:
        post.saveAs_CSV()
        
    if vtkFlag:
        post.saveAs_CSV()
    
    post.saveSlice_CSV()
    
    DataDict = Field.__dict__

    #Field.plotter()
    
