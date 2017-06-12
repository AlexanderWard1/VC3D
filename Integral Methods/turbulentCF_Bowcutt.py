
from math import exp, log, asin, sin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq
import time as time
from scipy import interpolate
from scipy.integrate import ode

#%% Inputs
verbose = 1

# Number of Tecplot generated streamlines {int}
nStreamlines = 1
#------------
# maximum number of steps in the the Tecplot streamlines {int}
maxSteps = 19999


T_wall = 400.

g_inf = 1.4
mu_0 = 1.716e-5
T_0 = 273.
sutherland_S = 111.

#%% Streamline Import
def importer(FieldFlag=0):
    """ This function imports the streamline data by iterating over the 
    Tecplot generated file. Since we are iterating over it anyway, we 
    calculate the length of the streamline for every point as well, saving 
    it in self.streamlineLengths.
    
    No flow properties along the streamline are saved - only the length, 
    and x, y, z coordinates. The flow properties for each area element are
    imported form the actual solution (.triq file).
    """
    
    print 'Importing streamline data...'
    
    filename = 'flowData_slice.csv'
    
    with open(filename, 'r') as dataFile:
#        data = dataFile.read().split('Streamtrace')
        data = dataFile.read()
    
    # Iterate through and construct the streamline dataframes
    streamlines = 1000.*np.ones((maxSteps, nStreamlines, 3))
    streamlineLengths = 1000.*np.ones([maxSteps, nStreamlines])
    
    count = 0 # if FieldFlag else 1
    
    length = 0 # Initialise maxLength counter to be small
    
    streamlineStepLengths = []
    streamlineTotalLengths = []
    
#    for zone in data:
    zone = data
    # This is iterating over the file, selecting the streamlines
    #print 'Zone', count
    zone = zone.strip().split('\n')
    
    streamlineStepLength = len(zone)
    streamlineStepLengths.append(int(streamlineStepLength))

    S = 0.
    rowCount = 0
    coords = []
    streamlineLength = []
    
    data = []
    
    for row in zone:
        # This is iterating over the individual streamlines
        row = row.split(',')
        
        
        if rowCount == 0:
            rowCount += 1
            continue
            
        elif rowCount == 1:
            x, y, z = (float(row[17]), float(row[18]), float(row[19]))
            
        else:
            xNew, yNew, zNew = (float(row[17]), float(row[18]), float(row[19]))
            S += ((x-xNew)**2 + (y-yNew)**2 + (z-zNew)**2)**0.5
            x, y, z = xNew, yNew, zNew
            
            if xNew < x:
                print "Warning: a streamline may be recirculating"
                names = ['Streamline', 'x', 'y', 'z', 'Length']
                properties = [count, xNew, yNew, zNew, S]
                for c1, c2 in zip(names, properties):
                    print "%-10s %s" % (c1, c2)
            
        M, T_e, U, V, W, p, rho = (float(row[4]), float(row[7]), 
                                   float(row[8]), float(row[9]), 
                                   float(row[11]), float(row[14]), 
                                   float(row[16]))
        
        coords.append((x, y, z, S))
        data.append((M, T_e, U, V, W, p, rho))
        streamlineLength.append(S)
        rowCount += 1
    
    streamlineCoords = np.array(coords).reshape((len(coords), 1, 4))
    streamlineData = np.array(data)


    return streamlineCoords, streamlineData

#%%
class turbulentCF(object):
    def __init__(self, streamlines, streamlineData):
        self.streamlineCoordinates = streamlines
        self.streamlineData = streamlineData
        
        # Get the streamline parameter in a single array.
        self.parameterisedStreamline = self.streamlineCoordinates[:, 0, 3]
        # Calculate the velocity along the streamline
        streamlineVelocity = np.linalg.norm(self.streamlineData[:, 2:4], axis=1)
        # Fit a cubic spline to the streamline velocity
        # Need this to calculate velocity derivatives
        self.parameterisedVelocity = interpolate.CubicSpline(self.parameterisedStreamline, streamlineVelocity, extrapolate=1)
        # Calculate the first derivative        
        self.parameterisedvelocityPrime = self.parameterisedVelocity.derivative(nu=1)
        # Calculate the second derivative
        self.parameterisedvelocityDoublePrime = self.parameterisedVelocity.derivative(nu=2)
        
        # Parameterise temperature, pressure and density along the streamline
        self.parameterisedM = interpolate.interp1d(self.parameterisedStreamline, self.streamlineData[:, 0], kind='linear', fill_value='extrapolate')
        self.parameterisedT = interpolate.interp1d(self.parameterisedStreamline, self.streamlineData[:, 1], kind='linear', fill_value='extrapolate')
        self.parameterisedP = interpolate.interp1d(self.parameterisedStreamline, self.streamlineData[:, 5], kind='linear', fill_value='extrapolate')
        self.parameterisedRho = interpolate.interp1d(self.parameterisedStreamline, self.streamlineData[:, 6], kind='linear', fill_value='extrapolate')
        
    def initialisation(self):
        # Get the streamline parameter in a single array.
        self.parameterisedStreamline = self.streamlineCoordinates[:, 0, 3]
        # Calculate the velocity along the streamline
        streamlineVelocity = np.linalg.norm(self.streamlineData[:, 2:4], axis=1)
        inverseStreamlineVelocity = 1./streamlineVelocity
        # Fit a cubic spline to the streamline velocity
        # Need this to calculate velocity derivatives
        self.parameterisedVelocity = interpolate.CubicSpline(self.parameterisedStreamline, streamlineVelocity, extrapolate=1)
        self.parameterisedInverseVelocity = interpolate.CubicSpline(self.parameterisedStreamline, inverseStreamlineVelocity, extrapolate=1)
        # Calculate the first derivative        
        self.parameterisedInverseVelocityPrime = self.parameterisedInverseVelocity.derivative(nu=1)
        # Calculate the second derivative
        self.parameterisedInverseVelocityDoublePrime = self.parameterisedInverseVelocity.derivative(nu=2)
        return self.parameterisedStreamline, streamlineVelocity, self.parameterisedVelocity, self.parameterisedInverseVelocityPrime, self.parameterisedInverseVelocityDoublePrime
    
    
    def getProperties(self, S, regime):
        """ This function sets all the local properties given a streamline 
        coordinate S 
        """
        if regime == 'turbulent':
            r = 0.89
        elif regime == 'laminar':
            self.r = 0.85

        # Get the local properties along the streamline
        self.T = self.parameterisedT(S)
        self.p = self.parameterisedP(S)
        self.rho = self.parameterisedRho(S)
        self.M = self.parameterisedM(S)
        self.velocity = self.parameterisedVelocity(S)
        self.inverseVelocity = self.parameterisedInverseVelocity(S)
        self.inverseVelocityPrime = self.parameterisedInverseVelocityPrime(S)
        self.inverseVelocityDoublePrime = self.parameterisedInverseVelocityDoublePrime(S)
        
        if self.T < 95.:
                self.mu = (1.488 * 10**-6.) * self.T**0.5 / (1. + 122.1*(10.**(-5/self.T))/self.T)
                self.mu_wall = (1.488 * 10**-6.) * T_wall**0.5 / (1. + 122.1*(10.**(-5/T_wall))/T_wall)
        else:
                self.mu = mu_0 * (self.T/T_0)**(3./2) * ((T_0 + sutherland_S) / (self.T + sutherland_S))
                self.mu_wall = mu_0 * (T_wall/T_0)**(3./2) * ((T_0 + sutherland_S) / (T_wall + sutherland_S))
    
        self.localReynolds = self.rho * self.velocity * S / self.mu
        
        self.T_adiabaticWall =  self.T * (1. + r*((g_inf - 1.)/2.) * self.M**2.)
        
        self.TawOnT = self.T_adiabaticWall/self.T
        self.TwOnT = T_wall/self.T
        
        if regime == 'laminar':
#            self.mu_wallPrime = 
#            self.muPrime = 
            pass
        elif regime == 'turbulent':
            self.Reynolds_L = self.rho/self.mu * self.mu/self.mu_wall * self.TwOnT**-0.5
            
            self.ReynoldsStar = self.Reynolds_L / self.inverseVelocityPrime
                
            a = (self.T_adiabaticWall + T_wall) / self.T - 2.
            b = (self.T_adiabaticWall - T_wall) / self.T
            c = ( ((self.T_adiabaticWall + T_wall) / self.T)**2. - 4.*self.TwOnT )**0.5
            
            A = a/c
            B = b/c
            
            self.Q = ( (self.TawOnT - 1.)**0.5 ) / (np.arcsin(A) + np.arcsin(B))
            
            if self.ReynoldsStar > 0:
                self.chi_max = 8.7 * self.Q * log(self.ReynoldsStar, 10)
        
    def calculateLaminar_cf(self, S_0):
        """ This function calculates the laminar skin friction coefficient 
        following the integral method by Walz.
        """
        
        thetaTilde = (self.T_adiabaticWall - T_wall) / (self.T_adiabaticWall - self.T)
        
        delta1 = 1
        delta2 = 2
        delta3 = 3
        
        
        WStar = delta3 / delta2
        
        bracket = 1. + self.r*(g_inf - 1.)/2. * self.M**2. * (1.16*WStar - 1.072 - thetaTilde*(2.*WStar - 2.581))
        chi = bracket**0.7 * (1. + self.r*(g_inf - 1.)/2. * self.M**2. * (1. - thetaTilde))**-0.7
                
        beta_mu = 0.1564 + 2.1921* (WStar - 1.515)**1.70
        
        g = 0.324 + 0.336 * (WStar - 1.515)**0.555
        
        delta1MuOnDelta = 0.42 * - (WStar - 1.515)**(0.424*WStar)
        
        psi_12 = (2. - delta1MuOnDelta * thetaTilde)/WStar + (1. - delta1MuOnDelta) / (WStar*g) * (1. - thetaTilde)
        psi_0Prime = 0.0144 * (2. - WStar) * (2. - thetaTilde)**0.8
        psi = 1. + self.M * (psi_12 - 1.) / ( self.M + (psi_12 - 1)/psi_0Prime )
        
        H_12 = 4.0306 - 4.2845 * (WStar - 1.515**0.3886)
        
        a = 1.7261 * (WStar - 1.515)**0.7158
        b = 1. + r * (g_inf - 1)/2. * self.M**2. * (W - thetaTilde) * (2. - W)
        H = b * H_12 + r * (g_inf - 1.)/2. * self.M**2. * (W - thetaTilde)

        if W <= 1.515:
            print "Uh oh, laminar boundary layer separated... I don't know what to do."

#        F_1 = 3. + 2.*H - self.M**2. + n * (self.mu_wallPrime/self.mu_wall) / (self.muPrime/self.mu)
        F_1 = 3. + 2.*H - self.M**2.
        F_2 = 2*a/b
        F_3 = 1. - H*r * (g_inf - 1.) * self.M**2. * (1. - thetaTilde/W)
        F_4 = (2. * beta - a * W) / b
        
        # Calculate the initial condition
        f = lambda beta, a, W: 2.*beta - a*W
        WStar_0 = brentq(f, -5., 5.)
        
        
        # Set up the ODE solver
        rZ = ode(self.walz).set_integrator('dopri5', atol=1e-2, rtol=1e-2, )
        rZ.set_initial_value(self.chi_0, S_0)
        
        rW = ode(self.walz).set_integrator('dopri5', atol=1e-2, rtol=1e-2, )
        rW.set_initial_value(self.chi_0, S_0)
        
        dt = 0.01
        
        while rZ.t < self.parameterisedStreamline[-1] and rW.t < self.parameterisedStreamline[-1]:
            if W <= 1.515:
                print "Uh oh, laminar boundary layer separated... I don't know what to do."
                rZ.integrate(rZ.t+dt)
                rW.integrate(rW.t+dt)
            
            else:
                Z = delta2 * self.rho*self.velocity*delta2/self.mu_wall
                W = delta3/delta2
                self.streamlineCfs.append(2. / r2.y**2.)
                self.streamlineXs.append(r2.t)

    
        return self.streamlineCfs, self.streamlineXs
    
    def walzZ(self, Z, F1, F2):
        """ Walz method for laminar compressible boundary layer """
    
        zPrime = -1. * self.velocityPrime/self.velocity*F1*Z + F2
        
        return ZPrime
    
    def walzW(self, W, F3, F4, Z):
        """ Walz method for laminar compressible boundary layer """
    
        zPrime = -1. * self.velocityPrime/self.velocity*F3*W + F4/Z
        
        return ZPrime
    
    def calculateTurbulent_cf(self, S_0):
        """
        Return the respective streamline given a starting position.
        point : [x y z] defining the starting point. MUST be one of the 
        starting points used to generate the streamlines in the first place.
        
        """
        
        self.getProperties(S_0, 'turbulent')
        
        cf_0 = 0.455 / ( self.Q**2. * log(0.06/self.Q * self.localReynolds * self.mu/self.mu_wall * (self.TwOnT)**-0.5 )**2.)
        cf_0 = 0.00165
        self.chi_0 = (2. / cf_0)**0.5
        
        self.streamlineCfs = []
        self.streamlineXs = []
        
        r1 = ode(self.whiteChristoph1).set_integrator('dopri5', atol=1e-2, rtol=1e-2, )
        r1.set_initial_value(self.chi_0, S_0)
        
        r2 = ode(self.whiteChristoph2).set_integrator('dopri5', atol=1e-2, rtol=1e-2, )
        r2.set_initial_value(self.chi_0, S_0)
        
        dt = 0.01
        
        # Integrate along the streamline
        while r1.t < self.parameterisedStreamline[-1] and r2.t < self.parameterisedStreamline[-1]:            
#            print r1.t + dt
#            print r2.t + dt
            r1.integrate(r1.t+dt)

            if self.ReynoldsStar < 0:
                self.streamlineCfs.append(2. / r1.y**2.)
                self.streamlineXs.append(r1.t)
            
            else:
#                print "r2"
                r2.integrate(r2.t+dt)
                
                if r1.y/self.chi_max < 0.36:
                    self.streamlineCfs.append(2. / r1.y**2.)
                    self.streamlineXs.append(r1.t)
                    
                elif r2.y/self.chi_max > 0.36:
                    self.streamlineCfs.append(2. / r2.y**2.)
                    self.streamlineXs.append(r2.t)
                else: 
                    print "Something stuffed up"
    
        return self.streamlineCfs, self.streamlineXs
    
    def whiteChristoph1(self, S, chi):
        """ The first of the two DEs """
        self.getProperties(S, 'turbulent')
        self.chi  = chi
        
        chiPrime = 1./8. * self.Reynolds_L * self.velocity * exp(-0.48 * self.chi/self.Q) + 5.5 * self.inverseVelocityPrime/self.inverseVelocity
        return chiPrime
        
    def whiteChristoph2(self, S, chi):
        """ The second of the two DEs """
        self.getProperties(S, 'turbulent')
        self.chi  = chi
        
        z = 1. - self.chi/self.chi_max
        fStar = (2.434 * z + 1.443 * z**2.) * exp(-44. * z**6.)
        gStar = 1. - 2.3 * z + 1.76 * z**3.
        numerator = self.inverseVelocityPrime/self.inverseVelocity * (1. + 9*self.Q**-2 * gStar * self.ReynoldsStar**0.07) + self.inverseVelocityDoublePrime/self.inverseVelocityPrime * (2. * self.Q**2. * gStar * self.ReynoldsStar**0.07)
        chiPrime = numerator / (0.16 * fStar * self.Q**3.)
        return chiPrime  

#%%
if __name__ == '__main__':
    streamlines, streamlineData = importer()
    
    cfClass = turbulentCF(streamlines, streamlineData)
    x, streamlineVelocity, parameterisedVelocity, parameterisedvelocityPrime, parameterisedvelocityDoublePrime = cfClass.initialisation()
    
    cfs, xs = cfClass.calculateTurbulent_cf(0.05)
    plt.plot(xs, cfs)
    