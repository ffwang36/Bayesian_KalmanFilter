import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from typing import Tuple
from statsmodels.graphics import tsaplots

class Diagnostics_plots():
    '''Class works for both simulation and Kalman Filter Estimation'''
    def __init__():
        pass
    
    def plot_TruePos(self, x):
        '''Simulation: Plots simulated true position of a particle'''
        plt.figure()
        plt.title('True Position')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Position (Meters)')
        plt.plot(range(self.t_steps), x[0,:])
        return plt.show()
    
    def plot_TrueVelocity(self, x):
        '''Simulation: Plots simulated true velocity of a particle'''
        plt.figure()
        plt.title('True Velocity')
        plt.xlabel('Velocity (Meters/Seconds)')
        plt.ylabel('Position (Meters)')
        plt.plot(range(self.t_steps), x[1,:])
        return plt.show()
    
    def plot_Obs(self, z):
        '''Simulation: Plots true observations of a particle'''
        plt.figure()
        plt.title('Position Observations')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Position Measurement (Meters)')
        plt.plot(range(self.t_steps), z[0,:])
        return plt.show()
    
    def plot_TruePos_Obs(self, x, z):
        '''Simulation: Compare true position and observations of a particle'''
        plt.figure()
        plt.title('Compare true position and observations')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Position (Meters)')
        plt.plot(range(self.t_steps), x[0,:])
        plt.plot(range(self.t_steps), z[0,:])
        return plt.show()
    
    def plot_est_predTrack(self, x, z, xpred, xest): #we can make true observation optional. 
        '''Kalman Filter: Compare estimated and predicted target track against true'''
        plt.figure()
        plt.title('Estimated and Predicted Target Track')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Position (Meters)')
        plt.legend()
        
        plt.plot(range(self.t_steps), x[0,:], label = 'True Position')
        plt.plot(range(self.t_steps), z[0,:], label = 'Observation Position', linestyle = '--')
        plt.plot(range(self.t_steps), xpred[0,:], label = 'Predicted Position', linestyle = ':')
        plt.plot(range(self.t_steps), xest[0,:], label = 'Estimated Position', linestyle = '-.')
        return plt.show()
    
    def plot_innov(self, xinnov):
        '''Plot innovation''' 
        innov_std = xinnov[0].std()

        plt.figure()
        plt.title('Innovation')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Innovation (Meters)')
        plt.legend(loc = 'upper right')
        
        plt.plot(range(self.t_steps), xinnov[0,:], label = 'Innovation')
        plt.axhline(innov_std, color = 'green', label = 'Innovation std', linestyle = ':')
        plt.axhline(-innov_std, color = 'green', linestyle = ':')
        plt.axhline(2*innov_std, color = 'lawngreen', label = 'Innovation 2 std')
        plt.axhline(-2*innov_std, color = 'lawngreen')
        return plt.show()

    def plot_autocorr(self, xinnov):
        '''Plot autocorrelation of innovation - test for whiteness'''
        tsaplots.plot_acf(xinnov[0,:])
        return plt.show()

class Simulate(Diagnostics_plots):
    '''
        Class computes true state-space history
        and true observations from a discrete-time
        model with no input. For use with a Kalman Filter.
    '''

    def __init__(self, F_init, G_init, H_init, Q_init, R_init, x0_init, t_steps_init):
        '''
        Initialise the class
        Args:
            F (numpy.ndarray):  Xsize*Xsize state transition matrix
            G (numpy.ndarray):  Xsize*Vsize state noise transition matrix
            H (numpy.ndarray):  Zsize*Xsize observation matrix
            Q (numpy.ndarray):  Vsize*Vsize process noise covariance matrix
            R (numpy.ndarray):  Zsize*Zsize observation noise covariance matrix
            x0 (numpy.ndarray): Xsize*1 initial state vector 
            t_steps (int): number of time-steps to be simulated

        Returns:
            z (numpy.ndarray): Zsize*t_steps Observation time history
            x (numpy.ndarray): Xsize*t_steps true state time history
        '''
        self.F = F_init
        self.G = G_init
        self.H = H_init
        self.Q = Q_init
        self.R = R_init
        self.x0 = x0_init
        self.t_steps = t_steps_init       
    
    def _validate_data(self):
        '''Validate input data.'''
        assert self.F.shape[0] == self.F.shape[1], 'F is non-square'
        assert self.x0.shape[0] == self.F.shape[0], 'x0 does not match dimension of F'
        assert self.G.shape[0] == self.x0.shape[0], 'G does not match dimension of x0 or F'
        assert self.Q.shape[0] == self.Q.shape[1], 'Q must be square'
        assert self.Q.shape[0] == self.G.shape[1], 'Q does not match dimension of G'
        assert self.H.shape[1] == self.x0.shape[0], 'H and Xsize do not match'
        assert self.R.shape[0] == self.R.shape[1], 'R must be square'      
        assert self.R.shape[0] == self.H.shape[0], 'R must match Zsize of H'
        return self.F, self.G, self.H, self.Q, self.R, self.x0
        
    def simulate(self):
        '''Purpose is to simulate the true position and velosity of a particle'''     
        
        # validate data first
        F, G, H, Q, R, x0 = self._validate_data()
        
        # define a few parameters for output matrix dimension specification
        self.Xsize = x0.shape[0]
        self.Zsize = H.shape[0]
        self.Vsize = G.shape[1]
        
        # fix up output matricies   
        x = np.zeros((self.Xsize, self.t_steps))
        z = np.zeros((self.Zsize, self.t_steps))

        # get some gaussian noise - rand('normal')
        v = sqrt(Q) * np.random.randn(self.Vsize, self.t_steps+1)
        w = sqrt(R) * np.random.randn(self.Zsize, self.t_steps+1)
    
        # initial value
        self.x0 = x[:,0].reshape(2,1)
        
        # now generate all the remaining states
        for i in range(self.t_steps-1):
            x[:, i+1] = F @ x[:,i] + G @ v[:,i] #Assuming no control inputs B and u

        # then all the observations
        for i in range(self.t_steps):
            z[:, i] = H @ x[:,i] + w[:, i]
        
        return z, x
    
    def __str__(self):
        return 'Simulate data.'

class KalmanFilter(Diagnostics_plots):
    '''A basic linear Kalman filter. Calls COVARS for gain history.'''
    
    def __init__(self, F_init, G_init, H_init, Q_init, R_init, P0_init, x0_init, t_steps_init, z_init):
        '''
        Initialise the class
        Args:
            F (numpy.ndarray):  Xsize*Xsize state transition matrix
            G (numpy.ndarray):  Xsize*Vsize state noise transition matrix
            H (numpy.ndarray):  Zsize*Xsize observation matrix
            Q (numpy.ndarray):  Vsize*Vsize process noise covariance matrix
            R (numpy.ndarray):  Zsize*Zsize observation noise covariance matrix
            P0 (numpy.ndarray): Xsize*Xsize initial state covariance
            x0 (numpy.ndarray): Xsize*1 initial state vector 
            t_steps (int):      number of time-steps to be simulated  
            z (numpy.ndarray):  Zsize*t_steps observation sequence to be filtered

        Returns:
            X (numpy.ndarray):     t_steps*(Xsize*Zsize): Gain history
            Pest (numpy.ndarray):  t_steps*(Xsize*Xsize): Estimate Covariance history
            Ppred (numpy.ndarray): t_steps*(Xsize*Xsize): Prediction Covariance history
            S (numpy.ndarray):     t_steps*(Xsize*Xsize): Innovation Covariance history
            
            xest (numpy.ndarray):  Xsize*t_steps estimated state time history
            xpred (numpy.ndarray): Xsize*t_steps predicted state time history
            innov (numpy.ndarray): Zsize*t_steps innovation time history
        '''
        self.F = F_init
        self.G = G_init
        self.H = H_init
        self.Q = Q_init
        self.R = R_init
        self.P0 = P0_init
        self.x0 = x0_init
        self.t_steps = t_steps_init    
        self.z = z_init
    
    def _validate_data(self):
        '''Validate input data.'''
        assert self.P0.shape[0] == self.P0.shape[1], 'P0 is non-square'
        assert self.F.shape[0] == self.F.shape[1], 'F is non-square'
        assert self.x0.shape[0] == self.F.shape[0], 'x0 does not match dimension of F'
        assert self.P0.shape[0] == self.F.shape[0], 'P0 does not match dimension of F'
        assert self.G.shape[0] == self.x0.shape[0], 'G does not match dimension of xo or F'
        assert self.Q.shape[0] == self.Q.shape[1], 'Q must be square'
        assert self.Q.shape[0] == self.G.shape[1], 'Q does not match dimension of G'
        assert self.H.shape[1] == self.x0.shape[0], 'H and Xsize do not match'
        assert self.R.shape[0] == self.R.shape[1], 'R must be square'      
        assert self.R.shape[0] == self.H.shape[0], 'R must match Zsize of H'
        assert self.P0.shape[0] == self.x0.shape[0], 'P0 must have dimensions of x0 (Xsize)'
        assert self.z.shape[0] == self.H.shape[0], 'Observation Sequence must have Zsize rows'
        return self.F, self.G, self.H, self.Q, self.R, self.P0, self.x0, self.z
    
    def _covars(self):
        '''Compute gain and covariance history for a Kalman Filter'''
        # validate data first
        F, G, H, Q, R, P0, x0, z = self._validate_data()
        
        # define a few parameters for output matrix dimension specification
        Xsize = P0.shape[0]
        Zsize = H.shape[0]
        
        # fix up output matricies
        W = np.zeros((self.t_steps, Xsize * Zsize))
        Pest = np.zeros((self.t_steps, Xsize * Xsize))
        Ppred = np.zeros((self.t_steps, Xsize * Xsize))
        S = np.zeros((self.t_steps, Zsize * Zsize))
        
        # initial value
        lPest = P0
        
        # ready to go !
        for i in range(self.t_steps-1): 
            # first the actual calculation in local variables
            lPpred = F @ lPest @ F.T + G @ Q @ G.T
            lS = H @ lPpred @ H.T + R
            lW = lPpred @ H.T @ np.linalg.inv(lS) 
            lPest = lPpred - lW @ lS @ lW.T
            # then record the results in columns of output states
            Pest[i+1, :] = lPest.reshape(1, Xsize*Xsize)
            Ppred[i+1, :] = lPpred.reshape(1, Xsize*Xsize)
            W[i+1, :] = lW.reshape(1, Xsize*Zsize)
            S[i+1, :]= lS.reshape(1, Zsize*Zsize)

        return W, Pest, Ppred, S
        
    def xestim(self):
        '''Estimate linear kalman filter. calls _covars for gain history.'''
        # validate data first
        F, G, H, Q, R, P0, x0, z = self._validate_data()
        
        # define a few parameters for output matrix dimension specification
        Xsize = P0.shape[0]
        Zsize = H.shape[0]

        # fix up output matricies
        xest = np.zeros((Xsize, self.t_steps))
        xpred = np.zeros((Xsize, self.t_steps))
        innov = np.zeros((Zsize, self.t_steps))

       # compute all the neccesary gain matricies a priori
        W, _, _, _ = self._covars()

        # initial prediction and estimate (done seperately because of X0)
        lW = W[0,:].reshape(Xsize,Zsize)
        xpred[:,0, None] = F @ x0
        innov[:,0, None] = z[:,0] - H @ xpred[:,0]
        xest[:,0] = xpred[:,0] + lW @ innov[:,0]

        # now generate all the remaining estimates
        for i in range(self.t_steps-1):
            xpred[:,i+1] = F @ xest[:,i]
            innov[:,i+1] = z[:,i+1] - H @ xpred[:,i+1]
            lW = W[i+1,:].reshape(Xsize,Zsize)
            xest[:,i+1] = xpred[:,i+1] + lW @ innov[:,i+1]

        return xest, xpred, innov
       
    def __str__(self):
        return 'Basic Kalman Filter.'