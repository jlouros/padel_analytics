"""
Improved EKF implementation with analytical Jacobians for better numerical stability.
"""

import numpy as np

class AnalyticalExtendedKalmanFilter:
    """
    EKF implementation with analytical Jacobians for better numerical stability.
    """
    
    def __init__(self, P, Q, R, x0):
        self.P = P
        self.Q = Q 
        self.R = R
        self.x = x0
        self.states = []
        self.state_uncertainties = []
        self.g = 9.81  # gravity constant
        
    def transition_jacobian(self, x, dt=1./30):
        """
        Analytical Jacobian of the transition function.
        State: [x, y, z, vx, vy, vz, 1]
        """
        F = np.array([
            [1, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0,  dt, 0,  0], 
            [0, 0, 1, 0,  0,  dt, -0.5*self.g*dt**2],
            [0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0,  0,  1,  -self.g*dt],
            [0, 0, 0, 0,  0,  0,  1]
        ])
        
        # Handle bouncing conditions analytically
        if x[2] < 0:  # Ball below ground
            # Reflection matrices would go here
            # For simplicity, using identity for bounce jacobian
            pass
            
        return F
    
    def observation_jacobian(self, x, projection_matrix):
        """
        Analytical Jacobian of the observation function (3D to 2D projection).
        """
        P = projection_matrix
        xyz1 = np.append(x[:3], 1)  # homogeneous coordinates
        
        # Project to get homogeneous image coordinates
        uvw = P @ xyz1
        u, v, w = uvw[0], uvw[1], uvw[2]
        
        # Analytical derivatives of [u/w, v/w] w.r.t [x, y, z]
        H = np.zeros((2, 7))
        
        # du/dx, du/dy, du/dz
        H[0, 0] = (P[0,0]*w - P[2,0]*u) / (w**2)
        H[0, 1] = (P[0,1]*w - P[2,1]*u) / (w**2)  
        H[0, 2] = (P[0,2]*w - P[2,2]*u) / (w**2)
        
        # dv/dx, dv/dy, dv/dz  
        H[1, 0] = (P[1,0]*w - P[2,0]*v) / (w**2)
        H[1, 1] = (P[1,1]*w - P[2,1]*v) / (w**2)
        H[1, 2] = (P[1,2]*w - P[2,2]*v) / (w**2)
        
        # Velocity components don't affect observation
        H[:, 3:6] = 0
        # Homogeneous coordinate doesn't affect observation  
        H[:, 6] = 0
        
        return H
        
    def predict(self, dt=1./30):
        """Prediction step with analytical Jacobian."""
        self.x = self.transition_function(self.x, dt)
        F = self.transition_jacobian(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q
        
        self.states.append(self.x.copy())
        self.state_uncertainties.append(self.P.copy())
        
    def update(self, z, projection_matrix):
        """Update step with analytical Jacobian."""
        if z == (0, 0):
            return  # Skip invalid measurements
            
        H = self.observation_jacobian(self.x, projection_matrix)
        predicted_z = self.observation_function(self.x, projection_matrix)
        
        y = z - predicted_z  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
