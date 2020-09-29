import pylds
from pylds.base import compute_lagrangian_descriptor
from pylds.tools import draw_lagrangian_descriptor

import numpy as np

# Define vector field
def cirque_vector_field(t, u, PARAMETERS=[0.5, 1, 1]):
    """
    Returns 1D vector field of the Cirque system, for an array of points in phase space.
    Number of model parameters: 0 . PARAMETERS = [None]
    Functional form: v = (y, x - x**3), with u = (x, y)
    
    Parameters
    ----------  
    t : float
        fixed time-point of vector field, for all points in phase space.
        
    u : array_like, shape(n,)
        Points in phase space.
        
    PARAMETERS : list of floats
        Vector field parameters.
    
    Returns
    -------
    v : array_like, shape(n,)
        Vector field corresponding to points u, in phase space at time t.
    """
    N_dims = u.shape[-1]
    points_positions = u.T[:int(N_dims/2)]
    points_momenta = u.T[int(N_dims/2):]
    x, y = points_positions
    p_x, p_y = points_momenta 
    
    # Hamiltonian Model Parameter
    W0, k, d = PARAMETERS
    
    # Vector field defintion
    v_x   =  p_x
    v_y   =  p_y
    v_p_x = -6*W0*(k**6)* ( (x + d)/((x + d)**2 + y**2 + k**2)**4 + (x - d)/((x - d)**2 + y**2 + k**2)**4 )
    v_p_y = -6*W0*(k**6)* y * ( 1/((x + d)**2 + y**2 + k**2)**4 + 1/((x - d)**2 + y**2 + k**2)**4 )
    v = np.array([v_x, v_y, v_p_x, v_p_y]).T
    return v

# Define PES
def cirque_potential(positions, PARAMETERS=[0.5, 1, 1]):
    x, y = positions.T
    
    # Function parameters
    W0, k, d = PARAMETERS
    
    # Potential energy function
    V = -W0*(k**6) * ( 1/((x + d)**2 + y**2 + k**2)**3 + 1/((x - d)**2 + y**2 + k**2)**3 )
    return V

################################################
#
# COMPUTATION OF LD MAPS
#
################################################
# Lp-norm, p-value
p_value = 1/2

ax1_min,ax1_max = [-2.5, 2.5]
ax2_min,ax2_max = [-1, 1]
N1, N2 = [600, 600]

# Box escape condition
box_boundaries = False

# Miscellaneous grid parameters
dims_fixed = [0,1,0,0] # Variable ordering (x y p_x p_y)
dims_fixed_values = [0] # This can also be an array of values
dims_slice = [1,0,1,0] # Visualisation slice
momentum_sign = 1 # Direction of momentum that defines the slice - (1) positive / (-1) negative

# potential parameters
n = -10 # double well character PES
dk = 0.1
W, k, d = [0.5, np.sqrt(7) + n*dk, 1]

potential_energy = lambda u: cirque_potential(u, PARAMETERS=[W, k, d])
vector_field = lambda t, u: cirque_vector_field(t, u, PARAMETERS=[W, k, d])

# Mesh visualisation slice parameters
dH = 0.27 # Chosen value to sample regular behaviour at kc
H0 = potential_energy(np.zeros(2)) + dH # Energy

slice_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]

grid_parameters = {
        'slice_parameters' : slice_parameters,
        'dims_slice' : dims_slice,
        'dims_fixed' : [dims_fixed, dims_fixed_values],
        'momentum_sign' : momentum_sign,
        'potential_energy': potential_energy,
        'energy_level': H0
    }

# Loop over tau values
tau_min = 10
tau_max = 102
d_tau = 2
for tau in np.arange(tau_min, tau_max, d_tau):
    # compute total LD
    LD_forward = compute_lagrangian_descriptor(grid_parameters, vector_field, tau, p_value)
    LD_backward = compute_lagrangian_descriptor(grid_parameters, vector_field, -tau, p_value)
    LD_total = LD_forward + LD_backward
    # save output data
    folder_name = "bifurcation/k_variable/"
    outfile_name = "LD_total_kn_n_"+str(n)+"_dk_"+str(dk)+"_tau_"+str(tau)+".dat"
    LD_total.dump(folder_name + outfile_name)
