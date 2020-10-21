#!/usr/bin python
import sys
import numpy as np
import pylds
from pylds.base import compute_lagrangian_descriptor, perturb_field
from pylds.tools import draw_lagrangian_descriptor

def cirque_vector_field(t, u, PARAMETERS=[0.5, 1, 1]):
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

def cirque_potential(positions, PARAMETERS=[0.5, 1, 1]):
    x, y = positions.T
    
    # Function parameters
    W0, k, d = PARAMETERS
    
    # Potential energy function
    V = -W0*(k**6) * ( 1/((x + d)**2 + y**2 + k**2)**3 + 1/((x - d)**2 + y**2 + k**2)**3 )
    return V

# potential parameters
W, k, d = [0.5, np.sqrt(7), 1]

# define potential and vector field
potential_energy = lambda u: cirque_potential(u, PARAMETERS=[W, k, d])
vector_field = lambda t, u: cirque_vector_field(t, u, PARAMETERS=[W, k, d])


H0 = float(sys.argv[1]) #energy
tau = int(sys.argv[2]) #integration time
ax1_min = float(sys.argv[3])
ax1_max = float(sys.argv[4])
N1 = int(sys.argv[5])
ax2_min = float(sys.argv[6])
ax2_max = float(sys.argv[7])
N2 = int(sys.argv[8])

# Lp-norm, p-value
p_value = 1/2
# box escape condition
box_boundaries = False
slice_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]

for slice_type in ['x-px', 'y-py']:    
    if slice_type == 'x-px':
        dims_fixed = [0,1,0,0]
        dims_slice = [1,0,1,0]
        momentum_sign = 1
    elif slice_type == 'y-py':
        dims_fixed = [1,0,0,0]
        dims_slice = [0,1,0,1]
    
    dims_fixed_values = [0]
    momentum_sign = 1
    # input grid params
    grid_parameters = {
            'slice_parameters' : slice_parameters,
            'dims_slice' : dims_slice,
            'dims_fixed' : [dims_fixed, dims_fixed_values],
            'momentum_sign' : momentum_sign,
            'potential_energy': potential_energy,
            'energy_level': H0
        }

    LD_forward = compute_lagrangian_descriptor(grid_parameters, vector_field, tau, p_value)
    folder_name = "bifurcation/k_critical/data/"
    outfile_name = "LD_forward_"+slice_type+"_tau_"+str(tau)+"_k_kc_E_"+str(H0)+".dat"
    LD_forward.dump(folder_name+"/"+outfile_name)

    LD_backward = compute_lagrangian_descriptor(grid_parameters, vector_field, -tau, p_value)
    folder_name = "bifurcation/k_critical/data/"
    outfile_name = "LD_backward_"+slice_type+"_tau_"+str(tau)+"_k_kc_E_"+str(H0)+".dat"
    LD_backward.dump(folder_name+"/"+outfile_name)
