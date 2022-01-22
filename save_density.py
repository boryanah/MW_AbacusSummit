import gc
import os

import numpy as np
import asdf

from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from tools.bitpacked import unpack_rvint
from tools.tsc import numba_tsc_3D

# redshifts
z_this = 0.1

# sample directories
sim_name = "AbacusSummit_highbase_c000_ph100"; Lbox = 1000.; N_dim = 1024; R = 1.; n_chunks = 16 # Mpc/h
#sim_name = "AbacusSummit_base_c000_ph002"; Lbox = 2000.; N_dim = 2048; R = 1.; n_chunks = 34 # Mpc/h

# simulation directory
sim_dir = f"/global/project/projectdirs/desi/cosmosim/Abacus/{sim_name:s}/halos/z{z_this:.3f}/halo_info/"
dens_smooth = np.load(f'/global/cscratch1/sd/boryanah/MW_AbacusSummit/smoothed_density_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy')

print("loading")
cat = CompaSOHaloCatalog(sim_dir, load_subsamples=False, fields = ['N', 'x_L2com'])
#part_mass = cat.header['ParticleMassHMsun']
#mass_halo = cat.halos['N'].astype(np.float64) * part_mass
pos_halo = cat.halos['x_L2com']+Lbox/2.
print("loaded")

cell_size = Lbox/N_dim
pos_halo /= cell_size
env = interpn((np.arange(N_dim)+0.5, np.arange(N_dim)+0.5, np.arange(N_dim)+0.5), dens_smooth, pos_halo, bounds_error=False, fill_value=None) # option 1

#pos_halo = (pos_halo.astype(int))%N_dim; env = dens_smooth[pos_halo[:, 0], pos_halo[:, 1], pos_halo[:, 2]] # option 2
print(env[:200])
#print(env2[:200])
np.save(f'/global/cscratch1/sd/boryanah/MW_AbacusSummit/environment_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy', env)
