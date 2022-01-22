import gc
import os

import numpy as np
import asdf

from scipy.ndimage import gaussian_filter

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from tools.bitpacked import unpack_rvint
from tools.tsc import numba_tsc_3D

# redshifts
zs = [0.1]

# sample directories
sim_name = "AbacusSummit_highbase_c000_ph100"; Lbox = 1000.; N_dim = 1024; R = 1.; n_chunks = 16 # Mpc/h
#sim_name = "AbacusSummit_base_c000_ph002"; Lbox = 2000.; N_dim = 2048; R = 1.; n_chunks = 34 # Mpc/h

# simulation directory
sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name

def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox/N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth

# for each redshift
for i in range(len(zs)):
    # this redshift
    z = zs[i]
    print("redshift = ", z)


    # if you don't have an unsmoothed density map
    dens = np.zeros((N_dim, N_dim, N_dim))

    # load the matter particles
    for i_chunk in range(n_chunks):
        print(i_chunk)
        # halo and field particles
        fn_halo = sim_dir+f'/halos/z{z:.3f}/halo_rv_A/halo_rv_A_{i_chunk:03d}.asdf'
        fn_field = sim_dir+f'/halos/z{z:.3f}/field_rv_A/field_rv_A_{i_chunk:03d}.asdf'

        # write out the halo (L0+L1) matter particles
        halo_data = (asdf.open(fn_halo)['data'])['rvint']
        pos_halo, _ = unpack_rvint(halo_data, Lbox, float_dtype=np.float32, velout=False)
        pos_halo += Lbox/2. # TESTING
        print("pos_halo = ", pos_halo[:5])
        numba_tsc_3D(pos_halo, dens, Lbox)
        del halo_data, pos_halo
        gc.collect()

        # write out the field matter particles
        field_data = (asdf.open(fn_field)['data'])['rvint']
        pos_field, _ = unpack_rvint(field_data, Lbox, float_dtype=np.float32, velout=False)
        pos_field += Lbox/2. # TESTING
        print("pos_field = ", pos_field[:5])
        numba_tsc_3D(pos_field, dens, Lbox)
        del field_data, pos_field
        gc.collect()

    dens_smooth = smooth_density(dens, R, N_dim, Lbox)
    #dens_smooth = dens
    print(dens_smooth[:2])
    print("saving")
    np.save(f'/global/cscratch1/sd/boryanah/MW_AbacusSummit/smoothed_density_R{R:.1f}_{sim_name:s}_z{z:.1f}.npy', dens_smooth)
