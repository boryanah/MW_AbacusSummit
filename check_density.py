# 3 questions: redshift ranges and main_progs = 0 (screws up the first halo) and unphysical halo
import glob
import os
from pathlib import Path

import asdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import spatial
from scipy.interpolate import interpn
from colossus.cosmology import cosmology as coco

from tools.merger import simple_load, get_halos_per_slab, get_one_header, unpack_inds, count_progenitors
from tools.compute_dist import dist

def extract_superslab(fn):
    '''
    Extract int(6) from "associations_z0.100.6.asdf"
    '''
    fn = str(fn)
    return int(fn.split('.')[-2])

def get_mt_info(fn_load, fields, minified):
    '''
    Load merger tree and progenitors information
    '''
    # loading merger tree info
    mt_data = simple_load(fn_load, fields=fields)
    
    # turn data into astropy table
    Merger = mt_data['merger']

    # if loading all progenitors
    if "Progenitors" in fields:
        num_progs = Merger["NumProgenitors"]
        # get an array with the starting indices of the progenitors array
        start_progs = np.zeros(len(num_progs), dtype=int)
        start_progs[1:] = num_progs.cumsum()[:-1]
        Merger.add_column(start_progs, name='StartProgenitors', copy=False)
    return mt_data

def correct_inds(halo_ids, N_halos_slabs, slabs, inds_fn):
    '''
    Reorder indices for given halo index array with 
    corresponding n halos and slabs for its time epoch
    '''
    # number of halos in the loaded superslabs
    N_halos_load = np.array([N_halos_slabs[i] for i in inds_fn])
    
    # unpack slab and index for each halo
    slab_ids, ids = unpack_inds(halo_ids)

    # total number of halos in the slabs that we have loaded
    N_halos = np.sum(N_halos_load)
    offsets = np.zeros(len(inds_fn), dtype=int)
    offsets[1:] = np.cumsum(N_halos_load)[:-1]
    
    # determine if unpacking halos for only one file (Merger_this['HaloIndex']) -- no need to offset 
    if len(inds_fn) == 1: return ids

    # select the halos belonging to given slab
    for i, ind_fn in enumerate(inds_fn):
        select = np.where(slab_ids == slabs[ind_fn])[0]
        ids[select] += offsets[i]
    return ids


# simulation parameters
sim_name = "AbacusSummit_highbase_c000_ph100" # smaller simulation
#sim_name = "AbacusSummit_base_c000_ph002" # larger simulation
merger_parent = Path("/global/project/projectdirs/desi/cosmosim/Abacus/merger")
catalog_parent = Path("/global/cscratch1/sd/boryanah/new_lc_halos/")
merger_dir = merger_parent / sim_name
header = get_one_header(merger_dir)
Lbox = header['BoxSize']
N_dim = 1024; R = 1.; cell_size = Lbox/N_dim
print("boxsize = ", Lbox)

# redshift of interest (AbacusSummit finishes at z = 0.1; higher z is chosen so as to encompass time of 3rd merger
z_start = 0.1
z_stop = 2.5

# all redshifts of the merger tree files from high to low z
zs_mt = np.load(Path("/global/homes/b/boryanah/repos/abacus_lc_cat/data_mt") / sim_name / "zs_mt.npy") # TODO: automatize
n_superslabs = len(list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0])))
print("number of superslabs = ",n_superslabs)

# starting and finishing redshift indices
ind_start = np.argmin(np.abs(zs_mt - z_start))
ind_stop = np.argmin(np.abs(zs_mt - z_stop))

# indices to load (loading all superslabs)
inds_fn_this = np.arange(n_superslabs, dtype=int)
inds_fn_prev = np.arange(n_superslabs, dtype=int)

# Constraint 1: mass today TODO: would need to be corrected for evolution between z = 0.1 and z = 0
h = 0.6736
mass_fin_lb = 0.9e12*h # Msun/h
mass_fin_hb = 1.2e12*h # Msun/h
print("final mass logM", np.log10(mass_fin_lb), np.log10(mass_fin_hb))
print(f"final mass M {mass_fin_lb:.2e} {mass_fin_hb:.2e}")

# fields to extract from the merger trees
fields_mt = ['HaloMass', 'Position', 'IsPotentialSplit', 'MainProgenitor']

# load recorded merger infomation
n_merger_m1 = np.load(f"data/n_merger_m1_{sim_name:s}.npy")
n_merger_m2 = np.load(f"data/n_merger_m2_{sim_name:s}.npy")
n_merger_m3 = np.load(f"data/n_merger_m3_{sim_name:s}.npy")
missing = np.load(f"data/missing_{sim_name:s}.npy")
# for testing purposes
#missing = np.ones(len(missing), dtype=bool)
int_selection = (n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & missing
all_selection = (missing)

# current redshift
z_this = zs_mt[ind_start]

# merger files
fns_this = list(sorted(merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf'), key=extract_superslab))
for counter in range(len(fns_this)):
    fns_this[counter] = str(fns_this[counter])
    
# get number of halos in each slab and number of slabs
N_halos_slabs_this, slabs_this = get_halos_per_slab(fns_this, minified=False)

# loading tree data
mt_data_this = get_mt_info(fns_this, fields=fields_mt, minified=False)
Merger_this = mt_data_this['merger']

# select halos by mass
initial_selection = (Merger_this['HaloMass'] > mass_fin_lb) & (Merger_this['HaloMass'] <= mass_fin_hb) & (Merger_this['MainProgenitor'] > 0) & (Merger_this['IsPotentialSplit'] == 0)
initial_inds = np.arange(len(Merger_this['HaloMass']), dtype=int)[initial_selection]
initial_position = Merger_this['Position'][initial_inds]+Lbox/2.
print(initial_position.min(), initial_position.max())

# compute environment at location of halo
if os.path.exists(f'data/environment_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy'):
    initial_env = np.load(f'data/environment_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy')
else:
    initial_position /= cell_size
    dens_smooth = np.load(f'data/smoothed_density_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy')
    initial_env = interpn((np.arange(N_dim), np.arange(N_dim), np.arange(N_dim)), dens_smooth, initial_position, bounds_error=False, fill_value=None) # option 1
    #initial_position = (initial_position.astype(int))%N_dim; initial_env = dens_smooth[initial_position[:, 0], initial_position[:, 1], initial_position[:, 2]] # option 2
    np.save(f'data/environment_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy', initial_env)

print("number of all halos = ", np.sum(all_selection))
print("number of all halos of interest = ", np.sum(int_selection))

print("mean env of all halos = ", np.mean(initial_env[all_selection]))
print("mean env of all halos of interest = ", np.mean(initial_env[int_selection]))

print("std env of all halos = ", np.std(initial_env[all_selection]))
print("std env of all halos of interest = ", np.std(initial_env[int_selection]))
