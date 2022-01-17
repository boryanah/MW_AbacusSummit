# 3 questions: redshift ranges and main_progs = 0 (screws up the first halo) and unphysical halo
import glob
import os
from pathlib import Path

import asdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import spatial
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
fields_mt = ['HaloIndex', 'HaloMass', 'Position', 'MainProgenitor', 'IsPotentialSplit', 'Progenitors', 'NumProgenitors']

n_merger_m1 = np.load("data/n_merger_m1.npy")
n_merger_m2 = np.load("data/n_merger_m2.npy")
n_merger_m3 = np.load("data/n_merger_m3.npy")
missing = np.load("data/missing.npy")
int_selection = (n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & missing
all_selection = (missing)

# loop over redshifts
for i in range(ind_start, ind_stop + 1):
    # current redshift
    z_this = zs_mt[i]
    z_prev = zs_mt[i+1]

    # merger files
    fns_this = list(sorted(merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf'), key=extract_superslab))
    fns_prev = list(sorted(merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf'), key=extract_superslab))
    for counter in range(len(fns_this)):
        fns_this[counter] = str(fns_this[counter])
        fns_prev[counter] = str(fns_prev[counter])
    
    # get number of halos in each slab and number of slabs
    N_halos_slabs_this, slabs_this = get_halos_per_slab(fns_this, minified=False)
    N_halos_slabs_prev, slabs_prev = get_halos_per_slab(fns_prev, minified=False)

    # loading tree data
    mt_data_this = get_mt_info(fns_this, fields=fields_mt, minified=False)
    Merger_this = mt_data_this['merger']

    if i == 0:
        initial_selection = (Merger_this['HaloMass'] > mass_fin_lb) & (Merger_this['HaloMass'] <= mass_fin_hb) & (Merger_this['MainProgenitor'] > 0) & (Merger_this['IsPotentialSplit'] == 0)
        merger_mass = np.zeros(len(zs_mt[:(ind_stop+1)]))
        merger_mass_int = np.zeros(len(zs_mt[:(ind_stop+1)]))
        main_progs = np.arange(len(Merger_this['HaloMass']), dtype=int)[initial_selection]
        main_progs_int = main_progs[int_selection]
        main_progs = main_progs[all_selection]
    merger_mass[i] = np.median(Merger_this['HaloMass'][main_progs[main_progs > 0]])
    merger_mass_int[i] = np.median(Merger_this['HaloMass'][main_progs_int[main_progs_int > 0]])

    # no info is denoted by 0 or -999, but -999 messes with unpacking, so we set it to 0
    Merger_this['MainProgenitor'][Merger_this['MainProgenitor'] < 0] = 0

    # rework the main progenitor and halo indices to return in proper order
    Merger_this['MainProgenitor'] = correct_inds(Merger_this['MainProgenitor'], N_halos_slabs_prev, slabs_prev, inds_fn_prev)
    
    # update the main progenitor indices for the next
    main_progs = Merger_this['MainProgenitor'][main_progs]
    main_progs_int = Merger_this['MainProgenitor'][main_progs_int]
    print("halos with no information", np.sum(main_progs <= 0)) # there should be no negative ones
    print("halos of interest with no information", np.sum(main_progs_int <= 0)) # there should be no negative ones
    #main_progs = main_progs[main_progs > 0]
np.save("data/merger_mass.npy", merger_mass)
np.save("data/merger_mass_int.npy", merger_mass_int)
np.save("data/zs_mt.npy", zs_mt[:(ind_stop+1)])
