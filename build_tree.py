# 3 questions: redshift ranges and main_progs = 0 (screws up the first halo) and unphysical halo
import glob
import os
import gc
import sys
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
#sim_name = sys.argv[1]
#sim_name = "AbacusSummit_base_c000_ph002" # larger simulation
merger_parent = Path("/global/project/projectdirs/desi/cosmosim/Abacus/merger")
catalog_parent = Path("/global/cscratch1/sd/boryanah/new_lc_halos/")
merger_dir = merger_parent / sim_name
header = get_one_header(merger_dir)
Lbox = header['BoxSize']

# redshift of interest (AbacusSummit finishes at z = 0.1; higher z is chosen so as to encompass time of 3rd merger
z_start = 0.1
z_stop = 2.5

# load zs from high to low
data_path = Path("/global/homes/b/boryanah/repos/abacus_lc_cat/data_mt")
# if merger tree redshift information has been saved, load it (if not, save it)
if not os.path.exists(data_path / sim_name / "zs_mt.npy"):
    # all merger tree snapshots and corresponding redshifts
    snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
    zs_mt = get_zs_from_headers(snaps_mt)
    os.makedirs(data_path / sim_name, exist_ok=True)
    np.save(data_path / sim_name / "zs_mt.npy", zs_mt)
zs_mt = np.load(data_path / sim_name / "zs_mt.npy")
n_superslabs = len(list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0])))
print("number of superslabs = ",n_superslabs)

# starting and finishing redshift indices
ind_start = np.argmin(np.abs(zs_mt - z_start))
ind_stop = np.argmin(np.abs(zs_mt - z_stop))

# indices to load (loading all superslabs)
inds_fn_this = np.arange(n_superslabs, dtype=int)
inds_fn_prev = np.arange(n_superslabs, dtype=int)

# cosmological parameters  # TODO: automatize (only issue is sigma8); could alternatively take the age from the header files instead of colossus, but error should be small
omega_b = 0.02237
omega_cdm = 0.1200
omega_ncdm = 0.00064420
ns = 0.9649
sigma8 = 0.807952
h = 0.6736
params = {'H0': h*100., 'w0': -1., 'wa': 0., 'Ob0': omega_b/h**2,
          'Om0': omega_cdm/h**2+omega_ncdm/h**2+omega_b/h**2,
          'Ode0': 1.-(omega_cdm/h**2+omega_ncdm/h**2+omega_b/h**2),
          'ns': ns, 'sigma8': sigma8, 'relspecies': False,
          'flat': True, 'de_model': 'lambda'}

# set abacus cosmology in colossus
cc_cosm = coco.setCosmology('colossus_cosm', params)
age = cc_cosm.age(0.) # sanity check
print("age of universe in Gyr", age)

def func(z, target):
    ''' Aiding function for solving for z '''
    return cc_cosm.lookbackTime(z) - target

# Constraint 1: mass today TODO: would need to be corrected for evolution between z = 0.1 and z = 0
#mass_fin_lb = 0.9e12*h # Msun/h # og 
mass_fin_hb = 1.2e12*h # Msun/h
mass_fin_lb = 0.8e12*h # Msun/h # TESTING
print("final mass logM", np.log10(mass_fin_lb), np.log10(mass_fin_hb))

# Constraint 2: mergers with > 1.e11 systems
# First merger LMC-like
age_m1_lb = 1 # Gyr
#age_m1_hb = 3 # Gyr # og
age_m1_hb = 4 # Gyr # TESTING
z_m1_lb = fsolve(func, args=(age_m1_lb), x0=0.)
z_m1_hb = fsolve(func, args=(age_m1_hb), x0=0.)
print("redshift range merger 1:", z_m1_lb, z_m1_hb)

# Second merger Sagittarius-like
#age_m2_lb = 5 # Gyr # og
age_m2_lb = 4 # Gyr # TESTING
#age_m2_hb = 7 # Gyr # og
age_m2_hb = 8 # Gyr # TESTING
z_m2_lb = fsolve(func, args=(age_m2_lb), x0=0.)
z_m2_hb = fsolve(func, args=(age_m2_hb), x0=0.)
print("redshift range merger 2:", z_m2_lb, z_m2_hb)

# Third merger Gaia-Sausage-Enceladus-like
#age_m3_lb = 9 # Gyr # og
age_m3_lb = 8 # Gyr # TESTING
age_m3_hb = 11 # Gyr
z_m3_lb = fsolve(func, args=(age_m3_lb), x0=0.)
z_m3_hb = fsolve(func, args=(age_m3_hb), x0=0.)
print("redshift range merger 3:", z_m3_lb, z_m3_hb)

# masses of all mergers
#mass_ms_lb = 0.8e11*h # Msun/h # og
mass_ms_lb = 1.e11*h # Msun/h # og
#mass_ms_hb = 2.0e11*h # Msun/h # og
mass_ms_hb = 3.0e11*h # Msun/h # TESTING

# Constraint 3: presence of Andromeda, single halo within 1 Mpc
mass_fut_lb = 0.9e12*h # Msun/h # og
#mass_fut_lb = 0.8e12*h # Msun/h # TESTING 
mass_fut_hb = 4.0e12*h # Msun/h
dist_fut_lb = 0.55*h # Mpc/h # 0.6
dist_fut_hb = 1.05*h # Mpc/h # 1.0

# radial velocity of andromeda
vel_rad = -301. # pm 1. km/s
time_start = 3.15e16*cc_cosm.lookbackTime(z_start) # sec time between last abacus redshift and now
dist_rad = vel_rad*time_start/3.086e19*h # Mpc/h
print("Andromeda has traveled in the last Gyr (z=0.1) = ", dist_rad, cc_cosm.lookbackTime(z_start))
dist_fut_lb -= dist_rad # Mpc/h
dist_fut_hb -= dist_rad # Mpc/h

# redshift selection arrays for the three mergers
zs_m1 = (z_m1_lb < zs_mt) & (z_m1_hb >= zs_mt)
zs_m2 = (z_m2_lb < zs_mt) & (z_m2_hb >= zs_mt)
zs_m3 = (z_m3_lb < zs_mt) & (z_m3_hb >= zs_mt)
"""
if np.argmax(zs_m1) != 0:
    zs_m1[np.argmax(zs_m1)-1] = True
if np.argmax(zs_m2) != 0:
    zs_m2[np.argmax(zs_m2)-1] = True
if np.argmax(zs_m3) != 0:
    zs_m3[np.argmax(zs_m3)-1] = True
"""
print("redshifts of interest for m1", zs_mt[zs_m1])
print("redshifts of interest for m2", zs_mt[zs_m2])
print("redshifts of interest for m3", zs_mt[zs_m3])


# thoughts on redshift ranges:
# the above redshift ranges refer to the upper and lower bounds when the mergers could have occurred
# otoh, the abacus epochs are limited and somewhat coarse. For a given z_this where we have a halo
# comprised of two smaller halos (1 = 2 + 3), and at z_prev we have 2 and 3 separate, that indicates
# that the merger has occurred between z_this and z_prev; hence we always want to take the (no, ceil and)
# floor of abacus redshifts to make sure we are thorough (this is not what I'm currently doing, but
# also I worry about being too lenient)

# fields to extract from the merger trees
fields_mt = ['HaloIndex', 'HaloMass', 'Position', 'MainProgenitor', 'IsPotentialSplit', 'Progenitors', 'NumProgenitors']

# starting redshift (this) and its progenitor (previous)
z_this = zs_mt[ind_start]
z_prev = zs_mt[ind_start+1]

# merger tree files
fns_this = list(sorted(merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf'), key=extract_superslab))
fns_prev = list(sorted(merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf'), key=extract_superslab))
for counter in range(len(fns_this)):
    fns_this[counter] = str(fns_this[counter])
    fns_prev[counter] = str(fns_prev[counter])

# get number of halos in each slab and the index of the slab
N_halos_slabs_this, slabs_this = get_halos_per_slab(fns_this, minified=False)
N_halos_slabs_prev, slabs_prev = get_halos_per_slab(fns_prev, minified=False)

# load tree data at this and previous redshift
mt_data_this = get_mt_info(fns_this, fields=fields_mt, minified=False)
mt_data_prev = get_mt_info(fns_prev, fields=fields_mt, minified=False)
Merger_this = mt_data_this['merger']
Merger_prev = mt_data_prev['merger']

# boolean array indicating whether merger information available for each halo
info_this = Merger_this['MainProgenitor'] > 0

# healthy halos (not potential splits)
notsplit_this = Merger_this['IsPotentialSplit'] == 0

# select Milky Way sized halos (by total mass)
mass_selection = (Merger_this['HaloMass'] > mass_fin_lb) & (Merger_this['HaloMass'] <= mass_fin_hb)

import time
t1 = time.time()
# halos to build main progenitor
Merger_this['MainProgenitor'] = correct_inds(Merger_this['MainProgenitor'], N_halos_slabs_prev, slabs_prev, inds_fn_prev)
print("time = ", time.time()-t1)

# select MW-sized halos with a single Andromeda-like halo in the vicinity
initial_selection = mass_selection & info_this & notsplit_this
position = Merger_this['Position'][initial_selection]+Lbox/2.
main_progs = Merger_this['MainProgenitor'][initial_selection]
N_mw = np.sum(initial_selection)
print("number of Milky Way sized halos with merger info", np.sum(initial_selection)) # teeny fraction don't have it


import time
initial_selection = mass_selection & info_this & notsplit_this
main_progs = Merger_this['MainProgenitor'][initial_selection]
t1 = time.time()
# halos to build main progenitor
main_progs = correct_inds(main_progs, N_halos_slabs_prev, slabs_prev, inds_fn_prev)
print("time = ", time.time()-t1)

# select MW-sized halos with a single Andromeda-like halo in the vicinity
position = Merger_this['Position'][initial_selection]+Lbox/2.

N_mw = np.sum(initial_selection)
print("number of Milky Way sized halos with merger info", np.sum(initial_selection)) # teeny fraction don't have it
quit()

# initialize arrays holding information about number of merger in the three redshift ranges
n_merger_m1 = np.zeros(N_mw, dtype=int)
n_merger_m2 = np.zeros(N_mw, dtype=int)
n_merger_m3 = np.zeros(N_mw, dtype=int)
n_merger_above_m1 = np.zeros(N_mw, dtype=int)
n_merger_above_m2 = np.zeros(N_mw, dtype=int)
n_merger_above_m3 = np.zeros(N_mw, dtype=int)
missing = np.ones(N_mw, dtype=bool)
andromeda = np.ones(N_mw, dtype=bool)

# would select Andromeda-like galaxies and find their positions and then compute the distance for each milky way one maybe by building a tree  and finding how many are within spherical ball but excluding innermost (and not potential splits) # tuks
mass_selection_and = (Merger_this['HaloMass'] > mass_fut_lb) & (Merger_this['HaloMass'] <= mass_fut_hb)
andromeda_selection = mass_selection_and & info_this & notsplit_this
pos_andro = Merger_this['Position'][andromeda_selection]+Lbox/2.
pos_andro %= Lbox
print("building tree")
print(pos_andro.shape)
print(position.shape)
tree = spatial.cKDTree(pos_andro, boxsize=Lbox)
print("querying tree")
list_hb = np.array(tree.query_ball_point(position, dist_fut_hb))
list_lb = np.array(tree.query_ball_point(position, dist_fut_lb))
n_andromeda = np.zeros(position.shape[0], dtype=int)
for i in range(len(list_hb)):
    if i % 100000 == 0: print(i)
    n_andromeda[i] = len(list_hb[i])-len(list_lb[i])
print("percentage MW-sized halos with Andromeda-sized halos beside them = ", np.sum(n_andromeda == 1)*100./len(n_andromeda))

# disqualify halos that don't have exactly one Andromeda-sized neighbor TODO: make prettier
andromeda[n_andromeda != 1] = False
np.save(f"data/andromeda_{sim_name:s}.npy", andromeda)

# we check for z_this (and not prev) because they were distinct in the previous redshift
if (z_this in zs_mt[zs_m1]) or (z_this in zs_mt[zs_m2]) or (z_this in zs_mt[zs_m3]):

    # all progenitors
    npout = Merger_this['NumProgenitors'][initial_selection]
    nstart = Merger_this['StartProgenitors'][initial_selection]
    progs = mt_data_this['progenitors']['Progenitors']
    masses_prev = Merger_prev['HaloMass']
    masses_prev[Merger_prev['IsPotentialSplit'] == 1] = 0. # set masses of potential splits to zero, so they don't make cut
    
    # this is for setting up the progenitors stuff
    offsets_prev = np.zeros(len(inds_fn_prev), dtype=np.int64)
    offsets_prev[1:] = np.cumsum(N_halos_slabs_prev)[:-1]
    
    # record number of progenitors in mass range
    n_merger, n_merger_above = count_progenitors(npout, nstart, main_progs, progs, masses_prev, offsets_prev, slabs_prev, mass_ms_lb, mass_ms_hb)
    if (z_this in zs_mt[zs_m1]):
        n_merger_m1 += n_merger
        n_merger_above_m1 += n_merger_above
    elif (z_this in zs_mt[zs_m2]):
        n_merger_m2 += n_merger
        n_merger_above_m2 += n_merger_above
    elif (z_this in zs_mt[zs_m3]):
        n_merger_m3 += n_merger
        n_merger_above_m3 += n_merger_above

# loop over redshifts
for i in range(ind_start+1, ind_stop + 1):
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
    
    # we check for z_this (and not prev) because they were distinct in the previous redshift
    if (z_this in zs_mt[zs_m1]) or (z_this in zs_mt[zs_m2]) or (z_this in zs_mt[zs_m3]):
        # all progenitors
        npout = Merger_this['NumProgenitors'][main_progs]
        nstart = Merger_this['StartProgenitors'][main_progs]
        progs = mt_data_this['progenitors']['Progenitors']
        mt_data_prev = get_mt_info(fns_prev, fields=fields_mt, minified=False)
        Merger_prev = mt_data_prev['merger']
        masses_prev = Merger_prev['HaloMass']
        masses_prev[Merger_prev['IsPotentialSplit'] == 1] = 0. # set masses of potential splits to zero, , so they don't make cut
        
        # this is for setting up the progenitors stuff
        offsets_prev = np.zeros(len(inds_fn_prev), dtype=np.int64)
        offsets_prev[1:] = np.cumsum(N_halos_slabs_prev)[:-1]

        # record number of progenitors in mass range of interest (i.e. merger sizes)
        n_merger, n_merger_above = count_progenitors(npout, nstart, main_progs, progs, masses_prev, offsets_prev, slabs_prev, mass_ms_lb, mass_ms_hb)
        if (z_this in zs_mt[zs_m1]):
            n_merger_m1 += n_merger
            n_merger_above_m1 += n_merger_above
        elif (z_this in zs_mt[zs_m2]):
            n_merger_m2 += n_merger
            n_merger_above_m2 += n_merger_above
        elif (z_this in zs_mt[zs_m3]):
            n_merger_m3 += n_merger
            n_merger_above_m3 += n_merger_above

        del mt_data_prev, Merger_prev, masses_prev, progs

    # no info is denoted by 0 or -999, but -999 messes with unpacking, so we set it to 0
    Merger_this['MainProgenitor'][Merger_this['MainProgenitor'] < 0] = 0
    
    # rework the main progenitor and halo indices to return in proper order
    Merger_this['MainProgenitor'] = correct_inds(Merger_this['MainProgenitor'], N_halos_slabs_prev, slabs_prev, inds_fn_prev)
            
    # update the main progenitor indices for the next
    main_progs = Merger_this['MainProgenitor'][main_progs]
    print("halos with no information", np.sum(main_progs <= 0)) # there should be no negative ones
    missing[main_progs <= 0] = False

    del mt_data_this, Merger_this
    gc.collect()

# save number of mergers per redshift range
np.save(f"data/n_merger_m1_{sim_name:s}.npy", n_merger_m1)
np.save(f"data/n_merger_m2_{sim_name:s}.npy", n_merger_m2)
np.save(f"data/n_merger_m3_{sim_name:s}.npy", n_merger_m3)
np.save(f"data/n_merger_above_m1_{sim_name:s}.npy", n_merger_above_m1)
np.save(f"data/n_merger_above_m2_{sim_name:s}.npy", n_merger_above_m2)
np.save(f"data/n_merger_above_m3_{sim_name:s}.npy", n_merger_above_m3)
np.save(f"data/missing_{sim_name:s}.npy", missing)
