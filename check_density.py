# 3 questions: redshift ranges and main_progs = 0 (screws up the first halo) and unphysical halo
import glob
import os
from pathlib import Path

import asdf
import numpy as np
import matplotlib.pyplot as plt

# sample directories
sim_name = "AbacusSummit_highbase_c000_ph100"; Lbox = 1000.; N_dim = 1024; R = 1.; n_chunks = 16 # Mpc/h
#sim_name = "AbacusSummit_base_c000_ph002"; Lbox = 2000.; N_dim = 2048; R = 1.; n_chunks = 34 # Mpc/h

# load environment
z_this = 0.1
env = np.load(f'/global/cscratch1/sd/boryanah/MW_AbacusSummit/environment_R{R:.1f}_{sim_name:s}_z{z_this:.1f}.npy')
inds_mw = np.load(f"data/inds_mw_{sim_name:s}.npy")
env = env[inds_mw]

# load merger and andromeda information
n_merger_m1 = np.load(f"data/n_merger_m1_{sim_name:s}.npy")
n_merger_m2 = np.load(f"data/n_merger_m2_{sim_name:s}.npy")
n_merger_m3 = np.load(f"data/n_merger_m3_{sim_name:s}.npy")
n_merger_above_m1 = np.load(f"data/n_merger_above_m1_{sim_name:s}.npy")
n_merger_above_m2 = np.load(f"data/n_merger_above_m2_{sim_name:s}.npy")
n_merger_above_m3 = np.load(f"data/n_merger_above_m3_{sim_name:s}.npy")
missing = np.load(f"data/missing_{sim_name:s}.npy")
andromeda = np.load(f"data/andromeda_{sim_name:s}.npy")

int_selection = (n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & (n_merger_above_m1 == 0) & (n_merger_above_m2 == 0) & (n_merger_above_m3 == 0) & andromeda


assert len(inds_mw) == len(int_selection), f"one is {len(inds_mw):d} and the other {len(int_selection):d}"

print("--------------------------------------")
print("number of all halos = ", len(inds_mw))
print("number of all halos of interest = ", np.sum(int_selection))

print("mean env of all halos = ", np.mean(env))
print("mean env of all halos of interest = ", np.mean(env[int_selection]))

print("std env of all halos = ", np.std(env))
print("std env of all halos of interest = ", np.std(env[int_selection]))
