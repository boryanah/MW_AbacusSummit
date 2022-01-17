import numpy as np
import sys

#sim_name = "AbacusSummit_highbase_c000_ph100"
#sim_name = "AbacusSummit_base_c000_ph002"
sim_name = sys.argv[1]

n_merger_m1 = np.load(f"data/n_merger_m1_{sim_name:s}.npy")[1:]
n_merger_m2 = np.load(f"data/n_merger_m2_{sim_name:s}.npy")[1:]
n_merger_m3 = np.load(f"data/n_merger_m3_{sim_name:s}.npy")[1:]
n_merger_above_m1 = np.load(f"data/n_merger_above_m1_{sim_name:s}.npy")[1:]
n_merger_above_m2 = np.load(f"data/n_merger_above_m2_{sim_name:s}.npy")[1:]
n_merger_above_m3 = np.load(f"data/n_merger_above_m3_{sim_name:s}.npy")[1:]
missing = np.load(f"data/missing_{sim_name:s}.npy")[1:]
andromeda = np.load(f"data/andromeda_{sim_name:s}.npy")[1:]

#n_merger_above_m1 = n_merger_above_m3
#n_merger_above_m2 = n_merger_above_m3
#n_merger_above_m3 = n_merger_above_m1

print("n_merger above m1 == 0 = ", np.sum(n_merger_above_m1 == 1))
print("n_merger above m2 == 0 = ", np.sum(n_merger_above_m2 == 1))
print("n_merger above m3 == 0 = ", np.sum(n_merger_above_m3 == 1))
print("n_merger m1 == 1 = ", np.sum(n_merger_m1 == 1))
print("n_merger m2 == 1 = ", np.sum(n_merger_m2 == 1))
print("n_merger m3 == 1 = ", np.sum(n_merger_m3 == 1))

print("all are one = ", np.sum((n_merger_m1+n_merger_above_m1 == 1) & (n_merger_m2+n_merger_above_m2 == 1) & (n_merger_m3+n_merger_above_m3 == 1)))

print("no above = ", np.sum((n_merger_above_m1 == 0) & (n_merger_above_m2 == 0) & (n_merger_above_m3 == 0)))
print("all are one = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1)))
print("all are one and andromeda = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & andromeda))
print("all are one and andromeda and have info = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & missing & andromeda))
print("all are one and no above = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & (n_merger_above_m1 == 0) & (n_merger_above_m2 == 0) & (n_merger_above_m3 == 0)))
print("all are one and no above and andromeda = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & (n_merger_above_m1 == 0) & (n_merger_above_m2 == 0) & (n_merger_above_m3 == 0) & andromeda))
print("all are one and no above and andromeda and have info = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1) & (n_merger_above_m1 == 0) & (n_merger_above_m2 == 0) & (n_merger_above_m3 == 0) & missing & andromeda))
print("total number = ", len(n_merger_m1))

