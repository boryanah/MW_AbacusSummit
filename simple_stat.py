import numpy as np

n_merger_m1 = np.load("data/n_merger_m1.npy")[1:]
n_merger_m2 = np.load("data/n_merger_m2.npy")[1:]
n_merger_m3 = np.load("data/n_merger_m3.npy")[1:]

print("n_merger m1 = 1", np.sum(n_merger_m1 == 1))
print("n_merger m2 = 1", np.sum(n_merger_m2 == 1))
print("n_merger m3 = 1", np.sum(n_merger_m3 == 1))
print("n_merger m1 > 0", np.sum(n_merger_m1 >= 1))
print("n_merger m2 > 0", np.sum(n_merger_m2 >= 1))
print("n_merger m3 > 0", np.sum(n_merger_m3 >= 1))
print("all are one = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1)))
print("total number = ", len(n_merger_m1))
print("percentage exactly one = ", np.sum((n_merger_m1 == 1) & (n_merger_m2 == 1) & (n_merger_m3 == 1))*100./len(n_merger_m1))

