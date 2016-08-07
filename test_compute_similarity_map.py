import cutils
import numpy
import multiprocessing
import json
from scipy import optimize

with open('config.json') as config_file:
    config = json.load(config_file)

subject = config["subject"]
session = config["session"]
scans = config["scans"]
hemisphere = config["hemisphere"]
atlas_name = config["atlas_name"]
roi_name = config["roi_name"]

out_path = config["out_path"] + \
           '/rfMRI_{0}_{1}_{2}'.format(subject, session, hemisphere)

fingerprints = numpy.load(out_path + "/fingerprints.npy")

num_fingerprints = fingerprints.shape[0]
num_processes = multiprocessing.cpu_count()

# Distribute load equally among all CPUs
pool = multiprocessing.Pool(num_processes)

# Approximation of the optimal fraction of the dataset to
# allocate to each CPU
processes_idx = numpy.arange(num_processes + 1)
fingerprints_ratio = processes_idx * 2 / (num_processes * (num_processes + 1))

ratio_loss = lambda vars: sum([(vars[i-1]*sum(vars[i-1:]) - vars[i]*sum(vars[i:]))**2 for i in range(1, len(vars))]) + (sum(vars) - 1)**2

res = optimize.minimize(ratio_loss, fingerprints_ratio[1:], method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

if all(res.x > 0) and all(res.x < 1):
    fingerprints_ratio[1:] = res.x

fingerprints_idx = (numpy.cumsum(fingerprints_ratio) * num_fingerprints).astype(int)

pool_idx = [numpy.arange(fingerprints_idx[i], fingerprints_idx[i+1])
            for i in range(len(fingerprints_idx)-1)]

starmap_input = [(fingerprints, idx) for idx in pool_idx]

# Run compute_similarity_map in parallel
eta2_chunks = pool.starmap(cutils.compute_similarity_map, starmap_input)

# Merge together results from all processes
eta2_coef = numpy.zeros((num_fingerprints, num_fingerprints))

for i_cpu in range(num_processes):
    for i_chunk in range(len(pool_idx[i_cpu])):

        i_eta = pool_idx[i_cpu][i_chunk]

        eta2_row = eta2_chunks[i_cpu][i_chunk, i_eta:]
        eta2_coef[i_eta, i_eta:] = eta2_row
        eta2_coef[i_eta:, i_eta] = eta2_row

numpy.save(out_path + "/eta2_coeff.npy", eta2_coef)
