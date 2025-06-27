import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook as cbook

import os

import sys
sys.path.append('../sim/')

import pandas as pd

from simfuncs import *

from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
from kde_ebm import plotting

import warnings 
# warnings.filterwarnings("ignore",category=cbook.mplDeprecation)

from pySuStaIn.MixtureSustain import MixtureSustain
from pySuStaIn.MixtureSustain import MixtureSustainData

import sklearn.model_selection
import time

import pylab

def run_sustain_alg(sustain_obj, mixture_obj, sequence_init, f_init, N_S_max, label=""):
    sequence_prev = sequence_init
    f_prev = f_init
    results = [{} for _ in range(N_S_max)]

    start_time = time.time()
    for s in range(N_S_max):
        sequence, f, likelihood, sequence_mat, f_mat, likelihood_mat = sustain_obj._estimate_ml_sustain_model_nplus1_clusters(
            mixture_obj, sequence_prev, f_prev)

        sequence_prev = sequence
        f_prev = f

        results[s].update({
            "S": s,
            "sequence": sequence,
            "f": f,
            "likelihood": likelihood,
            "sequence_mat": sequence_mat,
            "f_mat": f_mat,
            "likelihood_mat": likelihood_mat,
        })
    end_time = time.time()

    print(f'{label} Time: {end_time - start_time:.3f} seconds\n')
    return results, end_time - start_time


def main():
    N = 50
    M = 500
    N_S_ground_truth = 3
    ground_truth_fractions = np.array([0.5, 0.3, 0.2])
    BiomarkerNames = ['Biomarker ' + str(i) for i in range(N)]
    use_parallel_startpoints = False
    N_startpoints = 1
    N_S_max = 3
    N_iterations_MCMC = 1
    SuStaInLabels = BiomarkerNames
    validate = False
    sustainType = 'mixture_GMM'

    dataset_name = 'sim'
    output_folder = os.path.join(os.getcwd(), dataset_name + '_' + sustainType)
    os.makedirs(output_folder, exist_ok=True)

    dataset_name_dp = 'sim_dp'
    output_folder_dp = os.path.join(os.getcwd(), dataset_name_dp + '_' + sustainType)
    os.makedirs(output_folder_dp, exist_ok=True)

    for exp in range(5):
        ground_truth_subj_ids = list(np.arange(1, M+1).astype('str'))
        ground_truth_sequences = generate_random_mixture_sustain_model(N, N_S_ground_truth)
        ground_truth_subtypes = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)

        N_stages = N
        ground_truth_stages_control = np.zeros((int(np.round(M * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(M * 0.75)), 1))
        ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

        data, data_denoised = generate_data_mixture_sustain(
            ground_truth_subtypes, ground_truth_stages, ground_truth_sequences, sustainType)

        MIN_CASE_STAGE = np.round((N + 1) * 0.8)
        index_case = np.where(ground_truth_stages >= MIN_CASE_STAGE)[0]
        index_control = np.where(ground_truth_stages == 0)[0]

        labels = 2 * np.ones(data.shape[0], dtype=int)
        labels[index_case] = 1
        labels[index_control] = 0

        data_case_control = data[labels != 2, :]
        labels_case_control = labels[labels != 2]

        if sustainType == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels)

        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)

        for i in range(N):
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])

        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max,
                                N_iterations_MCMC, output_folder, dataset_name,
                                use_parallel_startpoints, use_dp=False, seed=1)

        sustain_dp = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max,
                                    N_iterations_MCMC, output_folder_dp, dataset_name_dp,
                                    use_parallel_startpoints, use_dp=True, seed=1)

        sustain_mixture_object = MixtureSustainData(L_yes, L_no, N_stages)

        # --- Run Regular and DP Models ---
        results_reg, time_reg = run_sustain_alg(
            sustain, sustain_mixture_object, [], [], N_S_max, label="Regular")

        results_dp, time_dp = run_sustain_alg(
            sustain_dp, sustain_mixture_object, [], [], N_S_max, label="DP")

        with open(f"sustain_reg{exp+1}.txt", "w") as f:
            for s in range(N_S_max):
                f.write(f"--- REG S = {s} ---\n")
                f.write(f"ml_sequence_EM_reg: {results_reg[s]['sequence']}\n")
                f.write(f"ml_f_EM_reg: {results_reg[s]['f']}\n")
                f.write(f"ml_likelihood_EM_reg: {results_reg[s]['likelihood']}\n")
                f.write(f"ml_sequence_mat_EM_reg: {results_reg[s]['sequence_mat']}\n")
                f.write(f"ml_f_mat_EM_reg: {results_reg[s]['f_mat']}\n")
                f.write(f"ml_likelihood_mat_EM_reg: {results_reg[s]['likelihood_mat']}\n\n")

        with open(f"sustain_dp{exp+1}.txt", "w") as f:
            for s in range(N_S_max):
                f.write(f"--- DP S = {s} ---\n")
                f.write(f"ml_sequence_EM_dp: {results_dp[s]['sequence']}\n")
                f.write(f"ml_f_EM_dp: {results_dp[s]['f']}\n")
                f.write(f"ml_likelihood_EM_dp: {results_dp[s]['likelihood']}\n")
                f.write(f"ml_sequence_mat_EM_dp: {results_dp[s]['sequence_mat']}\n")
                f.write(f"ml_f_mat_EM_dp: {results_dp[s]['f_mat']}\n")
                f.write(f"ml_likelihood_mat_EM_dp: {results_dp[s]['likelihood_mat']}\n\n")



        with open(f"compare{exp+1}.txt", "w") as f:
            f.write("Are all results the same?\n")
            keys = ['sequence', 'f', 'likelihood', 'sequence_mat', 'f_mat', 'likelihood_mat']

            for s in range(N_S_max):
                f.write(f'----- s = {s} -----\n')
                for key in keys:
                    result = np.allclose(results_reg[s][key], results_dp[s][key])
                    f.write(f"{key}: {result}\n")

            f.write(f"\ntime_reg: {time_reg:.3f}\ntime_dp: {time_dp:.3f}")
            

if __name__ == '__main__':
    main()
