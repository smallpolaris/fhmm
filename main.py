from __future__ import division
from math import pi
from torch.distributions import Normal, Beta, Gamma, MultivariateNormal
from torch.distributions.uniform import Uniform
# from itertools import count
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import numpy as np
import torch
import time
import numpy.random as nrd
import tool
import update
import multiprocessing


def do(o):
    torch.set_default_tensor_type(torch.DoubleTensor)
    name = multiprocessing.current_process().name
    print("name : %s staring, n:%d"% (name, o),  flush=True, file=open("sur" + str(o) + ".txt", "a"))
    N = 500
    NT = 4
    NX = 3
    NV = NX + 1
    # NV = NX + 1   # varying coefficient (include intercept)
    NK = 6  # number of knots
    NS = 2
    K = 2  # order (more explicit, degree (cubic)  (so number of spline is NK+K)
    D = 3  # degree = K + 1
    NH = 3      # transition model
    NB = NK + K - 1        # number of b-spline
    Iter = 6000
    burnin = 2000
    NG = 100               # number of grid
    Rep_s = 1 * o
    Rep_e = 1 * (o + 1)
    Rep = Rep_e - Rep_s
    G = 300
    #  -------------------------For saving parameters------------------------------
    #-----------------------------------Conditional model-----------------------
    all_beta = torch.zeros(Rep, Iter, NS, NV, NB)
    all_ptau = torch.zeros(Rep, Iter, NS, NV)  # penalty parameter for bspline
    all_tau_beta = torch.zeros(Rep, Iter, NS, NX)
    all_gamma_beta = torch.zeros(Rep, Iter, NS, NX)
    # all_sigma_beta = torch.zeros(Rep, Iter, NS)
    # ---------------------transition model------------------------
    all_tau = torch.zeros(Rep, Iter, NS - 1)
    all_phi = torch.zeros(Rep, Iter, NH)
    all_tau_phi = torch.zeros(Rep, Iter, NH)
    all_gamma_phi = torch.zeros(Rep, Iter, NH)
    # all_sigma_phi = torch.zeros(Rep, Iter, NH)
    all_zeta = torch.zeros(Rep, Iter, NS, NS)
    all_state = torch.zeros(Rep, Iter, N, NT)
    all_sigma = torch.zeros(Rep, Iter, NS)
    # --------------------About the process of random effect-----------------------
    all_P = torch.zeros(Rep, Iter, G)
    all_Z = torch.zeros(Rep, Iter, G, NG)
    all_nu = torch.zeros(Rep, Iter)
    all_psi = torch.zeros(Rep, Iter)
    all_nalpha = torch.zeros(Rep, Iter)
    all_L = torch.zeros(Rep, Iter, N, NT)
    all_thres = torch.zeros(Rep, Iter, NS, NV)
    all_accept_thres = torch.zeros(Rep, Iter, NS, NV)
    for rep in range(Rep_s, Rep_e):
        r = rep - Rep_s
        nrd.seed(77+3)
        #------------------------------------Load data----------------------------------------
        folder = ""
        x = torch.load(folder + "x.pt")[rep]
        y = torch.load(folder + "y.pt")[rep]
        t_state = torch.load(folder + "t_state.pt")[rep]
        t_state = t_state.long()
        h = torch.load(folder + "h.pt")[rep]
        ######--------------True value about the model--------------------
        # -------funtional part--------------
        grid = torch.linspace(0, 1, NG)
        t_beta = torch.zeros(NS, NX + 1, NG)  # -----------------zxx
        t_beta[0, 0] = grid
        # t_beta[0, 1] = torch.pow(grid - 0.3, 2) * (grid > 0.3)
        t_beta[0, 1] = 1 * (torch.cos(2.5 * pi * grid) + 1) * (grid > 0.4)
        t_beta[0, 2] = torch.sin(2 * pi * grid) + 1
        t_beta[1, 0] = 1 - grid
        t_beta[1, 1] = torch.cos(2 * pi * grid)
        t_beta[1, 2] = torch.sin(pi * (grid - 0.5)) * (grid <= 0.5)
        # -----------------parametric part---------------
        t_tau = torch.Tensor([0])
        t_sigma = torch.Tensor([0.25, 0.25])
        t_zeta = torch.Tensor([[-1, 0], [1, 0]])
        t_phi = torch.Tensor([1, -1, 0])
        ##--------------------------------------Initialization----------------------------------------------
        knots = np.zeros(shape=[NK + 2 * K])
        # knots[K:(NK + K)] = np.percentile(t, np.linspace(0, 100, NK + 1, endpoint=False)[1:])
        knots[K:(NK + K)] = np.linspace(0, 1, NK, endpoint=True)
        knots[:K] = 0
        knots[NK + K:] = 1
        #----------------------------Conditional model---------------------------------------
        grid = torch.linspace(0, 1, NG)
        dd = torch.unsqueeze(grid, 1) - torch.unsqueeze(grid, 0)  # NG * NG
        H_d = torch.exp(-torch.abs(dd))  # distance for the covariance
        # --------------------------------------- Initialize the parameters----------------------------------------------------
        # ---------------------------------Calculate the penalty matrix------------------------------------------#
        tool1 = tool.Tools(N=N, NT=NT, NS=NS, K=K, NB=NB, knots=knots)
        D = tool1.banded([1, -2, 1], NB - 2)
        PK = torch.from_numpy(np.matmul(np.transpose(D), D))
        # thres = torch.Tensor([[0.25, 0.5, 0.05, 0.05],
        #                       [0.5, 0.5, 0.05, 0.05]])  # soft threshold parameter  ( first for intercept, must be 0)
        thres = torch.Tensor(nrd.uniform(0, 0.15, (NS, NV)))
        thres[:, 0] = 0
        t_thres = torch.zeros(NS, NV)
        # c_thres = torch.Tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        c_thres = torch.ones((NS, NX)) * 0.05
        accept_thres = torch.zeros(NS, NV)
        beta = torch.randn(NS, NV, NB)  # coefficient of b-spline
        accept_beta = torch.zeros(NS, NV)
        c_beta = torch.Tensor([[0.005, 0.005, 0.005, 0.005], [0.005, 0.005, 0.005, 0.005]])
        # c_beta = torch.ones((NS, NV)) * 0.3
        tau_beta = torch.rand(NS, NX) * 10
        gamma_beta = torch.rand(NS, NX) * 10
        sigma_beta = torch.rand(NS) * 10
        ptau = torch.rand(NS, NV)  # penalty coefficient for Bspline
        sigma = torch.rand(NS)
        # # ------------------- hidden Markov model--------------------
        # tau
        tau = torch.rand(NS - 1)
        c_tau = 2
        accept_tau = torch.zeros(NS - 1)
        # phi
        phi = torch.randn(h.shape[2])
        c_phi = 3
        accept_phi = 0
        tau_phi = torch.Tensor(nrd.uniform(0, 10, NH))
        gamma_phi = torch.Tensor(nrd.uniform(0, 10, NH))
        sigma_phi = torch.Tensor([nrd.uniform(0, 10)])
        # zeta
        zeta = torch.randn(NS, NS)
        zeta[:, -1] = 0
        c_zeta = 6
        accept_zeta = torch.zeros(NS, NS)
        # state
        state = torch.multinomial(input=torch.rand(NS), num_samples=N * NT, replacement=True).reshape(N, NT)
        # ------------------------------Basis value(set) at grid---------------------------
        bs_value = tool1.bs_value(
            grid)  # NB * NG (NB spline at each grid)   #   Here spending time may be long; NG is large  (Assemble H^T in paper)
        bs_value = torch.from_numpy(bs_value)
        # PG = tool1.large_square(bs_value.t(), l=10) / NG  # NB * NB
        PG = tool1.large_square(bs_value.t(), l=10) # NB * NB
        IPG = torch.inverse(PG)
        data = update.MCMC(N=N, NT=NT, NS=NS, NV=NV, K=K, NG=NG, NB=NB, knots=knots, PK=PK, PG=PG, IPG=IPG, G=G,
                           H_d=H_d)
        # -----------------------------DPM  setting--------------------------------------
        Z = MultivariateNormal(torch.zeros(NG), torch.eye(NG)).sample((G,))  # G * NG
        c_z = 0.002
        accept_Z = torch.zeros(G)
        L = torch.multinomial(torch.rand(N, G), NT, replacement=True)
        dpm_a = torch.rand(1)
        V = torch.ones(G)
        V[:-1] = torch.squeeze(Beta(1, dpm_a).sample([G - 1]), 1)
        # mu_z = Normal(0, 10).sample((NG,))
        # sigma_z = 1 / Gamma(1, 0.01).sample((NG,))
        nu = torch.rand(1)
        psi = torch.rand(1) * 10
        c_psi = 1
        accept_psi = 0
        P = torch.zeros(G)
        P[1:] = torch.cumprod(1 - V, 0)[:-1] * V[1:]  # It is pi in the paper
        P[0] = V[0]
        non_alpha = Gamma(3, 5).sample()
        for iter in range(Iter):
            t0 = time.time()
            # # # #----------------------- hidden Markov model--------------------------------------
            tran = tool1.tran(zeta, phi, h)
            p0 = tool1.p0(tau)
            tran_p = tool1.tran_p(tran)
            tau, accept_tau = data.update_tau(tau, state, c_tau, accept_tau)
            all_tau[r, iter, :] = tau
            phi, accept_phi = data.update_phi(h, phi, zeta, state, c_phi, accept_phi, tau_phi)
            all_phi[r, iter, :] = phi
            tau_phi = data.update_tau_phi(gamma_phi, phi)
            all_tau_phi[r, iter, :] = tau_phi
            gamma_phi = data.update_gamma_phi(tau_phi)
            all_gamma_phi[r, iter, :] = gamma_phi
            zeta, accept_zeta = data.update_zeta(zeta, state, phi, h, c_zeta, accept_zeta)
            all_zeta[r, iter, :] = zeta
            eta = Z[L]
            y_likeli = data.y_likeli(x, y, beta, bs_value, eta, sigma, thres)
            state = data.update_state(p0, tran_p, y_likeli)
            # # # # --------------------update varying coefficient----------------------------------------------------------------------
            beta, accept_beta = data.update_vc(y, x, state, beta, sigma, thres, bs_value, eta, c_beta,
                                               accept_beta, tau_beta)  # 0.06
            thres, accept_thres = data.update_thres(y, x, state, beta, sigma, thres, bs_value, eta, c_thres,
                                                    accept_thres)  # 0.04
            tau_beta = data.update_tau_beta(gamma_beta, sigma, beta)
            gamma_beta = data.update_gamma_beta(tau_beta)
            mean = data.update_mean(x, state, beta, thres, bs_value)
            sigma = data.update_sigma(y, mean, eta, state, beta, ptau)
            # # #--------------------------------------Update random effect--------------------------------------
            P, Z, V, accept_Z = data.update_piz(y, mean, state, sigma, non_alpha, Z, L, nu, psi, c_z,
                                                accept_Z)
            all_P[r, iter] = P
            all_Z[r, iter] = Z
            L = data.update_L(P, Z, y, mean, state, sigma)  # 0.12s  (0.4s when G = 300)
            all_L[r, iter] = L
            nu = data.update_nu(Z, psi)
            all_nu[r, iter] = nu
            psi, accept_psi = data.update_psi(Z, psi, nu, c_psi, accept_psi)
            all_psi[r, iter] = psi
            non_alpha = data.update_nonalpha(V)
            all_nalpha[r, iter] = non_alpha
            # # #-----------------------------------------Label switch---------------------------------------------
            state, sigma, beta, tau_beta, gamma_beta, sigma_beta, thres = data.label_switch(state, sigma, beta, tau_beta,
                                                                                            gamma_beta, sigma_beta, thres,
                                                                                            bs_value)
            all_state[r, iter] = state
            all_sigma[r, iter] = sigma
            all_beta[r, iter] = beta
            all_ptau[r, iter] = ptau
            all_tau_beta[r, iter, :] = tau_beta
            all_gamma_beta[r, iter, :] = gamma_beta
            all_thres[r, iter] = thres
            all_accept_thres[r, iter] = accept_thres
            acc = (torch.sum(state == t_state)) / (N * NT)
            if iter > 40 and iter % 50 == 0:
                process = (r * Iter + iter) / (Rep * Iter)
                t1 = time.time()
                one_iter_time = t1 - t0
                print("%.3f seconds process time for one iter" % one_iter_time)
                rtime = Rep * Iter * one_iter_time * (1 - process) / 60
                print("Rep :%d, Iter : %d, acc: %.3f, process: %.5f, need %.1f min to complete" % (
                    r, iter, acc, process, rtime))
                print("Rep :%d, Iter : %d, acc: %.3f, process: %.3f, need %.1f min to complete" % (
                    r, iter, acc, process, rtime),
                      flush=True, file=open("sur" + str(o) + ".txt", "a"))
                # print(accept_thres, flush=True, file=open("sur" + str(o) + ".txt", "a"))
                # print(accept_thres)
        print("end of one")
    # #--------------------- parametric part-------------------# #
    e_tau = torch.mean(all_tau[:, burnin:], (0, 1))
    e_phi = torch.mean(all_phi[:, burnin:], (0, 1))
    e_zeta = torch.mean(all_zeta[:, burnin:], (0, 1))
    e_sigma = torch.mean(all_sigma[:, burnin:], (0, 1))
    b_tau = e_tau - e_tau
    rmse_tau = tool1.hrep_rmse(all_tau, t_tau, burnin)
    print("Bias for tau is")
    print(b_tau)
    print("RMSE for tau is")
    print(rmse_tau)
    b_phi = e_phi - t_phi
    rmse_phi = tool1.hrep_rmse(all_phi, t_phi, burnin)
    print("Bias for alpha is")
    print(b_phi)
    print("RMSE for alpha is")
    print(rmse_phi)
    b_zeta = e_zeta - t_zeta
    rmse_zeta = tool1.hrep_rmse(all_zeta, t_zeta, burnin)
    print("Bias for zeta is")
    print(b_zeta)
    print("RMSE for zeta is")
    print(rmse_zeta)
    b_sigma = e_sigma - t_sigma
    rmse_sigma = tool1.hrep_rmse(all_sigma, t_sigma, burnin)
    print(b_sigma)
    print(rmse_sigma)
    # #---------------------functional part------------------------# #
    e_beta = torch.mean(all_beta[:, burnin:], (0, 1))
    e_thres = torch.mean(all_thres[:, burnin:], (0, 1))
    g = 0
    sim_betam = torch.zeros(NS, NV, NG)
    soft_betam = torch.zeros(NS, NV, NG)
    plt.figure(figsize=(12, 12))
    for s in range(NS):
        for v in range(1, NV):
            sp_b = BSpline(knots, e_beta[s, v], K)
            sim_betam[s, v] = torch.Tensor(sp_b(grid))
            soft_betam[s, v] = tool1.soft_thres(sim_betam[s, v], e_thres[s, v])
            g += 1
            fig = plt.subplot(2, 3, g)
            # plt.plot(grid, sim_betam[s, v], lw=3, color='r', label='simulated spline')
            plt.plot(grid, soft_betam[s, v], lw=1, color='b', label='simulated soft')
            plt.plot(grid, t_beta[s, v], lw=1, color='k', label='True')
    # plt.savefig("soft.pdf")
    print("Estimator of varying coefficient in two states")
    plt.show()
    print("One simulation is done")



if __name__ == '__main__':
    numList = []
    for o in range(0, 1):
        p = multiprocessing.Process(target=do, args=(o,))
        numList.append(p)
        p.start()
        p.join()
