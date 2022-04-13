from __future__ import division
import numpy as np
import math
import tool
import time
from numpy.linalg import cholesky
from torch.distributions import Normal, MultivariateNormal, Gamma, Beta, Multinomial
from torch.distributions.uniform import Uniform
from scipy.stats import invgauss
from scipy.interpolate import BSpline
# from mpmath import *
import scipy.stats as ss
import time
import torch
import numpy.random as nrd
# import mpmath as mpm


class MCMC(tool.Tools):

    def __init__(self, N, NT, NS, NV, K, NG, NB, knots, PK, PG, IPG, G, H_d):
        self.N = N
        self.NT = NT
        self.NS = NS
        self.NV = NV
        self.K = K
        self.NG = NG
        self.NB = NB
        self.knots = knots
        self.PK = PK
        self.PG = PG
        self.IPG = IPG
        self.G = G
        self.H_d = H_d


    def update_state(self, p0, p_tran, y_likeli):
        # Args: p0: [NS-1]; tran_p: N * NT * NS * NS; y_likeli: N * NT * NG * NS  log_prob)
        #-----------------------This is truncation version----------------------------------------------#
        state = torch.ones(y_likeli.shape[0], y_likeli.shape[1]).long() * self.NS
        #-------------------------------- First step: Deal withe y_likeli so that the value can be not too small--------------
        y_likeli = torch.sum(y_likeli, dim=2)  # N * NT * NS
        # find the smallest integer to make y_likeli is between [0,1]
        M = torch.floor(y_likeli)    # (must be negative) N * NT * NS  different M for each state  downward direction for M
        M_min = torch.min(M, 2)[0]  # torch.max returns values and index, just need value.  N * NT
        my_likeli = y_likeli - torch.unsqueeze(M_min, 2)   # the small one must be [0, 1]
        ay_likeli = torch.where(my_likeli < 60.0, my_likeli, 60.0)      # adjusted y_likeli ; truncated too large y_likeli to 10.
        y_likeli = torch.exp(ay_likeli)  # N * NT * NS
        q_forward = torch.zeros(y_likeli.shape[0], y_likeli.shape[1], self.NS)
        q_backward = torch.zeros(y_likeli.shape[0], y_likeli.shape[1], self.NS)
        q_forward[:, 0, :] = y_likeli[:, 0, :] * p0
        q_backward[:, -1, :] = 1
        for t in range(1, self.NT):
            for s1 in range(0, self.NS):
                for s2 in range(self.NS):  # s2 is for sum
                    q_forward[:, t, s1] += q_forward[:, t - 1, s2] * p_tran[:, t, s2, s1] * y_likeli[:, t, s1]
        for t in range(self.NT-2, -1, -1):
            for s1 in range(0, self.NS):
                for s2 in range(0, self.NS):  # s2 is for sum
                    q_backward[:, t, s1] += q_backward[:, t+1, s2] * p_tran[:, t+1, s1, s2] * y_likeli[:, t+1, s2]
        prob = q_forward * q_backward  # N * NT * NS
        # prob_norm = prob / torch.unsqueeze(torch.sum(prob, dim=2), 2)
        for t in range(self.NT):
            state[:, t] = torch.squeeze(torch.multinomial(prob[:, t], 1, replacement=True), 1)
        return state


    def update_state_acc(self, p0, p_tran, y_likeli):
        # Args: p0: [NS-1]; tran_p: N * NT * NS * NS; y_likeli: N * NT * NG * NS  log_prob)
        #---------------------------This is for accuracy version (mpm)--------------------------------
        state = torch.ones(y_likeli.shape[0], y_likeli.shape[1]) * self.NS
        #-------------------------------- To numpy--------------
        y_likeli = torch.sum(y_likeli, dim=2)  # N * NT * NS
        p_tranarr = p_tran.numpy()
        p0_arr = p0.numpy()
        p0_mpm = mpm.matrix(p0_arr)
        y_likeliarr = y_likeli.numpy()
        # exp_ylikeli = np.zeros_like(y_likeliarr)
        # for t in range(self.NT):
        #     for s in range(self.NS):
        #         exp_ylikeli[:, t, s] = np.array(mpm.matrix(y_likeliarr[:, t, s]).apply(mpm.exp))
        # y_likeli = torch.exp(ay_likeli)  # N * NT * NS
        # q_forward = np.zeros(shape=[y_likeli.shape[0], y_likeli.shape[1], self.NS])
        # q_backward = np.zeros(shape=[y_likeli.shape[0], y_likeli.shape[1], self.NS])
        # q_forward[:, 0, :] = exp_ylikeli[:, 0, :] * p0_arr
        # q_backward[:, -1, :] = 1
        sample_prob = np.zeros(shape=[self.N, self.NT, self.NS])
        for i in range(self.N):
            q_forward = mpm.matrix(self.NT, self.NS)
            for s in range(self.NS):
                q_forward[0, s] = mpm.exp(y_likeliarr[i, 0, s] * p0_arr[s])
            for t in range(1, self.NT):
                for s1 in range(0, self.NS):
                    for s2 in range(self.NS):  # s2 is for sum
                        q_forward[t, s1] += q_forward[t - 1, s2] * p_tranarr[i, t, s2, s1] * mpm.exp(y_likeliarr[i, t, s1])
        #--------------------------------------- Backward----------------------------------------
            q_backward = mpm.matrix(self.NT, self.NS)
            q_backward[self.NT-1, :] = 1
            for t in range(self.NT-2, -1, -1):
                for s1 in range(0, self.NS):
                    for s2 in range(0, self.NS):  # s2 is for sum
                        q_backward[t, s1] += q_backward[t + 1, s2] * p_tranarr[i, t+1, s1, s2] * mpm.exp(y_likeliarr[i, t+1, s2])
            prob = mpm.matrix(self.NT, self.NS)
            norm_prob = mpm.matrix(self.NT, self.NS)
            sum = mpm.matrix(self.NT, 1) # = np.sum(prob, 1)
            for t in range(self.NT):
                for s in range(self.NS):
                    prob[t, s] = q_forward[t, s] * q_backward[t, s] # N * NT * NS
                    sum[t, 0] += prob[t, s]
                for s in range(self.NS):
                    norm_prob[t, s] = prob[t, s] / sum[t, 0]   # After standardize, the loss is OK
                sample_prob[i, t, :] = norm_prob[t, :]
        sample_prob = torch.from_numpy(sample_prob)
        # prob_norm = prob / torch.unsqueeze(torch.sum(prob, dim=2), 2)
        for t in range(self.NT):
            state[:, t] = torch.squeeze(torch.multinomial(sample_prob[:, t], 1, replacement=True), 1)
        return state

    def update_phi(self, d, phi, zeta, state, c_phi, accept_phi, tau_phi):
        #--------------------------Update the transition coefficient---------------------------------------------
        # phi_out = np.zeros(shape=self.ND, dtype=float)
        # phi1 = phi[np.newaxis, :]
        # temp_phi = np.zeros(shape=[self.ND, self.ND])
        # phi1 = torch.unsqueeze(phi, 0)
        temp_phi = torch.zeros(phi.shape[0], phi.shape[0])
        #----------- Test-------------
        # temp_phi1 = np.zeros(shape=[phi.shape[0], phi.shape[0]])
        # d_arr = d.numpy()
        for i in range(self.NS):
            for j in range(self.NS - 1):
                for k in range(j + 1):  # k be able to equal to j
                    for t in range(1, state.shape[1]):
                        # for n in range(state.shape[0]):
                        #     if state[n, t] == j and state[n, t - 1] == i:
                        #         temp_phi1 = temp_phi1 + np.dot(d_arr[n, t, :][:, np.newaxis], d_arr[n, t, :][np.newaxis, :]) * \
                        #                    math.exp(zeta[i, k]) / math.pow(1 + math.exp(zeta[i, k]), 2)
                        loc1 = torch.where(state[:, t] == j)
                        loc2 = torch.where(state[loc1[0], t-1] == i)
                        loc = loc1[0][loc2[0]]
                        temp_phi += torch.sum(torch.matmul(torch.unsqueeze(d[loc, t], 2), torch.unsqueeze(d[loc, t], 1)), 0) \
                                    * torch.exp(zeta[i, k]) / torch.square(1 + torch.exp(zeta[i, k]))
            for k in range(self.NS - 1):  # j=self.NS-1 special:    k can't be equal to self.NS -1
                for t in range(1, state.shape[1]):
                    loc3 = np.where(state[:, t] == self.NS-1)
                    loc4 = np.where(state[loc3[0], t - 1] == i)
                    loc = loc3[0][loc4[0]]
                    temp_phi += torch.sum(torch.matmul(torch.unsqueeze(d[loc, t], 2), torch.unsqueeze(d[loc, t], 1)), 0)
                                # * torch.exp(zeta[i, k]) / torch.square(1 + torch.exp(zeta[i, k]))
        # sigma_phi = np.linalg.inv(np.diag(v=np.ones(shape=phi.shape[0])) / 100 + temp_phi.numpy())
        bayes_sigma_inv = np.diag(1 / tau_phi)
        #     sigma_phi = np.linalg.inv(bayes_sigma_inv + temp_phi)
        Sigma_phi = np.linalg.inv(bayes_sigma_inv + temp_phi.numpy())
        try:
            phi_star = torch.Tensor(nrd.multivariate_normal(phi, c_phi * Sigma_phi))
        except ValueError:
            phi_star = torch.Tensor(nrd.multivariate_normal(phi, c_phi * np.eye(phi.shape[0])))
        # phi_star = torch.Tensor(nrd.multivariate_normal(phi, c_phi * Sigma_phi))
        ratio_phi = 0
        t1 = time.time()
        for s1 in range(self.NS):
            for s2 in range(self.NS - 1):
                # for n in range(state.shape[0]):
                for t in range(1, self.NT):
                    loc1 = torch.where(state[:, t] == s2)
                    loc2 = torch.where(state[loc1[0], t - 1] == s1)
                    loc = loc1[0][loc2[0]]
                    phix = torch.sum(phi * d[loc, t, :], 1)
                    phixstar = torch.sum(phi_star * d[loc, t, :], 1)
                    ratio_phi += torch.sum(phixstar - phix)
                    for k in range(s2 + 1):  # can be eaqul to s2
                        ratio_phi += torch.sum(np.log(1 + np.exp(phix + zeta[s1, k])) - np.log(1 + np.exp(phixstar + zeta[s1, k])))
            # for n in range(state.shape[0]):
            for t in range(1, self.NT):
                loc1 = torch.where(state[:, t] == self.NS - 1)
                loc2 = torch.where(state[loc1[0], t - 1] == s1)
                loc = loc1[0][loc2[0]]
                # if state[n, t] == self.NS - 1 and state[n, t - 1] == s1:
                phix = torch.sum(phi * d[loc, t, :], 1)
                phixstar = torch.sum(phi_star * d[loc, t, :], 1)
                for k in range(self.NS - 1):  # can't be equal to ns-1
                    ratio_phi += torch.sum(np.log(1 + np.exp(phix + zeta[s1, k])) - np.log(1 + np.exp(phixstar + zeta[s1, k])))
        # Prior I
        # a = ratio_phi + 0.5 *np.sum(phi*phi) -0.5 * np.sum(phi_star * phi_star)
        # Prior II
        one_iter_time = time.time() - t1
        # print("%.3f seconds process time for calculating ratio" % one_iter_time)
        # a = ratio_phi + np.sum((phi - 2) * (phi - 2)) / 100 - np.sum((phi_star - 2) * (phi_star - 2)) / 100
        a = ratio_phi + torch.sum(torch.square(phi) / tau_phi / 2) - torch.sum(torch.square(phi_star)/ tau_phi/2)
        # print("a:%.6f" % a)
        ratio_phi = 1 if a > 0 else np.exp(a)
        rand_ratio = torch.rand(1)
        if rand_ratio < ratio_phi:
            phi_out = phi_star
            accept_phi += 1
        else:
            phi_out = phi
        # print("phi1:%.6f" % phi_out[0])
        # print("phi1:%.6f" % phi_out[1])
        return phi_out, accept_phi

    def update_tau_phi(self, gamma_phi, phi):
        # tau_phi = invgauss.rvs(np.sqrt(gamma_phi * sigma_phi/ np.power(phi, 2)), np.sqrt(gamma_phi))
        tau_phi = invgauss.rvs(torch.sqrt(gamma_phi / torch.square(phi)), gamma_phi)   # I think no need to add sqrt
        # return torch.Tensor(1/tau_phi)    ### pay attention to here
        return torch.Tensor(tau_phi)

    def update_gamma_phi(self, tau_phi):
        # prior
        alpha_gamma = 1
        beta_gamma = 0.1
        gamma_phi = nrd.gamma(alpha_gamma + 1, 1 / (beta_gamma + (tau_phi / 2)))
        return torch.Tensor(gamma_phi)

    # def update_sigma_phi(self, phi, tau_phi):
    #     alpha_phi = 9
    #     beta_phi = 4
    #     beta_pos = torch.sum(torch.square(phi)/tau_phi) / 2 + beta_phi
    #     sigma_phi = 1 / np.random.gamma(alpha_phi + phi.shape[0] / 2, 1 / beta_pos, 1)
    #     # print("sigma1:%.6f"%sigma_out[0])
    #     # print("sigma2:%.6f" % sigma_out[1])
    #     return torch.Tensor(sigma_phi)

    #-----------------------------------Old version------------------------------------
    def update_phi_old(self, d, phi, zeta, state, c_phi, accept_phi):
        d = d.numpy()
        zeta = zeta.numpy()
        state = state.numpy()
        phi = phi.numpy()
        # phi_out = np.zeros_like(phi)
        # phi1 = phi[np.newaxis, :]
        # w_t = np.tile(w[:, np.newaxis, np.newaxis], (1, d.shape[1], 1))
        n_d = d
        temp_phi = np.zeros(shape=[phi.shape[0], phi.shape[0]])
        for i in range(self.NS):
            for j in range(self.NS - 1):
                for k in range(j + 1):  # k be able to equal to j
                    # for t in range(1, state.shape[1]):
                    #     for n in range(state.shape[0]):
                    for n in range(state.shape[0]):
                        for t in range(1, self.NT):
                            if state[n, t] == j and state[n, t - 1] == i:
                                temp_phi = temp_phi + np.dot(n_d[n, t, :][:, np.newaxis], n_d[n, t, :][np.newaxis, :]) * \
                                           math.exp(zeta[i, k]) / math.pow(1 + math.exp(zeta[i, k]), 2)

            for k in range(self.NS - 1):  # j=self.NS-1 special:    k can't be equal to self.NS -1
                # for t in range(1, state.shape[1]):
                #     for n in range(state.shape[0]):
                for n in range(state.shape[0]):
                    for t in range(1, self.NT):
                        if state[n, t] == self.NS - 1 and state[n, t - 1] == i:
                            temp_phi = temp_phi + np.dot(n_d[n, t, :][:, np.newaxis], n_d[n, t, :][np.newaxis, :]) * \
                                       math.exp(zeta[i, k]) / math.pow(1 + math.exp(zeta[i, k]), 2)
                            temp_phi = temp_phi + np.dot(np.transpose(n_d[n, t, :]), n_d[n, t, :]) * \
                                       math.exp(zeta[i, k]) / math.pow(1 + np.exp(zeta[i, k]), 2)
        # Prior I
        # sigma_phi = np.linalg.inv(np.diag(v=np.ones(shape=self.ND)) + temp_phi)
        # Prior I
        sigma_phi = np.linalg.inv(np.diag(v=np.ones(shape=phi.shape[0])) / 100 + temp_phi)
        phi_star = nrd.multivariate_normal(phi, c_phi * sigma_phi)
        ratio_phi = 0
        for s1 in range(self.NS):
            for s2 in range(self.NS - 1):
                for n in range(state.shape[0]):
                    for t in range(1, self.NT):
                        if state[n, t] == s2 and state[n, t - 1] == s1:
                            phix = np.sum(phi * n_d[n, t, :])
                            phixstar = np.sum(phi_star * n_d[n, t, :])
                            ratio_phi += phixstar - phix
                            # print("ratio_phi: %.6f" % ratio_phi)
                            for k in range(s2 + 1):  # can be eaqul to s2
                                ratio_phi += np.log(1 + np.exp(phix + zeta[s1, k])) \
                                             - np.log(1 + np.exp(phixstar + zeta[s1, k]))
                # print("ratio_phi: %.6f" % ratio_phi)
            for n in range(state.shape[0]):
                for t in range(1, self.NT):
                    if state[n, t] == self.NS - 1 and state[n, t - 1] == s1:
                        phix = np.sum(phi * n_d[n, t, :])
                        phixstar = np.sum(phi_star * n_d[n, t, :])
                        for k in range(self.NS - 1):  # can't be equal to ns-1
                            ratio_phi += np.log(1 + np.exp(phix + zeta[s1, k])) \
                                         - np.log(1 + np.exp(phixstar + zeta[s1, k]))
        # Prior I
        # a = ratio_phi + 0.5 *np.sum(phi*phi) -0.5 * np.sum(phi_star * phi_star)
        # Prior II
        a = ratio_phi + np.sum((phi - 2) * (phi - 2)) / 100 - np.sum((phi_star - 2) * (phi_star - 2)) / 100
        ratio_phi = 1 if a > 0 else np.exp(a)
        rand_ratio = np.random.rand(1)
        if rand_ratio < ratio_phi:
            phi_out = phi_star
            accept_phi += 1
        else:
            phi_out = phi
        # print("phi1:%.6f" % phi_out[0])
        # print("phi1:%.6f" % phi_out[1])
        return torch.from_numpy(phi_out), accept_phi

    def update_tau(self, tau, state, c_tau, accept_tau):
        tau_out = np.zeros(shape=[self.NS - 1])
        tau = tau.numpy()
        state = state.numpy()
        for i in range(self.NS - 1):
            sigma_tau = 0
            ratio_tau = 0
            for j in range(i, self.NS):
                sigma_tau += 0.25 * np.sum(state[:, 0] == j).item()
            # prior II
            Pri_sigma = 10000
            Pri_tau = 0
            sigma_tau = 1 / (sigma_tau + 1 / Pri_sigma)
            tau_star = nrd.normal(tau[i], np.sqrt(sigma_tau * c_tau))
            # calculate the ratio
            for j in range(i, self.NS):
                ratio_tau += np.sum(state[:, 0] == j) * (
                        np.log(1 + np.exp(tau[i])) - np.log(1 + np.exp(tau_star)))
            a = (tau_star - tau[i]) * np.sum(state[:, 0] == i) + ratio_tau - np.power((tau_star - Pri_tau), 2) / \
                (2 * Pri_sigma) + np.power(tau[i] - Pri_tau, 2) / (2 * Pri_sigma)
            ratio_tau = 1 if a > 0 else np.exp(a)
            rand_ratio = nrd.rand(1)
            if rand_ratio < ratio_tau:
                accept_tau[i] += 1
                tau_out[i] = tau_star
            else:
                tau_out[i] = tau[i]
        return torch.from_numpy(tau_out), accept_tau

    def update_zeta(self, zeta, state, phi, h, c_zeta, accept_zeta):
        # Args: phi is the coeff for transition model
        # tran = self.tran(zeta, beta, h, w)
        # n_d = np.concatenate((h, w), 2)  # N * NT
        n_d = h.numpy()
        phi = phi.numpy()
        state = state.numpy()
        zeta = zeta.numpy()
        c_zeta = c_zeta
        phi_d = np.sum(phi * n_d, 2)  # N * NT
        zeta_out = np.zeros(shape=[self.NS, self.NS])
        for s1 in range(self.NS):
            for s2 in range(self.NS - 1):
                temp_zeta = 0
                n_zeta = 0
                for k in range(s2, self.NS):
                    for t in range(1, self.NT):
                        loc1 = np.where(state[:, t] == k)
                        loc2 = np.where(state[loc1[0], t - 1] == s1)
                        loc = loc1[0][loc2[0]]
                        temp_zeta += np.sum(np.exp(phi_d[loc, t]) / np.square(1 + np.exp(phi_d[loc, t])), 0)
                        # if state[n, t] == k and state[n, t - 1] == s1:
                        #     temp_zeta += math.exp(phi_d[n, t]) / math.pow(1 + math.exp(phi_d[n, t]), 2)
                        if k == s2:
                            n_zeta += loc.shape[0]
                # Prior I
                # sigma_zeta = 1/(1 + temp_zeta)
                # Prior II
                pri_Sigma = 10000
                pri_zeta = 0
                sigma_zeta = 1 / (1 / pri_Sigma + temp_zeta)
                # zeta_star = np.random.randn(1) * math.sqrt(c_zeta * sigma_zeta) + zeta[s1, s2]
                zeta_star = nrd.normal(zeta[s1, s2], np.sqrt(c_zeta * sigma_zeta))
                ratio_1 = 0
                ratio_2 = 0
                for k in range(s2, self.NS):
                    for t in range(1, self.NT):
                        loc1 = np.where(state[:, t] == k)
                        loc2 = np.where(state[loc1[0], t - 1] == s1)
                        loc = loc1[0][loc2[0]]
                            # phi_d = tran[n, t, 0, 0] - zeta[0, 0]  # phi * d for n person t time fixed
                        ratio_1 += np.sum(np.log(1 + np.exp(phi_d[loc, t] + zeta[s1, s2])))
                        ratio_2 += np.sum(np.log(1 + np.exp(phi_d[loc, t] + zeta_star)))
                a = (zeta_star - zeta[s1, s2]) * n_zeta + ratio_1 - ratio_2 + \
                    math.pow(zeta[s1, s2] - pri_zeta, 2) / (2 * pri_Sigma) - math.pow(zeta_star - pri_zeta, 2) / (
                                2 * pri_Sigma)
                ratio_zeta = 1 if a > 0 else np.exp(a)
                # print("ratio_zeta : %6f" % ratio_zeta)
                rand_ratio = np.random.rand(1)
                if rand_ratio < ratio_zeta:
                    accept_zeta[s1, s2] += 1
                    zeta_out[s1, s2] = zeta_star
                else:
                    zeta_out[s1, s2] = zeta[s1, s2]
        return torch.from_numpy(zeta_out), accept_zeta

    def update_zeta_old(self, zeta, state, phi, h, c_zeta, accept_zeta):
        # Args: phi is the coeff for transition model
        # tran = self.tran(zeta, beta, h, w)
        n_d = h.numpy()  # N * NT
        zeta = zeta.numpy()
        phi = phi.numpy()
        state = state.numpy()
        phi_d = np.sum(phi * n_d, axis=2)  # N * NT
        zeta_out = np.zeros(shape=[self.NS, self.NS], dtype=float)
        for s1 in range(self.NS):
            for s2 in range(self.NS - 1):
                temp_zeta = 0
                n_zeta = 0
                for k in range(s2, self.NS):
                    for n in range(h.shape[0]):
                        for t in range(1, self.NT):
                            if state[n, t] == k and state[n, t - 1] == s1:
                                # phi_d = tran[n, t, 0, 0] - zeta[0, 0] #phi * d for n person t time fixed
                                temp_zeta += math.exp(phi_d[n, t]) / math.pow(1 + math.exp(phi_d[n, t]), 2)
                                if k == s2:
                                    n_zeta += 1
                # Prior I
                # sigma_zeta = 1/(1 + temp_zeta)
                # Prior II
                pri_Sigma = 100
                pri_zeta = 2
                sigma_zeta = 1 / (1 / pri_Sigma + temp_zeta)
                # print("n_zeta: %d"%n_zeta)
                zeta_star = np.random.randn(1) * math.sqrt(c_zeta * sigma_zeta) + zeta[s1, s2]
                # zeta_star = 1
                ratio_1 = 0
                ratio_2 = 0
                for k in range(s2, self.NS):
                    for n in range(self.N):
                        for t in range(1, self.NT):
                            if state[n, t] == k and state[n, t - 1] == s1:
                                # phi_d = tran[n, t, 0, 0] - zeta[0, 0]  # phi * d for n person t time fixed
                                ratio_1 += math.log(1 + math.exp(phi_d[n, t] + zeta[s1, s2]))
                                ratio_2 += math.log(1 + math.exp(phi_d[n, t] + zeta_star))
                # print("ratio1: %.6f" % ratio_1)
                # # Prior I
                # a = (zeta_star - zeta[s1, s2]) * n_zeta + ratio_1 - ratio_2 +\
                #              0.5 * math.pow(zeta[s1, s2], 2) - 0.5 * math.pow(zeta_star, 2)
                # Prior II
                a = (zeta_star - zeta[s1, s2]) * n_zeta + ratio_1 - ratio_2 + \
                    math.pow(zeta[s1, s2] - pri_zeta, 2) / (2 * pri_Sigma) - math.pow(zeta_star - pri_zeta, 2) / (
                                2 * pri_Sigma)
                # print("ratio:%.6f"% np.exp(a))
                # print("a:%.6f" % a)
                # print("zeta_star: %.6f" % zeta_star)
                # print("zeta: %.6f" % zeta[s1, s2])
                ratio_zeta = 1 if a > 0 else np.exp(a)
                # print("ratio_zeta : %6f" % ratio_zeta)
                rand_ratio = np.random.rand(1)
                if rand_ratio < ratio_zeta:
                    accept_zeta[s1, s2] += 1
                    zeta_out[s1, s2] = zeta_star
                else:
                    zeta_out[s1, s2] = zeta[s1, s2]
        # print("zeta1:%.6f"%zeta[0, 0])
        # print("zeta2:%.6f" % zeta[1, 0])
        return torch.from_numpy(zeta_out), accept_zeta


        # ------------------------------------------------Beta is the verying coefficient-----------------------------------
    def update_vc(self, y, x, state, beta, sigma, thres, bs_value, w, p_tau, c_beta, accept_beta, tau_beta):
        # ----------------Args: beta[NS * NV * NB] tau is penalty coefficient for Bspline coefficient--------------
        #-----------------Args: w: N * NT * NG   sigma_beta : [NS]  tau * beta [NS * NV]
        # state = state.numpy()
        for s in range(self.NS):
            location = torch.where(state == s)
            ty = y[location[0], location[1]]  # target y whose state is s  (shape: NUM * NG)
            tx = x[location[0], location[1]]  # shape: (Num * NX * NG)
            tw = w[location[0], location[1]]  # shape *(NUM * NG)
            # update beta one by one
            for v in range(self.NV):
                # beta_bs = [np.sum(beta[s, i: i+self.NB][:, np.newaxis] * bs_value, 0) for i in range(0, beta[s].shape[0], self.NB)]
                # each term is NG :(NB * 1) * (NB * NG) = NG  (NV * NG)
                beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1)
                # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
                #-----------------------------------New Version---------------------------------------------
                soft_beta_bs = torch.Tensor(
                    np.array([self.soft_thres(beta_bs[i], thres[s, i]).numpy() for i in range(self.NV)]))    # NV * NG
                # soft_xbeta = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs, 0) # NUM * NV * NG
                soft_xbeta = tx * torch.unsqueeze(soft_beta_bs, 0)  # NUM * NV * NG
                mean = torch.sum(soft_xbeta, 1)  # NUM * NG
                p1 = - torch.square(ty - mean - tw) / 2 / sigma[s]
                beta_star = beta[s].clone()
                # star = torch.Tensor(nrd.multivariate_normal(beta[s, v].numpy(), c_beta[s, v] * np.identity(self.NB)))  # replace v-th (NV * NB )
                # star = MultivariateNormal(beta[s, v], c_beta[s, v] * torch.eye(self.NB)).sample()
                star = MultivariateNormal(beta[s, v], c_beta[s, v] * torch.eye(self.NB)).sample()
                # star = MultivariateNormal(torch.zeros_like(beta[s, v]), c_beta[s, v] * torch.eye(self.NB)).sample()  # another proposal
                beta_star[v] = star
                beta_bs_star = torch.sum(torch.unsqueeze(beta_star, 2) * torch.unsqueeze(bs_value, 0), 1)  # (NV * NB * 1) * (1 * NB * NG)  NV * NG
                soft_beta_bs_star = torch.Tensor(
                    np.array([self.soft_thres(beta_bs_star[i], thres[s, i]).numpy() for i in range(self.NV)])) # NUM * NV * NG
                # soft_xbeta_star = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs_star, 0)
                soft_xbeta_star = tx * torch.unsqueeze(soft_beta_bs_star, 0)
                mean_star = torch.sum(soft_xbeta_star, 1)  # NUM * NG
                star1 = - np.square(ty - mean_star - tw) / 2 / sigma[s]
                # -------------- penalty probability------------------------
                p2 = 0
                star2 = 0
                # D = self.banded([1, -2, 1], self.NB - 2)
                # K = torch.Tensor(np.matmul(np.transpose(D), D))
                # p2 = - 0.5 / tau[s, v] * np.matmul(np.matmul(beta[s, v, np.newaxis], K), beta[s, v, :, np.newaxis])
                # star2 = - 0.5 / tau[s, v] * np.matmul(np.matmul(star[np.newaxis], K), star[:, np.newaxis])
                # p2 = - 0.5 / tau[s, v] * torch.mm(torch.mm(torch.unsqueeze(beta[s, v], 0), self.PK), torch.unsqueeze(beta[s, v], 1))   # this is for smooth
                # star2 = - 0.5 / tau[s, v] * torch.mm(torch.mm(torch.unsqueeze(star, 0), self.PK), torch.unsqueeze(star, 1))
                #--------------------------This is for adaptive Group Lasso----------------------------------
                p3 = 0
                star3 = 0
                # --------------------- Need to add intercept if not, for if v!==1...
                # if v > 0:
                #     # cov = sigma_beta[s] * tau_beta[s, v-1] * self.IPG
                #     cov = sigma[s] * tau_beta[s, v - 1] * self.IPG
                #     mm = MultivariateNormal(torch.zeros_like(beta[s, v]), cov)
                #     p3 = mm.log_prob(beta[s, v])
                #     star3 = mm.log_prob(star)
                # else:
                #     cov = torch.eye(self.NB)
                #     mm = MultivariateNormal(torch.zeros_like(beta[s, v]), cov)
                #     p3 = mm.log_prob(beta[s, v])
                #     star3 = mm.log_prob(star)
                ############--------------- smooth penalty---------------------------#######
                p4 = - 0.5 / p_tau[s, v] * torch.matmul(torch.matmul(beta[s, v], self.PK), torch.unsqueeze(beta[s, v], 1))
                star4 = - 0.5 / p_tau[s, v] * torch.matmul(torch.matmul(star, self.PK),
                                                        torch.unsqueeze(star, 1))
                # p4 = 0
                # star4 = 0
                log_ratio = torch.sum(star1) + star2 - torch.sum(p1) - p2 + star3 - p3 + star4 - p4
                ratio = torch.exp(log_ratio) if log_ratio < 0 else 1
                rand_ratio = nrd.rand(1)[0]
                # print("Before accepting")
                # print(beta)
                if ratio > rand_ratio:
                    beta[s, v] = star
                    accept_beta[s, v] += 1
                # print("After accepting:")
                # print(beta)
        return beta, accept_beta

    def update_thres(self, y, x, state, beta, sigma, thres, bs_value, w, c_thres, accept_thres):
        # ----------------Args: beta[NS * NV * NB] tau is penalty coefficient for Bspline coefficient--------------
        #-----------------Args: w: N * NT * NG   sigma_beta : [NS]  tau * beta [NS * NV]
        # state = state.numpy()
        for s in range(self.NS):
            location = torch.where(state == s)
            ty = y[location[0], location[1]]  # target y whose state is s  (shape: NUM * NG)
            tx = x[location[0], location[1]]  # shape: (Num * NX * NG)
            tw = w[location[0], location[1]]  # shape *(NUM * NG)
            # update beta one by one; fix thres for intercept is 0
            for v in range(1, self.NV):
                # beta_bs = [np.sum(beta[s, i: i+self.NB][:, np.newaxis] * bs_value, 0) for i in range(0, beta[s].shape[0], self.NB)]
                # each term is NG :(NB * 1) * (NB * NG) = NG  (NV * NG)
                beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1)
                # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
                #-----------------------------------New Version---------------------------------------------
                soft_beta_bs = torch.Tensor(
                    np.array([self.soft_thres(beta_bs[i], thres[s, i]).numpy() for i in range(self.NV)]))    # NV * NG
                # soft_xbeta = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs, 0) # NUM * NV * NG
                soft_xbeta = tx * torch.unsqueeze(soft_beta_bs, 0)  # NUM * NV * NG
                mean = torch.sum(soft_xbeta, 1)  # NUM * NG
                p1 = - torch.square(ty - mean - tw) / 2 / sigma[s]
                # beta_star = beta[s].clone()
                # star = torch.Tensor(nrd.multivariate_normal(beta[s, v].numpy(), c_beta[s, v] * np.identity(self.NB)))  # replace v-th (NV * NB )
                # beta_star[v] = star
                # beta_bs_star = torch.sum(torch.unsqueeze(beta_star, 2) * torch.unsqueeze(bs_value, 0), 1)  # (NV * NB * 1) * (1 * NB * NG)  NV * NG
                # soft_beta_bs_star = torch.Tensor(
                #     np.array([self.soft_thres(beta_bs_star[i], thres[s, i]).numpy() for i in range(self.NV)])) # NUM * NV * NG
                # # soft_xbeta_star = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs_star, 0)
                # soft_xbeta_star = tx * torch.unsqueeze(soft_beta_bs_star, 0)
                # mean_star = torch.sum(soft_xbeta_star, 1)  # NUM * NG
                # star1 = - np.square(ty - mean_star - tw) / 2 / sigma[s]
                thres_star = thres[s].clone()
                #----------------------------Pay attention to star! Must be positive----
                # star = thres[s, v] + nrd.uniform(max([-thres[s, v], -c_thres[s, v]]), c_thres[s, v])
                star = ss.truncnorm.rvs(-thres[s, v]/c_thres[s, v-1], (0.15 - thres[s, v])/c_thres[s, v-1],
                                        loc=thres[s, v], scale=c_thres[s, v-1])  # Truncated norm in [0. 0.15] with mean is original thres
                # star = ss.truncnorm.rvs(0 / c_thres[s, v - 1], (0.15 - 0) / c_thres[s, v - 1],
                #                  loc= 0, scale=c_thres[s, v - 1]) # Truncated norm in [0. 0.15] with mean is 0
                # thres_star[v] = star[0]
                thres_star[v] = star
                soft_beta_bs_star = torch.Tensor(
                    np.array([self.soft_thres(beta_bs[i], thres_star[i]).numpy() for i in range(self.NV)])) # NUM * NV * NG
                # soft_xbeta_star = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs_star, 0)
                soft_xbeta_star = tx * torch.unsqueeze(soft_beta_bs_star, 0)
                mean_star = torch.sum(soft_xbeta_star, 1)  # NUM * NG
                star1 = - np.square(ty - mean_star - tw) / 2 / sigma[s]
                # --------------prior probability------------------------
                p2 = ss.truncnorm.logpdf(thres[s, v], 0, 0.15) # Prior is Truncated norm in (0, 0.05)
                # star2 = ss.truncnorm.logpdf(star[0], 0, 0.05)
                star2 = ss.truncnorm.logpdf(star, 0, 0.15)
                log_ratio = torch.sum(star1) - torch.sum(p1) + star2 - p2
                ratio = torch.exp(log_ratio) if log_ratio < 0 else 1
                rand_ratio = nrd.rand(1)[0]
                # print("Before accepting")
                # print(beta)
                if ratio > rand_ratio:
                    # thres[s, v] = star[0]
                    thres[s, v] = star
                    accept_thres[s, v] += 1
                # print("After accepting:")
                # print(beta)
        return thres, accept_thres


    def update_mean(self, x, state, beta, thres, bs_value):
        mean = torch.zeros(self.N, self.NT, self.NG)
        for s in range(self.NS):
            location_s = torch.where(state == s)
            tx = x[location_s[0], location_s[1]]
            beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1) # NV * NG
            soft_beta_bs = torch.Tensor(
                np.array([self.soft_thres(beta_bs[i], thres[s, i]).numpy() for i in range(self.NV)]))  # NV * NG
            # soft_xbeta = torch.unsqueeze(tx, 2) * torch.unsqueeze(soft_beta_bs, 0)  # NUM * NV * NG
            soft_xbeta = tx * torch.unsqueeze(soft_beta_bs, 0)  # NUM * NV * NG
            mean[location_s[0], location_s[1]] = torch.sum(soft_xbeta, 1)  # NUM * NG
        return mean


    def update_tau_beta(self, gamma_beta, sigma, beta):
        # tau_phi = invgauss.rvs(np.sqrt(gamma_phi * sigma_phi/ np.power(phi, 2)), np.sqrt(gamma_phi))
        tau_beta = torch.zeros(self.NS, self.NV-1)
        for s in range(self.NS):
            for v in range(1, self.NV):
                # p_beta = torch.sqrt(torch.mm(torch.mm(torch.unsqueeze(beta[s, v], 0), self.PG), torch.unsqueeze(beta[s, v], 1)))# 1 * 1
                p_beta = torch.mm(torch.mm(torch.unsqueeze(beta[s, v], 0), self.PG), torch.unsqueeze(beta[s, v], 1))  # 1 * 1
                tau_beta[s, v-1] = invgauss.rvs(math.sqrt(gamma_beta[s, v-1] * sigma[s] / p_beta[0, 0]),
                               gamma_beta[s, v-1])  # I think no need to add sqrt
        # return 1 / tau_beta
        return 1 / tau_beta


    def update_gamma_beta(self, tau_beta):
        # prior
        alpha_gamma = 1
        beta_gamma = 0.1
        # print(tau_beta)
        gamma_beta = nrd.gamma(alpha_gamma + (self.NB + 1) / 2, 1 / (beta_gamma + (tau_beta.numpy() / 2)))
        # gamma_beta = nrd.gamma(alpha_gamma + 1, 1 / (beta_gamma + (tau_beta.numpy() / 2)))
        return torch.Tensor(gamma_beta)

    # def update_sigma_beta(self, beta, tau_beta):
    #     alpha_phi = 9
    #     beta_phi = 4
    #     p_beta = torch.matmul(torch.matmul(torch.unsqueeze(beta, 2), self.PG), torch.unsqueeze(beta, 3))# NS * NV * 1 * 1
    #     beta_pos = torch.sum(torch.squeeze(torch.squeeze(p_beta, 2), 2) / tau_beta/ 2, 1) + beta_phi  # NS
    #     sigma_beta = 1 / np.random.gamma(alpha_phi + self.NV / 2, 1 / beta_pos.numpy())
    #     # print("sigma1:%.6f"%sigma_out[0])
    #     # print("sigma2:%.6f" % sigma_out[1])
    #     return torch.Tensor(sigma_beta)

    def update_vc_cuda(self, y, x, state, beta, sigma, thres, bs_value, tau, w, c_beta, accept_beta):
        # ----------------Args: beta[NS * NV * NB] tau is penalty coefficient for Bspline coefficient--------------
        #-----------------Args: w: N * NT * NG
        # state = state.numpy()
        for s in range(self.NS):
            location = torch.where(state == s)
            ty = y[location[0], location[1]]  # target y whose state is s  (shape: NUM * NG)
            tx = x[location[0], location[1]]  # shape: (Num * NX)
            tw = w[location[0], location[1]]  # shape *(NUM * NG)
            # update beta one by one
            for v in range(self.NV):
                # beta_bs = [np.sum(beta[s, i: i+self.NB][:, np.newaxis] * bs_value, 0) for i in range(0, beta[s].shape[0], self.NB)]
                # each term is NG :(NB * 1) * (NB * NG) = NG  (NV * NG)
                beta_bs = torch.sum(torch.unsqueeze(beta[s].cuda(), 2) * torch.unsqueeze(bs_value.cuda(), 0), 1)
                # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
                x_beta = torch.unsqueeze(tx, 2) * torch.unsqueeze(beta_bs, 0)  # NUM * NV * NG
                # soft_xbeta = np.array(
                #     [self.soft_thres(x_beta[:, i], thres[s, i]) for i in range(self.NV)])  # NV * NUM * NG
                soft_xbeta = torch.Tensor(
                    np.array([self.soft_thres(x_beta[:, i], thres[s, i]).numpy() for i in range(self.NV)])) # NV * NUM * NG
                mean = torch.sum(soft_xbeta, 0)  # NUM * NG
                p1 = - torch.square(ty - mean - tw) / 2 / sigma[s]
                beta_star = beta[s].clone()
                star = torch.Tensor(nrd.multivariate_normal(beta[s, v].numpy(), c_beta[s, v] * np.identity(self.NB)))  # replace v-th (NV * NB )
                beta_star[v] = star
                beta_bs_star = torch.sum(torch.unsqueeze(beta_star, 2) * torch.unsqueeze(bs_value, 0), 1)  # (NV * NB * 1) * (1 * NB * NG)  NV * NG
                x_beta_star = torch.unsqueeze(tx, 2) * torch.unsqueeze(beta_bs_star, 0)  # NUM * NV * NG
                soft_xbeta_star =torch.Tensor(
                    np.array([self.soft_thres(x_beta_star[:, i], thres[s, i]).numpy() for i in range(self.NV)]))  # NV * NUM * NG
                mean_star = torch.sum(soft_xbeta_star, 0)  # NUM * NG
                star1 = - np.square(ty - mean_star - tw) / 2 / sigma[s]
                # -------------- penalty probability------------------------
                # D = self.banded([1, -2, 1], self.NB - 2)
                # K = torch.Tensor(np.matmul(np.transpose(D), D))
                # p2 = - 0.5 / tau[s, v] * np.matmul(np.matmul(beta[s, v, np.newaxis], K), beta[s, v, :, np.newaxis])
                # star2 = - 0.5 / tau[s, v] * np.matmul(np.matmul(star[np.newaxis], K), star[:, np.newaxis])
                p2 = - 0.5 / tau[s, v] * torch.mm(torch.mm(torch.unsqueeze(beta[s, v], 0), self.PK), torch.unsqueeze(beta[s, v], 1))
                star2 = - 0.5 / tau[s, v] * torch.mm(torch.mm(torch.unsqueeze(star, 0), self.PK), torch.unsqueeze(star, 1))
                log_ratio = torch.sum(star1) + star2 - torch.sum(p1) - p2
                ratio = torch.exp(log_ratio) if log_ratio < 0 else 1
                rand_ratio = nrd.rand(1)[0]
                # print("Before accepting")
                # print(beta)
                if ratio > rand_ratio:
                    beta[s, v] = star
                    accept_beta[s, v] += 1
                # print("After accepting:")
                # print(beta)
        return beta, accept_beta

    def update_ptau(self, bs):   # penalty for bspline coefficient
        tau = torch.zeros(self.NS, self.NV)
        a = 0.0001
        b = 0.0001
        r = self.NB - 2   # degree of PK
        a_star = a + 0.5 * r
        for s in range(self.NS):
            for v in range(self.NV):
                bs_temp = bs[s, v]
                b_star = b + 0.5 * torch.matmul(torch.matmul(torch.unsqueeze(bs_temp, 0), self.PK), torch.unsqueeze(bs_temp, 1))[0, 0]
                tau[s, v] = 1/nrd.gamma(a_star, 1/b_star)
        return tau


    # def update_ptau(self, bs):   # penalty for bspline coefficient
    #     tau = torch.zeros(self.NS, self.NV)
    #     a = 0.0001
    #     b = 0.0001
    #     r = self.NB - 2   # degree of PK
    #     a_star = a + 0.5 * r
    #     for s in range(self.NS):
    #         for v in range(self.NV):
    #             bs_temp = bs[s, v]
    #             b_star = b + 0.5 * np.matmul(np.matmul(bs_temp[np.newaxis, :], self.PK), bs_temp[:, np.newaxis])[0, 0]
    #             tau[s, v] = 1/nrd.gamma(a_star, 1/b_star)
    #     return tau

#---------------------------------For simulation label switch-----------------------------------
    def label_switch(self, state, sigma, beta, tau_beta, gamma_beta, p_tau, thres, bs_value):
        #----------------------Test grid----------------------------
        new_state = torch.zeros_like(state)
        new_sigma = torch.zeros_like(sigma)
        new_beta = torch.zeros_like(beta)
        # new_ptau = torch.zeros_like(ptau)
        new_tau_beta = torch.zeros_like(tau_beta)
        new_gamma_beta = torch.zeros_like(gamma_beta)
        new_ptau = torch.zeros_like(p_tau)
        new_thres = torch.zeros_like(thres)
        #------------- Here choose the first grid of intercept ----------------------------
        test_vc = torch.zeros(self.NS)
        for s in range(self.NS):
            test_vc[s] = torch.sum(beta[s, 0] * bs_value[:, 0])
            # test_vc[s] = torch.sum(beta[s, 1] * bs_value[:, -1])
        test = torch.arange(0, self.NS)  # 1, 2, 3...state
        test_order = torch.argsort(test_vc, descending=False)         # The intercept is in increasing order------------------------
        test_order_1 = torch.argsort(test_order)
        if (test_order != test).any():
            for s in range(sigma.shape[0]):
                new_sigma[s] = sigma[test_order[s]]
                new_beta[s] = beta[test_order[s]]
                # new_ptau[s] = ptau[test_order[s]]
                new_tau_beta[s] = tau_beta[test_order[s]]
                new_gamma_beta[s] = gamma_beta[test_order[s]]
                new_ptau[s] = p_tau[test_order[s]]
                new_thres[s] = thres[test_order[s]]
            for i in range(state.shape[0]):
                for t in range(state.shape[1]):
                    new_state[i, t] = test_order_1[state[i, t].int()]
                    # print("original_state:%1f"%state[i, t])
                    # print("new_state:%1f" % new_state[i, t])
        else:
            new_state = state
            new_sigma = sigma
            new_beta = beta
            # new_ptau = ptau
            new_tau_beta = tau_beta
            new_gamma_beta = gamma_beta
            new_ptau = p_tau
            new_thres = thres
        # print("o:%d,%d,%d,%d"%(np.sum(state==0),np.sum(state==1),np.sum(state==2),np.sum(state==3)))
        # print("n:%d,%d,%d,%d"%(np.sum(new_state == 0), np.sum(new_state == 1), np.sum(new_state == 2), np.sum(new_state == 3)))
        return new_state, new_sigma, new_beta, new_tau_beta, new_gamma_beta, new_ptau, new_thres

    def y_likeli_old(self, x, y, beta, bs_value, w, sigma, thres):
        mean = torch.zeros(self.N, self.NT, y.shape[2], self.NS)   # calculate the mean of y in each state
        # y_likeli = torch.zeros(self.N, self.NT, self.NG, self.NS)
        for s in range(self.NS):
            beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1)  # NV * NG
            # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
            x_beta = torch.unsqueeze(x, 3) * torch.unsqueeze(torch.unsqueeze(beta_bs, 0), 0)  # N * NT * NV * NG
            soft_xbeta = torch.Tensor(
                np.array([self.soft_thres(x_beta[:, :, i], thres[s, i]).numpy() for i in
                          range(self.NV)]))  # NV * N * NT * NG  #!!!!!!!!!!! Need to exclude intercept
            mean[:, :, :, s] = torch.sum(soft_xbeta, 0) + w  # N * NT * NG
        y_likeli_old = Normal(mean, sigma.sqrt()).log_prob(torch.unsqueeze(y, 3))
        return y_likeli_old

    # def y_likeli(self, x, y, beta, bs_value, w, sigma, thres):
    #     mean = torch.zeros(self.N, self.NT, y.shape[2], self.NS)   # calculate the mean of y in each state
    #     # y_likeli = torch.zeros(self.N, self.NT, self.NG, self.NS)
    #     for s in range(self.NS):
    #         beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1)  # NV * NG
    #         # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
    #         soft_beta_bs = torch.Tensor(
    #             np.array([self.soft_thres(beta_bs[i], thres[s, i]).numpy() for i in range(self.NV)]))  # NV * NG
    #         soft_xbeta = torch.unsqueeze(x, 3) * torch.unsqueeze(torch.unsqueeze(soft_beta_bs, 0), 1)  # N * NT * NV * 1  * ( NV * NG)
    #         mean[:, :, :, s] = torch.sum(soft_xbeta, 2) + w  # N * NT * NG
    #     y_likeli = Normal(mean, sigma.sqrt()).log_prob(torch.unsqueeze(y, 3))
    #     return y_likeli
    ##----------------------------------------- Modified version; x is functional predictor--------------------
    def y_likeli(self, x, y, beta, bs_value, w, sigma, thres):
        mean = torch.zeros(self.N, self.NT, y.shape[2], self.NS)   # calculate the mean of y in each state
        # y_likeli = torch.zeros(self.N, self.NT, self.NG, self.NS)
        for s in range(self.NS):
            beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1)  # NV * NG
            # (NV * NB * 1) * (1 * NB * NG)  NV * NG    Need to update in each cycle!!
            soft_beta_bs = torch.Tensor(
                np.array([self.soft_thres(beta_bs[i], thres[s, i]).numpy() for i in range(self.NV)]))  # NV * NG
            soft_xbeta = x * torch.unsqueeze(torch.unsqueeze(soft_beta_bs, 0), 1)  # N * NT * NV * 1  * ( NV * NG)
            mean[:, :, :, s] = torch.sum(soft_xbeta, 2) + w  # N * NT * NG
        y_likeli = Normal(mean, sigma.sqrt()).log_prob(torch.unsqueeze(y, 3))
        return y_likeli

    # def update_sigma(self, y, x, state, beta, bs_value, w, thres):
    #     for s in range(self.NS):
    #         location = torch.where(state == s)
    #         ty = y[location[0], location[1]]  # target y whose state is s  (shape: NUM * NG)
    #         tx = x[location[0], location[1]]  # shape: (Num * NX)
    #         tw = w[location[0], location[1]]  # shape *(NUM * NG)
    #         beta_bs = torch.sum(torch.unsqueeze(beta[s], 2) * torch.unsqueeze(bs_value, 0), 1) # NV * NG
    #         x_beta = torch.unsqueeze(tx, 2) * torch.unsqueeze(torch.unsqueeze(beta_bs, 0), 0)
    #         soft_xbeta = torch.Tensor(
    #             np.array([self.soft_thres(x_beta[:, :, i], thres[s, i]).numpy() for i in
    #                       range(self.NV)]))  # NV * N * NT * NG  #!!!!!!!!!!! Need to exclude intercept
    #         mean = torch.sum(soft_xbeta, 0) + w  # N * NT * NG

    def update_piz(self, y, mean, state, sigma, non_alpha, Z, L, nu, psi, c_z, accept_Z):
    #----------------------Z is the component for mixture distribution, L is the label for each subject N * NT-----
        V_star = torch.ones(self.G)
        P = torch.zeros(self.G)    # It is the pi in the paper
        l_k = torch.zeros(self.G)
        for g in range(self.G):
            l_k[g] = torch.sum(L == g)
        a_star = 1 + l_k  # G
        b_star = non_alpha + torch.flip(torch.cumsum(torch.flip(l_k, [0]), 0), [0])[1:]
        # --[l_G-1, .... L_0]  --------[L_G-1, L_G-1 + L_G-2, L_G-1 +L_G-2+ L_G-3, ..., l_G-1+...l_0] ---[l_G-1+...l_0,..., L_G-1]--
        V_star[:-1] = Beta(a_star[:-1], b_star).sample()
        P[1:] = torch.cumprod(1 - V_star, 0)[:-1] * V_star[1:]  # It is pi in the paper
        P[0] = V_star[0]
        # ----------------fo debug---------
        # sum_P = torch.sum(P)
        # p = torch.zeros(self.N, self.NT)
        # p_star = torch.zeros(self.N, self.NT)
        Z_out = torch.zeros_like(Z)
        H = torch.pow(self.H_d, psi)
        dd = MultivariateNormal(torch.zeros(self.NG), nu * H)
        # Z_star = dd.sample((self.G,))    # pay attention to here; Z is from mean 0
        Z_star = MultivariateNormal(Z, c_z * torch.eye(Z.shape[1])).sample() # NG * G
        # Z_star = MultivariateNormal(Z, c_z * nu * H).sample()
        Z_star = Z_star - torch.unsqueeze(torch.mean(Z_star, 0), 0)
        # Z_star = MultivariateNormal(torch.zeros(self.NG), c_z * torch.eye(self.NG)).sample((self.G,))
        uni_L = torch.unique(L)  # value is in[0. self.G)
        rem_L = torch.Tensor(list(set(range(self.G)).difference(set(uni_L.numpy().tolist())))).long()
        w = Z[L] # N * NT * NG
        if rem_L.shape[0] != 0:
            Z_out[rem_L] = dd.sample((rem_L.shape[0],))
        w_star = Z_star[L]  # only use some z_star   # N * NT * NG
        all_p = - torch.square(y - mean - w) / 2 / torch.unsqueeze(sigma[state], 2)  # N * NT * NG
        all_pstar = - torch.square(y - mean - w_star) / 2 / torch.unsqueeze(sigma[state], 2)
        # p0 = MultivariateNormal(mu_z, torch.diag(sigma_z)).log_prob(Z)  # G
        # star0 = MultivariateNormal(mu_z, torch.diag(sigma_z)).log_prob(Z_star)  # G
        p0 = dd.log_prob(Z)  # G
        star0 = dd.log_prob(Z_star)  # G
        for each_l in uni_L:
            loc = torch.where(L == each_l)
            p1 = torch.sum(all_p[loc[0], loc[1]])
            star1 = torch.sum(all_pstar[loc[0], loc[1]])
            log_ratio = star0[each_l] + star1 - p0[each_l] - p1
            ratio = torch.exp(log_ratio) if log_ratio < 0 else 1
            rand_ratio = nrd.rand(1)[0]
            if ratio > rand_ratio:
                Z_out[each_l] = Z_star[each_l]
                accept_Z[each_l] += 1
            else:
                Z_out[each_l] = Z[each_l]
        return P, Z_out, V_star, accept_Z

    def update_L(self, P, Z, y, mean, state, sigma):
        #-------------------------- L is the allocation label: N * NT------------------------------
        L = torch.ones(self.N, self.NT).long()
        # t0 = time.time()
        res = torch.unsqueeze(y - mean, 2) - torch.unsqueeze(torch.unsqueeze(Z, 0), 0) # N * NT * G * NG
        y_loglikeli = torch.sum(- torch.square(res) / 2 / torch.unsqueeze(torch.unsqueeze(sigma[state],2), 3), 3)  # N * NT * G
        aa = y_loglikeli - torch.unsqueeze(torch.max(y_loglikeli, 2)[0], 2)  # N * NT * G    # 0.09s here can be accelerated gpu
        # t1 = time.time()
        # aa = y_loglikeli - torch.unsqueeze(torch.mean(y_loglikeli, 1), 1)
        pos_p = torch.exp(aa) * P  # N * NT * G
        for t in range(self.NT):
            L[:, t] = torch.squeeze(torch.multinomial(pos_p[:,t], num_samples=1, replacement=True),1)
        # t2 = time.time()
        return L

    def update_sigma(self, y, mean, w, state, vc, p_tau):
        alpha = torch.zeros(self.NS)
        beta = torch.zeros(self.NS)
        res = y - mean - w  # N * NT * NG
        p_alpha = 0.001 # prior for gamma    zxx modified
        p_beta = 0.001
        p = torch.matmul(torch.matmul(torch.unsqueeze(vc, 2), self.PK), torch.unsqueeze(vc, 3))[:, :, 0, 0] / p_tau
        for s in range(self.NS):
            loc = torch.where(state == s)
            # alpha[s] = (loc[0].shape[0] * self.NG) / 2 + self.NV + p_alpha  # Prior II
            alpha[s] = (loc[0].shape[0] * self.NG) / 1.5 + p_alpha
            beta[s] = torch.sum(torch.square(res[loc[0], loc[1]])) / 2 + p_beta
            # print("s:%d, alpha_s:%.6f"%(s,len(sample[0])))
        sigma = 1 / Gamma(alpha, beta).sample()
        return sigma


    def update_muz(self, sigma_z, Z):
        #---------------------------------sigma_z : NS * NG ; Z: NS * G*  NG-------------------
        #------------------------------mu_z: NS * NG -----
        p_mean = torch.zeros(self.NG)  # prior for mean
        p_cov = torch.eye(self.NG) * 100               # prior for cov
        Sigma_z = torch.diag(sigma_z)  # this is matrix form for sigma_z
        sigma_mu = torch.inverse(self.G * torch.inverse(Sigma_z) + torch.inverse(p_cov))  #NG * NG
        sum_Z = torch.sum(Z, 0) # NG
        aa = torch.matmul(torch.inverse(p_cov), torch.unsqueeze(p_mean, 1)) + torch.matmul(torch.inverse(Sigma_z), torch.unsqueeze(sum_Z, 1)) # NG * 1
        mean = torch.matmul(sigma_mu, aa)  # NG * 1
        muz = MultivariateNormal(torch.squeeze(mean, 1), sigma_mu).sample()
        return muz

    def update_sigmaz(self, mu_z, Z):
        #-----------------------------Args: Z: NS * G*  NG; mu_z: NS * NG------------
        zeta_1 = 1
        zeta_2 = 0.01
        cov = torch.sum(torch.square(Z - torch.unsqueeze(mu_z, 0)), 0)  # NG
        # sigmaz = 1 / Gamma(zeta_1 + self.G/ 2, 1/ (zeta_2 + cov/2)).sample()
        sigmaz = 1 / Gamma(zeta_1 + self.G / 2, zeta_2 + cov / 2).sample()
        return sigmaz


    def update_nu(self, Z, psi):
        #---------------Args: nu is the variance in the distance covariance  Z: G * NG--------
        nu_1 = 1
        nu_2 = 0.01
        H = torch.pow(self.H_d, psi)
        inv_H = torch.inverse(H)  # NG * NG
        b = torch.matmul(torch.matmul(torch.unsqueeze(Z, 1), inv_H), torch.unsqueeze(Z, 2)) # G * 1 * 1
        nu = 1 / Gamma(nu_1 + self.G * self.NG / 2, nu_2 + torch.sum(b) / 2).sample()
        return nu

    def update_psi(self, Z, psi, nu, c_psi, accept_psi):
        #--------------Agrs: psi is the correlation coefficient in H_d---------------
        b_psi = 300  # 3/d/ max|s_i-s_j|  d is set to be 0,01 in paper
        H = torch.pow(self.H_d, psi)
        inv_H = torch.inverse(H)  # NG * NG
        b = torch.matmul(torch.matmul(torch.unsqueeze(Z, 1), inv_H), torch.unsqueeze(Z, 2)) # G * 1 * 1
        psi_star = ss.truncnorm.rvs(-psi / c_psi, (b_psi - psi) / c_psi, loc=psi, scale=c_psi)  # Truncated norm in [0, b_psi]
        H_star = torch.pow(self.H_d, psi_star)
        inv_Hs = torch.inverse(H_star)  # NG * NG
        b_star = torch.matmul(torch.matmul(torch.unsqueeze(Z, 1), inv_Hs), torch.unsqueeze(Z, 2)) # G * 1 * 1
        p = - self.G * torch.logdet(H) / 2 - torch.sum(b) / 2 / nu
        p_star = - self.G * torch.logdet(H_star) / 2 - torch.sum(b_star) / 2 / nu
        a = p_star - p
        ratio = torch.exp(a) if a < 0 else 1
        rand_ratio = torch.rand(1)
        if ratio > rand_ratio:
            return psi_star, accept_psi+1
        else:
            return psi, accept_psi


    # def update_nonalpha(self, K, nonalpha):
    #     # Args: K is the subject label;  N * NT
    #     tau_1 = 2
    #     tau_2 = 2
    #     l_k = torch.zeros(self.G)
    #     for g in range(self.G):
    #         l_k[g] = torch.sum(K == g)
    #     a_star = 1 + l_k
    #     b_star = nonalpha + torch.flip(torch.cumsum(torch.flip(l_k, [0]), 0), [0])[1:]
    #     #--[l_G-1, .... L_0]  --------[L_G-1, L_G-1 + L_G-2, L_G-1 +L_G-2+ L_G-3, ..., l_G-1+...l_0] ---[l_G-1+...l_0,..., L_G-1]--
    #     V_star = torch.ones(self.G)
    #     V_star[:-1] = Beta(a_star[:-1], b_star).sample()
    #     sgamma = torch.sum(torch.log(1 - V_star[:-1]))
    #     alpha = Gamma(self.G + tau_1 - 1, 1/ (tau_2 - sgamma)).sample()
    #     return alpha

    def update_nonalpha(self, V):  # V is V_star
        #------------------------Args: V: NS * NG; alpha: NS---------------------
        tau_1 = 2
        tau_2 = 2
        sgamma = torch.sum(torch.log(1 - V[:-1]), 0)
        # alpha = Gamma(self.G + tau_1 - 1, 1/ (tau_2 - sgamma)).sample()
        alpha = Gamma(self.G + tau_1 - 1, tau_2 - sgamma).sample()
        return alpha

    def w_likeli(self, w, nu, psi):
        H = torch.pow(self.H_d, psi)
        dd = MultivariateNormal(torch.zeros(self.NG), nu * H)
        w_likeli = dd.log_prob(w.float())
        return w_likeli     # N * NT



    def DIC(self, y, x, h, bs_value, all_state, all_zeta, all_tau, all_phi, all_tau_phi, all_gamma_phi, all_beta, all_sigma, all_thres,
            all_gamma_beta, all_tau_beta, all_Z, all_L, all_nu, all_psi, all_ptau, c_tau, c_phi, c_zeta,  c_beta,
            c_thres, c_psi, accept_tau, accept_phi, accept_zeta, accept_beta, accept_thres, accept_psi, dint, r, Rep, o):
        m_cdll1 = 0
        m_cdll2 = 0
        iter = all_state.shape[0]
        for i in range(iter):
            t0 = time.time()
            cdll1 = 0
            state = all_state[i]
            zeta = all_zeta[i]
            tau = all_tau[i]
            phi = all_phi[i]
            tau_phi = all_tau_phi[i]
            gamma_phi = all_gamma_phi[i]
            beta = all_beta[i]
            gamma_beta = all_gamma_beta[i]
            tau_beta = all_tau_beta[i]
            sigma = all_sigma[i]
            thres = all_thres[i]
            Z = all_Z[i]
            L = all_L[i]
            nu = all_nu[i]
            psi = all_psi[i]
            # non_alpha = all_nalpha[i]
            ptau = all_ptau[i]
            tran = self.tran(zeta, phi, h)
            p0 = self.p0(tau)
            p_tran = self.tran_p(tran)
            eta = Z[L]
            y_likeli = self.y_likeli(x, y, beta, bs_value, eta, sigma, thres)
            w_likeli = self.w_likeli(eta, nu, psi)
            # temp_y_likeli = y_likeli[:, :, :, state]   #???????????????????
            for n in range(y.shape[0]):  # t=0
                cdll1 += np.log(p0[state[n, 0]])  ## p0
                cdll1 += torch.sum(y_likeli[n, 0, :, state[n, 0]])
                for t in range(1, y.shape[1]):  # t = 1:last
                    cdll1 += np.log(p_tran[n, t, state[n, t - 1], state[n, t]])  # pitus
                    cdll1 += torch.sum(y_likeli[n, t, :, state[n, t]])  # likelihood
            cdll1 += torch.sum(w_likeli)
            ##---------------------DIC 2------------------------------------------##
            phi_dic = 0
            zeta_dic = 0
            tau_dic = 0
            beta_dic = 0
            sigma_dic = 0
            thres_dic = 0
            nu_dic = 0
            psi_dic = 0
            for ind in range(dint):
                tau, accept_tau = self.update_tau(tau, state, c_tau, accept_tau)
                phi, accept_phi = self.update_phi(h, phi, zeta, state, c_phi, accept_phi, tau_phi)
                tau_phi = self.update_tau_phi(gamma_phi, phi)
                gamma_phi = self.update_gamma_phi(tau_phi)
                zeta, accept_zeta = self.update_zeta(zeta, state, phi, h, c_zeta, accept_zeta)
                # eta = Z[L]
                # # # # --------------------update varying coefficient----------------------------------------------------------------------
                beta, accept_beta = self.update_vc(y, x, state, beta, sigma, thres, bs_value, eta, ptau, c_beta,
                                                   accept_beta, tau_beta)  # 0.06
                thres, accept_thres = self.update_thres(y, x, state, beta, sigma, thres, bs_value, eta, c_thres,
                                                        accept_thres)  # 0.04
                ptau = self.update_ptau(beta)
                tau_beta = self.update_tau_beta(gamma_beta, sigma, beta)
                gamma_beta = self.update_gamma_beta(tau_beta)
                mean = self.update_mean(x, state, beta, thres, bs_value)
                sigma = self.update_sigma(y, mean, eta, state)
                state, sigma, beta, tau_beta, gamma_beta, sigma_beta, thres = self.label_switch(state, sigma, beta, tau_beta,
                                                                                                gamma_beta, ptau, thres,
                                                                                                bs_value)
                # # #--------------------------------------Update para associated random effect--------------------------------------
                nu = self.update_nu(Z, psi)
                psi, accept_psi = self.update_psi(Z, psi, nu, c_psi, accept_psi)
                tau_dic += tau/dint
                phi_dic += phi/dint
                zeta_dic += zeta/dint
                beta_dic += beta/dint
                thres_dic += thres/dint
                sigma_dic += sigma/dint
                nu_dic += nu/dint
                psi_dic += psi/dint
            y_likeli_dic = self.y_likeli(x, y, beta_dic, bs_value, eta, sigma_dic, thres_dic)
            tran_pos = self.tran(zeta_dic, phi_dic, h)
            p0_dic_pos = self.p0(tau_dic)
            p_tran_dic_pos = self.tran_p(tran_pos)
            w_likeli_dic = self.w_likeli(eta, nu_dic, psi_dic)
            cdll2 = 0
            for n in range(y.shape[0]):  # t=0
                cdll2 += np.log(p0_dic_pos[state[n, 0]])  ## p0
                cdll2 += torch.sum(y_likeli_dic[n, 0, :, state[n, 0]])
                for t in range(1, y.shape[1]):  # t = 1:last
                    cdll2 += np.log(p_tran_dic_pos[n, t, state[n, t - 1], state[n, t]])  # pitus
                    cdll2 += torch.sum(y_likeli_dic[n, t, :, state[n, t]])  # likelihood
            cdll2 += torch.sum(w_likeli_dic)
            m_cdll1 += cdll1 / iter
            m_cdll2 += cdll2 / iter
            one_iter_time = time.clock() - t0
        process = (r * iter + i) / (Rep * iter)
        if i % 10 == 0:
            print("%.3f seconds process time for one iter" % one_iter_time)
            print("%.3f seconds process time for  one iter" % one_iter_time, flush=True, file=open("DIC" + str(o) + ".txt", "a"))
            rtime = Rep * iter * one_iter_time * (1 - process) / 60
            print("Total Rep :%d, now rep: %d, Iter : %d, process: %.3f, need %.1f min to complete" % (
            Rep, r, i, process, rtime))
            print("Rep :%d, now rep: %d, Iter : %d, process: %.3f, need %.1f min to complete" % (
            Rep, r, i, process, rtime),
                  flush=True, file=open("DIC" + str(o) + ".txt", "a"))
        DIC = -4 * m_cdll1 + 2 * m_cdll2
        return DIC, m_cdll1, m_cdll2

























