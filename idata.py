from __future__ import division
from math import pi, sqrt
from torch.distributions import Normal, Beta, Gamma, MultivariateNormal
from scipy.interpolate import BSpline
# from itertools import count
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import torch
import time
import numpy.random as nrd
import tool
import update

N = 500
NT = 4
NX = 3
# NV = NX + 1   # varying coefficient (include intercept)
NV = NX   # varying coefficient (include intercept)
NK = 6   # number of knots
NS = 2
K = 2  # order (more explicit, degree (cubic)  (so number of spline is NK+K)
D = 3  # degree = K + 1
NH = 3      # transition model
NB = NK + K - 1        # number of b-spline 
Rep = 1
Iter = 5000
burnin = 4000
NG = 100               # number of grid
#  -------------------------For saving parameters------------------------------
all_beta = torch.zeros(Rep, Iter, NS, NV, NB)
all_ptau = torch.zeros(Rep, Iter, NS, NV)   # penalty parameter for bspline
#---------------------transition model------------------------
all_tau = torch.zeros(Rep, Iter, NS-1)
all_phi = torch.zeros(Rep, Iter, NX)
all_zeta = torch.zeros(Rep, Iter, NS, NS)
#--------------------------About data storage----------------------------------
all_t_state = torch.zeros(Rep, N, NT)
all_x = torch.zeros(Rep, N, NT, NX+1, NG)
all_h = torch.zeros(Rep, N, NT, NH)
all_y = torch.zeros(Rep, N, NT, NG)
all_eta = torch.zeros(Rep, N, NT, NG)
torch.set_default_tensor_type(torch.DoubleTensor)
for r in range(Rep):
    nrd.seed(66+3)
    x = np.zeros(shape=[N, NT, NX+1, NG])
    # x[:, :, 0] = 1 # intercept
    # x[:, :, 0] = np.tile(nrd.binomial(1, 0.4, N)[:, np.newaxis, np.newaxis], (1, NT, NG))   # time-invariant and grid-invariant covariates
    # x[:, :, 0] = np.tile(nrd.binomial(1, 0.4, N)[:, np.newaxis, np.newaxis], (1, NT, NG))
    # x[:, :, 2] = nrd.uniform(0, 1, (N, NT))
    x[:, :, 0] = 1
    x[:, :, 1] = ss.t.rvs(3, size=(N, NT, NG))
    x[:, :, 2] = nrd.normal(0, 1, (N, NT, NG))
    x[:, :, 3] = nrd.uniform(-1, 1, (N, NT, NG))
    x = torch.from_numpy(x)   # numpy to tensor
    #####----------------------------Normalize x --------------
    x_mean = torch.mean(x[:, :, 1:], dim=(0, 1))   # NX * NG
    x_var = torch.std(x[:, :, 1:], dim=(0, 1))
    x_nor = (x[:, :, 1:] - torch.unsqueeze(torch.unsqueeze(x_mean, 0), 0)) / torch.unsqueeze(torch.unsqueeze(x_var, 0), 0)
    x[:, :, 1:] = x_nor
    all_x[r] = x
    ##--------------------------------------Transition  model----------------------------------------------
    h = np.zeros(shape=[N, NT, NH])
    h[:, :, 0] = np.tile(nrd.binomial(1, 0.4, N)[:, np.newaxis], (1, NT))   # time-invariant covariates
    h[:, :, 1] = nrd.uniform(0, 1, (N, NT))
    h[:, :, 2] = nrd.normal(0, 1, (N, NT))
    h = torch.from_numpy(h)
    all_h[r] = h
    t_tau = torch.Tensor([0])
    knots = np.zeros(shape=[NK + 2 * K])
    # knots[K:(NK + K)] = np.percentile(t, np.linspace(0, 100, NK + 1, endpoint=False)[1:])
    knots[K:(NK + K)] = np.linspace(0, 1, NK, endpoint=True)
    knots[:K] = 0
    knots[NK + K:] = 1
    # knots = torch.from_numpy(knots)
    tool1 = tool.Tools(N, NT, NS, K, NB, knots)
    t_p0 = tool1.p0(t_tau)  # the initial prob for each state (time 0)
    t_zeta = torch.Tensor([[-1, 0], [1, 0]])
    t_phi = torch.Tensor([1, -1, 0])
    t_tran = tool1.tran(t_zeta, t_phi, h)
    t_tran_p = tool1.tran_p(t_tran)
    # p_sum = np.sum(t_tran_p, axis=3)    # to check the probabiliy sum to 1
    t_state = tool1.tran_state(t_tran_p, t_p0)
    all_t_state[r] = t_state
    state0_pro = torch.sum(t_state == 0) / (t_state.shape[0] * t_state.shape[1])
    state1_pro = torch.sum(t_state == 1) / (t_state.shape[0] * t_state.shape[1])
    #----------------------------Conditional model---------------------------------------
    #----------First to test whether recursion is OK-------------------------------------
    t_sigma = torch.Tensor([0.25, 0.25])
    grid = torch.linspace(0, 1, NG)
    dd = torch.unsqueeze(grid, 1) - torch.unsqueeze(grid, 0) # NG * NG
    H_d = torch.exp(-torch.abs(dd)) # distance for the covariance
    y = torch.empty(N, NT, NG)
    location0 = torch.where(t_state == 0)
    location1 = torch.where(t_state == 1)
    t_beta = torch.zeros(NS, NX + 1, NG)                           #-----------------zxx
    t_beta[0, 0] = grid
    # t_beta[0, 1] = torch.pow(grid - 0.3, 2) * (grid > 0.3)
    t_beta[0, 1] = 1 * (torch.cos(2.5*pi*grid)+1) * (grid > 0.4)
    t_beta[0, 2] = torch.sin(2 * pi * grid) + 1
    t_beta[1, 0] = 1 - grid
    t_beta[1, 1] = torch.cos(2 * pi * grid)
    t_beta[1, 2] = torch.sin(pi * (grid-0.5)) * (grid <= 0.5)
    # t_beta[1, 3] = torch.square(2 * pi * grid)
    # t_beta[1, 0] = np.sqrt(2) * np.sin(4 * pi *grid)
    # t_beta[1, 1] = np.sin(pi * grid)
    # t_beta[1, 2] = np.cos(2 * pi * grid)
    # # #---------------------Plot the VC---------------------
    # plt.subplot(121)
    # plt.plot(grid, t_beta[0, 0], label='Intercept')
    # plt.plot(grid, t_beta[0, 1], label='Coeff1')
    # plt.plot(grid, t_beta[0, 2], label='Coeff2')
    # plt.subplot(122)
    # plt.plot(grid, t_beta[1, 0], label='Intercept')
    # plt.plot(grid, t_beta[1, 1], label='Coeff1')
    # plt.plot(grid, t_beta[1, 2], label='Coeff1')
    # plt.show()
    #---------------------------------------subject-specific devition-------------------------------------------
    #---------------------------------------subject-specific devition-------------------------------------------
    # eta_mean0 = torch.ones(NG)
    # eta_cov0 = torch.eye(NG) * 0.5
    # eta_mean1 = torch.zeros(NG)
    # eta_cov1 = torch.eye(NG)
    # t_eta = MultivariateNormal(eta_mean0, eta_cov0).sample((N, NT)) + MultivariateNormal(eta_mean1, eta_cov1).sample((N, NT))
    s1 = torch.normal(0, sqrt(1.2), (N, NT))
    s2 = torch.normal(0, sqrt(0.6), (N, NT))
    # t_eta = torch.unsqueeze(s1, 2) * sqrt(2) * torch.sin(2*pi*grid) + torch.unsqueeze(s2, 2) * (2) * torch.cos(2*pi*grid)
    # t_eta = Gamma(3, 5).sample((N, NT, NG))
    # mid_grid = torch.Tensor(math.ceil((0 + 99) / 2))
    # grid = torch.linspace(0, 1, NG)
    dd = torch.unsqueeze(grid, 1) - torch.unsqueeze(grid, 0)  # NG * NG
    H_d = torch.exp(-torch.abs(dd))  # distance for the covariance
    # H_d = torch.exp(-torch.square(dd))  # distance for the covariance
    sigma_s = torch.Tensor([0.5])
    psi_1, psi_2 = 0.25, 0.36
    cov_1 = torch.square(sigma_s) * torch.pow(H_d, psi_1)
    cov_2 = torch.square(sigma_s) * torch.pow(H_d, psi_2)
    label = torch.Tensor(nrd.binomial(1, 0.7, (N, NT, 1)))
    eta_1 = MultivariateNormal(torch.zeros(NG), cov_1).sample((N, NT))
    eta_2 = MultivariateNormal(torch.zeros(NG), cov_2).sample((N, NT))
    t_eta = label * eta_1 + (1-label) * eta_2
    all_eta[r] = t_eta
    # t_mean = torch.sum(t_beta[0] * torch.unsqueeze(x[location0[0], location0[1]], 2), 1)
    t_mean = torch.zeros(N, NT, NG)
    t_mean[location0[0], location0[1]] = torch.sum(t_beta[0] * x[location0[0], location0[1]], 1)
    t_mean[location1[0], location1[1]] = torch.sum(t_beta[1] * x[location1[0], location1[1]], 1)
    sample_mean = t_mean + t_eta
    y[location0[0], location0[1]] = torch.normal(sample_mean[location0[0], location0[1]], t_sigma[0].sqrt())   # Need to modify \eta
    y[location1[0], location1[1]] = torch.normal(sample_mean[location1[0], location1[1]], t_sigma[1].sqrt())
    all_mean = torch.sum(torch.unsqueeze(torch.unsqueeze(t_beta, 1), 1) * torch.unsqueeze(x, 0), 3)   #calculate the mean at each grid   # NS * N * NT * NG
    ty_likeli = torch.zeros(N, NT, NG, NS)                         # calculate the likelihood value at each grid   # N * NT * NG * NS
    ty_likeli[:, :, :, 0] = Normal(all_mean[0, :, :, ], t_sigma[0].sqrt()).log_prob(y)
    ty_likeli[:, :, :, 1] = Normal(all_mean[1, :, :, ], t_sigma[1].sqrt()).log_prob(y)
    all_tylikeli = torch.sum(ty_likeli, 2)
    diff_ylikeli = all_tylikeli[:,:,0] - all_tylikeli[:,:,1]
    # torch.sum(torch.abs(diff_ylikeli) < 1)
    all_y[r] = y
    print("End of one replication")
torch.save(all_x, "x.pt")
torch.save(all_y, "y.pt")
torch.save(all_t_state, 't_state.pt')
torch.save(all_h, "h.pt")
torch.save(all_eta, "t_eta.pt")
