from __future__ import division
from scipy.interpolate import BSpline
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import math
import time
import matplotlib.pyplot as plt


class Tools(object):
    def __init__(self, N, NT, NS, K, NB, knots):
        self.N = N
        self.NT = NT
        self.NS = NS
        self.K = K   # order
        self.NB = NB
        self.knots = knots

    # def p0(self,  tau):
    #     ## p0 is the initial state
    #     p0 = np.ones(self.NS)
    #     p0[0] = math.exp(tau[0])/(1+math.exp(tau[0]))
    #     # for s in range(self.NS-1):
    #     #     p0[-1] *= 1/(1+math.exp(tau[s]))
    #     p0[-1] = 1 / (np.cumprod(1+np.exp(tau[:])))[-1]
    #     for i in range(1, self.NS-1):
    #         # for m in range(0, i):
    #         #     p0[i] *= 1/(1+ math.exp(tau[m]))
    #         p0[i] = 1/ np.cumprod(1 + np.exp(tau[:i]))[-1]
    #         p0[i] *= math.exp(tau[i])/(1+math.exp(tau[i]))
    #     return p0


    def p0(self,  tau):
        ## p0 is the initial state
        p0 = torch.ones(self.NS)
        p0[0] = torch.exp(tau[0])/(1+torch.exp(tau[0]))
        # for s in range(self.NS-1):
        #     p0[-1] *= 1/(1+math.exp(tau[s]))
        p0[-1] = 1 / torch.cumprod(1+torch.exp(tau), 0)[-1]
        for i in range(1, self.NS-1):
            # for m in range(0, i):
            #     p0[i] *= 1/(1+ math.exp(tau[m]))
            p0[i] = 1 / torch.cumprod(1 + torch.exp(tau[:i]), 0)[-1]
            p0[i] *= torch.exp(tau[i])/(1+torch.exp(tau[i]), 0)
        return p0

    # calculate zeta + phi * d
    # def tran(self, zeta, phi, d):
    #     #-------------------------------------------Args: w and d are all N * NT * ND---------------------------
    #     tran = np.zeros(shape=[d.shape[0], d.shape[1], self.NS, self.NS])
    #     # n_d = np.concatenate((d, w), 2)  #N * NT
    #     phid = np.sum(phi * d, axis=2) # N * NT
    #     for u in range(self.NS):
    #         for s in range(self.NS):
    #             tran[:, :, u, s] = zeta[u, s] + phid
    #     return tran

    def tran(self, zeta, phi, d):
        # -------------------------------------------Args: w and d are all N * NT * ND---------------------------
        tran = torch.zeros(d.shape[0], d.shape[1], self.NS, self.NS)
        # n_d = np.concatenate((d, w), 2)  #N * NT
        phid = torch.sum(phi * d, dim=2)  # N * NT
        for u in range(self.NS):
            for s in range(self.NS):
                tran[:, :, u, s] = zeta[u, s] + phid
        return tran

    # def tran_p(self, tran):
    #     ## transition probability pitus
    #     p_tran = np.ones([tran.shape[0], tran.shape[1], self.NS, self.NS], dtype = float)
    #     #p_tran[:, 0, :, :] = 0
    #     for i in range(tran.shape[0]):
    #         for t in range(1, tran.shape[1]):
    #             for u in range(self.NS):
    #                 p_tran[i, t, u, 0] = math.exp(tran[i, t, u, 0])/(1+math.exp(tran[i, t, u, 0]))
    #                 for s1 in range(self.NS-1):
    #                     p_tran[i, t, u, -1] *= 1/(1+math.exp(tran[i, t, u, s1]))
    #                 for s in range(1, self.NS-1):
    #                     for s2 in range(s):
    #                         p_tran[i, t, u, s] *= 1/(1+math.exp(tran[i, t, u, s2]))
    #                     p_tran[i, t, u, s] *= math.exp(tran[i, t, u, s]) / (1 + math.exp(tran[i, t, u, s]))
    #     p_sum = np.sum(p_tran, axis=3)
    #     return p_tran

    # def tran_p(self, tran):
    #     ## transition probability pitus
    #     p_tran = torch.ones(tran.shape[0], tran.shape[1], self.NS, self.NS)
    #     #p_tran[:, 0, :, :] = 0
    #     for i in range(tran.shape[0]):
    #         for t in range(1, tran.shape[1]):
    #             for u in range(self.NS):
    #                 p_tran[i, t, u, 0] = torch.exp(tran[i, t, u, 0])/(1+torch.exp(tran[i, t, u, 0]))
    #                 for s1 in range(self.NS-1):
    #                     p_tran[i, t, u, -1] *= 1/(1+torch.exp(tran[i, t, u, s1]))
    #                 for s in range(1, self.NS-1):
    #                     for s2 in range(s):
    #                         p_tran[i, t, u, s] *= 1/(1+torch.exp(tran[i, t, u, s2]))
    #                     p_tran[i, t, u, s] *= torch.exp(tran[i, t, u, s]) / (1 + torch.exp(tran[i, t, u, s]))
    #     # p_sum = torch.sum(p_tran, dim=3)
    #     return p_tran

    def tran_p(self, tran):
        ## transition probability pitus
        p_tran = torch.ones(tran.shape[0], tran.shape[1], self.NS, self.NS)
        #p_tran[:, 0, :, :] = 0
        for u in range(self.NS):
            p_tran[:, :, u, 0] = torch.exp(tran[:, :,  u, 0])/(1+torch.exp(tran[:, :, u, 0]))
            # for s1 in range(self.NS-1):
            p_tran[:, :, u, -1] = 1/(torch.cumprod(1+torch.exp(tran[:, :, u, :-1]), -1)[:, :, -1])  # exclude itself
            for s in range(1, self.NS-1):
                # for s2 in range(s):
                #     p_tran[:, :, u, s] *= 1/(1+torch.exp(tran[:, :, u, s2]))
                p_tran[:, :, u, s] *= 1/(torch.cumprod(1+torch.exp(tran[:, :, u, :s], -1)[:, :, -1]))
                p_tran[:, :, u, s] *= torch.exp(tran[:, :, u, s]) / (1 + torch.exp(tran[:, :, u, s]))
        # p_sum = torch.sum(p_tran, dim=3)
        return p_tran


    # def tran_state(self, tran_p, p0):
    #     state = np.zeros(shape=[tran_p.shape[0], tran_p.shape[1]],dtype = int )
    #     for i in range(state.shape[0]):
    #         state[i, 0] = np.random.choice(range(self.NS), 1, p=p0)
    #         for t in range(1, state.shape[1]):
    #             state[i, t] = np.random.choice(range(0, self.NS), 1, p=tran_p[i, t, state[i, t-1], :])
    #     return state


    def tran_state(self, tran_p, p0):
        state = torch.zeros(tran_p.shape[0], tran_p.shape[1], dtype=int)
        state[:, 0] = torch.multinomial(p0, tran_p.shape[0], replacement=True)
        for t in range(1, state.shape[1]):
            x_axis = torch.arange(0, tran_p.shape[0])
            state[:, t] = torch.squeeze(torch.multinomial(tran_p[x_axis, t, state[:, t-1], :], 1), 1)
        return state


    # def bs_value(self, t):
    #     #--------Args: t (N) ;  return individual value(vector) of basis at t  (N * NB)
    #     bs_basis = np.zeros(shape=[self.NB, t.shape[0]])
    #     for n in range(self.NB):
    #         bs_c = np.zeros(shape=[self.NB])  # coefficient of basis
    #         bs_c[n] = 1
    #         sp_t = BSpline(self.knots, bs_c, self.K)
    #         bs_basis[n] = sp_t(t)
    #     return bs_basis

    def bs_value(self, t):
        #--------Args: t (N) ;  return individual value(vector) of basis at t  (N * NB)
        bs_basis = np.zeros( shape= [self.NB, t.shape[0]])
        for n in range(self.NB):
            bs_c = np.zeros(shape=[self.NB])  # coefficient of basis
            bs_c[n] = 1
            sp_t = BSpline(self.knots, bs_c, self.K)
            bs_basis[n] = sp_t(t)
        return bs_basis

    def soft_thres(self, value, thres):
        #---------------soft thresholding function------------------------
        return torch.sign(value) * torch.maximum(torch.Tensor([0.0]), torch.abs(value) - thres)

    def estimated_state(self, all_state, rep, burnin):
        estimated_state = torch.ones(rep, self.N, self.NT) * 99
        state_number = torch.ones(rep, self.N, self.NT, self.NS) * 99  # each state number
        for k in range(self.NS):
            state_number[:, :, :, k] = torch.sum(all_state[:, burnin:] == k, 1)
        estimated_state = state_number.argmax(axis=3)
        return estimated_state

    # def rep_sd(self, true, estimated, burnin):
    #     estimated = estimated[:, burnin:, ]
    #     sd = np.sqrt(np.sum(np.power(estimated - true, 2), axis=1) / estimated.shape[1])
    #     return sd

    def rep_sd(self, true, estimated, burnin):
        estimated = estimated[burnin:, ]
        sd = torch.sqrt(torch.sum(torch.square(estimated - true), axis=0) / estimated.shape[0])
        return sd


    def f_mean(self, bs, OT):
        sp = BSpline(self.knots, bs, self.K)
        f_mean = np.mean(sp(OT))
        return f_mean

    # def banded(self, g, N):
    #     """Creates a `g` generated banded matrix with 'N' rows"""
    #     n = len(g)
    #     T = torch.zeros(N, N + n - 1)
    #     for x in range(N):
    #         T[x][x:x + n] = g
    #     return T

    def banded(self, g, N):
        """Creates a `g` generated banded matrix with 'N' rows"""
        n = len(g)
        T = np.zeros((N, N + n - 1))
        for x in range(N):
            T[x][x:x + n] = g
        return T



    # =====================basis for bspline=====================
    # def B(x, k, i, t):
    #     if k == 0:
    #         return 1.0 if t[i] <= x < t[i + 1] else 0.0
    #     if t[i + k] == t[i]:
    #         c1 = 0.0
    #     else:
    #         c1 = (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t)
    #     if t[i + k + 1] == t[i + 1]:
    #         c2 = 0.0
    #     else:
    #         c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t)
    #     return c1 + c2



    #----------------------for analysis-------------------------------------------------
    # def hrep_rmse(self, estimated, true, burnin):  # for parameter
    #     estimated = estimated[:, burnin:, ]
    #     rmse = np.sqrt(np.sum(np.power(estimated - true[np.newaxis, np.newaxis, :], 2), axis=1) / estimated.shape[1])
    #     return np.mean(rmse, 0)

    def hrep_rmse(self, estimated, true, burnin):  # for parameter
        estimated = estimated[:, burnin:, ]
        rmse = torch.sqrt(torch.sum(torch.pow(estimated - torch.unsqueeze(torch.unsqueeze(true, 0), 0), 2), 1) / estimated.shape[1])
        return torch.mean(rmse, 0)


    def hrep_sd(self, estimated, burnin):    # for parameter
        estimated = estimated[:, burnin:, ]
        mean = np.mean(estimated[:, burnin:],1)
        sd = np.sqrt(np.sum(np.power(estimated - mean[:, np.newaxis], 2), axis=1) / estimated.shape[1])
        return np.mean(sd, 0)

    def rep_rmse(self, estimated, true, burnin):  # for effect
        estimated = estimated[:, burnin:, ]
        rmse = np.sqrt(np.sum(np.power(estimated - true[:, np.newaxis,], 2), axis=1) / estimated.shape[1])
        return rmse


    # def rep_sd(self, estimated, burnin):    # for effect
    #     estimated = estimated[:, burnin:, ]
    #     mean = np.mean(estimated[:, burnin:],1)
    #     sd = np.sqrt(np.sum(np.power(estimated - mean[:, np.newaxis], 2), axis=1) / estimated.shape[1])
    #     return np.mean(sd, 0)

    def cov_rate(self, t_value, p_value):
        ##------------------------t_value : shape:Rep------------------------
        ##------------------------p_value: shape: Rep * 3  # 0: true; 1: 5%percent; 2: 95% percent--------------------------##
        count = 0
        for i in range(p_value.shape[1]):
            if t_value[i] <= p_value[2, i] and t_value[i] >= p_value[1, i]:
                count += 1
        return count / t_value.shape[0]


    def hcov_rate(self, t_value, p_value):  # for vector
        ##------------------------t_value : shape:Rep------------------------
        ##------------------------p_value: shape: Rep * 3  # 0: true; 1: 5%percent; 2: 95% percent--------------------------##
        count = np.zeros_like(t_value)
        for j in range(t_value.shape[0]):
            for i in range(p_value.shape[1]):
                if t_value[j] <= p_value[1, i, j] and t_value[j] >= p_value[0, i, j]:
                    count[j] += 1
        return count / p_value.shape[1]

    # def large_square(self, y, l=100):
    #     # Args: y is p * n matrix, output: y' * y; l is the segmentation arg ( seg y to l parts); large l is memory-friendly
    #     # p is too large
    #     lp = int(y.shape[0] / l)         # each y_l is lp * n
    #     y_square = np.zeros(shape=[y.shape[1], y.shape[1]])
    #     for i in range(l):
    #         y_l = y[lp * i:lp * (i+1)]
    #         y_square += np.matmul(np.transpose(y_l), y_l)
    #     return y_square


    def large_square(self, y, l=100):
        # Args: y is p * n matrix, output: y' * y; l is the segmentation arg ( seg y to l parts); large l is memory-friendly
        # p is too large
        lp = int(y.shape[0] / l)         # each y_l is lp * n
        y_square = torch.zeros(y.shape[1], y.shape[1])
        for i in range(l):
            y_l = y[lp * i:lp * (i+1)]
            y_square += torch.mm(y_l.t(), y_l)
        return y_square

    # def large_square(self, y, l=100):
    #     # Args: y is nb * ng matrix, output: y * y'; l is the segmentation arg ( seg y to l parts); large l is memory-friendly
    #     # p is too large
    #     lp = int(y.shape[1] / l)         # each y_l is lp * n
    #     y_square = torch.zeros(y.shape[0], y.shape[0])
    #     for i in range(l):
    #         y_l = y[:, lp * i:lp * (i+1)]
    #         y_square += torch.mm(y_l, torch.transpose(y_l))
    #     return y_square

    def num_socre(self, eigenvalue, per):
        # Args: eigenvalue shape NP * TT (NP:number of predictors), per:percentage
        # Return the number of eigenvalues/ eigenimage of each predictor based on variance explained
        sum_var = np.sum(eigenvalue, 1)
        cu_per = np.cumsum(eigenvalue, 1) / sum_var[:, np.newaxis]
        num = np.sum(cu_per < per, 1)
        return num



    def plot_3d(self, voxel):
        # v_shape = voxel
        x, y, z = np.indices(np.array(voxel.shape))
        colorsvalues = np.empty(voxel.shape, dtype=object)
        alpha = 1
        # mycolormap = plt.get_cmap('hsv')
        # mycolormap = plt.get_cmap('winter')
        mycolormap = plt.get_cmap('cool')
        relative_value = np.round(voxel/np.max(voxel),1)
        # 透明度,视显示效果决定
        for i in range(0, voxel.shape[0]):
            for j in range(0, voxel.shape[1]):
                for k in range(0, voxel.shape[2]):
                    tempc = mycolormap(relative_value[i][j][k])
                    # tempc为tuple变量,存储当前数值的颜色值(R,G,B,Alpha)
                    colorreal = (tempc[0], tempc[1], tempc[2], alpha)
                    # tuple为不可变数据类型,所以替换自定义alpha值时需要重新定义
                    colorsvalues[i][j][k] = colorreal
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        b = time.time()
        ax.voxels(voxel, facecolors=colorsvalues)
        plt.show()
        e = time.time() - b
        print("Elapse %.3f"%e)





