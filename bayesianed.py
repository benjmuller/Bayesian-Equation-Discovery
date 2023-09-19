"""
Data driven Equation discovery algorithm (stable) - Bayesian Equation Discovery (BED)
Date: 2023-09-19
@author: Ben Muller
"""

import numpy as np

from scipy.integrate import odeint, solve_ivp

import torch
import gpytorch

from findiff import FinDiff

import jax.numpy as jnp
import jax.random as jrand

import numpyro 
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary

import matplotlib.pyplot as plt

##############################################################################
# Algorithm Outline:
#
# - Takes inputs X and U (first column of X is time derivative)
# - Calculates GP regression on U through self.GPRegression
# - Calculates derivatives of grid through self.dic and puts target functions into 
#   self.target and candidate functions into self.Pmat
# - MCMC with horeshoe prior and returns outcome in console. Note: if use_adam is True:
#   then for self.iters number of itterations, it loops through calculating MCMC then
#   minimising l2 norms of regression and equation discovery through optimising 
#   GP kernel parameters.
#
##############################################################################

# %%

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, X, U, likelihood):
        super(GPModel, self).__init__(X, U, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # 
        
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %%
class BED:
    
    def __init__(self, X, U, p_ord=1, d_ord=0, interact=True, grid_size=100, u_var=0.1, t_deriv = False, gp_train_itter=100, gp_lr=0.1, iters=1, adam_iters=30, use_adam=False, betas_min=0.05, include_const=False):           
        """
        X: Input nxp torch tensor - n datapoints, p parameters
        U: Output nxd torch tensor - n datapoints, d outputs
        p_ord: Order of polynomial included in candidate functions
        d_ord: Order of derivatives included in candidate functions
        interact: Boolian to include interactions between candidate functions (e.g. U[0] * dU[0]/dX[1])
        grid_size: number of points wanted in each input dimention to construct the grid for MCMC (e.g. for 3 dimentional input 100x100x100 grid)
        u_var: expected variance in the data (Note: data is standardised)
        t_deriv: Boolian to include time derivatives (i.e. first column in X) in the candidate functions (Note: use True if want to include derivatives for p=1)
        gp_train_itter: number of adam itterations in training gp
        iters: Number of "tuning" itterations (Note: see last bullet point of Algorithm outline for further details)
        adam_iters: Number of adam itterations in "tuning" (Note: see last bullet point of Algorithm outline for further details)
        use_adam: Boolian indicating using "tuning" (Note1: use iters=1 when use_adam=False. Note2: see last bullet point of Algorithm outline for further details)
        betas_min: minimum beta value to be included in the equation discovery output
        include_const: if include intercept in candidate functions
        """              
        try: # number of parameters
            self.param_num = X.shape[1]
        except IndexError:
            self.param_num = 1
                        
        self.data_num = X.shape[0] # number of datapoints
        
        self.out_num = U.shape[1] # number of output functions
        
        self.X = X # Data: rows datapoints, cols parameters
        self.U = U # Response: rows number of output functions, cols: datapoints
        
        self.Xmean = self.X.mean(0)
        self.Xstd = self.X.std(0)
        self.Umean = self.U.mean(0)
        self.Ustd = self.U.std(0)
        
        self.Xmin = X.min(0).values.detach().numpy()
        self.Xmax = X.max(0).values.detach().numpy()
        
        self.u_var = u_var # data expected variance
            
        self.training_iter = gp_train_itter # GP Regression learning itterations
        self.learning_rate = gp_lr # GP Regression learning rate
        
        self.t_deriv = t_deriv # True: include time derivative in candidate functions (use if want derivatives of one parameter functions). False: dont include time derivatives.
        
        self.grid_size = grid_size # Uniform Grid size for MCMC (each paramater has grid_size points)
        self.X_grid = np.vstack(list(map(np.ravel, np.meshgrid(*[np.linspace(self.Xmin[i], self.Xmax[i], self.grid_size) for i in range(self.param_num)], indexing="ij")))).T # Calculates grid and cooridinates in columns
        
        self.deriv_order = d_ord # derivative order
        self.poly_order = p_ord # polynomial order
        self.interactions = interact # True: inclue derivative, polynomial interactions. False: dont include interactions
        
        self.gplikelihood, self.gpmodel = self.GPR_calc() # GP over the training data
                        
        # Parameters for MCMC w/ Horseshoe prior
        self.lamb = None # Local variance
        self.tau = None # Global variance
        self.unscaled_beta = None # Mean (unscaled variance)
        self.beta = None # Mean (scaled variance)
        
        self.final_betas = []
        
        self.include_const = include_const
        self.dict_names = None # array containing strings of the mathematical formula of the corresponding value in self.Pmat
        self.target_names = None # array containing strings of the mathematical formula of the corresponding value in self.target
        self.dict_length = None # Number of functions in dictionary / potential functions
                                
        self.target, self.Pmat = self.dic(self.X_grid) # List of target functions and  dictionary functions
                        
        rng_key, rng_key_predict = jrand.split(jrand.PRNGKey(0)) # Random key for MCMC
        
        # Repeating inference through adam optimisation (Default off - still in beta)
        self.iters = iters # number of itterations of optimisation
        self.adamiters = adam_iters # number of adam itterations
        self.use_adam = use_adam 
        
        # MCMC inference + minimising l2 norms of regression and equation discovery
        for l in range(self.iters):
            for i, gpmodel in enumerate(self.gpmodel): # number of 1st order partial derivatives (Targets)
                # Running MCMC inference w/ Horeshoe prior
                nuts_kernel = NUTS(self.model)
                mcmc = MCMC(
                    nuts_kernel,
                    num_warmup=2000,
                    num_samples=2000,
                    progress_bar=False
                )
                mcmc.run(rng_key, jnp.asarray(self.target[i]), jnp.asarray(self.Pmat))
                # mcmc.print_summary(exclude_deterministic=False)
                
                # Saving final beta values
                if l == self.iters - 1:
                    posterior_samples = mcmc.get_samples()
                    summary_dict = summary(posterior_samples, group_by_chain=False)
                    betas = summary_dict["betas"]["mean"]
                    self.final_betas.extend([betas])                    
                
                # Minimising l2 norms
                if self.use_adam:    
                    posterior_samples = mcmc.get_samples()
                    summary_dict = summary(posterior_samples, group_by_chain=False)
                    betas = summary_dict["betas"]["mean"]
                    betas[abs(betas) < betas_min] = 0
                    betas = torch.from_numpy(betas)
                    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=self.learning_rate)
                    optimizer.zero_grad()
                    for j in range(self.adamiters):
                        optimizer.zero_grad()
                        loss = self.loss(i, betas)
                        loss.backward()
                        optimizer.step()
                        # print('iter', j, 'loss:{:.5f}'.format(loss.item()))
        
        if self.use_adam:    
            self.target, self.Pmat = self.dic(self.X_grid) # List of target functions and  dictionary functions
        

        # Printing discovered equation
        print("Discovered Equation(s):")
        for i, betas in enumerate(self.final_betas):
            indicies = np.asarray(abs(betas)>=betas_min).nonzero()[0]
            string = self.target_names[i] + " ="
            for j, index in enumerate(indicies):
                if j != len(indicies) - 1:
                    string +=  " {:.2f} ".format(betas[index]) + self.dict_names[index] + " +"
                else:
                    string += " {:.2f} ".format(betas[index]) + self.dict_names[index]
            print(string)
        print()
        
    def GPR_calc(self):
        """Calculates a GP Regression for each output (U)"""
        # Standardising X and U
        X = (self.X - self.Xmean.expand_as(self.X)) / (self.Xstd.expand_as(self.X))
        U = (self.U - self.Umean.expand_as(self.U)) / (self.Ustd.expand_as(self.U))
        
        models = []; likelihoods = []
        for i in range(self.out_num):
            U_temp = torch.reshape(U[:,i], (1, self.data_num))
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPModel(X, U_temp, likelihood)
            
            # Find optimal model hyperparameters
            model.train() 
            likelihood.train()
            
            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for i in range(self.training_iter):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, U_temp)
                loss.backward()
                optimizer.step()
                
            likelihood.eval(); model.eval()
            models.append(model); likelihoods.append(likelihood)
            
        return likelihoods, models
                
        
    def dic(self, X):
        """Calculates the array of the dictionary for the target function"""                        
        X = torch.from_numpy(X)
        X_std =  (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X))
        dict_temp = []; target_temp = []  # Stores dictionary values
        dict_names = []; target_names = [] # Stores dictionary values
        deriv_temp = []; deriv_names = [] # Stores derivative values
        dx = (self.Xmax - self.Xmin) / (self.grid_size - 1)
        
        for i, model in enumerate(self.gpmodel):
            # Getting GP regression mean
            f_mean = torch.reshape(model(X_std).mean, (self.grid_size ** self.param_num, 1))
            f_mean = (f_mean * self.Ustd[i].expand_as(f_mean) + self.Umean[i].expand_as(f_mean))
            f_mean = f_mean.detach().numpy()
                                                            
            # Calculating polynomial
            if self.poly_order == 1:
                dict_temp.extend([f_mean])
                dict_names.extend([f'U[{i}]'])
            else:
                dict_temp.extend([np.power(f_mean, j) for j in range(1, self.poly_order + 1)])
                dict_names.extend([f'U[{i}]^{j}' for j in range(1, self.poly_order + 1)])
            
            # Calculating derivatives            
            f_mean = np.reshape(f_mean, ([self.grid_size for _ in range(self.param_num)]))
            
            d1 = FinDiff(0, dx[0], 1)(f_mean).reshape((self.grid_size ** self.param_num, 1)) # Calculating (first order) time derivative 
            
            target_temp.extend([d1.reshape(-1,1)]) # Adding time derivative to target functions
            target_names.extend([f'dU[{i}]/dt'])
            
            # Adding first order derivatives if more than one parameter
            if self.deriv_order > 0:
                if self.param_num != 1:
                    for j in range(1, self.param_num):
                        df = FinDiff(j, dx[j], 1)(f_mean)
                        deriv_temp.extend([df.reshape((self.grid_size ** self.param_num, 1))])
                        deriv_names.extend([f'dU[{i}]/dX[{j}]'])
            
            # Adding second order and higher derivatives
            for deriv in range(2, self.deriv_order + 1):
                for j in range(1 - self.t_deriv, self.param_num):
                    df = FinDiff(j, dx[j], deriv)(f_mean)
                    deriv_temp.extend([df.reshape((self.grid_size ** self.param_num, 1))])
                    deriv_names.extend([f'd^{deriv}U[{i}]/dX[{j}]^{deriv}'])
        
                              
        # Calculating interactions
        
        if self.interactions is True:
            # Calculating products of polynomials for multiple outputs
            poly_int_temp = []; poly_int_names = []
            if self.out_num > 1:
                for i in range(self.out_num - 1):
                    interaction_ind = np.arange((i+1)*self.poly_order, self.out_num * self.poly_order)
                    for j in range(self.poly_order):
                        poly_int_temp.extend([dict_temp[i * self.poly_order + j] * dict_temp[k] for k in interaction_ind])
                        poly_int_names.extend([dict_names[i * self.poly_order + j] + '*' + dict_names[k] for k in interaction_ind])
            
            dict_temp.extend(poly_int_temp)
            dict_names.extend(poly_int_names)
            
            # Calculating polynomials * derivatives and derivatives * derivatives
            num_deriv_terms = len(deriv_temp)
            num_poly_terms = len(dict_temp)
            interaction_temp = []
            interaction_names = []
            for i in range(num_deriv_terms):
                for j in range(i + 1, num_deriv_terms):
                    interaction_temp.extend([deriv_temp[i] * deriv_temp[j]])
                    interaction_names.extend([deriv_names[i] + ' * ' + deriv_names[j]])
                for j in range(num_poly_terms):
                    interaction_temp.extend([deriv_temp[i] * dict_temp[j]])
                    interaction_names.extend([dict_names[j] + ' * ' + deriv_names[i]])
                    
        dict_temp.extend(deriv_temp)
        dict_names.extend(deriv_names)
        
        if self.interactions is True:
            dict_temp.extend(interaction_temp)
            dict_names.extend(interaction_names)
            
        if self.include_const:
            dict_temp.extend([np.ones((self.grid_size, 1))])
            dict_names.extend(['1'])
        
        self.dict_names = dict_names
        self.target_names = target_names
                    
        dictionary = np.hstack(dict_temp) 
        target = np.hstack(target_temp)
                
        self.dict_length = dictionary.shape[1] 
        
        return np.transpose(target), np.transpose(dictionary)
    
    def model(self, dU, X):
        """The probability prior (Horseshoe) for the spatial regression"""
        self.tau = numpyro.sample("tau", dist.HalfCauchy(scale=jnp.ones(1)))
        self.lamb = numpyro.sample("lamb", dist.HalfCauchy(scale=jnp.ones(self.dict_length)))
        self.unscaled_beta = numpyro.sample("unscaled betas", dist.Normal(0, jnp.ones(self.dict_length)))
        self.beta = numpyro.deterministic("betas", self.tau * self.lamb * self.unscaled_beta)
        mean = jnp.dot(self.beta, X)
        numpyro.sample('dU', dist.Normal(loc=mean, scale=self.u_var), obs=jnp.array(dU))
    
    def loss(self, gp_index, beta):
        """Returns l2 norm of regression and equation discovery"""
        u_norm = torch.sum((self.U - self.gpmodel[gp_index](self.X)[0]) ** 2)
        target, dictionary = self.dic(self.X_grid)
        diff_norm = np.sum((target[gp_index] - np.dot(beta, dictionary)) ** 2)
        return u_norm + diff_norm
    
    def return_Pmat(self):
        """Returns array of the candidate functions"""
        return self.Pmat
    
    def return_target(self):
        """Returns array of the target functions"""
        return self.target
    
    def return_grid(self):
        """Returns array of the target functions"""
        return self.X_grid

# %%

if __name__ == "__main__":
    
    print("Testing...")
    
    ###############################################################################
    ############################## Sine function ##################################
    ###############################################################################
    
    print('-' * 50)
    print("Sine function: (True: dU[0]/dt =  -1.00 d^3U[0]/dX[0]^3)\n")
    
    datalen = 15

    x1 = np.linspace(0,10,num=datalen)
    xtest = np.linspace(0,10,num=100)
    
    y1 = np.sin(x1)
    
    alg1 = BED(torch.from_numpy(np.array(x1).reshape(-1,1)), torch.from_numpy(np.array(y1).reshape(-1,1)), p_ord=1, d_ord=4, t_deriv=True, interact=False, betas_min=0.15)
    
    fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(8,4))
    
    ax1[0,0].plot(xtest, np.sin(xtest), linewidth=2.5)
    ax1[0,1].plot(x1, y1, "bo")
    ax1[0,1].plot(xtest,alg1.return_Pmat()[0], linewidth=2.5)
    ax1[0,1].plot(xtest, np.sin(xtest), "--", linewidth=2)
    ax1[0,2].plot(xtest,alg1.return_target()[0], linewidth=2.5)
    ax1[0,2].plot(xtest, np.cos(xtest), "--", linewidth=2)
    ax1[1,0].plot(xtest,alg1.return_Pmat()[1], linewidth=2.5)
    ax1[1,0].plot(xtest, - np.sin(xtest), "--", linewidth=2)
    ax1[1,1].plot(xtest,alg1.return_Pmat()[2], linewidth=2.5)
    ax1[1,1].plot(xtest,- np.cos(xtest), "--", linewidth=2)
    ax1[1,2].plot(xtest,alg1.return_Pmat()[3], linewidth=2.5)
    ax1[1,2].plot(xtest, np.sin(xtest), "--", linewidth=2)
        
    ylabs = ["Sin(x)", "U=Sin(x)", "dU=Cos(x)", "d2U=-Sin(x)", "d3U=-Cos(x)", "d4U=Sin(x)"]
    
    fig1.suptitle("Plot of GP Regression derivatives for Sine")
    
    for i, ax in enumerate(ax1.flat):
        ax.set_ylabel(ylabs[i]) 
        
    fig1.tight_layout()
        
    ###############################################################################
    ############################## Van der Pol ####################################
    ###############################################################################
    
    print('-' * 50)
    print("Van der Pol: (True: dU[0]/dt = 1.00 U[1]^1, dU[1]/dt = -1.00 U[0]^1 + 2.50 U[1]^1 + -2.50 U[0]^2*U[1]^1)\n")
    
    def vdp(t, z, mu):
        x, y = z
        return [y, mu*(1 - x**2)*y - x]
    
    a, b = 0, 8
    mu = 2.5
    t = np.linspace(a, b, 50)
    ts = np.linspace(a, b, 100)
    sol = solve_ivp(vdp, [a, b], [1, 1], t_eval=t, args=(mu,), rtol=1e-8)
    sols = solve_ivp(vdp, [a, b], [1, 1], t_eval=ts, args=(mu,), rtol=1e-8)
    
    alg2 = BED(torch.from_numpy(t.reshape(-1,1)), torch.from_numpy(np.array([sol.y[0], sol.y[1]]).T), p_ord=4, d_ord=0)
    target2 = alg2.return_target()
    dic2 = alg2.return_Pmat()
    
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    ax2[0].plot(ts, sols.y[0], linewidth=3)
    ax2[0].plot(ts, dic2[0], "--", linewidth=2)
    ax2[0].plot(ts, target2[0], linewidth=3)
    ax2[0].plot(ts, dic2[4], "--", linewidth=2)
    
    ax2[1].plot(ts, sols.y[1], linewidth=3)
    ax2[1].plot(ts, dic2[4], "--", linewidth=2)
    ax2[1].plot(ts, target2[1], linewidth=3)
    ax2[1].plot(ts, - dic2[0] + 2.5 * dic2[4] - 2.5 * dic2[12], "--", linewidth=2)
    
    ax2[0].legend(["True x", "GPR x", "Target (GPR)", "Equation Discovery (GPR)"], loc="upper left")
    ax2[1].legend(["True y", "GPR y", "Target (GPR)", "Equation Discovery (GPR)"], loc="upper left")
    
    fig2.tight_layout()
        
    ###############################################################################
    ############################ Burger's Equation ################################
    ###############################################################################    
    
    print('-' * 50)
    print("Burger's Equation: (True: dU[0]/dt = 0.10 d^2U[0]/dX[1]^2 + -1.00 U[0] * dU[0]/dX[1])\n")
    
    n_data = 30
    x = np.linspace(0,10,num=n_data)
    t = np.linspace(0,8,num=n_data)
    X = np.array([np.tile(t,n_data),np.repeat(x,n_data)])
    mu = 1; nu = 0.1
    u0 = np.exp(-(x-4) ** 2)
    k = 2*np.pi*np.fft.fftfreq(n_data, d = 10/n_data)
    
    n_datatest = 100
    xtest = np.linspace(0,10,num=n_datatest)
    ttest = np.linspace(0,8,num=n_datatest)
    Xtest = np.array([np.tile(ttest,n_datatest),np.repeat(xtest,n_datatest)])
    u0test = np.exp(-(xtest-4) ** 2)
    ktest = 2*np.pi*np.fft.fftfreq(n_datatest, d = 10/n_datatest)
    
    def burg_system(u,t,k,mu,nu):
        #Spatial derivative in the Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j*k*u_hat
        u_hat_xx = -k**2*u_hat
        
        #Switching in the spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)
        
        #ODE resolution
        u_t = -mu*u*u_x + nu*u_xx
        return u_t.real
        
    
    # PDE resolution (ODE system resolution)
    U = odeint(burg_system, u0, t, args=(k,mu,nu,)).T
    Utest = odeint(burg_system, u0test, ttest, args=(ktest,mu,nu,)).T
    
    alg3 = BED(torch.from_numpy(X.T), torch.from_numpy(U.reshape(-1,1)), p_ord=1, d_ord=4)
    dic3 = alg3.return_Pmat()
    target3 = alg3.return_target()
    
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    ax3[0].plot(x, U.reshape(1,-1)[0][0::n_data], "bo")
    ax3[0].plot(xtest, Utest.reshape(1,-1)[0][0::n_datatest], "r")
    ax3[0].plot(xtest, target3[0,:n_datatest], color="y")
    ax3[0].plot(xtest, 0.1 * dic3[2,:n_datatest] - dic3[8,:n_datatest])
    ax3[1].plot(x, U.reshape(1,-1)[0][n_data - 1::n_data], "bo")
    ax3[1].plot(xtest, Utest.reshape(1,-1)[0][n_datatest - 1::n_datatest], "r")
    ax3[1].plot(xtest, target3[0,-n_datatest:], color="y")
    ax3[1].plot(xtest, 0.1 * dic3[2,-n_datatest:] - dic3[8,-n_datatest:])
        
    ax3[0].legend(["Data", "GP Regression U (True)", "Target (GPR)", "Equation Discovery (GPR)"])
    ax3[1].legend(["Data", "GP Regression U (True)", "Target (GPR)", "Equation Discovery (GPR)"])
    
    ax3[0].set_xlabel("t=0")
    ax3[1].set_xlabel("t=8")
    
    fig3.tight_layout()
     
