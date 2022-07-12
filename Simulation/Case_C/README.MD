Overview

Simulation Case C extends from case B with a sampled population similar to that of the NBDPS case study population. A total of 10 subpopulations with respective sizes $n_s = (1600,1500,1500,1200,600,1000,1200,1200,1100,900)$ were generated. Each subpopulation was assigned to one local profile, with the exception of subpopulations 4 and 8. These two subpopulations were randomly split between two local profiles each. Global and local patterns were generated as in Cases A and B with an expected modal response probability of 0.85, and 0.05 otherwise. Allocation probability vector to the global level $(Pr(G_{ij}=1|s_i)$ for $s= (1,2,\ldots, 10)$ was defined as (0.7, 0.5,0.6, 0.4, 0.4, 0.6, 0.55, 0.55, 0.7, 0.45). Observed outcome values were generated as in Cases A and B, where $\xi = (0.8, 0.9, 1, 1.1, 1.2, 1.3, 0.7, 0.6, 1.4, 0.5, 1, -2, -1)$.  

Run Code

Input data source: sim_SRPCdataC250.mat

sim_sRPCC250.m - The fixed and adaptive sampling of the supervised RPC is implemented and condensed into a single code.

Output files - py_simResults_C250.mat
py_pred - Nx1 predicted probability vector of response outcome
pred_ci - Nx1 predicted global assignment for each subject calculated from posterior median model parameter estimates
pred_nu - Sxp matrix of posterior probability of a given variable being allocated to the global or local level for each subpopulation
DIC - Deviance information criterion as specified in Celeux, 2006
DIC_star - deviance information version 6 to penalize for data complexity as specified in Celeux, 2006
t_I0 - Sxp matrix of posterior mode for each exposure variable derived for each cluster
y_mse - mean squared error comparing true and predicted response outcome
nu_mse - mean squared error comparing true and predicted probability matrix of (global/local) variable allocation
G52_derivations.png - plot figure of median posterior probability matrix of predicted and true 
ν
theta0sRPC_simC250.fig - matlab figure illustrating posterior modal pattern derived for each cluster (t_I0)
theta0_simC250.fig - matlab figure illustrating true modal cluster pattern
sim_LCAprobitC250.m - The Gibbs sampler of the supervised LCA is implemented here.

Output files - simLCAprobitResults_C250.mat
theta0_med - Nx1 predicted probability vector of response outcome
pi_med - Kx1 posterior median probability of cluster assignment
pred_ci - Nx1 predicted global assignment for each subject calculated from posterior median model parameter estimates
xi_ci - credible interval of regression coefficients
dic_reg - Deviance information criterion as specified in Celeux, 2006
dic_star - deviance information version 6 to penalize for data complexity as specified in Celeux, 2006
t_I0 - Sxp matrix of posterior mode for each exposure variable derived for each cluster
y_mse - mean squared error comparing true and predicted response outcome
theta0slca_simC250.fig - matlab figure illustration posterior modal pattern derived for each cluster (t_I0)
