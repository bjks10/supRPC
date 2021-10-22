# Supervised Robust Profile Clustering (supRPC)

Supervised Robust Profile Clustering (supRPC) model is a population-based clustering technique that adjusts for differences that may exist within different subpopulations with consideration for their assocation with a binary outcome. This is an extension of Robust Profile Clustering (RPC) technique. In the RPC model, participants are clustered at two levels: (1) globally, where subjects are assigned to an overall population-level cluster using an overfitted mixture model, and (2) locally, where variations to global patterns are accommodated via a Beta-Bernoulli process dependent on subpopulation differences. This model is jointly modeled with a probit regression model, in order to generate robust global profiles that are informed by the probabity of a positive binary outcome. 


# Getting Started

The code and supporting materials are run using MATLAB software. To run the example data, you will need the files contained in the Simulation folder, as well as the following supporting function files provided here:

drchrnd.m - Dirichlet random generator function
truncnormrnd.m - function file to generate draw from truncated normal random distribution

The parameters of the supRPC model are estimated in a two-step sampling algorithm. 
* The adaptive sampling algorithm (supRPC_Part1dem_adaptive.m) allows the user to determine the appropriate number of nonempty clusters. 
* The fixed sampling algorithm (supRPC_Part2dem_fixed.m) allows the user to re-run the supervised RPC model with the number of nonempty clusters predetermined. 

When the number of clusters, globally and locally, is known a priori the adaptive sampling step can be skipped. The attached code uses data from the National Birth Defects Prevention Study, which is not publicly available. 


# Simulated Examples
The example dataset found in the Simulation folder is a MAT-file that contains the following variables:

* sampledata: 4800x50 matrix. This matrix is the input dataset containing subject level data for 50 variables. Each variable assigned a single categorical value (1,2,3,4).
* trueG: binary 4x50 matrix. This matrix is used as a reference to illustrate the true probability of allocation for each variable within each subpopulation to global (ν= 1) or local (ν= 0).
* subpop_samp: 4800x1 vector. This vector contains subpopulation ID for the 4800 subjects included in the dataset.
* true_global: 50x3 matrix. This matrix contains the 3 global profile patterns modally expected. 
* true_ci: 4800x1 matrix. This matrix contains the true global profile assignment to each subject.
* true_local: 8x50 matrix. This matrix contains the two local profile patterns modally expected for each subpopulation. Rows 1-2 correspond to subpopulation 1. Rows 3-4 correspond to subpopulation 2. Rows 5-6 correspond to subpopulation 3. Rows 7-8 correspond to subpopulation 4.  
* true_Li: 4800x1 matrix. This matrix contains the local profile assignment to each subject.
* true_xi: 1x7 vector. This row vector contains the true coefficients of the probit regression model
* true_y: 4800x1 vector. This column vector contains the true binary outcome for each subject: 1=case, 0=control.
* phi_WXtrue: 4800x1 vector. This column vector contains the true probability of outcome for each subject.

Two cases are demonstrated in the Simulation folder:
1. Case 1: Simulated population with no subpopulation-specific confounding present
2. Case 2: Simulated population with subpopulation-specific confounding present

Authors

Briana Stephenson, Amy Herring, Andrew Olshan
