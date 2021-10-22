%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SUPERVISED ROBUST PROFILE CLUSTERING prior  
%Programmer: Briana Stephenson
%Data: NBDPS 
% Remove DP and replace with OFM global/local
% Decrease concentration parameter to encourage sparsity
% adjust beta to simulate a t
% mu0~std normal, sig0 ~ IG(5/2,5/2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;


%% load NBDPS data %%
cc=readtable('lca_nbdps_dataout3MAR2021.csv');

food=table2array(cc(:,10:72));
alldems=table2array(cc(:,[1:9 73:76]));
% remove misreported foods (v10,43,50,51,56)

demfood=[alldems food];
    %drop all NaN foods
 data_clean=demfood(~any(isnan(demfood),2),:); %only keep subjects that are not missing 

food_clean=data_clean(:,14:end);
y_clean=data_clean(:,2);
sid_clean=data_clean(:,3); S=length(unique(sid_clean));
[n_clean,p]=size(food_clean); d=max(food_clean(:));

%% DEMOGRAPHIC DATA %%
w_sid=zeros(n_clean,S); 

for s=1:S
    w_sid(sid_clean==(s+9),s)=1;
end
 %make subpop4 = massachusetts (largest sample)
w_sidr=w_sid(:,[1:3 5:S]);

%demographic data
education=[(data_clean(:,5)==1) (data_clean(:,5)==2)]; % [noHS HSedu]
edu_vars=double(education);

smoke=double(data_clean(:,9)==1);
smoking=[(data_clean(:,9)==0) (data_clean(:,9)==1)]; %[nonsmoke smoke]
smoke_vars=double(smoking);

obese=double(data_clean(:,4)>=3);
underwt=double(data_clean(:,4)==2);
bmi_nih=[(data_clean(:,4)>=3) (data_clean(:,4)==1) (data_clean(:,4)==2)]; % NIH BMI (obese,normal,under)
bmi_vars=double(bmi_nih);

drinkr=double(data_clean(:,6)==1);
alcohol=[(data_clean(:,6)==0) (data_clean(:,6)==1)]; %[none alcohol]
alc_vars=double(alcohol);

age25=double(data_clean(:,7)==1);
age=[data_clean(:,7)>1 (data_clean(:,7)==1)]; %>25, under 25 yrs age;
age_vars=double(age);

race=data_clean(:,10:13); %white,black,hispanic,other;

fause=double(data_clean(:,8)==1);
FAuse=[(data_clean(:,8)==1) (data_clean(:,8)==0)]; % [FAuse noFAuse]
FA_vars=double(FAuse);

% %setup x matrix
% w_clean=[w_sid edu_vars smoke_vars bmi_vars alc_vars  race FA_vars age_vars]; 
w_dclean=[w_sid edu_vars smoke_vars age race ]; %drop age, alcohol, bmi - not significant

%%remove missing information rows%%
id_clean=data_clean(:,1);

%% vectorization of data %%
idz = repmat(1:p,n_clean,1); idz = idz(:);
x_d = food_clean(:); lin_idx = sub2ind([p,d],idz,x_d);
%subvectorization
idz_s=cell(S,1);
idz_s{S}=[];
lin_idxS=cell(S,1);
lin_idxS{S}=[];
x_s=cell(S,1);
x_s{S}=[];
n_s=zeros(S,1);
for s=1:S
    n_s(s)=length(food_clean(sid_clean==(s+9),:));
    idzs=repmat(1:p,n_s(s),1);
    idz_s{s}=idzs(:);
    food_s=food_clean(sid_clean==(s+9),:);
    xs=food_s(:);
    x_s{s}=xs;
    lin_idxS{s}=sub2ind([p,d],idz_s{s},xs);
end
%% -- RPC Predictor Model SETUP -- %%
k_max=50;

     

%% SET UP HYPERPRIORS %%
%beta
abe=1; bbe=1; %hypers for beta
beta=ones(1,S);


%nu_j^s
beta_s=1;
nu=betarnd(1,beta_s,[S,p]);


    %pi_h for all classes
alpha=ones(1,k_max)*(1/100);
pi_h=drchrnd(alpha,1);

    %phi - cluster index
x_Ci=mnrnd(1,pi_h,n_clean); [r, c]=find(x_Ci); gc=[r c];
    gc=sortrows(gc,1); Ci=gc(:,2);
n_Ci=sum(Ci);


%global theta0/1
eta=ones(1,d);
theta0=zeros(p,k_max,d);
theta1=zeros(S,p,k_max,d);

for k=1:k_max
    for j=1:p
        theta0(j,k,:)=drchrnd(eta,1);
        for s=1:S
        theta1(s,j,k,:)=drchrnd(eta,1);
        end
    end
end

    %global G_ij
    G_ij=zeros(n_clean,p);
 for s=1:S
     ns=n_s(s);
     nu_s=nu(s,:);
     G_ij(sid_clean==(s+9),:) = repmat(binornd(1,nu_s),[ns,1]);     % family index (0=global family,1=local family)
 end


%% SUBPOPULATION LPP NESTS %%
%determine number of global diets in each subpopulation
lambda_sk=drchrnd(alpha,S);

    L_ij=zeros(n_clean,p); %local cluster label variable

    n_Lij=zeros(S,p,k_max);
for s=1:S
    for j=1:p
        x_sl=mnrnd(1,lambda_sk(s,:),n_s(s)); [r, c]=find(x_sl); gc=[r c];
        gc=sortrows(gc,1); 
        L_ij(sid_clean==(s+9),j)=gc(:,2);
        n_Lij(s,j,:)=sum(x_sl);
    end
end


%% -- RESPONSE PROBIT MODEL SETUP -- %%
pcov=k_max+size(w_dclean,2);
X=zeros(n_clean,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid

Wmat=[w_dclean x_Ci];
ep_kp=zeros(n_clean,k_max); 

for k=1:k_max
   w_ip=zeros(n_clean,k_max);
   w_ip(:,k)=ones(n_clean,1);
   W_temp=[w_dclean w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_iter));
        phi_temp(phi_temp==1)=1-1e-6;
        phi_temp(phi_temp==0)=1e-6;
   probit_kp=y_clean.*log(phi_temp)+(1-y_clean).*log(1-phi_temp);
   ep_kp(:,k)=exp(probit_kp);
end


    

%% ------------ %%
%% data storage %%
%% ------------ %%
nrun=25000; burn=nrun/5; thin=10;

%predictor RPC model storage
pi_out=zeros(nrun,k_max);
z_probit=zeros(n_clean,1);

% Loglike0=zeros(nrun,1);
lambdas_out=zeros(nrun,S,k_max);
ks_out=zeros(nrun, S+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POSTERIOR COMPUTATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
   

%temporary storage
As=zeros(n_clean,p); 
p_ij=zeros(n_clean,p);
ZBrand=zeros(n_clean,1);
nL_ij=zeros(n_clean,k_max);
%% -- BEGIN MCMC -- %%

for iter=1:nrun
    
    %% -- update G_ij prob -- %%
 for s=1:S
     phrow1=L_ij(sid_clean==(s+9),:); phrow1=phrow1(:);
     theta1s=reshape(theta1(s,:,:,:),[p,k_max,d]);
     A = theta1s(sub2ind([p,k_max,d],idz_s{s},phrow1,x_s{s})); %index of subjects in theta1class
    A = reshape(A,[n_s(s),p]);
    As(sid_clean==(s+9),:)=A;
 end
 
 phrow0=repmat(Ci,[1,p]); phrow0=phrow0(:);
 B = theta0(sub2ind([p,k_max,d],idz,phrow0,x_d));
 B = reshape(B,[n_clean,p]);
 
 for s=1:S
     ns=n_s(s); nu_s=nu(s,:);
     p_ij(sid_clean==(s+9),:)=(repmat(nu_s,[ns,1]).*B(sid_clean==(s+9),:)./((repmat(nu_s,[ns,1]).*B(sid_clean==(s+9),:))+(repmat(1-nu_s,[ns,1]).*As(sid_clean==(s+9),:))));
 end
 
    G_ij=binornd(1,p_ij);
    
    %% -- update pi_h -- %%

    for h=1:k_max
        n_Ci(h)=sum(Ci==h);
    end
    alphaih=alpha+ n_Ci;
    pi_h=drchrnd(alphaih,1);
pi_out(iter,:)=pi_h;
ks_out(iter,S+1)=sum(pi_h>0.03);

  %%-- update lambda_sk --%%
    for s=1:S
      phi_s=L_ij(sid_clean==(s+9),:);
      for l=1:k_max
          nL_ij(s,l)=sum(phi_s(:)==l);
      end
      kn_Lij=alpha+nL_ij(s,:);
      lambda_sk(s,:)=drchrnd(kn_Lij,1);
      ks_out(iter,s)=sum(lambda_sk(s,:)>0.03);
    end
    lambdas_out(iter,:,:)=lambda_sk;

    
%   %% -- Ci ~multinomial(pi_h) -- %%

  Cp_k=zeros(n_clean,k_max);
for k=1:k_max
    t0h=reshape(theta0(:,k,:),p,d);
    tmpmat0=reshape(t0h(lin_idx),[n_clean,p]);
    Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2).*ep_kp(:,k);
end
% log_pc=log(Cp_k); log_sumpc=log(sum(Cp_k,2));
% log_pp=log_pc-log_sumpc;
% probCi=exp(log_pp);
probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));

    x_ci=mnrnd(1,probCi); [r, c]=find(x_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);

    for s=1:S
        Lijs=zeros(n_s(s),k_max,p);
             for h = 1:k_max
                theta1hs = reshape(theta1(s,:,h,:),p,d);
                tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
                 Lijs(:,h,:) = lambda_sk(s,h) * tmpmat1.^(G_ij(sid_clean==(s+9),:)==0);
             end  
            sumLijs=repmat(sum(Lijs,2),[1,k_max,1]);
            zupS = Lijs./sumLijs;
            for j=1:p
                sub_pj=reshape(zupS(:,:,j),[n_s(s),k_max]);
                l_ij=mnrnd(1,sub_pj);
                [r, c]=find(l_ij); x_l=[r c];
                x_sl=sortrows(x_l,1);
                L_ij(sid_clean==(s+9),j)=x_sl(:,2);
            end
    end 


% - update theta - %
dmat0=zeros(p,d);
dmat1=zeros(p,d);
for k=1:k_max
    Cis=repmat(Ci,[1,p]).*G_ij;
     ph0 = (Cis==k); %subj's in global cluster h
        for c = 1:d
             dmat0(:,c) = sum((food_clean==c).*ph0)';
        end
        for j=1:p
            a_tn0=eta+dmat0(j,:);
            theta0(j,k,:) = drchrnd(a_tn0,1);
        end
end
    
  for s=1:S
      phis=L_ij(sid_clean==(s+9),:).*(1-G_ij(sid_clean==(s+9),:));
      foods=food_clean(sid_clean==(s+9),:);
      for l=1:k_max
          ph1=(phis==l);
          for c=1:d
            dmat1(:,c) = sum((foods==c).*ph1);
          end
        for j=1:p
            a_tn1=eta+dmat1(j,:);
            theta1(s,j,l,:) = drchrnd(a_tn1,1);
        end 
      end
  end

  

    
    % update nu_j %
    for s=1:S
        Gs=G_ij(sid_clean==(s+9),:);
        nu(s,:) = betarnd(1 + sum(Gs), beta(s) + sum(1-Gs));
    end
  nu(nu==1) = 1-1e-06;
  nu(nu==0) = 1e-06; 


  % - update beta - %
  for s=1:S
    beta(s) = gamrnd(abe + p,1./( bbe - sum(log(1-nu(s,:)))));
  end



%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%


% Wup=[x_ci w_sid(:,2:end)]; %covariate matrix with state/global
Wup=[w_dclean x_ci];
%create latent z_probit
    %create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);
%truncation for cases (0,inf)
z_probit(y_clean==1)=truncnormrnd(1,WXi_now(y_clean==1),1,0,inf);
%truncation for controls (-inf,0)
z_probit(y_clean==0)=truncnormrnd(1,WXi_now(y_clean==0),1,-inf,0);

    %control extremes;
    
    if sum(z_probit==Inf)>0
        z_probit(z_probit==Inf)=norminv(1-1e-6);
    end 
    if sum(z_probit==-Inf)>0
        z_probit(z_probit==-Inf)=norminv(1e-6);
    end

% Response Xi(B) update

    sig0up=Sig0;
    xi_0up=xi_0;

    xi_sig_up=inv(sig0up)+(transpose(Wup)*Wup);
    xi_mu_up2=(sig0up\transpose(xi_0up))+(transpose(Wup)*z_probit);
    xi_mu_up=xi_sig_up\xi_mu_up2;
    xi_up=mvnrnd(xi_mu_up,inv(xi_sig_up));
    xi_iter=xi_up;
    wxi=Wup*transpose(xi_iter);

    phi_wxi=normcdf(wxi);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-6;
        phi_wxi(phi_wxi==1)=1-1e-6;
    probit_yi=y_clean.*log(phi_wxi) + (1-y_clean).*log(1-phi_wxi);


    %prep matrix for probit_Ci
    for k=1:k_max
        w_ip=zeros(n_clean,k_max);
        w_ip(:,k)=ones(n_clean,1);
        W_temp=[w_dclean w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_iter));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp<1e-15) = 1e-10; 
            p_kp=y_clean.*log(phi_temp)+(1-y_clean).*log(1-phi_temp);
            ep_kp(:,k)=exp(p_kp);
    end
%% RELABELLING STEP TO ENCOURAGE MIXING %%

    if mod(iter,10)==0
        new_order=randperm(k_max);
        newCi=Ci;
        for k=1:k_max
            new_k=new_order(k);
            newCi(Ci==k)=new_k;
        end
        Ci=newCi;
        theta0=theta0(:,new_order,:);
        ep_kp=ep_kp(:,new_order);

    end


end

lambdas_burn=lambdas_out(burn+1:end,:,:);
pi_burn=pi_out(burn+1:end,:);
ks_burn=ks_out(burn+1:end,:);

k_large=median(sum(pi_burn>0.06,2));
k_prob=median(sum(pi_burn>0.05,2));

bl=size(lambdas_burn,1);
ks_med=zeros(S,2);
for s=1:S
    s_lambda = reshape(lambdas_burn(:,s,:),[bl,k_max]);
    ks_med(s,1)=median(sum(s_lambda>0.05,2));
    ks_med(s,2)=median(sum(s_lambda>0.03,2));
end

save('sRPCdemred_MCMCadaptiveout25Aug21','lambdas_burn','pi_burn','ks_burn','k_prob','k_large','ks_med','-v7.3');




