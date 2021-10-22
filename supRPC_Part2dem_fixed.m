%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% POSTHOC ANALYSIS OF SUP-RPC    %%%
%%% Programmer: Briana Stephenson  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
%  load('sRPCdemred_MCMCout29apr2021')
% load('sRPCdemred_MCMCadaptiveout30Jul21')
load('sRPCdemred_MCMCadaptiveout25Aug21')
[m,S,~]=size(lambdas_burn);
%check number of nonempty clusters

% 
ks_keep=ks_med(:,1);
ks_max=max(ks_keep);
% Re-run supervised RPC with k_prob clusters - fixed algorithm 

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


%% SET UP HYPERPRIORS %%
%beta
abe=1; bbe=1; %hypers for beta
beta=ones(1,S);


%nu_j^s
beta_s=1;
nu=betarnd(1,beta_s,[S,p]);


    %pi_h for all classes
alpha0=ones(1,k_prob)*(1/k_prob);
pi_h=drchrnd(alpha0,1);

    %phi - cluster index
x_Ci=mnrnd(1,pi_h,n_clean); [r, c]=find(x_Ci); gc=[r c];
    gc=sortrows(gc,1); Ci=gc(:,2);
n_Ci=sum(Ci);


%global theta0/1
eta=ones(1,d);
theta0=zeros(p,k_prob,d);


for k=1:k_prob
    for j=1:p
        theta0(j,k,:)=drchrnd(eta,1);
    end
end

ks_max=max(ks_keep);
theta1=zeros(S,p,ks_max,d);
for s=1:S
    kss=ks_keep(s);
    for k=1:kss
        for j=1:p
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
%determine number of local diets in each subpopulation
alpha1=ones(1,ks_max)*(1/k_prob);
lambda_sk=zeros(S,ks_max);
for s=1:S
    kss=ks_keep(s);
    a1=alpha1(1:kss);
    lambda_sk(s,1:kss)=drchrnd(a1,1);
end


    L_ij=zeros(n_clean,p); %local cluster label variable

    n_Lij=zeros(S,p,ks_max);
for s=1:S
    for j=1:p
        x_sl=mnrnd(1,lambda_sk(s,:),n_s(s)); [r, c]=find(x_sl); gc=[r c];
        gc=sortrows(gc,1); 
        L_ij(sid_clean==(s+9),j)=gc(:,2);
        n_Lij(s,j,:)=sum(x_sl);
    end
end


%% -- RESPONSE PROBIT MODEL SETUP -- %%
pcov=k_prob+size(w_dclean,2);
X=zeros(n_clean,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid

Wmat=[w_dclean x_Ci];
ep_kp=zeros(n_clean,k_prob); 

for k=1:k_prob
   w_ip=zeros(n_clean,k_prob);
   w_ip(:,k)=ones(n_clean,1);
   W_temp=[w_dclean w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_iter));
        phi_temp(phi_temp==1)=1-1e-10;
        phi_temp(phi_temp==0)=1e-10;
   probit_kp=y_clean.*log(phi_temp)+(1-y_clean).*log(1-phi_temp);
   ep_kp(:,k)=exp(probit_kp);
end


    

%% ------------ %%
%% data storage %%
%% ------------ %%
nrun=25000; burn=15000; m_thin=10;

%predictor RPC model storage
beta_out=zeros(nrun,S);
nu_out=zeros(nrun,S,p);
pi_out=zeros(nrun,k_prob);
z_probit=zeros(n_clean,1);

% Loglike0=zeros(nrun,1);

theta0_out=zeros(nrun/m_thin,p,k_prob,d);
theta1_out=zeros(nrun/m_thin,S,p,ks_max,d);
lambdas_out=zeros(nrun,S,ks_max);
ci_out=zeros(nrun,n_clean);

%response spv model storage
xi_out=zeros(nrun,pcov);
loglike_out=zeros(nrun,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POSTERIOR COMPUTATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %temporary storage
As=zeros(n_clean,p); 
p_ij=zeros(n_clean,p);
ZBrand=zeros(n_clean,1);
nL_ij=zeros(n_clean,ks_max);
%% -- BEGIN MCMC -- %%

for iter=1:nrun
    
    %% -- update G_ij prob -- %%
 for s=1:S
     kss=ks_keep(s);
     phrow1=L_ij(sid_clean==(s+9),:); phrow1=phrow1(:);
     theta1s=reshape(theta1(s,:,1:kss,:),[p,kss,d]);
     A = theta1s(sub2ind([p,kss,d],idz_s{s},phrow1,x_s{s})); %index of subjects in theta1class
    A = reshape(A,[n_s(s),p]);
    As(sid_clean==(s+9),:)=A;
 end
 
 phrow0=repmat(Ci,[1,p]); phrow0=phrow0(:);
 B = theta0(sub2ind([p,k_prob,d],idz,phrow0,x_d));
 B = reshape(B,[n_clean,p]);
 
 for s=1:S
     ns=n_s(s); nu_s=nu(s,:);
     p_ij(sid_clean==(s+9),:)=(repmat(nu_s,[ns,1]).*B(sid_clean==(s+9),:)./((repmat(nu_s,[ns,1]).*B(sid_clean==(s+9),:))+(repmat(1-nu_s,[ns,1]).*As(sid_clean==(s+9),:))));
 end
 
    G_ij=binornd(1,p_ij);
    
    %% -- update pi_h -- %%

    for h=1:k_prob
        n_Ci(h)=sum(Ci==h);
    end
    alphaih=alpha0+ n_Ci;
    pi_h=drchrnd(alphaih,1);
pi_out(iter,:)=pi_h;

  %%-- update lambda_sk --%%
    for s=1:S
        kss=ks_keep(s);
      phi_s=L_ij(sid_clean==(s+9),:);
      for l=1:kss
          nL_ij(s,l)=sum(phi_s(:)==l);
      end
      kn_Lij=alpha1(1:kss)+nL_ij(s,:);
      lambda_sk(s,:)=drchrnd(kn_Lij,1);
    end
    lambdas_out(iter,:,:)=lambda_sk;

    
%   %% -- Ci ~multinomial(pi_h) -- %%

  Cp_k=zeros(n_clean,k_prob);
for k=1:k_prob
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
ci_out(iter,:)=Ci; %save cluster assignments for pairwise poseriors

    for s=1:S
        kss=ks_keep(s);
        Lijs=zeros(n_s(s),kss,p);
             for h = 1:kss
                theta1hs = reshape(theta1(s,:,h,:),p,d);
                tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
                 Lijs(:,h,:) = lambda_sk(s,h) * tmpmat1.^(G_ij(sid_clean==(s+9),:)==0);
             end  
            sumLijs=repmat(sum(Lijs,2),[1,kss,1]);
            zupS = Lijs./sumLijs;
            for j=1:p
                sub_pj=reshape(zupS(:,:,j),[n_s(s),kss]);
                l_ij=mnrnd(1,sub_pj);
                [r, c]=find(l_ij); x_l=[r c];
                x_sl=sortrows(x_l,1);
                L_ij(sid_clean==(s+9),j)=x_sl(:,2);
            end
    end 


% - update theta - %
dmat0=zeros(p,d);
dmat1=zeros(p,d);
for k=1:k_prob
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
      kss=ks_keep(s);
      phis=L_ij(sid_clean==(s+9),:).*(1-G_ij(sid_clean==(s+9),:));
      foods=food_clean(sid_clean==(s+9),:);
      for l=1:kss
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
  if mod(iter,m_thin)==0
     theta0_out(iter/m_thin,:,1:size(theta0,2),:)=theta0;
     theta1_out(iter/m_thin,:,:,1:size(theta1,3),:)=theta1;
  end
  

    
    % update nu_j %
    for s=1:S
        Gs=G_ij(sid_clean==(s+9),:);
        nu(s,:) = betarnd(1 + sum(Gs), beta(s) + sum(1-Gs));
    end
  nu(nu==1) = 1-1e-10;
  nu(nu==0) = 1e-10; 

 nu_out(iter,:,:)=nu;

  % - update beta - %
  for s=1:S
    beta(s) = gamrnd(abe + p,1./( bbe - sum(log(1-nu(s,:)))));
  end
 beta_out(iter,:)=beta;



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
        z_probit(z_probit==Inf)=norminv(1-1e-10);
    end 
    if sum(z_probit==-Inf)>0
        z_probit(z_probit==-Inf)=norminv(1e-10);
    end

% Response Xi(B) update

    sig0up=Sig0;
    xi_0up=xi_0;

    xi_sig_up=inv(sig0up)+(transpose(Wup)*Wup);
    xi_mu_up2=(sig0up\transpose(xi_0up))+(transpose(Wup)*z_probit);
    xi_mu_up=xi_sig_up\xi_mu_up2;
    ixi_sigup = inv(xi_sig_up);
    invxisig_rd = (ixi_sigup + ixi_sigup.') / 2; % fix the rounding issue
    xi_up=mvnrnd(xi_mu_up,invxisig_rd);
    xi_iter=xi_up;
    xi_out(iter,:)=xi_iter;
    wxi=Wup*transpose(xi_iter);

    phi_wxi=normcdf(wxi);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-10;
        phi_wxi(phi_wxi==1)=1-1e-10;
    probit_yi=y_clean.*log(phi_wxi) + (1-y_clean).*log(1-phi_wxi);

    loglike_out(iter)=sum(probit_yi);

    %prep matrix for probit_Ci
    for k=1:k_prob
        w_ip=zeros(n_clean,k_prob);
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
        new_order=randperm(k_prob);
        newCi=Ci;
        for k=1:k_prob
            new_k=new_order(k);
            newCi(Ci==k)=new_k;
        end
        Ci=newCi;
        theta0=theta0(:,new_order,:);
        ep_kp=ep_kp(:,new_order);

    end


end

%burn out early runs
beta_burn=beta_out(burn+1:end,:);
lambdas_burn=lambdas_out(burn+1:end,:,:);
pi_burn=pi_out(burn+1:end,:);
theta0_burn=theta0_out((burn/m_thin)+1:end,:,:,:);
theta1_burn=theta1_out((burn/m_thin)+1:end,:,:,:,:);
nu_burn=nu_out(burn+1:end,:,:);
xi_burn=xi_out(burn+1:end,:);
ci_burn=ci_out(burn+1:end,:);
loglike_burn=loglike_out(burn+1:end);
save('debug_adaptiveburns','pi_burn','lambdas_burn','beta_burn','lambdas_burn','theta0_burn','theta1_burn', 'nu_burn','xi_burn','ci_burn', 'loglike_burn','-v7.3')
%% Send burn-out parameters for post-processing %%
m_burn=size(ci_burn,1);
%%%%%%%%%%%%%%%%%%%%%
%% POST PROCESSING %%
%%%%%%%%%%%%%%%%%%%%%

pd=pdist(transpose(ci_burn),'hamming'); %prcnt differ
cdiff=squareform(pd); %Dij
Zci=linkage(cdiff,'complete');
dendrogram(Zci); % 6 groups selected based on results
saveas(gcf,'supRPCdemred_dendrogram28aug2021.png')
zclust=cluster(Zci,'maxclust',k_prob);


ci_relabel=zeros(m_burn,k_prob);

 for l=1:k_prob
     ci_relabel(:,l)=mode(ci_burn(:,zclust==l),2);
 end


%% reorder pi and theta0 parameters
% [~,ord_c]=sort(ci_relabel(1,:));
% ci_relabel=ci_relabel(:,[ord_c]);
%17 demographic covariates in model;
pi_order=zeros(m_burn,k_prob);
theta0_order=zeros(m_burn/m_thin,p,k_prob,d);
p_dem=size(xi_burn,2)-k_prob;
pcov=k_prob+p_dem;
xi_order=zeros(m_burn/m_thin,pcov);
% k_uni=zeros(m,1);
for iter=1:m_burn
   iter_order=ci_relabel(iter,:);
%    k_uni(iter)=length(unique(iter_order));
%    iter_uni=unique(iter_order);
   pi_order(iter,:)=pi_burn(iter,iter_order);
%     pi_order(iter,1:length(iter_uni))=pi_burn(iter,iter_uni);
    s_iteruni=p_dem+iter_order;
   xi_order(iter,:)=[xi_burn(iter,1:(p_dem)) xi_burn(iter,s_iteruni)];
   
   if mod(iter,m_thin)==0
       iter_m_thin=iter/m_thin;
%        theta0_order(iter_m_thin,:,1:length(iter_uni),:)=theta0_burn(iter_m_thin,:,iter_uni,:);
        theta0_order(iter_m_thin,:,:,:)=theta0_burn(iter_m_thin,:,iter_order,:);
   end
end

save('ordered_sRPCdemfixed_parms28aug2021','theta0_order','pi_order','xi_order','ci_relabel','Zci');
save('localburns_sRPCdemfixed_parms28aug2021','theta1_burn','lambdas_burn','beta_burn','nu_burn','-v7.3');
%%identify modal global patterns for each cluster
pi_med=median(pi_order);
pi_medi=pi_med/sum(pi_med);
xi_med=median(xi_order);
xi_prc=[prctile(xi_order,2.5); prctile(xi_order,97.5)];
% pi_med=pi_med(pi_med>0); k_in=sum(pi_med>0);
theta0_med=reshape(median(theta0_order),[p,k_prob,d]);
theta0_medi=theta0_med./sum(theta0_med,3);
[M0, I0]=max(theta0_med,[],3);
t_I0=transpose(I0);
[uc, ia, ic]=unique(t_I0,'rows');

%%identify modal local patterns for each cluster

theta1_med=reshape(median(theta1_burn),[S,p,ks_max,d]);
lambdas_med=reshape(median(lambdas_burn),[S,ks_max]);

val1=cell(S,1); ind1=cell(S,1);
for s=1:S
    t1s = theta1_med(s,:,1:ks_keep(s),:);
    s_t1=reshape(t1s,[p,ks_keep(s),d]);
    [val1{s},ind1{s}]=max(s_t1,[],3);
end

p_pos=transpose(sum(xi_order>0)/m_burn);

w_zclust=zeros(n_clean, k_prob);
for k=1:k_prob
    w_zclust(k_prob==k,k)=1;
end

    W_cz=[w_dclean w_zclust];
    loglike_yi=zeros(m_burn,n_clean);
    devprob_i=zeros(m,1);
    for iter=1:m_burn
        xi_i=xi_order(iter,:);
        WXi_i=W_cz*transpose(xi_i);
        phi_wxi=normcdf(WXi_i);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-10;
        phi_wxi(phi_wxi==1)=1-1e-10;
        lprobit_yi=y_clean.*log(phi_wxi) + (1-y_clean).*log(1-phi_wxi);
        loglike_yi(iter,:)=lprobit_yi;
        devprob_i(iter)=sum(lprobit_yi);
    end
    
like_srpc=zeros(n_clean,1);
ep_kp=zeros(n_clean,k_prob);
loglikesrpc_m_thin=zeros(m_burn,1);
G_ij=zeros(n_clean,p); L_ij=zeros(n_clean,p);

for ii=1:m_burn
    i_m_thin=ceil(ii/m_thin);
    pi_m_thin=pi_order(ii,:);
    theta0_m_thin=reshape(theta0_order(i_m_thin,:,:,:),[p,k_prob,d]);
    theta1_m_thin=reshape(theta1_burn(i_m_thin,:,:,:,:),[S,p,ks_max,d]);
    lambda_m_thin=reshape(lambdas_burn(ii,:,:),[S,ks_max]);
    nu_m_thin=reshape(nu_burn(ii,:,:),[S,p]);
    pi_h=pi_m_thin/sum(pi_m_thin);
    xi_itr=xi_order(ii,:);
    for s=1:S
        nu_s=reshape(nu_m_thin(s,:),[1,p]);
        G_ij(sid_clean==(s+9),:)=binornd(1,repmat(nu_s,[n_s(s),1]));
    end
    
    for k=1:k_prob
        w_ip=zeros(n_clean,k_prob);
        w_ip(:,k)=ones(n_clean,1);
        W_temp=[w_dclean w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_itr));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp<1e-15) = 1e-10; 
        p_kp=y_clean.*log(phi_temp)+(1-y_clean).*log(1-phi_temp);
        ep_kp(:,k)=exp(p_kp);
    end
    
        %assign global cluster
      Cp_k=zeros(n_clean,k_prob);
    for k=1:k_prob
        t0h=reshape(theta0_m_thin(:,k,:),p,d);
        tmpmat0=reshape(t0h(lin_idx),[n_clean,p]);
        Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2).*ep_kp(:,k);
    end

    probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));
    w_ci=mnrnd(1,probCi); [r, c]=find(w_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);
    
      %assign local cluster
    for s=1:S
    Lijs=zeros(n_s(s),ks_max,p);
         for h = 1:ks_max
            theta1hs = reshape(theta1_m_thin(s,:,h,:),p,d);
            tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
             Lijs(:,h,:) = lambda_m_thin(s,h) * tmpmat1.^(G_ij(sid_clean==(s+9),:)==0);
         end  
        sumLijs=repmat(sum(Lijs,2),[1,ks_max,1]);
        zupS = Lijs./sumLijs;
        for j=1:p
            sub_pj=reshape(zupS(:,:,j),[n_s(s),ks_max]);
            l_ij=mnrnd(1,sub_pj);
            [r, c]=find(l_ij); x_l=[r c];
            x_sl=sortrows(x_l,1);
            L_ij(sid_clean==(s+9),j)=x_sl(:,2);
        end  
    loglocal=prod(sum(Lijs,2),3);
    like_srpc(sid_clean==(s+9))=sum(Cp_k(sid_clean==(s+9),:),2).*loglocal;

    end
    
  %create truncated normal for latent z_probit model
    Wup=[w_dclean w_ci];
        WXi_now=Wup*transpose(xi_itr);

    phi_wxi=normcdf(WXi_now);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-10;
        phi_wxi(phi_wxi==1)=1-1e-10;
    lprobit_yi=y_clean.*log(phi_wxi) + (1-y_clean).*log(1-phi_wxi);


    loglikesrpc_m_thin(ii)=sum(log(like_srpc));

end

 %preset global clusters based on posterior medians
nu_med=reshape(median(nu_burn),[S,p]);
G_med=zeros(n_clean,p);
for s=1:S
   G_med(sid_clean==(s+9),:)=binornd(1,repmat(nu_med(s,:),[n_s(s),1]));
end

    for k=1:k_prob
        w_ip=zeros(n_clean,k_prob);
        w_ip(:,k)=ones(n_clean,1);
        W_temp=[w_dclean w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_med));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp<1e-15) = 1e-10; 
            p_kp=y_clean.*log(phi_temp)+(1-y_clean).*log(1-phi_temp);
            ep_kp(:,k)=exp(p_kp);
    end
loglike_srpcmed=zeros(n_clean,1);
delmed=zeros(n_clean,k_prob);
   for h = 1:k_prob
        t0h = reshape(theta0_medi(:,h,:),p,d);
        theta0h=bsxfun(@times,t0h,1./sum(t0h,2));
        tmpmat0 = reshape(theta0h(lin_idx),[n_clean,p]);
        delmed(:,h) = pi_medi(h)*prod(tmpmat0.^G_med,2).*ep_kp(:,h);
    end 
    zup0 = bsxfun(@times,delmed,1./(sum(delmed,2)));
    w_ci=mnrnd(1,zup0); [r, c]=find(w_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci_med=x_gc(:,2);
    [Ci_val,Ci_max]=max(zup0,[],2);

    for s=1:S
    Lijs=zeros(n_s(s),ks_max,p);
         for h = 1:ks_max
            theta1hs = reshape(theta1_med(s,:,h,:),p,d);
            tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
             Lijs(:,h,:) = lambdas_med(s,h) * tmpmat1.^(G_med(sid_clean==(s+9),:)==0);
         end  
        sumLijs=repmat(sum(Lijs,2),[1,ks_max,1]);
        zupS = Lijs./sumLijs;
        for j=1:p
            sub_pj=reshape(zupS(:,:,j),[n_s(s),ks_max]);
            l_ij=mnrnd(1,sub_pj);
            [r, c]=find(l_ij); x_l=[r c];
            x_sl=sortrows(x_l,1);
            L_ij(sid_clean==(s+9),j)=x_sl(:,2);
        end  
    loglocal=sum(log(sum(Lijs,2)),3);
    loglike_srpcmed(sid_clean==(s+9))=log(sum(delmed(sid_clean==(s+9),:),2))+loglocal;

    end


DIC_dem=-4*median(loglikesrpc_m_thin)+2*sum(loglike_srpcmed)
DIC_star=-6*median(loglikesrpc_m_thin)+4*sum(loglike_srpcmed)
z_rpc=Ci_max;
pat0_srpc=I0;
pat1_srpc=ind1;


%% CALCULATE AEBIC %%
tmel=zeros(S,1);
for s=1:S
    tmel(s)=ks_keep(s)*p*d;
end

S_srpc=numel(pi_medi)+numel(theta0_medi)+sum(ks_keep)+numel(nu_med)+numel(xi_med)+sum(tmel);


term1_ae_srpc=mean(loglike_burn);
term2_ae_srpc=S_srpc*log(n_clean);
term3_ae_srpc=2*S_srpc*log(pcov+k_prob);
tt_srpc=term1_ae_srpc+term2_ae_rpc+term3_ae_rpc;
aebic_srpc=-2*n_clean*mean(tt_srpc)


save('probitred_kp_sRPCdemfix28Aug2021','z_rpc','xi_med','xi_prc','p_pos','DIC_dem','DIC_star','aebic_srpc','pat0_srpc','pat1_srpc','val1','pi_medi','theta1_med','lambdas_med','theta0_medi','nu_med')


    %% referent cell coding transformation %%
xi_ref=zeros(m_burn,18);
xi_ref(:,1)=xi_order(:,4)+xi_order(:,12)+xi_order(:,13)+xi_order(:,16)+xi_order(:,17)+xi_order(:,23);
%subpop
xi_ref(:,2)=xi_order(:,1)-xi_order(:,4);
xi_ref(:,3)=xi_order(:,2)-xi_order(:,4);
xi_ref(:,4)=xi_order(:,3)-xi_order(:,4);
xi_ref(:,5)=xi_order(:,5)-xi_order(:,4);
xi_ref(:,6)=xi_order(:,6)-xi_order(:,4);
xi_ref(:,7)=xi_order(:,7)-xi_order(:,4);
xi_ref(:,8)=xi_order(:,8)-xi_order(:,4);
xi_ref(:,9)=xi_order(:,9)-xi_order(:,4);
xi_ref(:,10)=xi_order(:,10)-xi_order(:,4);

%education
xi_ref(:,11)=xi_order(:,11)-xi_order(:,12);
%smoking
xi_ref(:,12)=xi_order(:,14)-xi_order(:,13);
%age
xi_ref(:,13)=xi_order(:,15)-xi_order(:,16);
%race
xi_ref(:,14)=xi_order(:,18)-xi_order(:,17);
xi_ref(:,15)=xi_order(:,19)-xi_order(:,17);
xi_ref(:,16)=xi_order(:,20)-xi_order(:,17);
%Clusters
xi_ref(:,17)=xi_order(:,21)-xi_order(:,23);
xi_ref(:,18)=xi_order(:,22)-xi_order(:,23);
% xi_ref(:,19)=xi_order(:,24)-xi_order(:,21);
% xi_ref(:,20)=xi_order(:,25)-xi_order(:,21);
% xi_ref(:,21)=xi_order(:,26)-xi_order(:,21);
% for 9 cluster model
% xi_ref(:,22)=xi_order(:,27)-xi_order(:,21);
% xi_ref(:,23)=xi_order(:,28)-xi_order(:,21);
% xi_ref(:,24)=xi_order(:,29)-xi_order(:,21);


xi_cit=transpose(prctile(xi_ref,[50 2.5 97.5]));
p_pos=transpose(sum(xi_ref>0)/m_burn);



non_med=theta0_medi(:,:,1); 
low_med=theta0_medi(:,:,2); 
med_med=theta0_medi(:,:,3); 
hi_med=theta0_medi(:,:,4);  
alltheta=[non_med low_med med_med hi_med];

