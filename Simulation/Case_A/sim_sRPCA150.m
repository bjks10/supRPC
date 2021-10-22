bb = 151

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SUPERVISED ROBUST PROFILE CLUSTERING prior  
%Programmer: Briana Stephenson
%Data: NBDPS 
% Remove DP and replace with OFM global/local
% Decrease concentration parameter to encourage sparsity
% adjust beta to simulate a t
% mu0~std normal, sig0 ~ IG(5/2,5/2)
% Separate into Adaptive and Fixed Sampling algorithm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_n=bb;

%% load NBDPS data %%
 load(strcat('sim_sRPCdataA',num2str(sim_n),'.mat'))
% load('sim_sRPCtest3')
food=sampledata;

state=subpop_samp; S=length(unique(state));
[n,p]=size(food); d=max(food(:));

y_cc=true_y;
%vectorization of data
idz = repmat(1:p,n,1); idz = idz(:);
x_d = food(:); lin_idx = sub2ind([p,d],idz,x_d);
%subvectorization
idz_s=cell(S,1);
idz_s{S}=[];
lin_idxS=cell(S,1);
lin_idxS{S}=[];
x_s=cell(S,1);
x_s{S}=[];
n_s=zeros(S,1);
for s=1:S
    n_s(s)=length(food(state==(s),:));
    idzs=repmat(1:p,n_s(s),1);
    idz_s{s}=idzs(:);
    food_s=food(state==(s),:);
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

%gamma
% age=1; bge=1; %hypers for gamma
% % gamma_s=gamrnd(age,bge,[1,S]);
% gamma_s=ones(1,S);
% 
%nu_j^s
beta_s=1;
nu=betarnd(1,beta_s,[S,p]);


    %pi_h for all classes
alpha=ones(1,k_max)*(1/k_max);
pi_h=drchrnd(alpha,1);

    %phi - cluster index
x_Ci=mnrnd(1,pi_h,n); [r, c]=find(x_Ci); gc=[r c];
    gc=sortrows(gc,1); Ci=gc(:,2);
n_Ci=sum(x_Ci);


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
    G_ij=zeros(n,p);
 for s=1:S
     ns=n_s(s);
     nu_s=nu(s,:);
     G_ij(state==(s),:) = repmat(binornd(1,nu_s),[ns,1]);     % family index (0=global family,1=local family)
 end


%% SUBPOPULATION LPP NESTS %%
%determine number of global diets in each subpopulation
lambda_sk=drchrnd(alpha,S);

    L_ij=zeros(n,p); %local cluster label variable

    n_Lij=zeros(S,p,k_max);
for s=1:S
    for j=1:p
        x_sl=mnrnd(1,lambda_sk(s,:),n_s(s)); [r, c]=find(x_sl); gc=[r c];
        gc=sortrows(gc,1); 
        L_ij(state==(s),j)=gc(:,2);
        n_Lij(s,j,:)=sum(x_sl);
    end

end


%% -- RESPONSE PROBIT MODEL SETUP -- %%
pcov=k_max+S;
K0=k_max; X=zeros(n,pcov);
mu0=normrnd(0,1,[pcov,1]);
sigma0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
sig0=diag(sigma0);
xi_0=mvnrnd(mu0,sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid
w_sid=zeros(n,S); 

for s=1:S
    w_sid(state==(s),s)=1;
end
% Wmat=[w_sid x_Ci];
ep_kp=zeros(n,k_max); 

for k=1:k_max
   w_ip=zeros(n,k_max);
   w_ip(:,k)=ones(n,1);
   W_temp=[w_sid w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_iter));
   probit_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
   ep_kp(:,k)=exp(probit_kp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%  BEGIN ADAPTIVE SAMPLER FOR K-PROB GLOBAL AND KS-MED LOCAL %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

%% ------------ %%
%% data storage %%
%% ------------ %%
nrun=25000; 
burn=nrun/5; 

%GLOBAL/LOCAL PROFILE storage

pi_out=zeros(nrun,k_max);
z_probit=zeros(n,1);

lambdas_out=zeros(nrun,S,k_max);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POSTERIOR COMPUTATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %temporary storage
As=zeros(n,p); 
p_ij=zeros(n,p);
ZBrand=zeros(n,1);
   nL_ij=zeros(S,k_max); %store local cluster counts
   log_locrpc=zeros(n,1);

%% -- BEGIN MCMC -- %%

for iter=1:nrun
    
    %% -- update G_ij prob -- %%
 for s=1:S
     phrow1=L_ij(state==(s),:); phrow1=phrow1(:);
     theta1s=reshape(theta1(s,:,:,:),[p,k_max,d]);
     A = theta1s(sub2ind([p,k_max,d],idz_s{s},phrow1,x_s{s})); %index of subjects in theta1class
    A = reshape(A,[n_s(s),p]);
    As(state==(s),:)=A;
 end
 
 phrow0=repmat(Ci,[1,p]); phrow0=phrow0(:);
 B = theta0(sub2ind([p,k_max,d],idz,phrow0,x_d));
 B = reshape(B,[n,p]);
 
 for s=1:S
     ns=n_s(s); nu_s=nu(s,:);
     p_ij(state==(s),:)=(repmat(nu_s,[ns,1]).*B(state==(s),:)./((repmat(nu_s,[ns,1]).*B(state==(s),:))+(repmat(1-nu_s,[ns,1]).*As(state==(s),:))));
 end
 
    G_ij=binornd(1,p_ij);
    
    %% -- update pi_h -- %%

    for h=1:k_max
        n_Ci(h)=sum(Ci==h);
    end
    alphaih=alpha+ n_Ci;
    pi_h=drchrnd(alphaih,1);
pi_out(iter,:)=pi_h;

  %%-- update lambda_sk --%%
  
  
  for s=1:S
      phi_s=L_ij(state==(s),:);
      for l=1:k_max
          nL_ij(s,l)=sum(phi_s(:)==l);
      end
      kn_Lij=alpha+nL_ij(s,:);
      lambda_sk(s,:)=drchrnd(kn_Lij,1);
  end
    lambdas_out(iter,:,:)=lambda_sk;
 
    
%   %% -- Ci ~multinomial(pi_h) -- %%

  Cp_k=zeros(n,k_max);
for k=1:k_max
    t0h=reshape(theta0(:,k,:),p,d);
    tmpmat0=reshape(t0h(lin_idx),[n,p]);
    Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2).*ep_kp(:,k);
end
log_globrpc=sum(log(Cp_k),2);
probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));

    x_ci=mnrnd(1,probCi); [r, c]=find(x_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);



    for s=1:S
        Lijs=zeros(n_s(s),k_max,p);
             for h = 1:k_max
                theta1hs = reshape(theta1(s,:,h,:),p,d);
                tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
                 Lijs(:,h,:) = lambda_sk(s,h) * tmpmat1.^(G_ij(state==(s),:)==0);
             end
             log_locrpc(state==s,1)=sum(log(sum(Lijs,2)),3);
            sumLijs=repmat(sum(Lijs,2),[1,k_max,1]);
            zupS = Lijs./sumLijs;
            for j=1:p
                sub_pj=reshape(zupS(:,:,j),[n_s(s),k_max]);
                l_ij=mnrnd(1,sub_pj);
                [r, c]=find(l_ij); x_l=[r c];
                x_sl=sortrows(x_l,1);
                L_ij(state==(s),j)=x_sl(:,2);
            end     
    end 

    
% - update theta - %
dmat0=zeros(p,d);
dmat1=zeros(p,d);
for k=1:k_max
    Cis=repmat(Ci,[1,p]).*G_ij;
     ph0 = (Cis==k); %subj's in global cluster h
        for c = 1:d
             dmat0(:,c) = sum((food==c).*ph0)';
        end
        for j=1:p
            a_tn0=eta+dmat0(j,:);
            theta0(j,k,:) = drchrnd(a_tn0,1);
        end
end
    
  for s=1:S
      phis=L_ij(state==(s),:).*(1-G_ij(state==(s),:));
      foods=food(state==(s),:);
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
        Gs=G_ij(state==(s),:);
        nu(s,:) = betarnd(1 + sum(Gs), beta(s) + sum(1-Gs));
    end
  nu(nu==1) = 1-1e-06;
  nu(nu==0) = 1e-06; 

  
  % - update beta - %
  for s=1:S
    beta(s) = gamrnd(abe + p,1./( bbe - sum(log(1-nu(s,:)))));
  end


%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%


Wup=[w_sid x_ci];
%create latent z_probit
    %create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);
%truncation for cases (0,inf)
z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
%truncation for controls (-inf,0)
z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);

    %control extremes;
    
    if sum(z_probit==Inf)>0
        z_probit(z_probit==Inf)=norminv(1-1e-6);
    end 
    if sum(z_probit==-Inf)>0
        z_probit(z_probit==-Inf)=norminv(1e-6);
    end

% Response Xi(B) update



    xi_s_up1=inv(sig0)+(transpose(Wup)*Wup);
    xi_mu1=(sig0\transpose(xi_0))+(transpose(Wup)*z_probit); %%%%
    xi_mu_up=xi_s_up1\xi_mu1;
    xi_iter=mvnrnd(xi_mu_up,inv(xi_s_up1));
%     xi_out(iter,:)=xi_iter;
    wxi=Wup*transpose(xi_iter);

    phi_wxi=normcdf(wxi);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-6;
        phi_wxi(phi_wxi==1)=1-1e-6;
    lprobit_yi=y_cc.*log(phi_wxi) + (1-y_cc).*log(1-phi_wxi);




    %prep matrix for probit_Ci
    for k=1:k_max
        w_ip=zeros(n,k_max);
        w_ip(:,k)=ones(n,1);
        W_temp=[w_sid w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_iter));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp<1e-15) = 1e-10; 
            p_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
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

% beta_burn=beta_out(burn+1:end,:);
lambdas_burn=lambdas_out(burn+1:end,:,:);
pi_burn=pi_out(burn+1:end,:);

k_prob=median(sum(pi_burn>0.05,2));

bl=size(lambdas_burn,1);
ks_med=zeros(S,1);
for s=1:S
    s_lambda = reshape(lambdas_burn(:,s,:),[bl,k_max]);
    ks_med(s)=median(sum(s_lambda>0.05,2));
end
ks_max=max(ks_med);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  BEGIN FIXED SAMPLER FOR K-PROB GLOBAL AND KS-MED LOCAL %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nrun=25000; 
burn=15000; 
thin=10;

%predictor RPC model storage
beta_out=zeros(nrun,S);
nu_out=zeros(nrun,S,p);
pi_out=zeros(nrun,k_prob);
z_probit=zeros(n,1);

theta0_out=zeros(nrun/thin,p,k_prob,d);
theta1_out=zeros(nrun/thin,S,p,ks_max,d);
lambdas_out=zeros(nrun,S,ks_max);
ks_out=zeros(nrun,S+1);
ci_out=zeros(nrun,n);
%response spv model storage
xi_out=zeros(nrun,S+k_prob);
loglike_out=zeros(nrun,1);
   nL_ij=zeros(S,ks_max); %store local cluster counts


%% INITIAL PARAMETERS FOR FIXED SAMPLER %%
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
x_Ci=mnrnd(1,pi_h,n); [r, c]=find(x_Ci); gc=[r c];
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

theta1=zeros(S,p,ks_max,d);
for s=1:S
    kss=ks_med(s);
    for k=1:kss
        for j=1:p
    theta1(s,j,k,:)=drchrnd(eta,1);
        end
    end
end

    %global G_ij
    G_ij=zeros(n,p);
 for s=1:S
     ns=n_s(s);
     nu_s=nu(s,:);
     G_ij(state==(s),:) = repmat(binornd(1,nu_s),[ns,1]);     % family index (0=global family,1=local family)
 end


%% SUBPOPULATION LPP NESTS %%
%determine number of local diets in each subpopulation
alpha1=ones(1,ks_max)*(1/k_prob);
lambda_sk=zeros(S,ks_max);
for s=1:S
    kss=ks_med(s);
    a1=alpha1(1:kss);
    lambda_sk(s,1:kss)=drchrnd(a1,1);
end


    L_ij=zeros(n,p); %local cluster label variable

    n_Lij=zeros(S,p,ks_max);
for s=1:S
    for j=1:p
        x_sl=mnrnd(1,lambda_sk(s,:),n_s(s)); [r, c]=find(x_sl); gc=[r c];
        gc=sortrows(gc,1); 
        L_ij(state==(s),j)=gc(:,2);
        n_Lij(s,j,:)=sum(x_sl);
    end
end


%% -- RESPONSE PROBIT MODEL SETUP -- %%
pcov=k_prob+size(w_sid,2);
X=zeros(n,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid

Wmat=[w_sid x_Ci];
ep_kp=zeros(n,k_prob); 

for k=1:k_prob
   w_ip=zeros(n,k_prob);
   w_ip(:,k)=ones(n,1);
   W_temp=[w_sid w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_iter));
        phi_temp(phi_temp==1)=1-1e-10;
        phi_temp(phi_temp==0)=1e-10;
   probit_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
   ep_kp(:,k)=exp(probit_kp);
end

for iter=1:nrun
    
    %% -- update G_ij prob -- %%
 for s=1:S
     
     phrow1=L_ij(state==(s),:); phrow1=phrow1(:);
     theta1s=reshape(theta1(s,:,:,:),[p,ks_max,d]);
     A = theta1s(sub2ind([p,ks_max,d],idz_s{s},phrow1,x_s{s})); %index of subjects in theta1class
    A = reshape(A,[n_s(s),p]);
    As(state==(s),:)=A;
 end
 
 phrow0=repmat(Ci,[1,p]); phrow0=phrow0(:);
 B = theta0(sub2ind([p,k_prob,d],idz,phrow0,x_d));
 B = reshape(B,[n,p]);
 
 for s=1:S
     ns=n_s(s); nu_s=nu(s,:);
     p_ij(state==(s),:)=(repmat(nu_s,[ns,1]).*B(state==(s),:)./((repmat(nu_s,[ns,1]).*B(state==(s),:))+(repmat(1-nu_s,[ns,1]).*As(state==(s),:))));
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
      phi_s=L_ij(state==(s),:);
      kss=ks_med(s);
      for l=1:kss
          nL_ij(s,l)=sum(phi_s(:)==l);
      end
      kn_Lij=alpha1(1:kss)+nL_ij(s,1:kss);
     lambda_sk(s,1:kss)=drchrnd(kn_Lij,1);
  end
    lambdas_out(iter,:,:)=lambda_sk;
 
    
   %% -- Ci ~multinomial(pi_h) -- %%
    %% GLOBAL PROFILE ASSIGNMENT %%

  Cp_k=zeros(n,k_prob);
for k=1:k_prob
    t0h=reshape(theta0(:,k,:),p,d);
    tmpmat0=reshape(t0h(lin_idx),[n,p]);
    Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2).*ep_kp(:,k);
end
log_globrpc=sum(log(Cp_k),2);
probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));

    x_ci=mnrnd(1,probCi); [r, c]=find(x_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);
ci_out(iter,:)=Ci;

    %% -- Lij ~multinomial(lambda_sk) -- %%
        %% LOCAL PROFILE ASSIGNMENT %%
    for s=1:S
        ks1=ks_med(s);
        Lijs=zeros(n_s(s),ks1,p);
             for h = 1:ks1
                theta1hs = reshape(theta1(s,:,h,:),p,d);
                tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
                 Lijs(:,h,:) = lambda_sk(s,h) * tmpmat1.^(G_ij(state==(s),:)==0);
             end
             log_locrpc(state==s,1)=sum(log(sum(Lijs,2)),3);
            sumLijs=repmat(sum(Lijs,2),[1,ks1,1]);
            zupS = Lijs./sumLijs;
            for j=1:p
                sub_pj=reshape(zupS(:,:,j),[n_s(s),ks1]);
                l_ij=mnrnd(1,sub_pj);
                [r, c]=find(l_ij); x_l=[r c];
                x_sl=sortrows(x_l,1);
                L_ij(state==(s),j)=x_sl(:,2);
            end     
    end 

    
% - update theta - %
dmat0=zeros(p,d);
dmat1=zeros(p,d);
for k=1:k_prob
    Cis=repmat(Ci,[1,p]).*G_ij;
     ph0 = (Cis==k); %subj's in global cluster h
        for c = 1:d
             dmat0(:,c) = sum((food==c).*ph0)';
        end
        for j=1:p
            a_tn0=eta+dmat0(j,:);
            theta0(j,k,:) = drchrnd(a_tn0,1);
        end
end
    
  for s=1:S
      kss=ks_med(s);
      phis=L_ij(state==(s),:).*(1-G_ij(state==(s),:));
      foods=food(state==(s),:);
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

  if mod(iter,thin)==0
     theta0_out(iter/thin,:,1:size(theta0,2),:)=theta0;
     theta1_out(iter/thin,:,:,1:size(theta1,3),:)=theta1;
  end
  
  
  % update nu_j %
    for s=1:S
        Gs=G_ij(state==(s),:);
        nu(s,:) = betarnd(1 + sum(Gs), beta(s) + sum(1-Gs));
    end
  nu(nu==1) = 1-1e-06;
  nu(nu==0) = 1e-06; 

  nu_out(iter,:,:)=nu;
  
  % - update beta - %
  for s=1:S
    beta(s) = gamrnd(abe + p,1./( bbe - sum(log(1-nu(s,:)))));
  end
 beta_out(iter,:)=beta;

%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%


Wup=[w_sid x_ci];
%create latent z_probit
    %create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);
%truncation for cases (0,inf)
z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
%truncation for controls (-inf,0)
z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);

    %control extremes;
    
    if sum(z_probit==Inf)>0
        z_probit(z_probit==Inf)=norminv(1-1e-6);
    end 
    if sum(z_probit==-Inf)>0
        z_probit(z_probit==-Inf)=norminv(1e-6);
    end

% Response Xi(B) update



    xi_s_up1=inv(Sig0)+(transpose(Wup)*Wup);
    xi_mu1=(Sig0\transpose(xi_0))+(transpose(Wup)*z_probit); %%%%
    xi_mu_up=xi_s_up1\xi_mu1;
    xi_iter=mvnrnd(xi_mu_up,inv(xi_s_up1));
    xi_out(iter,:)=xi_iter;
    
    wxi=Wup*transpose(xi_iter);

    phi_wxi=normcdf(wxi);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-6;
        phi_wxi(phi_wxi==1)=1-1e-6;
    lprobit_yi=y_cc.*log(phi_wxi) + (1-y_cc).*log(1-phi_wxi);




    %prep matrix for probit_Ci
    for k=1:k_prob
        w_ip=zeros(n,k_prob);
        w_ip(:,k)=ones(n,1);
        W_temp=[w_sid w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_iter));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp<1e-15) = 1e-10; 
            p_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
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

beta_burn=beta_out(burn+1:end,:);
lambdas_burn=lambdas_out(burn+1:end,:,:);
pi_burn=pi_out(burn+1:end,:);
theta0_burn=theta0_out((burn/thin)+1:end,:,:,:);
theta1_burn=theta1_out((burn/thin)+1:end,:,:,:,:);
nu_burn=nu_out(burn+1:end,:,:);
xi_burn=xi_out(burn+1:end,:);
ci_burn=ci_out(burn+1:end,:);
% dic_burn=dic_out(burn+1:end);



[m_perm,S,p,ks_max,d]=size(theta1_burn);
[m,n]=size(ci_burn);
thin=m/m_perm;
%check number of nonempty clusters

% 

theta1_med=reshape(median(theta1_burn),[S,p,ks_max,d]);
lambdas_med=reshape(median(lambdas_burn),[S,ks_max]);


lambdas_x=cell(S,1); lambdas_x{S}=[];
theta1_x=cell(S,1); theta1_x{S}=[];
for s=1:S
    wss=lambdas_med(s,:);
lambdas_x{s}=wss(wss>0.01);
theta1_x{s}=reshape(theta1_med(s,:,wss>0.01,:),[p,length(lambdas_x{s}),d]);
end
% 
ks=zeros(S,1);
val1=cell(S,1); ind1=cell(S,1);
for s=1:S
    ks(s)=size(theta1_x{s},2);
    [val1{s},ind1{s}]=max(theta1_x{s},[],3);
end

%% PAPASPILOULIS POSTERIOR PAIRWISE MIXING %%
pd=pdist(transpose(ci_burn),'hamming'); %prcnt differ
cdiff=squareform(pd); %Dij
Zci=linkage(cdiff,'complete');
dendrogram(Zci); % 6 groups selected based on results
% saveas(gcf,'supRPCsim_dendrogram.png')

clustk1=k_prob; %as determined by dendrogram
zclust=cluster(Zci,'maxclust',clustk1);

% zclust_full=cluster(Zci,'maxclust',k_med);

% use mode to identify relabel order for each iteration
 ci_relabel=zeros(m,clustk1);

 for l=1:clustk1
     ci_relabel(:,l)=mode(ci_burn(:,zclust==l),2);
 end


%% reorder pi and theta0 parameters
% [~,ord_c]=sort(ci_relabel(1,:));
% ci_relabel=ci_relabel(:,[ord_c]);
pi_order=zeros(m,clustk1);
theta0_order=zeros(m_perm,p,clustk1,d);
xi_order=zeros(m,clustk1+S);
for iter=1:m
   iter_order=ci_relabel(iter,:);
%    iter_uni=unique(iter_order);
   pi_order(iter,:)=pi_burn(iter,iter_order);
%     pi_order(iter,1:length(iter_uni))=pi_burn(iter,iter_uni);
    s_iteruni=S+iter_order;
   xi_order(iter,:)=[xi_burn(iter,1:S) xi_burn(iter,s_iteruni)];
   
   if mod(iter,thin)==0
       iter_thin=iter/thin;
       theta0_order(iter_thin,:,:,:)=theta0_burn(iter_thin,:,iter_order,:);
   end
end


pi_med=median(pi_order);
pi_medi=pi_med/sum(pi_med);
xi_med=median(xi_order);
xi_prc=[prctile(xi_order,2.5); prctile(xi_order,97.5)];
% pi_med=pi_med(pi_med>0); k_in=sum(pi_med>0);
theta0_med=reshape(median(theta0_order),[p,clustk1,d]);
% theta0_med=theta0_med(:,pi_med>0.05,:);
[M0, I0]=max(theta0_med,[],3);
t_I0=transpose(I0);
[uc, ia, ic]=unique(t_I0,'rows');

pi_med=pi_order(:,ia);
theta0_med=theta0_med(:,ia,:);
sia=S+ia;
kia=length(ia);
xi_med=[xi_med(1:S) xi_med(sia)];
xi_prc=[xi_prc(:,1:S) xi_prc(:,sia)];

theta0_order=theta0_order(:,:,ia,:);
pi_order=pi_order(:,ia);
xi_order=[xi_order(:,1:S) xi_order(:,sia)];
p_pos=transpose(sum(xi_order>0)/m);


loglike_srpclocal=zeros(n,1);
ep_kp=zeros(n,kia);
loglikesrpc_thin=zeros(m_perm,1);
G_ij=zeros(n,p); L_ij=zeros(n,p);

for ii=1:m_perm
    i_thin=ii*thin;
    pi_thin=pi_order(i_thin,:);
    theta0_thin=reshape(theta0_order(ii,:,:,:),[p,kia,d]);
    theta1_thin=reshape(theta1_burn(ii,:,:,:,:),[S,p,ks_max,d]);
    lambda_thin=reshape(lambdas_burn(i_thin,:,:),[S,ks_max]);
    nu_thin=reshape(nu_burn(i_thin,:,:),[S,p]);
    pi_h=pi_thin/sum(pi_thin);
    xi_itr=xi_order(i_thin,:);
    for s=1:S
        nu_s=reshape(nu_thin(s,:),[1,p]);
        G_ij(state==(s),:)=binornd(1,repmat(nu_s,[n_s(s),1]));
    end
    
    for k=1:kia
        w_ip=zeros(n,kia);
        w_ip(:,k)=ones(n,1);
        W_temp=[w_sid w_ip];
        phi_temp=normcdf(W_temp*transpose(xi_itr));
        %correct for MATLAB precision error
            phi_temp(phi_temp==1) = 1-1e-10;
            phi_temp(phi_temp==0) = 1e-10; 
        p_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
        ep_kp(:,k)=exp(p_kp);
    end
    
        %assign global cluster
      Cp_k=zeros(n,kia);
    for k=1:kia
        t0h=reshape(theta0_thin(:,k,:),p,d);
        tmpmat0=reshape(t0h(lin_idx),[n,p]);
        Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2).*ep_kp(:,k);
    end

    probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));
    log_globrpc=log(sum(Cp_k,2));
    tij_logglobrpc=sum(probCi.*log(Cp_k),2);
    w_ci=mnrnd(1,probCi); 
    
      %assign local cluster
    for s=1:S
        lts=lambda_thin(s,:);
        lambda_ii=lts(lts>0.05);
        lambda_is=lambda_ii/sum(lambda_ii);
        ks=length(lambda_is);
        theta1_is=reshape(theta1_thin(s,:,lts>0.05,:),[p,ks,d]);
    Lijs=zeros(n_s(s),ks,p);
         for h = 1:ks
            theta1hs = reshape(theta1_is(:,h,:),p,d);
            tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
             Lijs(:,h,:) = lambda_is(h) * tmpmat1.^(G_ij(state==(s),:)==0);
         end  
        sumLijs=repmat(sum(Lijs,2),[1,ks,1]);
        zupS = Lijs./sumLijs;
        for j=1:p
            sub_pj=reshape(zupS(:,:,j),[n_s(s),ks]);
            l_ij=mnrnd(1,sub_pj);
            [r, c]=find(l_ij); x_l=[r c];
            x_sl=sortrows(x_l,1);
            L_ij(state==(s),j)=x_sl(:,2);
        end  
    locallike=sum(Lijs,2);
    loglike_srpclocal(state==(s))=sum(log(locallike),3);
    end
    
  %create truncated normal for latent z_probit model
    Wup=[w_sid w_ci];
        WXi_now=Wup*transpose(xi_itr);

    phi_wxi=normcdf(WXi_now);
        %remove extremes 
        phi_wxi(phi_wxi==0)=1e-10;
        phi_wxi(phi_wxi==1)=1-1e-10;
    lprobit_yi=y_cc.*log(phi_wxi) + (1-y_cc).*log(1-phi_wxi);

    loglike_srpc=tij_logglobrpc+loglike_srpclocal;
    loglikesrpc_thin(ii)=sum(loglike_srpc);
end


nu_med=reshape(median(nu_burn),[S,p]);
G_med=zeros(n,p);
for s=1:S
   G_med(state==(s),:)=binornd(1,repmat(nu_med(s,:),[n_s(s),1]));
end

ep_kp=zeros(n,kia);
for k=1:kia
   w_ip=zeros(n,kia);
   w_ip(:,k)=ones(n,1);
   W_temp=[w_sid w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_med));
    phi_temp(phi_temp==1) = 1-1e-10;
    phi_temp(phi_temp==0) = 1e-10; 
   p_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
   ep_kp(:,k)=exp(p_kp);
 
end

like_srpcmed=zeros(n,1);
delmed=zeros(n,kia);
   for h = 1:kia
        t0h = reshape(theta0_med(:,h,:),p,d);
        theta0h=bsxfun(@times,t0h,1./sum(t0h,2));
        tmpmat0 = reshape(theta0h(lin_idx),[n,p]);
        delmed(:,h) = pi_med(h)*prod(tmpmat0.^G_med,2).*ep_kp(:,h);
    end 
    zup0 = bsxfun(@times,delmed,1./(sum(delmed,2)));
    tij_logglobrpcmed=sum(zup0.*log(delmed),2);

    wm_ci=mnrnd(1,zup0); [r, c]=find(wm_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);
loglike_locsrpcmed=zeros(n,1);
for s=1:S
        lambda_m=lambdas_x{s};
        lambda_med=lambda_m/sum(lambda_m);
        theta1_ms=theta1_x{s};
        ks=length(lambda_med);
        
    Lijs=zeros(n_s(s),ks,p);
         for h = 1:ks
            theta1hs = reshape(theta1_ms(:,h,:),p,d);
            tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
             Lijs(:,h,:) = lambda_med(h) * tmpmat1.^(G_med(state==(s),:)==0);
         end  
        sumLijs=repmat(sum(Lijs,2),[1,ks,1]);
        zupS = Lijs./sumLijs;
        for j=1:p
            sub_pj=reshape(zupS(:,:,j),[n_s(s),ks]);
            l_ij=mnrnd(1,sub_pj);
            [r, c]=find(l_ij); x_l=[r c];
            x_sl=sortrows(x_l,1);
            L_ij(state==(s),j)=x_sl(:,2);
        end  
    locallike_med=sum(Lijs,2);
    loglike_locsrpcmed(state==(s))=sum(log(locallike_med),3);
end

W_med=[w_sid wm_ci];
WX_med=W_med*transpose(xi_med);
py_pred=normcdf(WX_med);
pred_ci=Ci;
y_mse=immse(py_pred,phi_WXtrue);

pred_nu=nu_med;
loglike_srpcmed=tij_logglobrpcmed+loglike_locsrpcmed;

DIC_star=-6*mean(loglikesrpc_thin)+4*sum(loglike_srpcmed);
DIC=-4*median(loglikesrpc_thin)+2*sum(loglike_srpcmed);
nu_mse=immse(nu_med,trueG);
save(strcat('py_simResults_A',num2str(sim_n)),'py_pred','pred_ci','pred_nu','DIC','DIC_star','t_I0','y_mse','nu_mse');

    clf  
    %plot comparing predicted to true nu
subplot(1,2,1);
heatmap(transpose(nu_med));
title('Derived Local Deviations');

subplot(1,2,2);
heatmap(transpose(trueG));
title('True Local Deviations');
saveas(gcf,strcat('G',num2str(sim_n),'_deviations.png'))


% Heatmap of theta0 - global pattern mode
thetafile=strcat('theta0sRPC_simA',num2str(sim_n),'.fig');
figure; 
    h=heatmap(t_I0(2:4,:))
    h.YLabel = "supRPC Global";
    h.XLabel = "Exposure variables";
    h.Colormap = parula
saveas(gcf,thetafile)

truethetafile=strcat('theta0simA',num2str(sim_n),'.fig');
figure; 
    h=heatmap(transpose(true_global))
    h.YLabel = "True Global";
    h.XLabel = "Exposure variables";
    h.Colormap = parula
saveas(gcf,truethetafile)