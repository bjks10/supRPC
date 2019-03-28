%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SUPERVISED ROBUST PROFILE CLUSTERING prior  
%Programmer: Briana Stephenson
%Data: Simulated Set 
% adjust beta to simulate a t
% mu0~std normal, sig0 ~  IG(5/2,5/2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_n=bb;


%% load Simulated data %%
load(strcat('Sim_sRPCdata',num2str(sim_n),'.mat'))


food=probdata;
% remove misreported foods (v10,43,50,51,56)

y_cc=cc;
state=subpop_id; S=length(unique(state));
[n,p]=size(food); d=max(food(:));

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
n_s(s)=length(food(state==s,:));
idzs=repmat(1:p,n_s(s),1);
idz_s{s}=idzs(:);
food_s=food(state==s,:);
xs=food_s(:);
x_s{s}=xs;
lin_idxS{s}=sub2ind([p,d],idz_s{s},xs);
end

%% -- RPC Predictor Model SETUP -- %%
k_max=8;

     

%% SET UP HYPERPRIORS %%

    %pi_h for all classes
alpha=ones(1,k_max);
pi_h=drchrnd(alpha,1);

    %phi - cluster index
Ci=mnrnd(1,pi_h,n); [r, c]=find(Ci); gc=[r c];
    gc=sortrows(gc,1); Ci=gc(:,2);
n_Ci=sum(Ci);


%global theta0/1
 eta=ones(1,d);
theta0=zeros(p,k_max,d);
for k=1:k_max
    for j=1:p
        theta0(j,k,:)=drchrnd(eta,1);
    end
end



%% -- RESPONSE PROBIT MODEL SETUP -- %%
K0=k_max;
pcov=K0+S;
 X=zeros(n,pcov);
mu0=zeros(pcov,1);
% mu0=normrnd(0,1,[K0+S,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid
w_sid=zeros(n,S); 

for s=1:S
    w_sid(state==(s),s)=1;
end
%  w_sid(:,1)=1; %subpop 1 serve as state intercept



%% ------------ %%
%% data storage %%
%% ------------ %%
nrun=25000; burn=nrun/5; thin=100;

%predictor RPC model storage
pi_out=zeros(nrun,k_max);

dev_tot=zeros(nrun,1);
log_prob=zeros(nrun,1);
theta0_out=zeros(nrun/thin,p,k_max,d);
ci_out=zeros(nrun,n);

%response spv model storage
xi_out=zeros(nrun,pcov);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POSTERIOR COMPUTATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %temporary storage
ZBrand=zeros(n,1);
z_probit=zeros(n,1);

%% -- BEGIN MCMC -- %%

for iter=1:nrun
     
    %% -- update pi_h -- %%

    for h=1:k_max
        n_Ci(h)=sum(Ci==h);
    end
    alphaih=alpha+ n_Ci;
    pi_h=drchrnd(alphaih,1);
pi_out(iter,:)=pi_h;

  
    
%   %% -- Ci ~multinomial(pi_h) -- %%

  Cp_k=zeros(n,k_max);
for k=1:k_max
    t0h=reshape(theta0(:,k,:),p,d);
    tmpmat0=reshape(t0h(lin_idx),[n,p]);
    Cp_k(:,k)=pi_h(k)*prod(tmpmat0,2);
end
probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));

    t_ij=Cp_k./sum(Cp_k,2);
    t_logp=t_ij.*log(Cp_k);
    devlca_i=sum(t_logp,2);

    x_ci=mnrnd(1,probCi); [r, c]=find(x_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);

ci_out(iter,:)=Ci; %save cluster assignments for pairwise poseriors

% - update theta - %
dmat0=zeros(p,d);
for k=1:k_max
    Cis=repmat(Ci,[1,p]);
     ph0 = (Cis==k); %subj's in global cluster h
        for c = 1:d
             dmat0(:,c) = sum((food==c).*ph0)';
        end
        for j=1:p
            a_tn0=eta+dmat0(j,:);
            theta0(j,k,:) = drchrnd(a_tn0,1);
        end
end
    
  if mod(iter,thin)==0
     theta0_out(iter/thin,:,:,:)=theta0;
  end


%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%
%     if sum(isnan(xi_iter(:)))>0
%         break; 
%     end

Wup=[w_sid x_ci]; %covariate matrix with state/global
% Wup=[ones(n,1) w_sid(:,2:end) x_ci(:,2:end)];
  %add cluster L to referent group
%create latent z_probit
    %create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);
    
         
    %truncation for cases (0,inf)
    z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
    %truncation for controls (-inf,0)
    z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);
        %fix bounds for extreme cases 
            ninf=sum(z_probit==inf);
            nninf=sum(z_probit==-inf);
            if ninf>0
                wx_max=min(WXi_now(y_cc==1));
                r=rand(ninf,1);
                plo=normcdf(8); phi=1;
                % scale to [plo,phi]
                r=plo+(phi-plo)*r;
                r(r==1)=1-1e-15;
                % Invert through standard normal
                z=norminv(r);
                % apply shift and scale
                z_probit(z_probit==inf)=wx_max+z;
            end
            if nninf>0
                wx_min=max(WXi_now(y_cc==0));
                r=rand(nninf,1);
                plo=0; phi=normcdf(-38);
                % scale to [plo,phi]
                r=plo+(phi-plo)*r;
                r(r==0)=1e-15;
                % Invert through standard normal
                z=norminv(r);
                % apply shift and scale
                z_probit(z_probit==-inf)=wx_min+z;
            end
                
    

% Response Xi(B) update

sig0up=Sig0(1:pcov,1:pcov);
xi_0up=xi_0(1:pcov);

xi_sig_up=inv(sig0up)+(transpose(Wup)*Wup);
xi_mu_up2=(sig0up*transpose(xi_0up))+(transpose(Wup)*z_probit);
xi_mu_up=xi_sig_up\xi_mu_up2;
xi_up=mvnrnd(xi_mu_up,inv(xi_sig_up));
xi_iter(1:length(xi_up))=xi_up;
xi_out(iter,:)=xi_iter;

phiWXilca=normcdf(Wup*transpose(xi_up));
phiWXilca(phiWXilca==0)=1e-10;
   phiWXilca(phiWXilca==1)=1-1e-10;
devprob_i=y_cc.*log(phiWXilca) + (1-y_cc).*log(1-phiWXilca);
dev_tot(iter)=sum(devprob_i + devlca_i);

%% RELABELLING STEP TO ENCOURAGE MIXING %%

if mod(iter,10)==0
new_order=randperm(k_max);
newCi=Ci;
    for k=1:k_max
    newCi(Ci==k)=new_order(k);
    end
Ci=newCi;
theta0=theta0(:,new_order,:);
end


end

pi_burn=pi_out(burn+1:end,:);
theta0_burn=theta0_out((burn/thin)+1:end,:,:,:);
xi_burn=xi_out(burn+1:end,:);
ci_burn=ci_out(burn+1:end,:);
devtot_burn=dev_tot(burn+1:end);
% Logprob_burn=log_prob(burn+1:end);

m_perm=size(theta0_burn,1);
m=size(pi_burn,1);
m_thin=m/m_perm;
% 
 k_iter=sum(pi_burn>0.01,2);
 k_all=max(k_iter);
 k_in=median(k_iter);
k_med=median(sum(pi_burn>0.05,2))
%% PAPASPILOULIS POSTERIOR PAIRWISE MIXING %%
pd=pdist(transpose(ci_burn),'hamming'); %prcnt differ
cdiff=squareform(pd); %Dij
Zci=linkage(cdiff,'complete');
 dendrogram(Zci); % 5 groups selected based on results
% saveas(gcf,'supRPC_dendrogram.png')

zclust=cluster(Zci,'maxclust',k_max);

% use mode to identify relabel order for each iteration
 ci_relabel=zeros(m,k_max);
 for l=1:k_max
     ci_relabel(:,l)=mode(ci_burn(:,zclust==l),2);
 end

 
% save('LCA_SimSave1','pi_burn','theta0_burn','ci_relabel','Zci','ci_burn','Loglike0_burn','-v7.3');


 
%% reorder pi and theta0 parameters
pi_order=zeros(m,k_max);
theta0_order=zeros(m_perm,p,k_max,d);
xi_order=zeros(m,K0+S);
for iter=1:m
   iter_order=ci_relabel(iter,:);
   pi_order(iter,:)=pi_burn(iter,iter_order);
   if mod(iter,m_thin)==0
       iter_thin=iter/m_thin;
       theta0_order(iter_thin,:,:,:)=theta0_burn(iter_thin,:,iter_order,:);
   end
   cov_order=[1:S S+iter_order];
   xi_order(iter,:)=xi_burn(iter,cov_order);
end
% 
% %%identify modal patterns for each cluster
pi_med=median(pi_order);
theta0_med=reshape(median(theta0_order),[p,k_max,d]);
[M0, I0]=max(theta0_med,[],3);
% t_I0=transpose(I0);
% [g, ia, ic]=unique(t_I0,'rows');
% 
% pi_med=pi_med(ia)/sum(pi_med(ia));
% theta0_med=theta0_med(:,ia,:);

%  save('Simordered_LCAparms','theta0_order','pi_order','ci_relabel','Zci');
% 
% 
% 
% %% RE-RUN SUP-RPC WITH CORRECT LABELING ORDER %%
%reduced clustering set from dendrogram
w_ciz=zeros(n,k_max);
for k=1:k_max
    w_ciz(zclust==k,k)=1;
end
pcov=S+k_max;
xi_order=zeros(m,pcov);
% 
%  
  %% -- RESPONSE PROBIT -- %%
K0=k_max; %pcov=K0+S-1;
X=zeros(n,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;
n=size(zclust,1);

% Xi=zeros(n,pcov);

% devianceLCA0=zeros(m,1);
% z_probit=zeros(n,1);

%% -- BEGIN MCMC -- %%
% for iter=1:m
% 
% %Wup=[ones(n,1) w_sid(:,2:end) w_ciz(:,2:end)];
%   Wup=[w_sid w_ciz];
%     %probit model y=xb+e
%     
%     %create latent z_probit
% 
%         %create truncated normal for latent z_probit model
%         WXi_now=Wup*transpose(xi_iter);
% 
%         z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);
%         z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
%     if sum(isnan(z_probit)>0)
%         break;
%     end
%        
% 
%     % Response beta update
%     xi_mu_up1=inv(Sig0)+(transpose(Wup)*Wup);
%     xi_mu_up2=(Sig0*mu0)+(transpose(Wup)*z_probit); %%%%
%     xi_mu_up=xi_mu_up1\xi_mu_up2;
%     xi_sig_up=inv(inv(Sig0)+(transpose(Wup)*Wup));
%     xi_iter=mvnrnd(xi_mu_up,xi_sig_up);
%     xi_order(iter,:)=xi_iter;
% 
% phiWXilca=normcdf(Wup*transpose(xi_iter));
%    phiWXilca(phiWXilca==0)=1e-10;
%    phiWXilca(phiWXilca==1)=1-1e-10;
%     loglikei=y_cc.*log(phiWXilca) + (1-y_cc).*log(1-phiWXilca);
%     devianceLCA0(iter)=sum(loglikei);
% end
% burn=5000;
% xi_burn=xi_order(burn+1:end,:);
% xi_prc_nodem=transpose(prctile(xi_burn,[2.5 50 97.5]))


xi_med=median(xi_order);


% devianceLCAburn=devianceLCA0(burn+1:end);

p_pos=transpose(sum(xi_order>0)/m);

py_pred= normcdf(Wup*transpose(xi_med));

xi_mean=mean(xi_order);

theta0_mean=reshape(mean(theta0_order),[p,k_max,d]);

pi_m=mean(pi_order);
pi_mean=pi_m./sum(pi_m);

delmean=zeros(n,k_max);
   for h = 1:k_max
        theta0h = reshape(theta0_mean(:,h,:),p,d);
        tmpmat0 = reshape(theta0h(lin_idx),[n,p]);
         delmean(:,h) = pi_mean(h) * prod(tmpmat0,2);
   end 
    
    zup0 = bsxfun(@times,delmean,1./(sum(delmean,2)));
%dic-lca term 2-2    
   t_mean=zup0./sum(zup0,2);
    t_logpmean=t_mean.*log(zup0);
    devlca_mean=sum(t_logpmean,2);
    
    w_ci=mnrnd(1,zup0); [r, c]=find(w_ci); w_gc=[r c];
    w_gc=sortrows(w_gc,1); pred_ci=w_gc(:,2);

    
    Xmean=[w_sid w_ci]; %covariate matrix with state/global
      %add cluster 1 to referent group
WXi_mean=Xmean*transpose(xi_mean);     

phiWXimean=normcdf(WXi_mean);
% - pull extremes off the bounds - %
phiWXimean(phiWXimean==0)=1e-10;
   phiWXimean(phiWXimean==1)=1-1e-10;
   
    devprob_mean=y_cc.*log(phiWXimean) + (1-y_cc).*log(1-phiWXimean);
    %DIC based only on probit model
%     dic_probit=-4*mean(devianceLCAburn)+2*sum(loglike_mean);
dic_full2=-4*mean(devtot_burn)+2*sum(devprob_mean+devlca_mean);

save(strcat('py_simLCAResults',num2str(sim_n)),'py_pred','pred_ci','k_in','dic_full2','I0');
