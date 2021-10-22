bb = 151

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRADITIONAL LCA (6-classes)  
%Programmer: Briana Stephenson
%Data: NBDPS 
%Parameters estimated: 
% pi (cluster membership)
% theta (cluster multinomial density)
% save output for import to R for label switch postprocess
%EDIT -- Remove Croissants, fruit punch, veg/soy burger, soy milk
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

K=4; %six classes

%PRIOR FOR PI (ai = 1) -- cluster membership probability
alpha = ones(1,K);
pi0=drchrnd(alpha,1);

Ci=mnrnd(1,pi0,n); [r, c]=find(Ci); gc=[r c];
    gc=sortrows(gc,1); Ci=gc(:,2);
n_Ci=sum(Ci);

%PRIOR FOR THETA (JEFFREY'S PRIOR - a_p =1/2)
 eta=ones(1,d);
theta0=zeros(p,K,d);
for h=1:K
    for j=1:p
        theta0(j,h,:)=drchrnd(eta,1);
    end
end


%%%%%% MCMC %%%%%%
nrun = 25000; burn = 5000;
%Matrix Storage for MCMC

pi_out = zeros(nrun,K); %simulated set of gammas

theta0_out = zeros(nrun,p,K,d); %simulated rhos

%% -- RESPONSE PROBIT MODEL SETUP -- %%
pcov=K+S;
pdem=S;
X=zeros(n,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid
w_sid=zeros(n,S); 

for s=1:S
    w_sid(state==(s),s)=1;
end

ep_kp=zeros(n,K); 

for k=1:K
   w_ip=zeros(n,K);
   w_ip(:,k)=ones(n,1);
   W_temp=[w_sid w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_iter));
        phi_temp(phi_temp==1)=1-1e-10;
        phi_temp(phi_temp==0)=1e-10;
   probit_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
   ep_kp(:,k)=exp(probit_kp);
end



%response spv model storage
xi_out=zeros(nrun,pcov);
ci_out=zeros(nrun,n);
loglik_lca=zeros(nrun,1);

    %temporary storage
ZBrand=zeros(n,1);
z_probit=zeros(n,1);
%-----MCMC COMPUTATION------%

for iter=1:nrun
      %% -- update pi_h -- %%

    for h=1:K
        n_Ci(h)=sum(Ci==h);
    end
    alphaih=alpha+ n_Ci;
    pi_h=drchrnd(alphaih,1);
pi_out(iter,:)=pi_h;

  
    
%   %% -- Ci ~multinomial(pi_h) -- %%

  Cp_k=zeros(n,K);
for l=1:K
    t0h=reshape(theta0(:,l,:),p,d);
    tmpmat0=reshape(t0h(lin_idx),[n,p]);
    Cp_k(:,l)=pi_h(l)*prod(tmpmat0,2).*ep_kp(:,l);
end

probCi = bsxfun(@times,Cp_k,1./(sum(Cp_k,2)));
log_lca=log(sum(Cp_k,2));
    x_ci=mnrnd(1,probCi); [r, c]=find(x_ci); x_gc=[r c];
    x_gc=sortrows(x_gc,1); Ci=x_gc(:,2);

ci_out(iter,:)=Ci; %save cluster assignments for pairwise poseriors

% - update theta - %
dmat0=zeros(p,d);
for h=1:K
    Cis=repmat(Ci,[1,p]);
     ph0 = (Cis==h); %subj's in global cluster h
        for c = 1:d
             dmat0(:,c) = sum((food==c).*ph0)';
        end
        for j=1:p
            a_tn0=eta+dmat0(j,:);
            theta0(j,h,:) = drchrnd(a_tn0,1);
        end
end
    
  
     theta0_out(iter,:,:,:)=theta0;



%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%


% Wup=[w_sid x_ci(:,2:end)]; %covariate matrix with state/global
Wup=[w_sid x_ci];
  %add cluster L to referent group
% pcov=size(Wup,2);
%create latent z_probit
    %create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);
%truncation for cases (0,inf)
z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
%truncation for controls (-inf,0)
z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);

    %control extremes;
    
    if sum(z_probit==Inf)>0
        z_probit(z_probit==Inf)=norminv(1-1e-10);
    end 
    if sum(z_probit==-Inf)>0
        z_probit(z_probit==-Inf)=norminv(1e-10);
    end

% Response Xi(B) update

    xi_s_up1=inv(Sig0)+(transpose(Wup)*Wup);
    xi_mu1=(Sig0\transpose(xi_0))+(transpose(Wup)*z_probit); %%%%
    xi_mu_up=xi_s_up1\xi_mu1;
    xi_iter=mvnrnd(xi_mu_up,inv(xi_s_up1));
    xi_out(iter,:)=xi_iter;
    wxi=Wup*transpose(xi_iter);

phiWXilca=normcdf(Wup*transpose(xi_iter));
    phiWXilca(phiWXilca==0)=1e-10;
    phiWXilca(phiWXilca==1)=1-1e-10;
loglikei=y_cc.*log(phiWXilca) + (1-y_cc).*log(1-phiWXilca);
loglik_lca(iter)=sum(log_lca);

%% RELABELLING STEP TO ENCOURAGE MIXING %%

    if mod(iter,10)==0
        new_order=randperm(K);
        newCi=Ci;
        for h=1:K
            newCi(Ci==h)=new_order(h);
        end
        Ci=newCi;
        theta0=theta0(:,new_order,:);
        ep_kp=ep_kp(:,new_order);
    end


end


%%%%%%%%%%%%%%%% SAVE OUTPUT FOR LABEL SWITCH POSTPROCESS %%%%%%%%
piburn=pi_out(burn+1:nrun,:);
thetaburn=theta0_out(burn+1:nrun,:,:,:);
ciburn=ci_out(burn+1:end,:);
loglik_lcaburn=loglik_lca(burn+1:end);
xiburn=xi_out(burn+1:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

% save(strcat('NBDPS_BLCAout',num2str(K)),'piburn','thetaburn','ciburn','Loglike0_burn','loglik_lcaburn','xiburn');


m_perm=size(thetaburn,1);
m=size(piburn,1);
m_thin=m/m_perm;

%% PAPASPILOULIS POSTERIOR PAIRWISE MIXING %%
k_in=median(sum(piburn>0.01,2));
pd=pdist(transpose(ciburn),'hamming'); %prcnt differ
cdiff=squareform(pd); %Dij
Zci=linkage(cdiff,'complete');

% zclust=cluster(Zci,'maxclust',k_in);
zclustK=cluster(Zci,'maxclust',K);


%  ci_relabel=zeros(m,k_in);
 ci_relabel=zeros(m,K);

%  for l=1:k_in
for l=1:K
     ci_relabel(:,l)=mode(ciburn(:,zclustK==l),2);
end

 
 %% reorder pi and theta0 parameters
pi_order=zeros(m,K);
theta0_order=zeros(m,p,K,d);
xi_order=zeros(m,pcov);
for iter=1:m
   iter_order=ci_relabel(iter,:);
   pi_order(iter,:)=piburn(iter,iter_order);
   theta0_order(iter,:,:,:)=thetaburn(iter,:,iter_order,:);
    s_iterorder=pdem+iter_order;
   xi_order(iter,:)=[xiburn(iter,1:pdem) xiburn(iter,s_iterorder)];
end

% %%identify modal patterns for each cluster
pi_med=median(pi_order);
theta0_med=reshape(median(theta0_order),[p,K,d]);
[M0, I0]=max(theta0_med,[],3);
t_I0=transpose(I0);
[g, ia, ic]=unique(t_I0,'rows');

pi_med=pi_med(ia)/sum(pi_med(ia));
kia=length(ia);
theta0_med=theta0_med(:,ia,:);
p_ia=pdem+transpose(ia);
p_order=[1:4 p_ia];
xi_ord=xi_order(:,p_order);
xi_med=median(xi_ord);
xi_ci=prctile(xi_ord,[50 2.5 97.5]);


ep_kpl=zeros(n,kia);
for k=1:kia
   w_ip=zeros(n,kia);
   w_ip(:,k)=ones(n,1);
   W_temp=[w_sid w_ip];
   phi_temp=normcdf(W_temp*transpose(xi_med));
   probit_kp=y_cc.*log(phi_temp)+(1-y_cc).*log(1-phi_temp);
   ep_kpl(:,k)=exp(probit_kp);
end

delmean=zeros(n,kia);
   for h = 1:kia
        theta0h = reshape(theta0_med(:,h,:),p,d);
        tmpmat0 = reshape(theta0h(lin_idx),[n,p]);
         delmean(:,h) = pi_med(h) * prod(tmpmat0,2).*ep_kpl(:,h);
    end 
    zup0 = bsxfun(@times,delmean,1./(sum(delmean,2)));
    loglca_med=log(sum(delmean,2));
    med_ci=mnrnd(1,zup0); [r, c]=find(med_ci); w_gc=[r c];
    w_gc=sortrows(w_gc,1); pred_ci=w_gc(:,2);

    
    Wmed=[w_sid med_ci]; %covariate matrix with state/global
      %add cluster 1 to referent group
WXi_med=Wmed*transpose(xi_med);     

phiWXimed=normcdf(WXi_med);
% - pull extremes off the bounds - %
phiWXimed(phiWXimed==0)=1e-10;
   phiWXimed(phiWXimed==1)=1-1e-10;
 py_pred=phiWXimed; 
y_mse=immse(py_pred,phi_WXtrue)

    %DIC based only on probit model
       dic_star=-6*median(loglik_lcaburn)+4*sum(loglca_med)
       dic_reg=-4*median(loglik_lcaburn)+2*sum(loglca_med)
    

save(strcat('simLCAprobitResults_A',num2str(bb)),'theta0_med','pi_med','pred_ci','xi_ci','t_I0','y_mse','dic_star','dic_reg');

lcathetafile=strcat('theta0slca_simA',num2str(sim_n),'.fig');
figure; 
    h=heatmap(t_I0(2:4,:))
    h.YLabel = "supLCA Profile";
    h.XLabel = "Exposure variables";
    h.Colormap = parula
saveas(gcf,lcathetafile)
