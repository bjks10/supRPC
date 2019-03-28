%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SUPERVISED ROBUST PROFILE CLUSTERING 
%Programmer: Briana Stephenson
%Data: Simulated Set #343
% adjust beta to simulate a t
% mu0~std normal, sig0 ~ IG(5/2,5/2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_n=343;


%% load Simulated data %%
load(strcat('Sim_sRPCdata',num2str(sim_n),'.mat'))


food=probdata;

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
k_max=50;



%% -- SET UP HYPERPRIORS -- %%
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
alpha=ones(1,k_max)*(1/1000);
pi_h=drchrnd(alpha,1);

%phi - cluster index
Ci=mnrnd(1,pi_h,n); [r, c]=find(Ci); gc=[r c];
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
G_ij=zeros(n,p);
for s=1:S
    ns=n_s(s);
    nu_s=nu(s,:);
    G_ij(state==(s),:) = repmat(binornd(1,nu_s),[ns,1]);     % family index (0=global family,1=local family)
end


%% SUBPOPULATION LEVEL %%

%lambda_k^s
lambda_sk=drchrnd(alpha,S);
lambda_sk(lambda_sk==0)=1e-10;
lambda_sk(lambda_sk==1)=1-1e-10;
L_ij=zeros(n,p); %local cluster label variable

n_Lij=zeros(S,p,k_max);
for s=1:S
    lambda_sk(s,:)=lambda_sk(s,:)/sum(lambda_sk(s,:));
    for j=1:p
        x_sl=mnrnd(1,lambda_sk(s,:),n_s(s)); [r, c]=find(x_sl); gc=[r c];
        gc=sortrows(gc,1);
        L_ij(state==(s),j)=gc(:,2);
        n_Lij(s,j,:)=sum(x_sl);
    end
end


%% -- RESPONSE PROBIT MODEL SETUP -- %%
K0=k_max; X=zeros(n,K0+S);
mu0=normrnd(0,1,[K0+S,1]);
sig0=1./gamrnd(5/2,2/5,[K0+S,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;

%subpopulation design matrix: w_sid
w_sid=zeros(n,S);

for s=1:S
    w_sid(state==(s),s)=1;
end
%  w_sid(:,1)=1; %subpop 1 serve as state intercept



%% ----------------- %%
%% MCMC data storage %%
%% ----------------- %%
nrun=25000; burn=nrun/5; thin=100;

%predictor RPC model storage
beta_out=zeros(nrun,S);
nu_out=zeros(nrun,S,p);
pi_out=zeros(nrun,k_max);

Loglike0=zeros(nrun,1);

theta0_out=zeros(nrun/thin,p,k_max,d);
theta1_out=zeros(nrun/thin,S,p,k_max,d);
lambdas_out=zeros(nrun,S,k_max);
ks_out=zeros(nrun,S+1);
ci_out=zeros(nrun,n);

%response spv model storage
xi_out=zeros(nrun,S+K0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POSTERIOR COMPUTATION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%temporary storage
As=zeros(n,p);
p_ij=zeros(n,p);
ZBrand=zeros(n,1);
z_probit=zeros(n,1);
nL_ij=zeros(S,k_max); %store local cluster counts

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


%%-- Ci ~multinomial(pi_h) --%%

    Cp_k=zeros(n,k_max);
    for k=1:k_max
        t0h=reshape(theta0(:,k,:),p,d);
        tmpmat0=reshape(t0h(lin_idx),[n,p]);
        Cp_k(:,k)=pi_h(k)*prod(tmpmat0.^G_ij,2);
    end
    
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

    ci_out(iter,:)=Ci; %save cluster assignments for pairwise poseriors

%%-- update theta --%%
    dmat0=zeros(p,d);
    dmat1=zeros(p,d);
    
    for k=1:k_max
        Cis=repmat(Ci,[1,p]).*G_ij;
        ph0 = (Cis==k); %subj's in global cluster h
    
        for c = 1:d
            sum((food==c).*ph0)';
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
    
    if mod(iter,thin)==0
        theta0_out(iter/thin,:,1:size(theta0,2),:)=theta0;
        theta1_out(iter/thin,:,:,1:size(theta1,3),:)=theta1;
    end



%%-- update nu_j --%%
    for s=1:S
        Gs=G_ij(state==(s),:);
        nu(s,:) = betarnd(1 + sum(Gs), beta(s) + sum(1-Gs));
    end
    nu(nu==1) = 1-1e-06;
    nu(nu==0) = 1e-06;

    nu_out(iter,:,:)=nu;

%%-- update beta --%%
    for s=1:S
        beta(s) = gamrnd(abe + p,1./( bbe - sum(log(1-nu(s,:)))));
    end
    
    beta_out(iter,:)=beta;



%% -- RESPONSE MODEL PARAMETERS UPDATE -- %%


    Wup=[w_sid x_ci];

    pcov=size(Wup,2);
%create latent z_probit
%create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter(1:pcov));
%truncation for cases (0,inf)
    z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
%truncation for controls (-inf,0)
    z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);


% Response Xi(B) update

    sig0up=Sig0(1:pcov,1:pcov);
    xi_0up=xi_0(1:pcov);

    xi_sig_up=inv(sig0up)+(transpose(Wup)*Wup);
    xi_mu_up2=(sig0up*transpose(xi_0up))+(transpose(Wup)*z_probit);
    xi_mu_up=xi_sig_up\xi_mu_up2;
    xi_up=mvnrnd(xi_mu_up,inv(xi_sig_up));
    xi_iter(1:length(xi_up))=xi_up;
    xi_out(iter,:)=xi_iter;



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

%% burn-out %%
beta_burn=beta_out(burn+1:end,:);
lambdas_burn=lambdas_out(burn+1:end,:,:);
pi_burn=pi_out(burn+1:end,:);
theta0_burn=theta0_out((burn/thin)+1:end,:,:,:);
theta1_burn=theta1_out((burn/thin)+1:end,:,:,:,:);
nu_burn=nu_out(burn+1:end,:,:);
xi_burn=xi_out(burn+1:end,:);
ci_burn=ci_out(burn+1:end,:);

beta_med=median(beta_burn);
nu=reshape(median(nu_burn,1),[S,p]);

%posterior medians of local cluster model
lambdas=reshape(median(lambdas_burn,1),[S,k_max]);
theta1=reshape(median(theta1_burn,1),[S,p,k_max,d]);

lambdas_x=cell(S,1); lambdas_x{S}=[];
theta1_x=cell(S,1); theta1_x{S}=[];
for s=1:S
    wss=lambdas(s,:);
    lambdas_x{s}=wss(wss>0.01);
    theta1_x{s}=reshape(theta1(s,:,wss>0.01,:),[p,length(lambdas_x{s}),d]);
end

nu_med=transpose(nu);

m_perm=size(theta1_burn,1);
m=size(pi_burn,1);
m_thin=m/m_perm;

k_iter=sum(pi_burn>0.01,2);
k_all=max(k_iter);
k_in=median(k_iter);
k_med=median(sum(pi_burn>0.05,2));

%% PAPASPILOULIS POSTERIOR PAIRWISE MIXING %%
pd=pdist(transpose(ci_burn),'hamming'); %prcnt differ
cdiff=squareform(pd); %Dij
Zci=linkage(cdiff,'complete');

zclust=cluster(Zci,'maxclust',k_in);

% use mode to identify relabel order for each iteration
ci_relabel=zeros(m,k_in);
for l=1:k_in
    ci_relabel(:,l)=mode(ci_burn(:,zclust==l),2);
end

k_keep=zeros(m,1);
for it=1:m
    k_keep(it)=length(unique(ci_relabel(it,:)));
end

k_up=median(k_keep);
ci_relabel2=zeros(m,k_up);
for l=1:k_up
    ci_relabel2(:,l)=mode(ci_burn(:,zclust==l),2);
end


%save('sRP_SimSave1','beta_burn','lambdas_burn','pi_burn','theta0_burn',...
      %    'theta1_burn','nu_burn','ci_relabel2','Zci','xi_burn','Loglike0_burn','-v7.3');


%% reorder pi and theta0 parameters
k0=k_up;
[m_perm,S,p,~,d]=size(theta1_burn);
pi_order=zeros(m,k0);
theta0_order=zeros(m_perm,p,k0,d);
k_uni=zeros(m,1);
for iter=1:m
    iter_order=ci_relabel2(iter,:);
    pi_order(iter,:)=pi_burn(iter,iter_order);

   
    k_uni(iter)=length(unique(iter_order));
    iter_uni=unique(iter_order);

    if mod(iter,m_thin)==0
        iter_thin=iter/m_thin;
        theta0_order(iter_thin,:,:,:)=theta0_burn(iter_thin,:,iter_order,:);
    end
end

% %%identify modal patterns for each cluster
pi_med=median(pi_order);
theta0_med=reshape(median(theta0_order),[p,k0,d]);
[M0, I0]=max(theta0_med,[],3);
t_I0=transpose(I0);
[c, ia, ic]=unique(t_I0,'rows');

pi_med=pi_med(ia)/sum(pi_med(ia));
theta0_med=theta0_med(:,ia,:);
k_ia=length(pi_med);

zclust=cluster(Zci,'maxclust',k_ia);


%save('Simordered_sRPCparms','theta0_order','pi_order','ci_relabel2','Zci');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RE-RUN SUP-RPC WITH CORRECT LABELING ORDER %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


w_ciz=zeros(n,k_ia);
for k=1:k_ia
    w_ciz(zclust==k,k)=1;
end
pcov=S+k_ia;


%% -- RESPONSE PROBIT -- %%
% K0=k_ia; pcov=K0+S-1;
X=zeros(n,pcov);
mu0=normrnd(0,1,[pcov,1]);
sig0=1./gamrnd(5/2,2/5,[pcov,1]); %shape=5/2, scale=5/2
Sig0=diag(sig0);
xi_0=mvnrnd(mu0,Sig0);
xi_iter=xi_0;
n=size(zclust,1);

% Xi=zeros(n,pcov);

devprob0=zeros(m,n);
z_probit=zeros(n,1);
xi_order=zeros(m,pcov);

%% -- BEGIN MCMC -- %%
Wup=[w_sid w_ciz];

for iter=1:m


    
%probit model y=xb+e

%create latent z_probit

%create truncated normal for latent z_probit model
    WXi_now=Wup*transpose(xi_iter);

    z_probit(y_cc==0)=truncnormrnd(1,WXi_now(y_cc==0),1,-inf,0);
    z_probit(y_cc==1)=truncnormrnd(1,WXi_now(y_cc==1),1,0,inf);
    if sum(isnan(z_probit))>0
        break
    end


% Response beta update
    xi_mu_up1=inv(Sig0)+(transpose(Wup)*Wup);
    xi_mu_up2=(Sig0*mu0)+(transpose(Wup)*z_probit); %%%%
    xi_mu_up=xi_mu_up1\xi_mu_up2;
    xi_sig_up=inv(inv(Sig0)+(transpose(Wup)*Wup));
    xi_iter=mvnrnd(xi_mu_up,xi_sig_up);
    xi_order(iter,:)=xi_iter;

    phiWXirpc=normcdf(Wup*transpose(xi_iter));
    phiWXirpc(phiWXirpc==0)=1e-10;
    phiWXirpc(phiWXirpc==1)=1-1e-10;
    devprob_i=y_cc.*log(phiWXirpc) + (1-y_cc).*log(1-phiWXirpc);
    devprob0(iter,:)=devprob_i;


end
burn=5000;
xi_burn=xi_order(burn+1:end,:);
xi_prc_nodem=transpose(prctile(xi_burn,[2.5 50 97.5]))
xi_med=median(xi_burn);
devianceProbitburn=devprob0(burn+1:end,:);

p_pos=transpose(sum(xi_burn>0)/(m-burn));

py_pred= normcdf(Wup*xi_prc_nodem(:,2));

xi_mean=mean(xi_burn);
nu_mean=reshape(mean(nu_burn),[S,p]);
lambdas_mean=reshape(mean(lambdas_burn),[S,k_max]);
theta1_mean=reshape(mean(theta1_burn),[S,p,k_max,d]);
d_mean=zeros(n,p);
for s=1:S
    d_mean(state==s,:)=binornd(1,repmat(nu_mean(s,:),[n_s(s),1]));
end

theta0_mean=reshape(mean(theta0_order),[p,k0,d]);
pisum=sum(pi_order,2);
pinorm=bsxfun(@times,pi_order,1./pisum);

pi_mean=mean(pinorm); 
delmean=zeros(n,k0);
for h = 1:k0
    theta0h = reshape(theta0_mean(:,h,:),p,d);
    tmpmat0 = reshape(theta0h(lin_idx),[n,p]);
    delmean(:,h) = pi_mean(h) * prod(tmpmat0.^d_mean,2);
end
zup0 = bsxfun(@times,delmean,1./(sum(delmean,2)));

t0_mean=zup0./sum(zup0,2);
t_logpmean=t0_mean.*log(zup0);
dev_globmean=sum(t_logpmean,2);

w_ci=mnrnd(1,zup0); [r, c]=find(w_ci); w_gc=[r c];
w_gc=sortrows(w_gc,1); pred_ci=w_gc(:,2);

%local likelihood part%

dev_locmean=zeros(n,1);
for s=1:S

    ws_x=reshape(lambdas_mean(s,:),[1,k_max]);
    ws_xx=ws_x(ws_x>0.01); ks=length(ws_xx);
    ws_xt1=ws_xx./sum(ws_xx);
    theta1_s=reshape(theta1_mean(s,:,:,:),[p,k_max,d]);
    theta1_sx=theta1_s(:,ws_x>0.01,:);
    delS=zeros(n_s(s),ks,p);
    for h = 1:ks
        t1hs = reshape(theta1_sx(:,h,:),p,d);
        theta1hs=bsxfun(@times,t1hs,1./sum(t1hs,2));
        tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
        delS(:,h,:) = ws_xt1(h) * tmpmat1.^(1-d_mean(state==(s),:));
    end
    t1_mean=bsxfun(@times,delS,1./sum(delS,2));
    t1_logpmean=t1_mean.*log(delS);

    dev_locmean(state==(s))=sum(sum(t1_logpmean,2),3);
end

dev_rpcmean=dev_globmean + dev_locmean;

Xmean=[w_sid w_ci]; %covariate matrix with state/global

WXi_mean=Xmean*transpose(xi_mean);

phiWXimean=normcdf(WXi_mean);
phiWXimean(phiWXimean==0)=1e-10;
phiWXimean(phiWXimean==1)=1-1e-10;
dev_probmean=y_cc.*log(phiWXimean) + (1-y_cc).*log(1-phiWXimean);

y_mse=immse(py_pred,py_true)


%calcuate iterated loglikelihood of RPC
loc_like=zeros(n,1);

nu_iter=zeros(n,p); d_iter=zeros(n,p);
loglikRPC=zeros(m_perm,1);
dev_rpciter=zeros(m_perm,n);
pi_thin=pi_order(1:m_thin:end,:);

for iter=1:m_perm
    t_iter=iter*m_thin;
    nu_iter=reshape(nu_burn(t_iter,:,:),[S,p]);
    for s=1:S
        d_iter(state==(s),:)=binornd(1,repmat(nu_iter(s,:),[n_s(s),1]));
    end
    pi_t=pi_thin(iter,:)./sum(pi_thin(iter,:));
%global likelihood part
    del=zeros(n,k0);
    for h = 1:k0
        t0h = reshape(theta0_order(iter,:,h,:),p,d);
        theta0h = bsxfun(@times, t0h,1./sum(t0h,2));
        tmpmat0 = reshape(theta0h(lin_idx),[n,p]);
        del(:,h) = pi_t(h) * prod(tmpmat0.^d_iter,2);
    end
    
    t0_iter=del./sum(del,2);
    t_logpiter=t0_iter.*log(del);
    dev_globiter=sum(t_logpiter,2);


%local likelihood part

    dev_lociter=zeros(n,1);
    for s=1:S 
        ws_x=reshape(lambdas_burn(iter*m_thin,s,:),[1,k_max]);
        ws_xx=ws_x(ws_x>0.01); ks=length(ws_xx);
        ws_xt=ws_xx./sum(ws_xx);
        theta1_s=reshape(theta1_burn(iter,s,:,:,:),[p,k_max,d]);
        theta1_sx=theta1_s(:,ws_x>0.01,:);
        
        delS=zeros(n_s(s),ks,p);
        for h = 1:ks
            t1hs = reshape(theta1_sx(:,h,:),p,d);
            theta1hs=bsxfun(@times,t1hs,1./sum(t1hs,2));
            tmpmat1 = reshape(theta1hs(lin_idxS{s}),[n_s(s),p]);
            d_its=d_iter(state==s,:);
            delS(:,h,:) = ws_xt(h) * tmpmat1.^(1-d_its);
        end  
    t1_iter=bsxfun(@times,delS,1./sum(delS,2));
    t1_logpiter=t1_iter.*log(delS);


    dev_lociter(state==(s))=sum(sum(t1_logpiter,2),3);

    end
    
    total_like=dev_globiter+dev_lociter;
    dev_rpciter(iter,:)=total_like;
end

dev_rpciterburn=dev_rpciter((burn/m_thin)+1:end,:);
dev_probburn=devianceProbitburn(1:m_thin:end,:);


DIC_full1=-4*mean(sum(dev_rpciterburn+dev_probburn,2))+2*sum(dev_rpcmean+dev_probmean)

save(strcat('py_simResults',num2str(sim_n)),'py_pred','pred_ci','nu_med','DIC_full1','t_I0','y_mse');

%nu

    clf  
    %plot comparing predicted to true nu
subplot(1,2,1);
heatmap(transpose(nu_med), 1:S,1:p);
title('Derived Local Deviations');

subplot(1,2,2);
heatmap(transpose(trueG), 1:S, 1:p);
title('True Local Deviations');
saveas(gcf,strcat('G_deviations.png'))
