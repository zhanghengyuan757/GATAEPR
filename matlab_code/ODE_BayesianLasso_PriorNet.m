function [Para_Post_pdf,S,AM]=ODE_BayesianLasso_PriorNet(Data_ordered,TimeSampled,PriorNet)
% Input:
% Data_smooth -- (pseudo)time-course gene expression data
% TimeSampled -- Time points associate with (pseudo)time-course gene expression data
% PriorNet -- a matrix representing the prior network information. (p_ij) for prior connection from gene j to gene i 
% Output:
% Para_Post_pdf -- posterior distribution over the coefficients in the GRN model
% S --  a matrix saving the presence probability. 
% AM -- Adjacent matrix of the inferred network i.e., (a_ij) for the regulatory coefficient from gene j to gene i. 

%% data reformation
Time=TimeSampled;
x=Data_ordered;  % original data 
Time=(Time-min(Time))/(max(Time)-min(Time));
y=diff((x)')./diff(Time)';  % for linear ODE model
% y=diff(log(x'))./diff(Time)';  % for mass-action model
y=y';
x=x(:,1:end-1);

%% Calculate Post_pdf for each gene's parameters
Para_Post_pdf=cell(size(x,1));
for ind_gene=1:size(x,1)
     ind_gene
y_output=y(ind_gene,:);
x_input=[x',ones(size(x,2),1)].*x(ind_gene,:)';
% if nargin>2
    x_input(:,PriorNet(ind_gene,:)==0)=[]; % exclude genes not regulating this gene (priorPPI) 
% end
% x_input(:,ind_gene)=[];  % Because the above line has riled to exclude 'ind_gene'

%% Lasso
[A,FitInfo] = lasso_Sun(x_input,y_output,'Alpha',1,'CV',10,'Standardize',1,'Intercept_Flag',0);  % modified from matlab function by adding Intercept_Flag argument associated with lassoFit_Sun (redefined function)
idxLambdaMinMSE = FitInfo.IndexMinMSE;  %Predict exam scores for the test data. 
coef = A(:,idxLambdaMinMSE);
coef0 = FitInfo.Intercept(idxLambdaMinMSE);

%% Bayesian Lasso
% predictornames={'Deg','B','C'};
p=size(x_input,2);

% L=ones(p+1,1)*lambda; L(ind_gene)=L(ind_gene)*1e5;
PriorMdl = bayeslm(p,'ModelType','lasso','Intercept',false);  %,'Lambda',L  %% ,'VarNames',predictornames; %% ,'Mu',[coef],'V',eye(p)
% table(PriorMdl.Lambda,'RowNames',PriorMdl.VarNames)

ismissing = any(isnan(x_input),2);
n = sum(~ismissing); % Effective sample size
% lambda = FitInfo.Lambda*n./sqrt(FitInfo.MSE); 


% Preallocate
% BayesLassoCoefficients = zeros(p,1);
% BayesLassoCI95 = zeros(p,2,1);
% fmseBayesLasso = zeros(1,1);
% BLCPlot = zeros(p,1);

% Estimate and forecast
rng(3); % For reproducibility
Y_train=y_output;
X_train=x_input;
    lambda = FitInfo.LambdaMinMSE*n./sqrt(min(FitInfo.MSE));
%     PriorMdl.Lambda = lambda;  % set PriorMdl.Lambda as FitInfo.Lambda*n./sqrt(FitInfo.MSE) from Lasso results
%     coef(ind_gene)=0;
    [EstMdl,Summary] = estimate(PriorMdl,X_train,Y_train,'Display',false,'BetaStart',[coef]);  %,'Sampler',"hmc",'NumDraws',1e6,'Burnin',1e3
%     BayesLassoCoefficients = Summary.Mean(1:(end - 1));
%     BLCPlot = Summary.Mean(1:(end - 1));
%     BayesLassoCI95 = Summary.CI95(1:(end - 1),:);
%%%% Plot the posterior distribution of estimated  parameters
% figure, plot(EstMdl)

Summary = summarize(EstMdl);
Summary = Summary.MarginalDistributions;

    Summary_new=zeros(size(x,1),2);
    Summary_new(find(PriorNet(ind_gene,:)==1),:)=[Summary.Mean(1:end-2),Summary.Std(1:end-2)];
    Mean=Summary_new(:,1);
    Std=Summary_new(:,2);
Para_Post_pdf{ind_gene}=table(Mean,Std);   % save the parameter estimation for targeting each gene into a cell structure
end

%%  Calculate Confidence Interval
    alpha=0.01:0.01:1;
    CI_alpha=zeros(size(x,1),length(alpha),size(x,1),2);
for i=1:size(x,1)
    Para_Post_pdf_each=Para_Post_pdf{i};
%     Para=Para_Post_pdf_each.BetaDraws';
Summary = Para_Post_pdf_each;
    alpha=0.01:0.01:1;
    ind_alpha=1;
 for k_alpha=alpha
%     MUCI = CI_Sun(Para,k_alpha);  % 
    MUCI = CI_Sun_Normal_PriorNet(Summary,k_alpha);  % 
    CI_alpha(i,ind_alpha,:,:)=MUCI;
    ind_alpha=ind_alpha+1;
 end
    size(CI_alpha)
end

S=ones(size(x,1),size(x,1));  % probability that the estimated parameter interval contains 0 ; Intercept term counted
for i=1:size(x,1) 
    for k=length(alpha):-1:1
        CI_genei_alpha=reshape(CI_alpha(i,k,:,:),size(x,1),2);
        for j=1:size(x,1)
             if CI_genei_alpha(j,1)<=0 & CI_genei_alpha(j,2)>=0   % for coeficient matric               
                 S(i,j)=min(S(i,j),1-alpha(k));  % S=1-max{alpha:CI_alpha containing 0}
%                  break;
             end        
       end

   end
end

% %% Reform probability matrix for interation coefficients
%    for i=1:size(Data_smooth,1)
%         S_new(i,i)=0;
%         S_new(i,setdiff(1:size(Data_smooth,1),i))=S(i,1:end-1);
%    end
%    S=S_new;

%  %% Output for network visualization
% clear Act_Inh Act_Inh_strength
% for i=1:size(S,1)
% Summary = summarize(Para_Post_pdf{i});  %(Para_Post_pdf{i}.Mean)';
% Summary = Summary.MarginalDistributions;
% Act_Inh(i,:) = Summary.Mean(1:end-2);
% Act_Inh_strength(i,i)=0;
% Act_Inh_strength(i,setdiff(1:size(S,1),i))=Act_Inh(i,:);
% end
% % Adjacent matrix
% AM=Act_Inh_strength.*(S>0.95);
