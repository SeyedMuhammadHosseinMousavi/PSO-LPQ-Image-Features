%% PSO-LPQ Image Features - Created in 22 Jan 2022 by Seyed Muhammad Hossein Mousavi
% This code extracts Local Phase Quantization (LPQ) features out of 100
% samples of images in 10 classes. LPQ features are in the family of frequency based features.
% Then desired number of PSO features
% will be selected out of extracted LPQ features which have highest
% impact. Actually, you can select n strongest features. Results show,
% however number of selected features goes down, but recognition accuracy
% is almost intact. 'nf' is number of selected features by PSO. Images are
% stores in 'data' folder. 
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)

%% Making Things Ready !!!
clc;
clear;
warning('off');

%% LPQ Feature Extraction
% Read input images
path='data';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;
% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
gray{i}=rgb2gray(images{i});
disp(['To Gray :   ' num2str(i) ]);
end;
% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted2{i}=imadjust(gray{i});
disp(['Contrast Adjust :   ' num2str(i) ]);
end;
% Resize Image
for i = 1 : filesnumber(1,1)
resized2{i}=imresize(adjusted2{i}, [256 256]);
disp(['Image Resized :   ' num2str(i) ]);
end;

%% LPQ Features
clear LPQ_tmp;
clear LPQ_Features;
winsize=9;
for i = 1 : filesnumber(1,1)
LPQ_tmp{i}=lpq(resized2{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
LPQ_Features(i,:)=LPQ_tmp{i};
end;

%% Labeling for Classification
sizefinal=size(LPQ_Features);
sizefinal=sizefinal(1,2);
%
LPQ_Features(1:10,sizefinal+1)=1;
LPQ_Features(11:20,sizefinal+1)=2;
LPQ_Features(21:30,sizefinal+1)=3;
LPQ_Features(31:40,sizefinal+1)=4;
LPQ_Features(41:50,sizefinal+1)=5;
LPQ_Features(51:60,sizefinal+1)=6;
LPQ_Features(61:70,sizefinal+1)=7;
LPQ_Features(71:80,sizefinal+1)=8;
LPQ_Features(81:90,sizefinal+1)=9;
LPQ_Features(91:100,sizefinal+1)=10;

%% PSO Feature Selection
% Data Preparation
x=LPQ_Features(:,1:end-1)';
t=LPQ_Features(:,end)';
data.x=x;
data.t=t;
data.nx=size(x,1);
data.nt=size(t,1);
data.nSample=size(x,2);

%% Number of Desired PSO Features

nf=50;

%% Cost Function
CostFunction=@(u) FeatureSelectionCost(u,nf,data);
% Number of Decision Variables
nVar=data.nx;
% Size of Decision Variables Matrix
VarSize=[1 nVar];
% Lower Bound of Variables
VarMin=0;
% Upper Bound of Variables
VarMax=1;

%% PSO Parameters
% Maximum Number of Iterations
MaxIt=10;
% Population Size (Swarm Size)
nPop=5;
% Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
% Inertia Weight
w=chi;
% Inertia Weight Damping Ratio
wdamp=1;
% Personal Learning Coefficient
c1=chi*phi1;
% Global Learning Coefficient
c2=chi*phi2;
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Basics
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Out=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.Out=[];
particle=repmat(empty_particle,nPop,1);
BestSol.Cost=inf;
for i=1:nPop
% Begin Position
particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
% Begin Velocity
particle(i).Velocity=zeros(VarSize);
% Evaluation
[particle(i).Cost, particle(i).Out]=CostFunction(particle(i).Position);
% Update Personal Best
particle(i).Best.Position=particle(i).Position;
particle(i).Best.Cost=particle(i).Cost;
particle(i).Best.Out=particle(i).Out;
% Update Global Best
if particle(i).Best.Cost<BestSol.Cost
BestSol=particle(i).Best;
end
end
%
BestCost=zeros(MaxIt,1);

%% PSO Body Part

for it=1:MaxIt
for i=1:nPop
% Update Velocity
particle(i).Velocity = w*particle(i).Velocity ...
+c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
+c2*rand(VarSize).*(BestSol.Position-particle(i).Position);
% Apply Velocity Limits
particle(i).Velocity = max(particle(i).Velocity,VelMin);
particle(i).Velocity = min(particle(i).Velocity,VelMax);
% Update Position
particle(i).Position = particle(i).Position + particle(i).Velocity;
% Velocity Mirror Effect
IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
% Apply Position Limits
particle(i).Position = max(particle(i).Position,VarMin);
particle(i).Position = min(particle(i).Position,VarMax);
% Evaluation
[particle(i).Cost, particle(i).Out] = CostFunction(particle(i).Position);
% Update Personal Best
if particle(i).Cost<particle(i).Best.Cost
particle(i).Best.Position=particle(i).Position;
particle(i).Best.Cost=particle(i).Cost;
particle(i).Best.Out=particle(i).Out;
% Update Global Best
if particle(i).Best.Cost<BestSol.Cost
BestSol=particle(i).Best;
end
end
end
%
BestCost(it)=BestSol.Cost;
%
disp(['In Iteration ' num2str(it) ': PSO Fittest Value Is : ' num2str(BestCost(it))]);
w=w*wdamp;
end

%% Creating PSO + LPQ Features Matrix
% Extracting Data
RealData=data.x';
% Extracting Labels
RealLbl=data.t';
FinalFeaturesInd=BestSol.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
PSO_LPQ_Features=RealData(:,FFI);
% Adding Labels
PSO_LPQ_Features_Lbl=PSO_LPQ_Features;
PSO_LPQ_Features_Lbl(:,end+1)=RealLbl;

%% SVM For LPQ Features Only
sizenet=size(RealData);
sizenet=sizenet(1,1);
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(RealData,RealLbl,'Learners',tsvm);
svmerror = resubLoss(svmclass);
% Predict the labels of the training data.
predictedlpq = resubPredict(svmclass);
ct=0;
for i = 1 : sizenet(1,1)
if RealLbl(i) ~= predictedlpq(i)
    ct=ct+1;
end;
end;
% Compute Accuracy
finsvm=ct*100/ sizenet;
LPQ_SVM=(100-finsvm)-svmerror;

%% SVM For PSO + LPQ Features
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(PSO_LPQ_Features,RealLbl,'Learners',tsvm);
svmerror = resubLoss(svmclass);
% Predict the labels of the training data.
predictedpso = resubPredict(svmclass);
ct=0;
for i = 1 : sizenet(1,1)
if RealLbl(i) ~= predictedpso(i)
    ct=ct+1;
end;
end;
% Compute Accuracy
finsvm=ct*100/ sizenet;
PSO_LPQ_SVM=(100-finsvm)-svmerror;

%% Statistics
fprintf('The LPQ SVM Accuracy Is =  %0.4f.\n',LPQ_SVM)
fprintf('The PSO + LPQ SVM Accuracy Is =  %0.4f.\n',PSO_LPQ_SVM)

%% Confusion Matrix of Classification
figure
set(gcf, 'Position',  [50, 100, 1300, 500])
subplot(1,2,1)
lpqsvm = confusionchart(RealLbl,predictedlpq, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
lpqsvm.Title = (['LPQ SVM =  ' num2str(LPQ_SVM) '%, Features #: ' num2str(sizefinal)]);
subplot(1,2,2)
psolpqsvm = confusionchart(RealLbl,predictedpso, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
psolpqsvm.Title = (['PSO LPQ SVM =  ' num2str(PSO_LPQ_SVM) '%, Features #: ' num2str(nf)]);

%% Plot PSO Training Stage
figure;
set(gcf, 'Position',  [600, 300, 600, 300])
plot(BestCost,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','g',...
'Color',[0.9,0.2,0.1]);
title('PSO Algorithm Training')
xlabel('PSO Iteration','FontSize',10,...
'FontWeight','bold','Color','k');
ylabel('Fittest Value','FontSize',10,...
'FontWeight','bold','Color','k');
legend({'PSO Train'});

% I think That Is It !!!... :|
