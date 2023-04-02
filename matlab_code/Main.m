cancer = 'luad';
Data = readmatrix(cancer+"_fpkm.csv");
Data=Data(:,2:size(Data,2));
stage = readmatrix(cancer+"_stage.csv");
stage=stage(:,2:size(stage,2));
%  stage have two types of label :type1 is I,II,III,IV;type2 is IA,IB,IC,IIA,...
%  we use type2 for cancer progression prediction

Data=cat(1,Data,transpose(stage(:,2)));
Progression_Inference(Data,cancer);



