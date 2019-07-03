% Decompose fMRI and EEG
% Call penCTFhals for single regularization parameter
clc;clear;
% LOAD THE DATA
% Replace this with your own directory
maindir = pwd;
datadir = fullfile(maindir,'data');

datanameB = 'fmri';
datanameE = 'eeg';

load(fullfile(datadir,datanameB));
load(fullfile(datadir,datanameE));
load(fullfile(datadir,'LeadField.mat'));
load(fullfile(datadir,'LapMat.mat'));

% Since power spectrum of the EEG is used we are using K.^2
% For more info check Miwakeichi et. al., Neuroimage, 2004.
K = K.^2;

% CMTF
% smoothness/sparsity/orthogonality/nonnegativity on spatial sigantures

al1  = 1e-3;
al2  = 10;
alorth=0;
gamma= 10;

couple.x.nn        = [1 1 1];
couple.x.alphaorth = 0;
couple.x.alphaL2   = al2;
couple.x.alphaL1   = al1;
couple.x.L         = L;

couple.y.nn        = [1 0];
couple.y.alphaorth = alorth;
couple.y.alphaL2   = al2;
couple.y.alphaL1   = al1;
couple.y.L         = L;

couple.common.alphaorth = alorth;
couple.common.alphaL2   = al2;
couple.common.alphaL1   = al1;
couple.common.L         = L;

couple.cdim      = [1 1];
couple.gamma     = gamma;
couple.maxiters  = 300;
couple.K         = K;

P =10* P/sqrt(sum(P(:).^2));
B =10* B/sqrt(sum(B(:).^2));
Rc= 1; Rx = 2; Ry = 2;

% Initial Factors
load(fullfile(datadir,[datanameE '_initFac']));%U0x
load(fullfile(datadir,[datanameB '_initFac']));%U0y

[Ux,Uy,output] = penCTFhals(S,B,[Rx Ry],Rc,couple,U0x,U0y);

% Plot the Factors
figure,
for i=1:length(Ux)
    subplot(length(Ux),1,i),plot(Ux{i})
    title(sprintf('EEG signature - %d',i))
end
figure,
for i=1:length(Uy)
    subplot(length(Uy),1,i),plot(Uy{i})
    title(sprintf('fMRI signature - %d',i))
end

