clear;clc
addpath(genpath('source'))
mex('source/evaluateFitness.c', '-lOpenCL', '-outdir', 'source', 'LDFLAGS="\$LDFLAGS -z noexecstack"')
% load('initialWeights.mat')

%% Parameters
% Population parameters
param.populationSize = 256; % Population size
param.generations = 150; % Number of generations
param.mutationRate = 0.1; % Mutation rate
% param.initialWeights = bestWeights;

% Neural network parameters
param.nHiddenLayers = 3; % Neural network architecture
param.hiddenLayerSize = 100; % Hidden layer size

% Game parameters
param.gridSize = [20 20]; % Playing area size
param.maxSteps = 500; % Maximum steps in the game
param.bonusSteps = 100; % Amount of steps rewarded for eating an apple

% Miscellaneous parameters
param.plotFitness = 1; % Plot fitness scores after each generation
param.saveBestWeights = 1;

%% Initialization
ga = GeneticAlgorithm(param);

%% Training
ga = ga.runEvolution();

%% Play
% snake(bestWeights', architecture)
