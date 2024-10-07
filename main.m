clear;clc
addpath(genpath('source'))
mex source/evaluateFitness.c -lOpenCL
% load('initialWeights.mat')

%% Parameters
% Population parameters
param.populationSize = 128; % Population size
param.generations = 1000; % Number of generations
param.mutationRate = 0.10; % Mutation rate
% param.initialWeights = bestWeights;

% Neural network parameters
param.nHiddenLayers = 3; % Neural network architecture
param.hiddenLayerSize = 100; % Hidden layer size
param.dropoutRate = 0.01; % Neuron drop out rate (optional, default=0)

% Game parameters
param.gridSize = [20 20]; % Playing area size
param.initialLength = 3; % Initial snake length
param.maxSteps = 500; % Maximum steps in the game
param.bonusSteps = 100; % Amount of steps rewarded for eating an apple

% Miscellaneous parameters
param.plotFitness = 1; % Plot fitness scores after each generation
param.saveBestWeights = 1;

%% Initialization
ga = GeneticAlgorithm(param);

%% Training
ga = ga.runEvolution();