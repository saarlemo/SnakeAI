clear;clc
addpath(genpath('source'))
mex source/evaluateFitness.c -lOpenCL
%% Parameters
% Population parameters
param.populationSize = 1024; % Population size
% param.topNreproduce = 4; % Top N genomes by fitness are selected for reproduction
param.generations = 2500; % Number of generations
param.mutationRate = 0.10; % Mutation rate

% Neural network parameters
param.nHiddenLayers = 3; % Neural network architecture
param.hiddenLayerSize = 100; % Hidden layer size
param.dropoutRate = 0.01; % Neuron drop out rate (optional, default=0)

% Game parameters
param.gridSize = [20 20]; % Playing area size
param.initialLength = 3; % Initial snake length
param.maxSteps = 500; % Maximum steps in the game
param.bonusSteps = 100; % Amount of steps rewarded for eating an apple

% Plot fitness scores after each generation
param.plotFitness = 1;

%% Initialization
ga = GeneticAlgorithm(param);

%% Training
ga = ga.runEvolution();