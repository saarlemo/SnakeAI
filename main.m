clear
addpath(genpath('source'))
load('initialGenome.mat');

%% Parameters
% Population parameters
param.populationSize = 10; % Population size
param.topNreproduce = 4; % Top N genomes by fitness are selected for reproduction
param.generations = 75; % Number of generations
param.mutationRate = 0.10; % Mutation rate
param.fitnessFun = ... % Fitness function of steps and apples
    @(s, a) log1p(s + (2^a + 500*a^(2.1)) - (a^(1.2)*(0.25*s)^(1.3))); 

% Neural network parameters
param.architecture = [11, 100, 100, 100, 3]; % Neural network architecture
param.dropoutRate = 0.01; % Neuron drop out rate (optional, default=0)
param.initialGenome = gen; % Cell array of initial weights (optional)
% Activation function options: ReLU, step, sigmoid, tanh, softplus,
% gaussian. If not selected, identity activation function is used.
param.activationFunction = 'ReLU'; % Activation function (optional)

% Game parameters
param.gridSize = [20 20]; % Playing area size
param.initialLength = 3; % Initial snake length
param.maxSteps = 500; % Maximum steps in the game
param.bonusSteps = 100; % Amount of steps rewarded for eating an apple

% Miscellaneous parameters
param.plotFitness = 1; % 1=Plot fitness at each generation

%% Initialization
ga = GeneticAlgorithm(@MatoPeli, @Agent, param);

%% Training
ga = ga.runEvolution();

% Save best genome
ga.saveBestGenome('initialGenome.mat')

%% Playback
playbackSpeed = 0.01;
playbackSteps = 5000;
% bestGen = ga.extractBestGenome(); % Extract the genome with the best fitness
ga.playGenome(bestGen, playbackSpeed, playbackSteps) % Play game 