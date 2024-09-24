clear
addpath(genpath('source'))

%% Parameters
% Population parameters
param.populationSize = 500; % Population size
param.topNreproduce = 4; % Top N genomes by fitness are selected for reproduction
param.generations = 10; % Number of generations
param.architecture = [11, 100, 100, 100, 3]; % Neural network architecture
param.mutationRate = 0.10; % Mutation rate

% Game parameters
param.gridSize = [20 20]; % Playing area size
param.initialLength = 3; % Initial snake length
param.maxSteps = 500; % Maximum steps in the game
param.bonusSteps = sum(param.gridSize); % Amount of steps rewarded for eating an apple

% Miscellaneous parameters
param.plotFitness = 1; % Plot fitness at each generation

%% Initialization
ga = GeneticAlgorithm(@MatoPeli, @Agent, param);

%% Training
ga = ga.runEvolution();

%% Playback
playbackSpeed = 0.01;
playbackSteps = 5000;
bestGen = ga.extractBestGenome(); % Extract the genome with the best fitness
ga.playGenome(bestGen, playbackSpeed, playbackSteps) % Play game 