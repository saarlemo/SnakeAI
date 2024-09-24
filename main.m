clear
addpath(genpath('source'))

%% Parameters
% Population parameters
param.populationSize = 500;
param.topNreproduce = 2;
param.generations = 10;
param.architecture = [11, 100, 100, 100, 3];
param.mutationRate = 0.10;

% Game parameters
param.gridSize = [20 20];
param.initialLength = 3;
param.maxSteps = 500;

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