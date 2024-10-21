classdef GeneticAlgorithm
    properties
        population  % Weight matrix: each column is one genome
        param       % Parameters
    end
    methods
        function obj = GeneticAlgorithm(param)
            obj.param = param;

            % Calculate amount of weights in one neural network
            nWeights = 0;
            layers = [12, param.hiddenLayerSize * ones(1,param.nHiddenLayers), 3];
            for ii = 1:param.nHiddenLayers+1
                nWeights = nWeights + layers(ii) * layers(ii+1); % Bias included in layer size
            end
            
            % Initialize population
            if isfield(param, 'initialWeights')
                fprintf("test")
                obj.population = repmat(param.initialWeights, 1, param.populationSize);
            else
                obj.population = single(randn(nWeights, param.populationSize));
            end
            % Initialize figure
            if param.plotFitness == 1
                f1 = figure(1);
                scatter([],[])
                xlim([0 param.generations])
                xlabel('Generation', 'interpreter', 'latex')
                ylabel('Fitness', 'interpreter', 'latex')
                grid on
                set(f1, 'defaulttextinterpreter', 'latex')
                set(gca,'TickLabelInterpreter','latex')
            end
        end

        function obj = saveBestWeights(obj, topFitnessIdx)
            architecture = [12, obj.param.hiddenLayerSize * ones(1, obj.param.nHiddenLayers), 3];
            bestWeights = obj.population(:, topFitnessIdx);
            save('initialWeights.mat', 'architecture', 'bestWeights');
        end

        function obj = runEvolution(obj)
            % Run the genetic algorithm for a specified number of generations
            paramToPass.inputSize = 12;
            paramToPass.hiddenSize = obj.param.hiddenLayerSize;
            paramToPass.outputSize = 3;
            paramToPass.nHidden = obj.param.nHiddenLayers;
            paramToPass.gridWidth = obj.param.gridSize(1);
            paramToPass.gridHeight = obj.param.gridSize(2);
            paramToPass.maxSteps = obj.param.maxSteps;
            paramToPass.bonusSteps = obj.param.bonusSteps;

            for gen = 1:obj.param.generations
                tic
                fitnessScores = evaluateFitness(single(obj.population), paramToPass);

                [maxFitness, topFitness] = max(fitnessScores);

                % obj.population = repmat(obj.population(:, topFitness), 1, obj.param.populationSize);
                parents = selectParents(obj.population, fitnessScores);
                obj.population = crossover(parents);
                obj.population = mutate(obj.population, obj.param.mutationRate);

                % Optionally, display progress
                if obj.param.plotFitness
                    hold on;
                    scatter(gen * ones(size(fitnessScores)), fitnessScores, '.', 'MarkerEdgeColor', '#0072BD')
                    hold off
                    drawnow
                end
            
                dt = toc;
                fprintf('Generation %d: best fitness = %f, dt = %f s\n', gen, maxFitness, dt);
            end

            if obj.param.saveBestWeights == 1
                obj.saveBestWeights(topFitness);
            end

            function parents = selectParents(population, fitnessScores)
                parents = zeros([size(population), 2]);
                fitnessScores = double(fitnessScores);
                fitnessScores(isnan(fitnessScores)) = 0;

                probs = exp(fitnessScores - max(fitnessScores));
                probs = probs ./ sum(probs);
                
                for ii = 1:numel(fitnessScores)
                    parentIdx = randsample(1:size(population, 2), 2, true, probs);
                    parents(:, ii, 1) = population(:, parentIdx(1));
                    parents(:, ii, 2) = population(:, parentIdx(2));
                end
            end
            function population = crossover(parents)
                population = zeros(size(parents, [1, 2]));
                for ii = 1:size(population, 2)
                    crossoverIdx = randsample(3:size(population, 1)-2, 1);
                    population(1:crossoverIdx, ii) = parents(1:crossoverIdx, ii, 1);
                    population(crossoverIdx+1:end, ii) = parents(crossoverIdx+1:end, ii, 2);
                end
            end
            function newPopulation = mutate(population, mutationRate)
                mutationMask = rand(size(population)) <= mutationRate;
                newPopulation = population + mutationMask .* randn(size(population));
            end
        end
    end
end