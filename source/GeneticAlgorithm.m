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

            for gen = 1:obj.param.generations
                tic
                fitnessScores = evaluateFitness(obj.population);

                [maxFitness, topFitness] = max(fitnessScores);

                obj.population = repmat(obj.population(:, topFitness), 1, obj.param.populationSize);
                mutationMask = rand(size(obj.population)) <= obj.param.mutationRate;
                obj.population = obj.population + mutationMask .* randn(size(obj.population));
                
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
        end
    end
end