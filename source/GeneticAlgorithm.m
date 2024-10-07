classdef GeneticAlgorithm
    properties
        population  % Weight matrix: each row is one genome
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
            obj.population = single(randn(nWeights, param.populationSize));
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

        function obj = runEvolution(obj)
            % Run the genetic algorithm for a specified number of generations

            for gen = 1:obj.param.generations
                tic
                fitnessScores = evaluateFitness(obj.population);
                % obj.population = obj.population.generateNextGeneration(fitnessScores);
                dt = toc;

                [~, topFitness] = max(fitnessScores);
            
                obj.population = repmat(obj.population(:, topFitness), 1, obj.param.populationSize);
                mutationMask = rand(size(obj.population)) <= obj.param.mutationRate;
                obj.population = obj.population + mutationMask .* randn(size(obj.population));
                
                % Optionally, display progress
                fprintf('Generation %d: best fitness = %f, dt = %f s\n', gen, max(fitnessScores), dt);
                if obj.param.plotFitness
                    hold on;
                    scatter(gen * ones(size(fitnessScores)), fitnessScores, '.', 'MarkerEdgeColor', '#0072BD')
                    hold off
                    drawnow
                end
            end
        end
    end
end