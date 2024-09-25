classdef Genome
    properties
        architecture    % The neural network layer sizes, e.g., [16 100 100 3]
        weights         % Cell array of weight matrices
    end
    methods
        function obj = Genome(architecture, weights)
            if nargin < 2 || isempty(weights)
                obj.architecture = architecture;
                obj.weights = obj.initializeRandomWeights(architecture);
            else
                obj.architecture = architecture;
                obj.weights = weights;
            end
        end

        function weights = initializeRandomWeights(~, architecture)
            nLayers = length(architecture);
            weights = cell(1, nLayers - 1);
            for ii = 1:nLayers - 1
                inputSize = architecture(ii) + 1;
                outputSize = architecture(ii + 1);
                weights{ii} = randn(outputSize, inputSize) * 0.1;
            end
        end

        function obj = mutate(obj, mutationRate)
            for ii = 1:length(obj.weights)
                mutationMask = rand(size(obj.weights{ii})) < mutationRate;
                obj.weights{ii}(mutationMask) = obj.weights{ii}(mutationMask) + randn(sum(mutationMask(:)), 1) * 0.1;
            end
        end

        function offspring = crossover(obj, other)
            offspringWeights = obj.weights;
            crossoverPoint = randi(length(obj.weights));
            for ii = crossoverPoint:length(obj.weights)
                offspringWeights{ii} = other.weights{ii};
            end
            offspring = Genome(obj.architecture, offspringWeights);
        end
    end
end
