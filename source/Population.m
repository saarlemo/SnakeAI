classdef Population
    properties
        genomes % Cell array of Genome objects
        param % Parameters
    end
    methods
        function obj = Population(param)
            obj.param = param;
            obj.genomes = cell(1, param.populationSize);
            for i = 1:param.populationSize
                obj.genomes{i} = Genome(obj.param.architecture);
            end
        end
        function parents = selectParents(obj, fitnessScores)
            [~, idx] = maxk(fitnessScores, obj.param.topNreproduce);
            parents = cell(1, length(obj.genomes));
            for ii = 1:length(obj.genomes)
                parentGenomeIdx = idx(mod(ii, obj.param.topNreproduce) + 1);
                parents{ii} = obj.genomes{parentGenomeIdx};
            end
        end
        function obj = generateNextGeneration(obj, fitnessScores)
            % Produce the next generation of genomes
            parents = obj.selectParents(fitnessScores);
            newGenomes = cell(1, length(parents));
            for i = 1:2:length(parents)
                parent1 = parents{i};
                if i + 1 <= length(parents)
                    parent2 = parents{i + 1};
                else
                    parent2 = parents{1};
                end
                offspring1 = parent1.crossover(parent2);
                offspring2 = parent2.crossover(parent1);
                offspring1 = offspring1.mutate(obj.param.mutationRate);
                offspring2 = offspring2.mutate(obj.param.mutationRate);
                newGenomes{i} = offspring1;
                if i + 1 <= length(parents)
                    newGenomes{i + 1} = offspring2;
                end
            end
            obj.genomes = newGenomes;
        end
    end
end
