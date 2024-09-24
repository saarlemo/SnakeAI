classdef GeneticAlgorithm
    properties
        gameClass   % Handle to the game class
        agentClass  % Handle to the agent class
        population  % Population object
        param       % Parameters
    end
    methods
        function obj = GeneticAlgorithm(gameClass, agentClass, param)
            % Constructor for the GeneticAlgorithm class.

            obj.gameClass = gameClass;
            obj.agentClass = agentClass;
            obj.param = param;
            obj.population = Population(param);
            if param.plotFitness == 1
                f1 = figure(1);
                scatter([],[])
                xlim([0 param.generations])
                grid on
            end
        end

        function fitnessScores = evaluateFitness(obj)
            % Evaluates the fitness scores of the whole population.

            numGenomes = length(obj.population.genomes);
            fitnessScores = zeros(1, numGenomes);
            for ii = 1:numGenomes
                genome = obj.population.genomes{ii};
                fitnessScores(ii) = obj.evaluateSingleGenomeFitness(genome);
            end
        end

        function fitness = evaluateSingleGenomeFitness(obj, genome)
            % Evaluates the fitness score of a single genome.

            gameInstance = obj.gameClass(obj.param);
            agentInstance = obj.agentClass(genome, obj.param);
            gameInstance = gameInstance.reset();
            while ~gameInstance.isOver()
                state = gameInstance.getState();
                action = agentInstance.decideAction(state);
                gameInstance = gameInstance.applyAction(action);
            end
            fitness = gameInstance.getReward();
        end

        function obj = runEvolution(obj)
            % Run the genetic algorithm for a specified number of generations

            for gen = 1:obj.param.generations
                fitnessScores = obj.evaluateFitness();
                obj.population = obj.population.generateNextGeneration(fitnessScores);
                
                % Optionally, display progress
                fprintf('Generation %d: Best Fitness = %f\n', gen, max(fitnessScores));
                if obj.param.plotFitness
                    hold on;
                    scatter(gen * ones(size(fitnessScores)), fitnessScores, '.')
                    hold off
                    drawnow
                end
            end
        end

        function gen = extractBestGenome(obj)
            % Returns the genome with best fitness from the population.

            fitnessScores = obj.evaluateFitness();
            [~, idx] = max(fitnessScores);
            gen = obj.population.genomes(idx);
            gen = gen{1};
        end

        function saveBestGenome(obj, fname)
            gen = extractBestGenome(obj);
            save(fname, 'gen')
        end

        function playGenome(obj, genome, playBackSpeed, playbackSteps)
            % Visualizes the snake game played by the agent with the input genome.

            tmpParam = obj.param;
            tmpParam.maxSteps = playbackSteps;
            tmpParam.dropoutRate = 0;
            gameInstance = obj.gameClass(tmpParam);
            agentInstance = obj.agentClass(genome, tmpParam);
            gameInstance = gameInstance.reset();

            % Figure handle for rendering
            f2 = figure(2);
            
            % Run the game loop
            while ~gameInstance.isOver()
                state = gameInstance.getState();
                action = agentInstance.decideAction(state);
                gameInstance = gameInstance.applyAction(action);
                gameInstance.render(f2); % Render
                pause(playBackSpeed); % Pause
            end

            fprintf('Final Score: %d\n', gameInstance.score); % Display final score
        end
    end
end
