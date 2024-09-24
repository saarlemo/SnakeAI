classdef Agent
    % Agent An agent to play the snake game.
    properties
        genome  % The genome containing the neural network weights
    end
    methods
        function obj = Agent(genome)
            % Constructor of the class
            obj.genome = genome;
        end
        function action = decideAction(obj, state)
            % Decide on an action given the current game state using the neural network
            inputVector = obj.preprocessState(state);
            outputVector = obj.forwardPass(inputVector);
            action = obj.selectAction(outputVector);
        end
        function inputVector = preprocessState(~, state)
            % Preprocess the game state into an input vector of 11 binary values
            
            head = state.snake(end, :);             % Snake head position
            direction = state.direction;            % Current head direction
            food = state.food;                      % Food location
            gridSize = state.gridSize;              % Grid size
            snakeBody = state.snake(1:end-1, :);    % Exclude head for collision checks

            % Precompute possible directions
            leftDir = turnLeft(direction);
            rightDir = turnRight(direction);
            
            % Compute next positions for danger detection
            nextLeft = head + leftDir;
            nextRight = head + rightDir;
            nextAhead = head + direction;
            
            % Check for wall collisions
            dangerLeft = isOutOfBounds(nextLeft, gridSize) || isSnakeCollision(nextLeft, snakeBody);
            dangerRight = isOutOfBounds(nextRight, gridSize) || isSnakeCollision(nextRight, snakeBody);
            dangerAhead = isOutOfBounds(nextAhead, gridSize) || isSnakeCollision(nextAhead, snakeBody);
            
            % Moving direction (one-hot encoding)
            movingLeft = double(all(direction == [-1, 0]));
            movingRight = double(all(direction == [1, 0]));
            movingUp = double(all(direction == [0, 1]));
            movingDown = double(all(direction == [0, -1]));

            % Food relative position
            foodLeft = food(1) < head(1);
            foodRight = food(1) > head(1);
            foodUp = food(2) > head(2);
            foodDown = food(2) < head(2);
            
            % Create input vector
            inputVector = double([
                dangerLeft;
                dangerRight;
                dangerAhead;
                movingLeft;
                movingRight;
                movingUp;
                movingDown;
                foodLeft;
                foodRight;
                foodUp;
                foodDown
            ]);

            % Helper function to check if a position is out of bounds
            function out = isOutOfBounds(pos, gridSize)
                out = pos(1) < 1 || pos(1) > gridSize(1) || pos(2) < 1 || pos(2) > gridSize(2);
            end
            
            % Helper function to check if a position collides with the snake
            function collision = isSnakeCollision(pos, snakeBody)
                % Utilize logical indexing for faster collision detection
                collision = any(all(bsxfun(@eq, snakeBody, pos), 2));
            end

            % Helper functions to determine new direction vectors
            function newDirection = turnLeft(currentDirection)
                rotationMatrix = [0, -1; 1, 0];
                newDirection = (rotationMatrix * currentDirection')';
            end
            
            function newDirection = turnRight(currentDirection)
                rotationMatrix = [0, 1; -1, 0];
                newDirection = (rotationMatrix * currentDirection')';
            end
        end

        function outputVector = forwardPass(obj, inputVector)
            % Forward pass through the neural network
            activation = inputVector;
            for i = 1:length(obj.genome.weights)
                weights = obj.genome.weights{i};
                % Compute weighted sum
                z = weights * activation;
                % Apply activation function (e.g., ReLU)
                activation = obj.activationFunction(z);
            end
            outputVector = activation;
        end
        function a = activationFunction(~, z)
            % ReLU activation function
            a = max(0, z);
        end
        function action = selectAction(~, outputVector)
            % Select the action based on the neural network's output
            [~, idx] = max(outputVector);
            possibleActions = {'left', 'straight', 'right'};
            action = possibleActions{idx};
        end
    end
end
