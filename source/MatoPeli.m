classdef MatoPeli
    properties
        gridSize         % Size of the game grid [rows, columns]
        snake            % Matrix of snake body positions
        direction        % Current direction vector
        snakeLength      % Current length of the snake
        food             % Position of the food
        isGameOver       % Flag indicating if the game is over
        initialLength    % Initial length of the snake
        possibleActions  % Possible actions: 'left', 'straight', 'right'
        score            % Current score
        maxSteps         % Maximum number of steps allowed
        stepsTaken       % Number of steps taken so far
    end
    
    methods
        function obj = MatoPeli(param)
            % Constructor to initialize the game
            if ~isfield(param, 'gridSize')
                gridSize = [20, 20];
            end
            if ~isfield(param, 'initialLength')
                initialLength = 3;
            end
            if ~isfield(param, 'maxSteps')
                maxSteps = 500;
            end
            obj.gridSize = param.gridSize;
            obj.initialLength = param.initialLength;
            obj.maxSteps = param.maxSteps;
            obj.possibleActions = {'left', 'straight', 'right'};
            obj = obj.reset();
        end
        
        function obj = reset(obj)
            % Initialize or reset the game state
            obj.snake = [floor(obj.gridSize(1)/2), floor(obj.gridSize(2)/2)]; % Start position
            obj.direction = [0, 1];  % Initial direction (up)
            obj.snakeLength = obj.initialLength;
            obj.food = obj.placeFood();
            obj.isGameOver = false;
            obj.score = 0;
            obj.stepsTaken = 0;
        end
        
        function possibleActions = getPossibleActions(obj)
            % Return a list of possible actions at the current state
            possibleActions = obj.possibleActions;
        end
        
        function isOver = isOver(obj)
            % Check if the game has ended
            isOver = obj.isGameOver || (obj.stepsTaken >= obj.maxSteps);
        end
        
        function obj = applyAction(obj, action)
            % Apply an action to the game state
            obj.direction = obj.updateDirection(obj.direction, action);
            
            % Move snake
            obj.snake = [obj.snake; obj.snake(end, :) + obj.direction];
            if size(obj.snake, 1) > obj.snakeLength
                obj.snake(1, :) = [];  % Remove tail
            end
            
            % Check collisions
            head = obj.snake(end, :);
            if head(1) < 1 || head(1) > obj.gridSize(1) || ... % Wall collision
               head(2) < 1 || head(2) > obj.gridSize(2) || ...
               ismember(head, obj.snake(1:end-1, :), 'rows')    % Self collision
                obj.isGameOver = true;
            end
            
            % Check if food is eaten
            if isequal(head, obj.food)
                obj.snakeLength = obj.snakeLength + 1;
                obj.score = obj.score + 1;
                obj.food = obj.placeFood();
            end
            
            obj.stepsTaken = obj.stepsTaken + 1;
        end
        
        function reward = getReward(obj)
            % Calculate the reward or score for the current state
            % Reward can be based on the score and survival time
            reward = 10 * obj.score + obj.stepsTaken * 0.01; % Adjust weighting as needed
        end
        
        function state = getState(obj)
            % Get the current state of the game
            % State can include the snake's head position, direction, food position, etc.
            state.snake = obj.snake;
            state.direction = obj.direction;
            state.food = obj.food;
            state.gridSize = obj.gridSize;
        end

        function render(obj, figureHandle)
            % Render the current state of the game
            figure(figureHandle);
            clf; % Clear the figure
            hold on;
            axis([1 obj.gridSize(1)+1 1 obj.gridSize(2)+1]);
            axis square;
            set(gca, 'XTick', 1:obj.gridSize(1)+1, 'YTick', 1:obj.gridSize(2)+1);
            grid on;
            
            % Draw the snake
            plot(obj.snake(:, 1) + 0.5, obj.snake(:, 2) + 0.5, 'ks', 'MarkerFaceColor', 'g', 'MarkerSize', 20);
            
            % Draw the food
            plot(obj.food(1) + 0.5, obj.food(2) + 0.5, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 20);
            
            % Update the title with the current score
            title(['Score: ' num2str(obj.score)]);
            
            drawnow;
        end
    end
    
    methods (Access = private)
        function newDirection = updateDirection(~, currentDirection, action)
            % Update direction based on the action
            switch action
                case 'left' % Left
                    rotationMatrix = [0, -1; 1, 0];
                    newDirection = (rotationMatrix * currentDirection')';
                case 'right' % Right
                    rotationMatrix = [0, 1; -1, 0];
                    newDirection = (rotationMatrix * currentDirection')';
                case 'straight' % Straight
                    newDirection = currentDirection;
                otherwise
                    error('Invalid action: %s', action);
            end
        end
        
        function foodPosition = placeFood(obj)
            % Place food in a random empty cell
            emptySpaces = setdiff(obj.allComb(1:obj.gridSize(1), 1:obj.gridSize(2)), obj.snake, 'rows');
            if isempty(emptySpaces)
                foodPosition = [];
            else
                idx = randi(size(emptySpaces, 1));
                foodPosition = emptySpaces(idx, :);
            end
        end
        
        function combos = allComb(~, v1, v2)
            % Generate all combinations of grid positions
            [A, B] = meshgrid(v1, v2);
            combos = [A(:), B(:)];
        end
    end
end
