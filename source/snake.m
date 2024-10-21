function snake(weights, architecture)
    % Constants
    GRID_WIDTH = 20;
    GRID_HEIGHT = 20;
    MAX_STEPS = 200;
    BONUS_STEPS = 100;
    MAX_SNAKE_LENGTH = GRID_WIDTH * GRID_HEIGHT;

    % Direction vectors
    dx = [-1, 0, 1, 0];
    dy = [0, -1, 0, 1];

    % Initialize snake
    snake_x = zeros(MAX_SNAKE_LENGTH,1);
    snake_y = zeros(MAX_SNAKE_LENGTH,1);
    snake_length = 1;
    direction = 2; % Initial direction: right
    steps = 0;
    alive = true;
    score = 0;

    % Initialize snake position at center
    snake_x(1) = floor(GRID_WIDTH / 2);
    snake_y(1) = floor(GRID_HEIGHT / 2);

    % Initialize food position
    rng('shuffle');
    food_x = randi(GRID_WIDTH);
    food_y = randi(GRID_HEIGHT);

    % Ensure food is not placed on the snake
    while food_x == snake_x(1) && food_y == snake_y(1)
        food_x = randi(GRID_WIDTH);
        food_y = randi(GRID_HEIGHT);
    end

    % Unpack weights
    weights_cell = unpack_weights(weights, architecture);

    % Initialize figure
    figure;
    axis([0 GRID_WIDTH 0 GRID_HEIGHT]);
    grid on;
    hold on;

    currentMaxSteps = MAX_STEPS;

    % Game loop
    while alive && steps < currentMaxSteps
        % Calculate possible directions
        left_direction = mod(direction + 1, 4);
        right_direction = mod(direction + 3, 4);

        % Set neural network inputs
        inputs = zeros(12,1);
        inputs(1) = check_collision(left_direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
        inputs(2) = check_collision(right_direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
        inputs(3) = check_collision(direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
        inputs(4) = double(direction == 0); % Moving left
        inputs(5) = double(direction == 2); % Moving right
        inputs(6) = double(direction == 1); % Moving up
        inputs(7) = double(direction == 3); % Moving down
        inputs(8) = double(food_x < snake_x(1)); % Food left
        inputs(9) = double(food_x > snake_x(1)); % Food right
        inputs(10) = double(food_y < snake_y(1)); % Food above
        inputs(11) = double(food_y > snake_y(1)); % Food below
        inputs(12) = 1.0; % Bias

        % Forward pass through neural network
        hidden = cell(length(architecture)-2,1);

        % First hidden layer
        sum_layer = weights_cell{1}' * inputs;
        hidden{1} = max(0, sum_layer); % ReLU activation
        % Add bias unit
        hidden{1}(architecture(2)) = 1.0;

        % Hidden layers
        for l = 2:length(architecture)-2
            sum_layer = weights_cell{l}' * hidden{l-1};
            hidden{l} = max(0, sum_layer);
            % Add bias unit
            hidden{l}(architecture(l+1)) = 1.0;
        end

        % Output layer
        outputs = max(0, weights_cell{end}' * hidden{end});

        % Determine action
        [~, action] = max(outputs);
        action = action - 1; % Adjust for zero-based indexing

        % Update direction
        if action == 0
            direction = mod(direction + 1, 4); % Turn left
        elseif action == 1
            direction = mod(direction + 3, 4); % Turn right
        end
        % Else continue straight

        % Move the snake: shift positions
        for i = snake_length:-1:2
            snake_x(i) = snake_x(i-1);
            snake_y(i) = snake_y(i-1);
        end
        snake_x(1) = snake_x(1) + dx(direction+1);
        snake_y(1) = snake_y(1) + dy(direction+1);

        % Check for collisions
        if snake_x(1) < 1 || snake_x(1) > GRID_WIDTH || snake_y(1) < 1 || snake_y(1) > GRID_HEIGHT
            alive = false;
            break;
        end
        for i = 2:snake_length
            if snake_x(1) == snake_x(i) && snake_y(1) == snake_y(i)
                alive = false;
                break;
            end
        end

        % Check for food consumption
        if snake_x(1) == food_x && snake_y(1) == food_y
            if snake_length < MAX_SNAKE_LENGTH
                snake_length = snake_length + 1;
            end
            score = score + 1;
            currentMaxSteps = currentMaxSteps + BONUS_STEPS;

            % Place new food
            placed = false;
            while ~placed
                food_x = randi(GRID_WIDTH);
                food_y = randi(GRID_HEIGHT);
                placed = true;
                for i = 1:snake_length
                    if snake_x(i) == food_x && snake_y(i) == food_y
                        placed = false;
                        break;
                    end
                end
            end
        end

        steps = steps + 1;

        % Update visualization
        clf;
        hold on;
        for i = 1:snake_length
            rectangle('Position', [snake_x(i)-1, snake_y(i)-1, 1, 1], 'FaceColor', 'green');
        end
        rectangle('Position', [food_x-1, food_y-1, 1, 1], 'FaceColor', 'red');
        axis([0 GRID_WIDTH 0 GRID_HEIGHT]);
        set(gca, 'xtick', 0:GRID_WIDTH, 'ytick', 0:GRID_HEIGHT);
        grid on;
        drawnow;
        % Optionally, add a pause to control speed
        % pause(0.1);
    end

    fprintf('Game over. Score: %d\n', score);
end

function collision = check_collision(dir, head_x, head_y, snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT)
    dx = [-1, 0, 1, 0];
    dy = [0, -1, 0, 1];

    nx = head_x + dx(dir+1);
    ny = head_y + dy(dir+1);

    % Check wall collision
    if nx < 1 || nx > GRID_WIDTH || ny < 1 || ny > GRID_HEIGHT
        collision = 1;
        return;
    end

    % Check self-collision
    for i = 1:snake_length
        if snake_x(i) == nx && snake_y(i) == ny
            collision = 1;
            return;
        end
    end

    collision = 0;
end

function weights_cell = unpack_weights(weights_vector, architecture)
    idx = 1;
    num_layers = length(architecture) - 1;
    weights_cell = cell(num_layers,1);

    % Layer 1
    input_size = architecture(1);
    output_size = architecture(2) - 1;
    num_weights = input_size * output_size;
    weights_cell{1} = reshape(weights_vector(idx : idx + num_weights -1), input_size, output_size);
    idx = idx + num_weights;

    % Hidden layers
    for l = 2 : num_layers - 1
        input_size = architecture(l);
        output_size = architecture(l+1) -1;
        num_weights = input_size * output_size;
        weights_cell{l} = reshape(weights_vector(idx : idx + num_weights -1), input_size, output_size);
        idx = idx + num_weights;
    end

    % Output layer
    input_size = architecture(end -1);
    output_size = architecture(end);
    num_weights = input_size * output_size;
    weights_cell{num_layers} = reshape(weights_vector(idx : idx + num_weights -1), input_size, output_size);
end
