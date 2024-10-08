function fitnessValue = snake(weights, architecture)
% SNAKE - Plays the snake game using the given weights and network architecture
%   weights: 1 x n array of network weights
%   architecture: 1 x n array of layer sizes
%   Visualizes the snake game and returns the fitness value

% Extract network architecture parameters
INPUT_SIZE = architecture(1);
HIDDEN_SIZE = architecture(2);
N_HIDDEN = numel(architecture)-2;
OUTPUT_SIZE = architecture(end);

% Define constants
GRID_WIDTH = 20;
GRID_HEIGHT = 20;
MAX_SNAKE_LENGTH = GRID_WIDTH * GRID_HEIGHT;
MAX_STEPS = 200;
BONUS_STEPS = 100;

% Direction vectors: 0 - left, 1 - up, 2 - right, 3 - down
dx = [-1, 0, 1, 0];
dy = [0, -1, 0, 1];

% Initialize game state variables
snake_x = zeros(MAX_SNAKE_LENGTH, 1);
snake_y = zeros(MAX_SNAKE_LENGTH, 1);
snake_length = 1;
direction = 2; % Initial direction: right
steps = 0;
alive = true;
score = 0;

% Initialize snake position at the center
snake_x(1) = floor(GRID_WIDTH / 2);
snake_y(1) = floor(GRID_HEIGHT / 2);

% Initialize random seed based on weights (simple method)
seed = uint32(weights(1) * 10000.0);

% Initialize food position
rng(seed); % Set random seed
food_x = randi(GRID_WIDTH) - 1; % Zero-based index
food_y = randi(GRID_HEIGHT) - 1;

% Ensure food is not placed on the snake
while food_x == snake_x(1) && food_y == snake_y(1)
    food_x = randi(GRID_WIDTH) - 1;
    food_y = randi(GRID_HEIGHT) - 1;
end

% Neural network arrays
inputs = zeros(INPUT_SIZE, 1);
hidden = zeros(HIDDEN_SIZE, N_HIDDEN);
outputs = zeros(OUTPUT_SIZE, 1);

% Reshape weights
idx = 1;

% Weights from input to first hidden layer
w_input_hidden = reshape(weights(idx:idx+INPUT_SIZE*HIDDEN_SIZE-1), INPUT_SIZE, HIDDEN_SIZE);
idx = idx + INPUT_SIZE * HIDDEN_SIZE;

% Weights between hidden layers
w_hidden_hidden = cell(N_HIDDEN - 1, 1);
for l = 1:N_HIDDEN - 1
    w_hidden_hidden{l} = reshape(weights(idx:idx+HIDDEN_SIZE*HIDDEN_SIZE-1), HIDDEN_SIZE, HIDDEN_SIZE);
    idx = idx + HIDDEN_SIZE * HIDDEN_SIZE;
end

% Weights from last hidden to output layer
w_hidden_output = reshape(weights(idx:idx+HIDDEN_SIZE*OUTPUT_SIZE-1), HIDDEN_SIZE, OUTPUT_SIZE);
idx = idx + HIDDEN_SIZE * OUTPUT_SIZE;

% Game loop
currentMaxSteps = MAX_STEPS;

% Visualization setup
figure;
axis([0 GRID_WIDTH 0 GRID_HEIGHT]);
axis square;
hold on;

while alive && steps < currentMaxSteps
    % Calculate possible directions
    left_direction = mod(direction + 1, 4);
    right_direction = mod(direction + 3, 4);

    % Set neural network inputs
    inputs(1) = check_collision(left_direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
    inputs(2) = check_collision(right_direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
    inputs(3) = check_collision(direction, snake_x(1), snake_y(1), snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT);
    inputs(4) = (direction == 0); % Moving left
    inputs(5) = (direction == 2); % Moving right
    inputs(6) = (direction == 1); % Moving up
    inputs(7) = (direction == 3); % Moving down
    inputs(8) = (food_x < snake_x(1)); % Food left
    inputs(9) = (food_x > snake_x(1)); % Food right
    inputs(10)= (food_y < snake_y(1)); % Food above
    inputs(11)= (food_y > snake_y(1)); % Food below
    inputs(12)= 1.0; % Bias term

    % Forward pass through neural network
    % First hidden layer
    hidden(:,1) = activation(w_input_hidden' * inputs);

    % Hidden layers
    for l = 2:N_HIDDEN
        hidden(:,l) = activation(w_hidden_hidden{l-1}' * hidden(:,l-1));
    end

    % Output layer
    outputs = activation(w_hidden_output' * hidden(:,N_HIDDEN));

    % Determine action based on highest output
    [~, action] = max(outputs);

    % Update direction based on action
    if action == 1
        direction = mod(direction + 1, 4); % Turn left
    elseif action == 2
        direction = mod(direction + 3, 4); % Turn right
    % Else continue straight
    end

    % Move the snake: shift positions
    snake_x(2:snake_length) = snake_x(1:snake_length-1);
    snake_y(2:snake_length) = snake_y(1:snake_length-1);
    snake_x(1) = snake_x(1) + dx(direction+1);
    snake_y(1) = snake_y(1) + dy(direction+1);

    % Check for collisions
    if snake_x(1) < 0 || snake_x(1) >= GRID_WIDTH || snake_y(1) < 0 || snake_y(1) >= GRID_HEIGHT
        alive = false;
        break;
    end
    for i = 2:snake_length
        if snake_x(1) == snake_x(i) && snake_y(1) == snake_y(i)
            alive = false;
            break;
        end
    end
    if ~alive
        break;
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
            food_x = randi(GRID_WIDTH) - 1;
            food_y = randi(GRID_HEIGHT) - 1;
            placed = true;
            for i = 1:snake_length
                if snake_x(i) == food_x && snake_y(i) == food_y
                    placed = false;
                    break;
                end
            end
        end
    end

    % Visualization
    cla;
    plot(snake_x(1:snake_length)+0.5, snake_y(1:snake_length)+0.5, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    hold on;
    plot(food_x+0.5, food_y+0.5, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    axis([0 GRID_WIDTH 0 GRID_HEIGHT]);
    axis square;
    drawnow;

    steps = steps + 1;
end

a_d = double(score);
s_d = double(steps);

fitnessValue = log1p(2.0^a_d + 500.0 * a_d^2.1 - a_d^1.2 * (0.25 * s_d)^1.3);

end

function y = activation(x)
% ReLU activation function
y = max(0, x);
end

function collision = check_collision(dir, head_x, head_y, snake_x, snake_y, snake_length, GRID_WIDTH, GRID_HEIGHT)
% Check collision in a given direction
dx = [-1, 0, 1, 0];
dy = [0, -1, 0, 1];
nx = head_x + dx(dir+1);
ny = head_y + dy(dir+1);

% Check wall collision
if nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT
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
