function simulation
    
    function [dx, dy] = computeForce(x, y)
        %   dx_i = sum_{j != i} f(r_ij)*(x_i - x_j)
        %   dy_i = sum_{j != i} f(r_ij)*(y_i - y_j)
        %   where r_ij = distance between cell i and cell j in 2D,
        %   and x,y are each N x 1 with N cells in 2D.
        
        % Define parameters
        A = 0.5;       % amplitude of repulsion
        alpha = 0.5; % decay rate of repulsion
        B = 1;       % amplitude of attraction
        beta = 2;  % decay rate of attraction
        
        % Define f(r) as an anonymous function
        f = @(r) A*exp(-alpha*r) - B*exp(-beta*r);
    
        N = length(x);
        dx = zeros(N,1);
        dy = zeros(N,1);
    
        for i = 1:N
            for j = 1:N
                if j ~= i
                    % Differences in x,y
                    diffx = x(i) - x(j);
                    diffy = y(i) - y(j);
    
                    % Euclidean distance between (x_i,y_i) and (x_j,y_j)
                    r = sqrt(diffx^2 + diffy^2);
    
                    % Optionally check r>0 to avoid dividing by zero
                    % if your f(r) needs that. Here it's not shown.
                    val = f(r);  % scalar from your function f
    
                    % Sum up contributions
                    dx(i) = dx(i) + val * diffx;
                    dy(i) = dy(i) + val * diffy;
                end
            end
        end
    end

    % Parameters
    numCells = 4;
    radius   = 50;
    numSteps = 150;
    
    % Allocate array to store positions (numCells x 2 x numSteps)
    % posHistory = zeros(numCells, 2, numSteps);
    history = cell(1, numSteps);
    
    % Initialize cell positions near center
    spawnRadius = 1;
    angles = 2*pi*rand(numCells,1);
    rDist  = spawnRadius * sqrt(rand(numCells,1));
    x = rDist .* cos(angles);
    y = rDist .* sin(angles);
    p = rand(numCells,1);
 
    
    % Store initial positions
    % posHistory(:,1,1) = x;
    % posHistory(:,2,1) = y;
    history{1} = [x, y, p];
    
    % Run the simulation in a loop, but NOT animating yet
    for t = 2:numSteps
        
        coords = [x, y];
        distMat = squareform(pdist(coords)); 

        % WEIGHTED MEAN DIST
        % Choose a bandwidth parameter h that defines the neighborhood size
        h = 7; 
        
        % Compute weights that decay with distance (Gaussian kernel)
        w = exp(-((distMat/h).^2));
        % w = exp(-distMat);
        
        % Exclude self-contributions by zeroing the diagonal
        w(eye(size(w))==1) = 0;
        
        % Compute a weighted mean distance for each cell
        weightedMeanDist = sum(distMat .* w, 2) ./ sum(w, 2);
        disp(weightedMeanDist)

        % Define local density as the inverse of the weighted mean distance
        locDensity = 1 ./ weightedMeanDist;
        % locDensity = exp(-((weightedMeanDist/h).^2));
        % disp(weightedMeanDist)

        % UNWEIGHTED MEAN DIST
        % N = size(coords,1);
        % % sum distance from each row i to all columns j
        % sumDist = sum(distMat, 2);
        % % average distance to the other cells
        % meanDist = sumDist / (N-1);
        % % Define local density as inverse of mean distance
        % locDensity = 1 ./ meanDist;

        % locDensity = (locDensity - min(locDensity)) / (max(locDensity) - min(locDensity));
        % disp(locDensity)

        lambda = 0.05;

        [dx, dy] = computeForce(x, y);
        dp = lambda*(1-locDensity);
        % disp(p)
        % disp(dp)
        x = x + dx;
        y = y + dy;
        p = p +dp;
        % disp(t/numSteps)
        % disp(length(x))

        dividing = find(p >= 1);  % Indices of cells that reached p=1

        for idx = dividing'
            p(idx) = 0;           % Reset parent's p to 0
            epsDist = 0.75;        % small distance for new cell
            randAngle = 2*pi*rand;
    
            % compute new cell's position
            newX = x(idx) + epsDist*cos(randAngle);
            newY = y(idx) + epsDist*sin(randAngle);
    
            % Append to x, y, p
            x  = [x;  newX];
            y  = [y;  newY];
            p  = [p;  0];         % new cell starts with p=0
        end
        
        % Enforce circular boundary
        distFromCenter = sqrt(x.^2 + y.^2);
        outside = distFromCenter > radius;
        if any(outside)
            scaleFactor   = radius ./ distFromCenter(outside);
            x(outside)    = x(outside) .* scaleFactor;
            y(outside)    = y(outside) .* scaleFactor;
        end

        numCells = length(x);  % Update to include newly divided cells
        % posHistory(1:numCells,1,t) = x;
        % posHistory(1:numCells,2,t) = y;
        history{t} = [x, y, p];
    end
    
    % Figure
    hFig = figure('Name','Cell Simulation','NumberTitle','off');
    hAx  = axes('Parent', hFig);
    hold(hAx, 'on');
    axis(hAx, 'equal');
    caxis(hAx, [0, 1]);
    xlim(hAx, [-radius, radius]);
    ylim(hAx, [-radius, radius]);
    title(hAx, '2D Circle + Cells');
    
    % Draw boundary circle
    theta = linspace(0, 2*pi, 100);
    plot(hAx, radius*cos(theta), radius*sin(theta), 'k'); 
    
    % Plot initial positions
    scat = scatter(hAx, history{1}(:,1), history{1}(:,2), [], history{1}(:,3), 'filled');
    
    % Slider
    % Places a slider at the bottom of the figure
    % 'Min'=1, 'Max'=numSteps, start at 1
    hSlider = uicontrol('Style','slider', ...
        'Min',1, 'Max',numSteps, 'Value',1, ...
        'SliderStep',[1/(numSteps-1), 10/(numSteps-1)], ...
        'Units','normalized', 'Position',[0.1 0.02 0.6 0.05], ...
        'Callback', @sliderCallback);
    
    % Play/Pause
    % This button toggles a timer that automatically increments the slider
    hPlayPause = uicontrol('Style','pushbutton', 'String','Play', ...
        'Units','normalized', 'Position',[0.72 0.02 0.15 0.05], ...
        'Callback', @playPauseCallback);
    
    % Timer
    % We'll use a MATLAB timer object to periodically increment the slider
    playTimer = timer('ExecutionMode','fixedRate', ...
        'Period', 0.01, ...         % seconds between timer callbacks
        'TimerFcn', @autoAdvance); % function to call on each tick
   
    % Callback
    function sliderCallback(src, ~)
        % Called when the slider is moved
        t = round(get(src, 'Value'));  % current timestep
        updateScatter(t);
    end

    function playPauseCallback(src, ~)
        % Called when the "Play"/"Pause" button is clicked
        if strcmp(get(src, 'String'), 'Play')
            % Switch to "Pause" mode & start the timer
            set(src, 'String', 'Pause');
            start(playTimer);
        else
            % Switch to "Play" mode & stop the timer
            set(src, 'String', 'Play');
            stop(playTimer);
        end
    end

    function autoAdvance(~, ~)
        currentVal = round(get(hSlider, 'Value'));  % <-- round here
        if currentVal < numSteps
            newVal = currentVal + 1;
        else
            newVal = 1;  % loop back to start, or you could stop the timer
        end
        set(hSlider, 'Value', newVal);
        updateScatter(newVal);
    end

    function updateScatter(t)
        % Updates the scatter plot to show positions at time t
        data = history{t};
        xData = data(:,1);
        yData = data(:,2);
        cData = data(:,3);
        set(scat, 'XData', xData, 'YData', yData, 'CData', cData);
        drawnow;
    end

    % Cleanup
    % If the figure is closed while playing, stop the timer
    addlistener(hFig, 'ObjectBeingDestroyed', @(~,~) stopIfRunning());

    function stopIfRunning()
        if strcmp(playTimer.Running, 'on')
            stop(playTimer);
        end
        delete(playTimer);
    end

end
