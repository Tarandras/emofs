%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vectorized EMoFS Multivariate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pdf_index, Mixture_Out] = emofs( ...
    inputArray, HalfPeriod_L, Mixture_Count_N, Trimming_Factor, ...
    Convergence_Factor, Convergence_Debug)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Design decision
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MAX_ITERATION = 1000; 
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Input sizes
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, rCount] = size(inputArray);
    if ~isscalar(Mixture_Count_N)
        error("Mixture_Count_N must be scalar");
    end
    if length(HalfPeriod_L) ~= rCount
        error("HalfPeriod_L must match number of variables");
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Construct pdf_index using ndgrid (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MaxHalfPeriod = max(HalfPeriod_L);
    MaxPeriod = 2 * MaxHalfPeriod;
    
    rangeVec = -MaxHalfPeriod:MaxHalfPeriod;
    %rangeVec = 1:MaxHalfPeriod
    gridCells = cell(1, rCount);
    [gridCells{:}] = ndgrid(rangeVec);
    pdf_index = reshape(cat(rCount+1, gridCells{:}), [], rCount);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Construct mixture indices (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gridCells = cell(1, rCount);
    %[gridCells{:}] = ndgrid(1:Mixture_Count_N);
    [gridCells{:}] = ndgrid(-Mixture_Count_N:Mixture_Count_N);
    mixture_indices = reshape(cat(rCount+1, gridCells{:}), [], rCount);
    mixtureCount = size(mixture_indices, 1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Initialize mixtures
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tao_MixturesA = ones(mixtureCount,1) / Mixture_Count_N;
    tao_MixturesB = ones(mixtureCount,1) / Mixture_Count_N;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Precompute scaling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    scaleVec = pi ./ HalfPeriod_L;                 % 1 x rCount
    scaledInput = inputArray .* scaleVec;          % inputSize x rCount
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% EM Iterations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Convergence_Debug
        error_diff_A_history = zeros(MAX_ITERATION, mixtureCount);
        error_diff_B_history = zeros(MAX_ITERATION, mixtureCount);
        rate_A_history = zeros(MAX_ITERATION, mixtureCount);
        rate_B_history = zeros(MAX_ITERATION, mixtureCount);
        tao_MixturesA_bckup = zeros(MAX_ITERATION, mixtureCount);
    end
    for iteration = 1:MAX_ITERATION % Design decision taking IGMM as reference
        % Convergence backup
        tao_MixturesA_convergence  = tao_MixturesA;
        tao_MixturesB_convergence  = tao_MixturesB;
    
        % Phase matrix (inputSize x mixtureCount)
        phaseMatrix = scaledInput * mixture_indices.';   
    
        cosMat = cos(phaseMatrix);
        sinMat = sin(phaseMatrix);
    
        all_absolute_sum = sum(abs(tao_MixturesA) + abs(tao_MixturesB));
    
        % Denominator per row
        denominator = all_absolute_sum + ...
            cosMat * tao_MixturesA + ...
            sinMat * tao_MixturesB;      % inputSize x 1
    
        % T_p and hat_T_p
        T_p = sum(cosMat ./ denominator, 1).';      % column vector
        hat_T_p = sum(sinMat ./ denominator, 1).';  % column vector
    
        % Sums
        T_p_sum = sum(tao_MixturesA .* T_p + abs(tao_MixturesA));
        hat_T_p_sum = sum(tao_MixturesA .* hat_T_p + abs(tao_MixturesA));
    
        % C_n and S_n
        C_n = all_absolute_sum - abs(tao_MixturesA);
        S_n = all_absolute_sum - abs(tao_MixturesB);
    
        % Update A
        tao_MixturesA = (T_p .* C_n) ./ ...
            (T_p_sum + hat_T_p_sum - sign(tao_MixturesA) .* T_p);
    
        % Update B
        tao_MixturesB = (hat_T_p .* S_n) ./ ...
            (T_p_sum + hat_T_p_sum - sign(tao_MixturesB) .* hat_T_p);

        % Convergence check
        error_diff_A = tao_MixturesA_convergence - tao_MixturesA;
        error_diff_B = tao_MixturesB_convergence - tao_MixturesB;
        error_rms_A = sqrt(sum(error_diff_A.^2, "all")); 
        error_rms_B = sqrt(sum(error_diff_B.^2, "all")); 
        if Convergence_Debug
            error_diff_A_history(iteration, :) = error_diff_A.';
            error_diff_B_history(iteration, :) = error_diff_B.';
            rate_A_history(iteration, :) = (tao_MixturesA_convergence ...
                ./ tao_MixturesA).';
            rate_B_history(iteration, :) = (tao_MixturesB_convergence ...
                ./ tao_MixturesB).';
            tao_MixturesA_bckup(iteration, :) = tao_MixturesA;
        end
        %fprintf("Convergence RMS, A and B, at %d th cycle:" + ...
        %    " %f; %f\n", iteration, error_rms_A, error_rms_B);
        if( error_rms_A <= Convergence_Factor && ...
                error_rms_B <= Convergence_Factor)
            fprintf("Converged at %d th iteration\n", iteration);
            break;
        end
            
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Check Convergence
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if( iteration == MAX_ITERATION )
        fprintf("Failed to converge in %d iterations for emofs" + ...
            " with %d components\n", iteration, Mixture_Count_N);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Convergence Debug
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Convergence_Debug
        figure;
        colors = parula(Mixture_Count_N);
        for plotter = 1:Mixture_Count_N
            plot( 1:iteration, ...
              abs( tao_MixturesA_bckup(1:iteration, plotter) ...
                   - tao_MixturesA(plotter) ), ...
              'Color', colors(plotter,:), ...
              'LineWidth', 0.7 );
    
            hold on
        end
        xlabel('Iteration');
        ylabel('Error Per Iteration');
        colormap(parula);
        cb = colorbar;
        cb.Label.String = 'Mixture Index';
        caxis([1 Mixture_Count_N]);
        hold off
    
        
        size(tao_MixturesA_bckup(1,:))
        size(tao_MixturesA)
        err = zeros(MAX_ITERATION, mixtureCount);
        ratio = zeros(MAX_ITERATION-1, mixtureCount);
        for k = 1:iteration
            err(k,:) = abs(tao_MixturesA_bckup(k,:) - tao_MixturesA');
        end
        for k = 1:iteration-1
            ratio(k,:) = err(k+1,:) ./ err(k,:);
        end
        figure;
        plot(ratio)
        ylabel('Convergence Ratio')
        xlabel('Iteration')
  
        figure;
        colors = parula(Mixture_Count_N);
        for plotter = 1:Mixture_Count_N
            plot(abs(error_diff_A_history(1:(floor(iteration/2)), ...
                plotter)), 'Color', colors(plotter,:), 'LineWidth', 0.7 );
            hold on
        end
        xlabel('Iteration');
        ylabel('a(t+1) / a(t) Rate');
        colormap(parula);
        cb = colorbar;
        cb.Label.String = 'Mixture Index';
        caxis([1 Mixture_Count_N]);
        hold off
    
        figure;
        colors = parula(Mixture_Count_N);
        for plotter = 1:Mixture_Count_N
            plot( abs(rate_A_history(1:floor(iteration/2), plotter)), ...
                  'Color', colors(plotter,:), ...
                  'LineWidth', 0.7 );
            hold on
        end
        xlabel('Iteration');
        ylabel('a(t+1) / a(t) Rate');
        colormap(parula);
        cb = colorbar;
        cb.Label.String = 'Mixture Index';
        caxis([1 Mixture_Count_N]);
        hold off
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute Mixture_Out (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    scaledPDF = pdf_index .* scaleVec;
    phasePDF = scaledPDF * mixture_indices.';   
    
    Mixture_Out = ...
        cos(phasePDF) * tao_MixturesA + ...
        sin(phasePDF) * tao_MixturesB;
    
    Mixture_Out = Mixture_Out + (4 / MaxPeriod);
    %Mixture_Out = smooth(Mixture_Out);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Remove Bias & Normalize (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DC_below_zero = sum(abs(Mixture_Out(Mixture_Out < 0)));
    DC_average_below_0 = DC_below_zero / numel(Mixture_Out);
    Mixture_Out = Mixture_Out + DC_average_below_0;
    Mixture_Out(Mixture_Out < Trimming_Factor) = 0;
    Total_Area_Mixture_Out = sum(Mixture_Out);
    Mixture_Out = Mixture_Out / Total_Area_Mixture_Out;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Debug Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %fprintf("DC_average_below_0:  %f\n", DC_average_below_0);
    %fprintf("DC_below_zero:  %f\n", DC_below_zero);
    %fprintf("Total_Area_Mixture_Out:  %f\n", Total_Area_Mixture_Out);

end