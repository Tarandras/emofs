%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pdf_index, Mixture_Out]=emofs_multivariate_educational( ...
    inputArray, HalfPeriod_L, Mixture_Count_N, Trimming_Factor)
%function [tao_MixturesA tao_MixturesB]=emofs_multivariate_educational( ...
%    inputArray, HalfPeriod_L, Mixture_Count_N, Trimming_Factor)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the input size and the count of random variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[inputSize, rCount] = size(inputArray);

% TODO: remove the following, we shall support only the case of having same
% amount of components for all dimensions. 

% It is expected that the Mixture_Count_N has the dimension of either [1 1]
% as the same amount of sinusoidal component or [1 rCount] as 
% [N_x, N_y, N_z, ...] if it should differ per random variable. 
[tmpCheckR, tmpCheckC] = size(Mixture_Count_N);
if (tmpCheckR == 1) && (tmpCheckC == 1) 
    % this is a regular condition
    fprintf("Same sinusoidal count for all random variables as:\n");
else
    error("Unexpected size for count of sinusoidal terms")
end

% Half period can differ per each random variable
[tmpCheckR, tmpCheckC] = size(HalfPeriod_L);
if (tmpCheckR == 1) && (tmpCheckC == rCount)
    % this is a regular condition 
    fprintf("Multiple half periods for random variables as:\n");
else
    error("Unexpected half period input")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construct mixture index Matrix [ (2*MaxHalfPeriod)^rCount  rCount ]
%% Note that this matrix holds the input points for mixture output.
%% It spans the points between -MaxHalfPeriod to MaxHalfPeriod
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxHalfPeriod = max( HalfPeriod_L );
MaxPeriod = 2*MaxHalfPeriod;
% initiate index matrix
pdf_index = zeros( (MaxPeriod+1)^rCount, rCount );
% initiate counter
index_counter = 0;
for i=1:( (MaxPeriod+1)^rCount )
    % increment index_counter and distribute result into pdf_index
    transient_divider = index_counter;
    for last_index=rCount:-1:1
        % we operate based on module / base ( MaxPeriod + 1)
        last_element = mod( transient_divider, (MaxPeriod + 1) );
        pdf_index(i, last_index) = last_element;
        transient_divider = floor( transient_divider / (MaxPeriod+1) ); 
    end
    index_counter = index_counter + 1;
end
pdf_index = pdf_index - MaxHalfPeriod;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construct index Matrix [ (Mixture_Count_N ^ rCount)    rCount ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initiate index matrix
mixture_indices = zeros( (Mixture_Count_N^rCount), rCount );
% initiate counter
index_counter = 0;
% start from the second row as the first is assigend to be one. 
for i=1:(Mixture_Count_N^rCount)
    % distribute result into mixture_indices
    transient_divider = index_counter;
    for last_index=rCount:-1:1
        % we operate based on module / base ( Mixture_Count_N )
        last_element = mod( transient_divider, Mixture_Count_N );
        mixture_indices(i, last_index) = last_element + 1;
        transient_divider = floor( transient_divider / Mixture_Count_N ); 
    end
    % increment index_counter
    index_counter = index_counter + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initially, the mixture components are distributed homogenously.
%% Note that the mixture_indices holds the indexes for both vectors. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tao_MixturesA = ones((Mixture_Count_N^rCount),1) / Mixture_Count_N;
tao_MixturesB = ones((Mixture_Count_N^rCount),1) / Mixture_Count_N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% apply the gEMoFS solution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TODO: replace the following with convergence check... Note that it is 
% ignored in this script for experimental purposes
for iteration=1: (13)
    fprintf("Current iteration:  %d\n", iteration);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate the T_p for sine and cosine
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    all_absolute_sum = sum(abs(tao_MixturesA) + abs(tao_MixturesB), "all");
    % Calculate the T_p(indexer) and hat_T_p(indexer) per hidden var.
    % Initialize output variables for T_p and hat_T_p
    T_p = zeros(1, Mixture_Count_N^rCount);
    hat_T_p = zeros(1, Mixture_Count_N^rCount);
    for indexer=1:(Mixture_Count_N^rCount)
        % initiate both, TODO: remove it as initiated above 
        T_p(indexer) = 0;
        hat_T_p(indexer) = 0;
        for row=1:inputSize % over the input 
            % denominator
            tmpDenominator = 0;
            for k=1:(Mixture_Count_N^rCount) % over the mixture
                cos_sin_term = 0;
                for col=1:rCount
                    cos_sin_term = cos_sin_term + ...
                        pi / HalfPeriod_L(col) * inputArray(row,col) * ...
                        mixture_indices(k, col);
                end
                tmpDenominator = tmpDenominator + ...
                    tao_MixturesA(k) * cos( cos_sin_term )+ ...
                    tao_MixturesB(k) * sin( cos_sin_term );
            end
            tmpDenominator = all_absolute_sum + tmpDenominator;

            % nominator
            cos_sin_term = 0; 
            for col=1:rCount
                cos_sin_term = cos_sin_term + ...
                    pi / HalfPeriod_L(col) * inputArray(row,col) * ...
                    mixture_indices(indexer, col);
            end
            T_p_tmp = cos( cos_sin_term ) / tmpDenominator;
            hat_T_p_tmp = sin( cos_sin_term ) / tmpDenominator;
            T_p(indexer) = T_p(indexer) + T_p_tmp;
            hat_T_p(indexer) = hat_T_p(indexer) + hat_T_p_tmp ;
        end
    end 
    % calculate overall sum
    T_p_sum = 0;
    hat_T_p_sum = 0;
    for indexer=1:(Mixture_Count_N^rCount)
        T_p_sum = T_p_sum + tao_MixturesA(indexer) * ...
            T_p(indexer) + abs(tao_MixturesA(indexer));
        hat_T_p_sum = hat_T_p_sum + tao_MixturesA(indexer) * ...
            hat_T_p(indexer) + abs(tao_MixturesA(indexer));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate the C_n for cosine units
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C_n = zeros(1, (Mixture_Count_N^rCount));
    for indexer=1:(Mixture_Count_N^rCount)
        C_n(indexer) = all_absolute_sum - abs(tao_MixturesA(indexer)) ;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate the S_n for cosine units
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S_n = zeros(1, (Mixture_Count_N^rCount));
    for indexer=1:(Mixture_Count_N^rCount)
        S_n(indexer) = all_absolute_sum - abs(tao_MixturesB(indexer)) ;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% re-Calculate the tao_MixturesA
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:(Mixture_Count_N^rCount)
        tao_MixturesA(i) = ( T_p(i) * C_n(i) ) /...
            (T_p_sum + hat_T_p_sum - sign(tao_MixturesA(i)) * T_p(i));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% re-Calculate the tao_MixturesB
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:(Mixture_Count_N^rCount)
        tao_MixturesB(i) = ( hat_T_p(i) * S_n(i) ) / ...
            ( T_p_sum + hat_T_p_sum - sign(tao_MixturesB(i)) * hat_T_p(i));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate Mixture_Out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mixture_Out = zeros(1, (MaxPeriod+1)^rCount );
% remember pdf_index = zeros( (MaxPeriod+1)^rCount, rCount );
for row=1:( (MaxPeriod+1)^rCount )
    Mixture_Out(row) = 0;
    for indexer=1:(Mixture_Count_N^rCount)
        % calculate cos / sin term
        cos_sin_term = 0;
        for col=1:rCount
            cos_sin_term = cos_sin_term + ...
                pi / HalfPeriod_L(col) * pdf_index(row,col) * ...
                mixture_indices(indexer,col);
        end
        % calculate the mixture out
        Mixture_Out(row) = Mixture_Out(row) + ...
                tao_MixturesA(indexer) * cos( cos_sin_term ) + ...
                tao_MixturesB(indexer) * sin( cos_sin_term );
    end
    Mixture_Out(row) = Mixture_Out(row) + ( 4 / MaxPeriod );
end
Mixture_Out = smooth(Mixture_Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Remove the Bias & Normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the size of output
outputSize = length(Mixture_Out);
% find out the total DC component below 0
DC_below_zero = 0;
for i=1: (outputSize)
    if Mixture_Out(i) < 0
      DC_below_zero = abs(Mixture_Out(i)) + DC_below_zero;
    end
end
% Average the DC and pull up Mixture output
DC_average_below_0 = DC_below_zero / outputSize;
for i=1: (outputSize)
    Mixture_Out(i) = Mixture_Out(i) + DC_average_below_0;
end
% Trim out all values below zero
for i=1: (outputSize)
    if Mixture_Out(i) < Trimming_Factor
      Mixture_Out(i) = 0;
    end
end
% Normalize the PDF
Total_Area_Mixture_Out = 0;
for i=1: (outputSize)
    Total_Area_Mixture_Out = Mixture_Out(i) + Total_Area_Mixture_Out;
end
% normalize with the total area
for i=1: (outputSize)
    Mixture_Out(i) = Mixture_Out(i) / Total_Area_Mixture_Out;
end
fprintf("DC_average_below_0:  %d\n", DC_average_below_0);
fprintf("DC_below_zero:  %d\n", DC_below_zero);
fprintf("Total_Area_Mixture_Out:  %d\n", Total_Area_Mixture_Out);
