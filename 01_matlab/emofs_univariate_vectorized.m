%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Mixture_Out = emofs_univariate_vectorized(inputArray, ...
    HalfPeriod_L, Mixture_Count_N, Trimming_Factor)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the input size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputSize = length(inputArray);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initially, the mixture components are distributed homogenously
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tao_MixturesA = ones(1, Mixture_Count_N) / Mixture_Count_N;
tao_MixturesB = ones(1, Mixture_Count_N) / Mixture_Count_N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Precompute reusable quantities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega = pi / HalfPeriod_L;
p = (1:Mixture_Count_N)';         % K×1
x = inputArray(:)';               % 1×N

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Iterative Update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iteration = 1:13

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Build cosine & sine matrices (K × inputSize)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C = cos(omega * (p * x));
    S = sin(omega * (p * x));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Denominator (1 × inputSize)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    all_absolute_sum = sum(abs(tao_MixturesA) + abs(tao_MixturesB));

    tmpDenominator = all_absolute_sum ...
        + tao_MixturesA * C ...
        + tao_MixturesB * S;      % 1×inputSize

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute T_p and hat_T_p  (K×1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_p     = sum(C ./ tmpDenominator, 2);
    hat_T_p = sum(S ./ tmpDenominator, 2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute T_p_sum and hat_T_p_sum (scalars)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_p_sum     = sum(tao_MixturesA .* T_p' + abs(tao_MixturesA));
    hat_T_p_sum = sum(tao_MixturesB .* hat_T_p' + abs(tao_MixturesB));

    total_sum = T_p_sum + hat_T_p_sum;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute C_n and S_n (1×K)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C_n = all_absolute_sum - abs(tao_MixturesA);
    S_n = all_absolute_sum - abs(tao_MixturesB);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update tao_MixturesB (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tao_MixturesB = ...
        (hat_T_p' .* S_n) ./ ...
        (total_sum - sign(tao_MixturesB) .* hat_T_p');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update tao_MixturesA (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tao_MixturesA = ...
        (T_p' .* C_n) ./ ...
        (total_sum - sign(tao_MixturesA) .* T_p');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Recalculate Mixture_Out (vectorized)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    i_vals = (-HalfPeriod_L):(HalfPeriod_L-1);  % 1×(2L)
    basisC = cos(omega * (p * i_vals));
    basisS = sin(omega * (p * i_vals));

    Mixture_Out = ...
        tao_MixturesA * basisC + ...
        tao_MixturesB * basisS + ...
        (1 / HalfPeriod_L);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Remove Bias & Normalize (Fully Vectorized)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputSize = length(Mixture_Out);
% DC below zero and its avarage
negative_mask = Mixture_Out < 0;
DC_below_zero = sum(abs(Mixture_Out(negative_mask)));
DC_average_below_0 = DC_below_zero / outputSize;

% Shift upward by means of the average
Mixture_Out = Mixture_Out + DC_average_below_0;

% Trim the negative values of the mixture output
Mixture_Out(Mixture_Out < Trimming_Factor) = 0;

% Normalize based on the calculated area to meet up pdf requirement
Total_Area_Mixture_Out = sum(Mixture_Out);
Mixture_Out = Mixture_Out / Total_Area_Mixture_Out;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Log out 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("DC_average_below_0:\n");
disp(DC_average_below_0)
fprintf("DC_below_zero:\n");
disp(DC_below_zero)
fprintf("Total_Area_Mixture_Out:\n");
disp(Total_Area_Mixture_Out)

end