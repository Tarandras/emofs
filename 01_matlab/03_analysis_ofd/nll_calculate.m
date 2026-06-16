function nll = nll_calculate(inputArray, pdf_index, Z_pdf)

    [inputSize, dim] = size(inputArray);
    offsets = zeros(1, dim);
    for d = 1:dim
        offsets(d) = round(1 - min(pdf_index(:,d)));
    end

    nll = 0;
    invalid_case_counter = 0;
    gridSize = size(Z_pdf);

    for i = 1:inputSize
        % Shift to grid coordinates
        query = inputArray(i,:) + offsets;
        % Convert to nearest grid index (DIRECT lookup attempt)
        idx = round(query);
        idxCell = num2cell(idx);
        % --------------------------------------------------
        % 1. DIRECT probability lookup
        % --------------------------------------------------
        inBounds = true;
        for d = 1:dim
            if idx(d) < 1 || idx(d) > gridSize(d)
                inBounds = false;
                break;
            end
        end
        if inBounds == false
            continue;
        end
        prob = Z_pdf(idxCell{:});
        % --------------------------------------------------
        % handle invalid probabilities
        % --------------------------------------------------
        if isempty(prob) || isnan(prob) || prob <= 0
            invalid_case_counter = invalid_case_counter + 1;
            %queryCell = num2cell(query);
            %prob = interpn(Z_pdf, queryCell{:}, 'linear');
            prob = 1e-5;
        end
        nll = nll + log(prob);
    end
    nll = -nll / inputSize;
    fprintf("Found %d invalid condition\n", invalid_case_counter);
end