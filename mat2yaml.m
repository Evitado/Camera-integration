function mat2yaml(matFile, yamlFile, yamlKey)
% mat2yaml  Load a .mat, extract first 4×4 matrix, write as YAML.
%
%   mat2yaml(matFile, yamlFile, yamlKey)
%
%   - matFile : path to .mat (e.g. 'cam0_T.mat' or 'cam0_tform.mat')
%   - yamlFile: output YAML path (e.g. 'cam0_T.yaml')
%   - yamlKey : the top‑level key, e.g. 'T_lidar_to_cam0'

    % 1) Load
    S = load(matFile);
    vars = fieldnames(S);
    if isempty(vars)
        error("No variables found in %s", matFile);
    end

    % 2) Get matrix or tform
    M = S.(vars{1});
    if isa(M, 'rigidtform3d')
        % extract 4×4
        R = M.Rotation;
        t = M.Translation(:);
        T = eye(4);
        T(1:3,1:3) = R;
        T(1:3,4  ) = t;
    elseif isnumeric(M) && all(size(M) == [4 4])
        T = M;
    else
        error("Variable '%s' is not a 4×4 numeric matrix or rigidtform3d", vars{1});
    end

    % 3) Write YAML
    fid = fopen(yamlFile, 'w');
    if fid < 0, error("Cannot open %s for writing", yamlFile); end

    fprintf(fid, "%s:\n", yamlKey);
    for i = 1:4
        fprintf(fid, "  - [%.9f, %.9f, %.9f, %.9f]\n", T(i,1),T(i,2),T(i,3),T(i,4));
    end
    fclose(fid);

    fprintf("Wrote %s from %s (var: %s)\n", yamlFile, matFile, vars{1});
end
