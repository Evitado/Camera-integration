function prep_cam(camX)
% Prep one camera (cam0..cam3) for MATLAB Lidar Camera Calibrator.

% --- root folder where everything lives ---
root = "C:\Users\mehta\OneDrive\Desktop\Sensor-fusion";

% --- build all the key paths portably ---
imgDir  = fullfile(root, sprintf("calib_imgs_cam%d", camX));
yamlF   = fullfile(root, "calib", sprintf("calib_imgs_cam%d_intrinsics.yaml", camX));
pcapF   = fullfile(root, "pcap", sprintf("cam%d.pcap", camX));
metaF   = fullfile(root, "pcap", sprintf("cam%d.json", camX));

outDir  = fullfile(root, "out", sprintf("cam%d", camX));
pcDir   = fullfile(outDir, "pointclouds");
intrDir = fullfile(outDir, "intrinsics");
expDir  = fullfile(outDir, "export");

% --- ensure output folders exist ---
if ~isfolder(pcDir),   mkdir(pcDir);   end
if ~isfolder(intrDir), mkdir(intrDir); end
if ~isfolder(expDir),  mkdir(expDir);  end

% --- sanity checks ---
assert(isfolder(imgDir), "Missing images folder: %s", imgDir);
assert(isfile(yamlF),    "Missing YAML file:      %s", yamlF);
assert(isfile(pcapF),    "Missing PCAP file:      %s", pcapF);
assert(isfile(metaF),    "Missing JSON file:      %s", metaF);

% ---------- images ----------
imgs = [dir(fullfile(imgDir,"*.png")); dir(fullfile(imgDir,"*.jpg")); dir(fullfile(imgDir,"*.jpeg"))];
assert(~isempty(imgs), "No images found in %s", imgDir);
[~,ord] = sort({imgs.name}); imgs = imgs(ord);
probe = imread(fullfile(imgDir, imgs(1).name));
[H,W,~] = size(probe);
fprintf("Image size: %d x %d\n", H, W);

% ---------- YAML -> intrinsics ----------
intr = parse_yaml_opencv(yamlF);
fx = intr.fx; fy = intr.fy; cx = intr.cx; cy = intr.cy;
k1 = intr.k1; k2 = intr.k2; p1 = intr.p1; p2 = intr.p2; k3 = intr.k3;

intrinsics = cameraIntrinsics([fx fy], [cx cy], [H W], ...
    RadialDistortion=[k1 k2 k3], TangentialDistortion=[p1 p2]);

save(intrDir + "\cam_intrinsics.mat","intrinsics");
fprintf("Saved intrinsics → %s\\cam_intrinsics.mat\n", intrDir);

% ---------- PCAP -> PCD (count-matched to images) ----------
r = ousterFileReader(pcapF, metaF);
N = r.NumberOfFrames;
numImgs = numel(imgs);
idx = round(linspace(1, N, numImgs));
idx = unique(max(1, min(N, idx)));

fprintf("PCAP frames: %d, exporting %d PCDs to match images...\n", N, numel(idx));
for i = 1:numel(idx)
    k = idx(i);
    pt = readFrame(r, k);            % pointCloud object
    % name by sequence index i (matched to image i)
    fn = fullfile(pcDir, sprintf("pc_%04d.pcd", i));
    pcwrite(pt, fn, "Encoding","binary");
end
fprintf("Saved %d PCDs → %s\n", numel(idx), pcDir);

fprintf("\n=== CAM%d ready ===\n", camX);
fprintf("Images:      %s\n", imgDir);
fprintf("PointClouds: %s\n", pcDir);
fprintf("Intrinsics:  %s\\cam_intrinsics.mat\n", intrDir);
fprintf("Next: run lidarCameraCalibrator and load these paths.\n");
end

% -------- helper: parse this YAML format --------
function intr = parse_yaml_opencv(yamlPath)
txt = fileread(yamlPath);

K = get_opencv_matrix(txt, "camera_matrix");
if numel(K) ~= 9, error("camera_matrix must have 9 elements"); end
intr.fx = K(1); intr.cx = K(3);
intr.fy = K(5); intr.cy = K(6);

D = get_opencv_matrix(txt, "dist_coeff");  % OpenCV order: [k1 k2 p1 p2 k3]
if numel(D) < 4
    error("dist_coeff must have at least 4 elements [k1 k2 p1 p2 (k3)]");
end
intr.k1 = D(1); intr.k2 = D(2);
intr.p1 = D(3); intr.p2 = D(4);
intr.k3 = 0;
if numel(D) >= 5, intr.k3 = D(5); end
end

function arr = get_opencv_matrix(txt, key)
% Extract OpenCV '!!opencv-matrix' data array
pat = sprintf('%s[\\s\\S]*?data\\s*:\\s*\\[([^\\]]+)\\]', regexptranslate('escape', key));
m = regexp(txt, pat, 'tokens','once');
if isempty(m)
    error("Key not found: %s", key);
end
nums = regexp(m{1}, '[-+0-9.eE]+', 'match');
arr = str2double(nums);
end
