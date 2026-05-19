%%
load('histology\probe_ccf.mat');

% remove session with only 2 neurons
probe_ccf(20) = [];

addpath('histology\npy-matlab');
addpath('histology\AP_histology-master');
addpath('histology\AP_histology-master\allenCCF_repo_functions');
addpath('histology\BrowsingFunctions');
allen_atlas_path = 'histology\AllenCCF';
addpath(allen_atlas_path);

%% 3D
gui_data.tv = readNPY(fullfile(allen_atlas_path, 'template_volume_10um.npy'));
gui_data.av = readNPY(fullfile(allen_atlas_path, 'annotation_volume_10um_by_index.npy'));
gui_data.st = ap_histology.loadStructureTree(fullfile(allen_atlas_path, 'structure_tree_safe_2017.csv'));
gui_data.probe_color = lines(size(probe_ccf, 1));
% plot a 3D brain
figure(1); clf;
axes_atlas = axes;
[~, brain_outline] = plotBrainGrid([], axes_atlas);
set(axes_atlas, 'ZDir', 'reverse');
hold(axes_atlas, 'on');
axis vis3d equal off manual;
view([-30, 25]);
caxis([0, 300]);
[ap_max, dv_max, ml_max] = size(gui_data.tv);
xlim([-10, ap_max + 10])
ylim([-10, ml_max + 10])
zlim([-10, dv_max + 10])
h = rotate3d(gca);
h.Enable = 'on';
% plot the probes
for curr_probe = 1:length(probe_ccf)
    % Plot points and line of best fit
    points = probe_ccf(curr_probe).points;
    r0 = mean(points, 1);
    xyz = bsxfun(@minus, probe_ccf(curr_probe).points, r0);
    [~, ~, V] = svd(xyz, 0);
    histology_probe_direction = V(:, 1);
    % (make sure the direction goes down in DV - flip if it's going up)
    if histology_probe_direction(2) < 0
        histology_probe_direction = -histology_probe_direction;
    end
    line_eval = [-100, 100];
    probe_fit_line = bsxfun(@plus, bsxfun(@times, line_eval', histology_probe_direction'), r0);
%     plot3(probe_ccf(curr_probe).points(:, 1), ...
%         probe_ccf(curr_probe).points(:, 3), ...
%         probe_ccf(curr_probe).points(:, 2), ...
%         '.', 'color', 'k', 'MarkerSize', 20);
%     line(probe_fit_line(:, 1), probe_fit_line(:, 3), probe_fit_line(:, 2), ...
%         'color', 'k', 'linewidth', 1);
    plot3(points(:, 1), points(:, 3), points(:, 2), 'color' ,'k', 'linewidth', 1);
end

set(gcf, 'renderer', 'painters');
% print('histology3d', '-depsc');

%% 2D
aps = NaN(1, length(probe_ccf));
for i_probe = 1:length(probe_ccf)
    r0 = mean(probe_ccf(i_probe).points, 1);
    aps(i_probe) = r0(1);
end
ap_min = floor(min(aps));
ap_mean = round(mean(aps));
ap_max = ceil(max(aps));

figure(2); clf;

subplot(1, 3, 1); hold all;
slice = squeeze(gui_data.av(ap_min, :, :));
imagesc(flipud(slice));
for i_probe = 1:length(probe_ccf)
    points = probe_ccf(i_probe).points;
    % only plot probes closest to this slice
    [~, ind] = min(abs(mean(points(:, 1)) - [ap_min, ap_mean, ap_max]));
    if ind ~= 1, continue; end
    plot(size(slice, 2) - points(:, 3), size(slice, 1) - points(:, 2), 'k');
end
title(sprintf('Minimum AP: %i', ap_min));
xlim([0, size(slice, 2)]);
ylim([0, size(slice, 1)]);

subplot(1, 3, 2); hold all;
slice = squeeze(gui_data.av(ap_mean, :, :));
imagesc(flipud(slice));
for i_probe = 1:length(probe_ccf)
    points = probe_ccf(i_probe).points;
    % only plot probes closest to this slice
    [~, ind] = min(abs(mean(points(:, 1)) - [ap_min, ap_mean, ap_max]));
    if ind ~= 2, continue; end
    plot(size(slice, 2) - points(:, 3), size(slice, 1) - points(:, 2), 'k');
end
title(sprintf('Mean AP: %i', ap_mean));
xlim([0, size(slice, 2)]);
ylim([0, size(slice, 1)]);

subplot(1, 3, 3); hold all;
slice = squeeze(gui_data.av(ap_max, :, :));
imagesc(flipud(slice));
for i_probe = 1:length(probe_ccf)
    points = probe_ccf(i_probe).points;
    % only plot probes closest to this slice
    [~, ind] = min(abs(mean(points(:, 1)) - [ap_min, ap_mean, ap_max]));
    if ind ~= 3, continue; end
    plot(size(slice, 2) - points(:, 3), size(slice, 1) - points(:, 2), 'k');
end
title(sprintf('Maximum AP: %i', ap_max));
xlim([0, size(slice, 2)]);
ylim([0, size(slice, 1)]);

% set(gcf, 'renderer', 'painters');
% print('histology', '-depsc');

%% atlas browser
tv = readNPY('template_volume_10um.npy'); % grey-scale "background signal intensity"
av = readNPY('annotation_volume_10um_by_index.npy'); % the number at each pixel labels the area, see note below
st = loadStructureTree('structure_tree_safe_2017.csv'); % a table of what all the labels mean

file_save_location = 'histology\test'; % where the probe locations will be saved
probe_name = 'test'; % name probe to avoid overwriting

fig = figure(3);
f = allenAtlasBrowser(fig, tv, av, st, file_save_location, probe_name, 'coronal');

% min ap: 398 --> 142
% mean ap: 437 --> 103
% max ap: 472 --> 68
