%% ============== batch_mfdfa_subject_overview_matOnly.m (MEAN+SHADE v6) ==============
% Subject overview of Dq–hq curves with EZ (red) vs Non-EZ (blue).
% Reads EZ only from each .mat (no CSV / fuzzy).
% Exports:

clear; clc;

%% =================== USER SETTINGS ===================
input_folder        = fullfile(pwd, 'matfiles');           % folder of .mat
base_output_dir     = fullfile(pwd, 'MFDFA_Dq_plots_mat_fonts20');       % output root

Fs                  = 1000;        % Hz
segment_duration    = 300;         % seconds
scale               = 2.^(4:12);   % MFDFA scales
q                   = -5:0.1:5;    % MFDFA q-range
m                   = 1;           % detrending order
Fig                 = 0;           % turn off MFDFA1 plotting

dpi                 = 300;         % export resolution
fig_size_inches     = [1, 1, 7.5, 6.0];
font_size           = 44;

% --- Plot controls ---
SMOOTH_WINDOW       = 0;           % 0/[] off; else odd integer >=3 (e.g., 5 or 7)
SHADE_ALPHA         = 0.22;        % opacity of the bands (0–1)

% Colors (Non-EZ=blue, EZ=red)
C_NON_SHADE = [0.12 0.42 0.95];
C_EZ_SHADE  = [0.88 0.12 0.12];
C_NON_LINE  = [0.00 0.30 0.90];    % slightly darker for the mean line
C_EZ_LINE   = [0.85 0.00 0.00];

MEAN_LW            = 2.3;          % line width for mean curve

% Parallel
OPEN_PARPOOL_IF_NEEDED = true;
NUM_WORKERS_LIMIT       = [];

% Console verbosity
VERBOSE_WORKER_WARNINGS = false;
PRINT_PROGRESS          = true;

%% =================== PREP ===================
if ~exist(base_output_dir, 'dir'), mkdir(base_output_dir); end
if OPEN_PARPOOL_IF_NEEDED
    p = gcp('nocreate');
    if isempty(p)
        if isempty(NUM_WORKERS_LIMIT)
            parpool;
        else
            parpool('local', NUM_WORKERS_LIMIT);
        end
    end
end

matFiles = dir(fullfile(input_folder, '*.mat'));
if isempty(matFiles), error('No .mat files found in: %s', input_folder); end

% Broadcast constants to workers
q_const     = parallel.pool.Constant(q);
scale_const = parallel.pool.Constant(scale);
Lq          = numel(q) - 1;  % expected length from your MFDFA1

if PRINT_PROGRESS
    fprintf('Processing %d subject file(s)...\n\n', numel(matFiles));
end

% NEW: keep track of produced per-subject CSVs for final merge
produced_subject_csvs = strings(0,1);

%% =================== SUBJECT LOOP ===================
for f = 1:numel(matFiles)
    t0 = tic;
    subj_file = fullfile(matFiles(f).folder, matFiles(f).name);
    [~, subj_name] = fileparts(subj_file);

    subj_out = fullfile(base_output_dir, subj_name);
    if ~exist(subj_out, 'dir'), mkdir(subj_out); end

    % ----- Load data + channel names -----
    S = load(subj_file);
    X = extractDataMatrix(S);
    if isempty(X)
        warning('Skipping %s (no suitable data matrix).', subj_name);
        continue;
    end
    if size(X,1) > size(X,2), X = X.'; end   % enforce [channels x samples]
    [C, T] = size(X);

    % IMPORTANT FIX: ALWAYS returns exactly C names
    raw_names  = getRawChannelNamesForMatch(S, C);
    raw_names  = raw_names(:).'; % ensure 1×C row

    % ----- EZ from .mat only (robust detector) -----
    ez_mask = getEZFromMatStructRobust(S, raw_names, C);
    fprintf('Subject %-28s : EZ=%d / %d (from .mat)\n', subj_name, nnz(ez_mask), C);

    % ----- Segmentation plan -----
    seg_len      = Fs * segment_duration;
    num_segments = floor(T / seg_len);
    if num_segments < 1
        warning('Skipping %s (not enough samples for one %gs segment).', subj_name, segment_duration);
        continue;
    end
    seg_starts = (0:(num_segments-1))*seg_len + 1;
    seg_ends   = seg_starts + seg_len - 1;
    if seg_ends(end) > T, seg_ends(end) = T; end

    fprintf('  Channels=%d, Segments=%d (%.1f s each)\n', C, num_segments, segment_duration);

    % ----- Prealloc outputs -----
    H_mean = nan(Lq, C);
    D_mean = nan(Lq, C);
    L_used = zeros(C,1,'uint16');

    % ---------- PARFOR (robust to variable-length hq/Dq) ----------
    parfor ch = 1:C
        try
            x = X(ch, :);
            sum_hq   = zeros(Lq,1);
            sum_Dq   = zeros(Lq,1);
            used_len = uint16(0);
            count    = uint16(0);
            printed_once = false;

            for sidx = 1:num_segments
                seg_data = x(seg_starts(sidx):seg_ends(sidx));
                try
                    [~, ~, hq, Dq_] = MFDFA1(seg_data, scale_const.Value, q_const.Value, m, Fig);
                    hq  = hq(:);
                    Dq_ = Dq_(:);
                    if isempty(hq) || isempty(Dq_) || ~all(isfinite(hq)) || ~all(isfinite(Dq_))
                        continue;
                    end
                catch ME
                    if ~printed_once && VERBOSE_WORKER_WARNINGS
                        fprintf('[%s] ch %d seg %d: %s\n', subj_name, ch, sidx, ME.message);
                        printed_once = true;
                    end
                    continue;
                end

                Lcur = min([Lq, numel(hq), numel(Dq_)]);
                if Lcur == 0, continue; end

                sum_hq(1:Lcur) = sum_hq(1:Lcur) + hq(1:Lcur);
                sum_Dq(1:Lcur) = sum_Dq(1:Lcur) + Dq_(1:Lcur);
                used_len       = max(used_len, uint16(Lcur));
                count          = count + 1;
            end

            h_out = nan(Lq,1);
            d_out = nan(Lq,1);

            if count > 0 && used_len > 0
                ul = double(used_len);
                h_out(1:ul) = sum_hq(1:ul) ./ double(count);
                d_out(1:ul) = sum_Dq(1:ul) ./ double(count);
            end

            H_mean(:, ch) = h_out;
            D_mean(:, ch) = d_out;
            L_used(ch)    = used_len;

        catch ME
            error('PARFOR_FAIL: Subject=%s | ch=%d | msg=%s', subj_name, ch, ME.message);
        end
    end

    % Crop to longest valid prefix
    L_final = double(max(L_used));
    if L_final == 0
        warning('No valid MFDFA results for subject %s.', subj_name);
        continue;
    end
    H_mean = H_mean(1:L_final, :);
    D_mean = D_mean(1:L_final, :);

    % =================== OVERLAY FIGURE (SHADE + MEAN) ===================
    valid_cols = all(isfinite(H_mean),1) & all(isfinite(D_mean),1);
    non_idx    = find(valid_cols & (~ez_mask(:)).');
    ez_idx     = find(valid_cols & ( ez_mask(:)).');

    fig = figure('Visible','off','Units','inches','Position',fig_size_inches);
    ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); set(ax,'FontSize',font_size);

    handles = []; labels = {};

    % --- Non-EZ shaded band + mean line ---
    if ~isempty(non_idx)
        mean_non_h = mean(H_mean(:, non_idx), 2, 'omitnan');
        mean_non_d = mean(D_mean(:, non_idx), 2, 'omitnan');
        std_non_d  = std (D_mean(:, non_idx), 0, 2, 'omitnan');

        mean_non_h = smoothopt(mean_non_h, SMOOTH_WINDOW);
        mean_non_d = smoothopt(mean_non_d, SMOOTH_WINDOW);
        std_non_d  = smoothopt(std_non_d , SMOOTH_WINDOW);

        mask = isfinite(mean_non_h) & isfinite(mean_non_d) & isfinite(std_non_d);
        x = mean_non_h(mask);
        up = mean_non_d(mask) + std_non_d(mask);
        lo = mean_non_d(mask) - std_non_d(mask);

        [x, ord] = sort(x); up = up(ord); lo = lo(ord);

        hPatchNon = fill(ax, [x; flipud(x)], [up; flipud(lo)], C_NON_SHADE, ...
                         'FaceAlpha', SHADE_ALPHA, 'EdgeColor','none', ...
                         'DisplayName','Non-EZ mean±SD');
        handles(end+1) = hPatchNon; labels{end+1} = 'Non-EZ mean±SD';

        y = mean_non_d(mask); y = y(ord);
        hMeanNon = plot(ax, x, y, '-', 'Color', C_NON_LINE, 'LineWidth', MEAN_LW, ...
                        'DisplayName','Non-EZ mean');
        handles(end+1) = hMeanNon; labels{end+1} = 'Non-EZ mean';
    end

    % --- EZ shaded band + mean line ---
    if ~isempty(ez_idx)
        mean_ez_h = mean(H_mean(:, ez_idx), 2, 'omitnan');
        mean_ez_d = mean(D_mean(:, ez_idx), 2, 'omitnan');
        std_ez_d  = std (D_mean(:, ez_idx), 0, 2, 'omitnan');

        mean_ez_h = smoothopt(mean_ez_h, SMOOTH_WINDOW);
        mean_ez_d = smoothopt(mean_ez_d, SMOOTH_WINDOW);
        std_ez_d  = smoothopt(std_ez_d , SMOOTH_WINDOW);

        mask = isfinite(mean_ez_h) & isfinite(mean_ez_d) & isfinite(std_ez_d);
        x = mean_ez_h(mask);
        up = mean_ez_d(mask) + std_ez_d(mask);
        lo = mean_ez_d(mask) - std_ez_d(mask);

        [x, ord] = sort(x); up = up(ord); lo = lo(ord);

        hPatchEZ = fill(ax, [x; flipud(x)], [up; flipud(lo)], C_EZ_SHADE, ...
                        'FaceAlpha', SHADE_ALPHA, 'EdgeColor','none', ...
                        'DisplayName','EZ mean±SD');
        handles(end+1) = hPatchEZ; labels{end+1} = 'EZ mean±SD';

        y = mean_ez_d(mask); y = y(ord);
        hMeanEZ = plot(ax, x, y, '-', 'Color', C_EZ_LINE, 'LineWidth', MEAN_LW, ...
                       'DisplayName','EZ mean');
        handles(end+1) = hMeanEZ; labels{end+1} = 'EZ mean';
    end

    % Tight limits
    allH = [H_mean(:,non_idx), H_mean(:,ez_idx)];
    allD = [D_mean(:,non_idx), D_mean(:,ez_idx)];
    if ~isempty(allH)
        x_valid = allH(isfinite(allH));
        y_valid = allD(isfinite(allD));
        if ~isempty(x_valid) && ~isempty(y_valid)
            rx = range(x_valid); ry = range(y_valid);
            padX = 0.02 * (rx + (rx==0));
            padY = 0.02 * (ry + (ry==0));
            xlim(ax, [min(x_valid)-padX, max(x_valid)+padX]);
            ylim(ax, [min(y_valid)-padY, max(y_valid)+padY]);
        end
    end

    xlabel(ax, '\beta');
    ylabel(ax, 'f(\beta)', 'Interpreter', 'tex');
    title(ax, sprintf('%s', subj_name), 'FontSize', font_size+1, 'FontWeight','bold');

    % =================== PATCH: LEGEND REMOVED ===================
    % Legend intentionally disabled for clean overview plots.
    % (This removes the bottom legend block you circled.)
    %
    % if ~isempty(handles)
    %     legend(ax, handles, labels, 'Location','southoutside','NumColumns',2,'Box','off');
    % end
    % ============================================================

    out_png = fullfile(subj_out, sprintf('%s__overview_EZ_vs_NonEZ.png', subj_name));
    try
        exportgraphics(fig, out_png, 'Resolution', dpi);
    catch
        set(fig, 'PaperUnits','inches','PaperPosition',fig_size_inches);
        print(fig, out_png, '-dpng', sprintf('-r%d',dpi));
    end
    close(fig);

    % ----- Tiny EZ summary next to figure -----
    Tsum = table((1:C).', string(raw_names(:)), logical(ez_mask(:)), ...
                 'VariableNames', {'channel_index','channel_label','is_ez'});
    writetable(Tsum, fullfile(subj_out, sprintf('%s__ez_summary.csv', subj_name)));

    % =================== NEW: SAVE PER-SUBJECT FULL CURVES (FIXED) ===================
    q_used = q(1:L_final).';                 % (L_final×1)
    [QQ, CH] = ndgrid(q_used, 1:C);          % (L_final×C)

    nrows = numel(QQ);

    subj_col       = repmat(string(subj_name), nrows, 1);   % (nrows×1)
    chan_idx_col   = CH(:);                                % (nrows×1)
    chan_label_col = string(raw_names(chan_idx_col));      % (nrows×1)
    chan_label_col = chan_label_col(:);                    % force column

    q_col  = QQ(:);        q_col  = q_col(:);
    hq_col = H_mean(:);    hq_col = hq_col(:);
    dq_col = D_mean(:);    dq_col = dq_col(:);

    % Safety assert (optional, but helpful)
    if ~( numel(subj_col)==nrows && numel(chan_idx_col)==nrows && numel(chan_label_col)==nrows && ...
          numel(q_col)==nrows && numel(hq_col)==nrows && numel(dq_col)==nrows )
        error('Table columns length mismatch for subject %s.', subj_name);
    end

    Tsubject = table( ...
        subj_col, chan_idx_col, chan_label_col, q_col, hq_col, dq_col, ...
        'VariableNames', {'subject','channel_index','channel_label','q','hq_mean','Dq_mean'} ...
    );

    out_ch_csv = fullfile(subj_out, sprintf('%s__channel_DqHq.csv', subj_name));
    writetable(Tsubject, out_ch_csv);

    produced_subject_csvs(end+1,1) = string(out_ch_csv);

    if PRINT_PROGRESS
        fprintf('  ➜ Overview saved: %s (%.1fs)\n', out_png, toc(t0));
        fprintf('  ➜ Curves CSV     : %s\n', out_ch_csv);
    end
end

% =================== NEW: MERGE ALL PER-SUBJECT CSVs ===================
if ~isempty(produced_subject_csvs)
    combined_dir = fullfile(base_output_dir, '_combined');
    if ~exist(combined_dir, 'dir'), mkdir(combined_dir); end

    Tall = table();
    for k = 1:numel(produced_subject_csvs)
        try
            Tk = readtable(produced_subject_csvs(k));
            req = {'subject','channel_index','channel_label','q','hq_mean','Dq_mean'};
            if all(ismember(req, Tk.Properties.VariableNames))
                Tall = [Tall; Tk]; %#ok<AGROW>
            else
                warning('Skipping malformed CSV: %s', produced_subject_csvs(k));
            end
        catch ME
            warning('Failed reading %s: %s', produced_subject_csvs(k), ME.message);
        end
    end

    if ~isempty(Tall)
        out_all = fullfile(combined_dir, 'ALL__channel_DqHq_combined.csv');
        writetable(Tall, out_all);
        fprintf('\n✅ Combined CSV saved: %s  (rows=%d)\n', out_all, height(Tall));
    else
        fprintf('\nℹ️ No per-subject CSVs to combine.\n');
    end
else
    fprintf('\nℹ️ No per-subject CSVs were produced.\n');
end

fprintf('\n✅ Done. Subject overview plots saved under: %s\n', base_output_dir);

%% ======================= HELPERS =======================
function y = smoothopt(v, w)
    if nargin < 2 || isempty(w) || ~isscalar(w) || ~isfinite(w) || w < 3 || mod(w,2) == 0
        y = v;
    else
        y = movmean(v, w, 'omitnan');
    end
end

function tf_mask = getEZFromMatStructRobust(S, raw_names, C)
    tf_mask = false(C,1);
    if isempty(S), return; end
    rn = lower(regexprep(string(raw_names(:)), '[\s\-\.:_/\\]+', ''));
    pairs = {};
    fn = fieldnames(S);
    for i = 1:numel(fn)
        f = fn{i}; fL = lower(f); v = S.(f);
        if isstruct(v)
            fn2 = fieldnames(v);
            for j = 1:numel(fn2)
                f2  = fn2{j}; f2L = lower(f2); v2 = v.(f2);
                if contains(f2L,'ez') || contains(f2L,'soz')
                    pairs(end+1,:) = {sprintf('%s.%s',f,f2), v2}; %#ok<AGROW>
                end
            end
        end
        if contains(fL,'ez') || contains(fL,'soz')
            pairs(end+1,:) = {f, v}; %#ok<AGROW>
        end
    end
    if isempty(pairs), return; end

    for k = 1:size(pairs,1)
        val = pairs{k,2};
        if islogical(val) && isvector(val) && numel(val)==C
            tf_mask = tf_mask | val(:); if any(tf_mask), return; end; continue;
        end
        if isnumeric(val) && isvector(val)
            idx = val(:); idx = idx(~isnan(idx));
            if isempty(idx), continue; end
            if max(idx) <= C-1 && any(idx==0), idx = idx + 1; end
            idx = idx(idx>=1 & idx<=C);
            if ~isempty(idx), tf_mask(idx) = true; return; end
            continue;
        end
        if ischar(val) || isstring(val) || iscellstr(val) || ...
           (iscell(val) && all(cellfun(@(x)ischar(x)||isstring(x), val)))
            s = string(val(:));
            s = lower(regexprep(s, '[\s\-\.:_/\\]+', ''));
            for t = 1:numel(s), tf_mask = tf_mask | strcmp(rn, s(t)); end
            if any(tf_mask), return; end
        end
    end
end

function raw_names = getRawChannelNamesForMatch(S, C)

    raw = {};
    if isfield(S,'ch_names') && ~isempty(S.ch_names)
        raw = S.ch_names;
        if isstring(raw), raw = cellstr(raw); end
        if ischar(raw),   raw = {raw};       end
        raw = raw(:)';
    end

    nraw = numel(raw);
    if nraw < C
        pad = arrayfun(@(k) sprintf('Channel_%03d', k), (nraw+1):C, 'UniformOutput', false);
        raw = [raw, pad];
    elseif nraw > C
        raw = raw(1:C);
    end

    raw_names = raw;
end

function X = extractDataMatrix(S)
    if isfield(S,'data') && isnumeric(S.data) && ismatrix(S.data)
        X = S.data;
        return;
    end
    X = [];
    fn = fieldnames(S);
    for k = 1:numel(fn)
        val = S.(fn{k});
        if isnumeric(val) && ismatrix(val)
            if size(val,1) <= size(val,2)
                X = val;
            else
                X = val.';
            end
            return;
        end
    end
end
