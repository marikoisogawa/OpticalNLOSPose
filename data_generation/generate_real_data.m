% -------------------------------------------------------------------------
% Reconstruct Stanford interactive data provided by the following paper
% 
% This code highly refers the following implementation: 
%    D.B. Lindell, G. Wetzstein, M. O'Toole
%    "Wave-based non-line-of-sight imaging using fast f-k migration", 
%    ACM Trans. Graph. (SIGGRAPH), 2019.
% 
% -------------------------------------------------------------------------

function generate_real_data(param)

    load('results/realtime2_32/tofgrid_32.mat');
    
    N = 32;
    temporal_upsample = 4./30; % Convert from 4 Hz to 30 Hz

    transient_max = -realmax;
    transient_min =  realmax;

    idx_start = 70.0 / 4.0 * 30.0;
    idx = idx_start;
    
    for ii = 70:temporal_upsample:200
        
        fprintf('%d\n', idx);
        if (0)
            meas = AccumSingleSavedTimestamps(32, 'realtime2_32', floor(ii),ii-floor(ii),1) + ...
                   AccumSingleSavedTimestamps(32, 'realtime2_32', floor(ii)+1,0,ii-floor(ii));
        else
            tmp0 = AccumSingleSavedTimestamps(32, 'realtime2_32', floor(ii)-1,0,1);
            timeval0 = ones(N,1)*linspace(0,1,N) - 1;
            tmp1 = AccumSingleSavedTimestamps(32, 'realtime2_32', floor(ii),0,1);
            timeval1 = ones(N,1)*linspace(0,1,N);
            tmp2 = AccumSingleSavedTimestamps(32, 'realtime2_32', floor(ii)+1,0,1);
            timeval2 = ones(N,1)*linspace(0,1,N) + 1;

            timeval = (ii-floor(ii));
            wgh0 = max(1-abs(timeval - timeval0),0);
            wgh1 = max(1-abs(timeval - timeval1),0);
            wgh2 = max(1-abs(timeval - timeval2),0);

            wgh0 = repmat(permute(wgh0, [3 1 2]),[4096 1 1]);
            wgh1 = repmat(permute(wgh1, [3 1 2]),[4096 1 1]);
            wgh2 = repmat(permute(wgh2, [3 1 2]),[4096 1 1]);

            meas = wgh0.*tmp0 + wgh1.*tmp1 + wgh2.*tmp2;
        end

        meas = meas(1:2:end, :, :) + meas(2:2:end, :, :);
        meas = permute(meas, [2, 3, 1]);

        for kk = 1:size(meas, 1)
            for ll = 1:size(meas,2 )
                meas(kk, ll, :) = circshift(meas(kk, ll, :), [0 0 -floor(tofgrid(kk, ll) / (param.bin_resolution*1e12))]);
            end
        end

        meas = meas(:, :, 1:1024);

        if ~exist(param.dstdir_name, 'dir'); mkdir(param.dstdir_name); end
        transient_img = double(meas);
        tmp_max = max(transient_img(:));
        tmp_min = min(transient_img(:));
        if tmp_max > transient_max
            transient_max = tmp_max;
        end
        if tmp_min < transient_min
            transient_min = tmp_min;
        end
        dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.dstdir_name, idx);
        save(dst_transient_image_name_mat, 'transient_img');
        idx = idx + 1;
    end

    %% Convert from .mat to .npy transient image
    %  Use specific range of frames (please modify as you want) to avoid memory allocation error
    for idx1 = idx_start : idx-1
        enable_process = 0;
        if 811<=idx1 && idx1 <=1471
            enable_process = 1;
        end

        margin = 5;
        if idx1 - margin <= 811
            start_ind = 811;
              end_ind = 811+2*margin;
        elseif idx1 + margin >= 1471
            start_ind = 1471-2*margin;
              end_ind = 1471;
        else
            start_ind = idx1-margin;
              end_ind = idx1+margin;
        end

        if enable_process == 1
            dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.dstdir_name, idx1);
            dst_transient_image_name_npy = sprintf('%s/%08d.npy', param.dstdir_name, idx1);
                command = sprintf('[your python path] utils/mat_to_numpy.py --mat_name %s --npy_name %s --matrix_size 32 --enable_scale1 --dstdir %s --start_ind %d --end_ind %d', ...
                dst_transient_image_name_mat, dst_transient_image_name_npy, param.dstdir_name, start_ind, end_ind);
                dst_transient_image_name_mat, dst_transient_image_name_npy, param.dstdir_name, start_ind, end_ind);
            dos(command);       
        end
    end
    for idx1 = idx_start : idx-1
        if 811<=idx1 && idx1 <=1471
            dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.dstdir_name, idx1);
            command = sprintf('rm %s', dst_transient_image_name_mat);
            dos(command);
        end
    end
end



