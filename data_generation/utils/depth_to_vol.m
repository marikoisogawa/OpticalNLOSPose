%% Convert depth image to 3D depth volume

function depth_vol = depth_to_vol(param, depth_img)
    depth_vol = zeros(param.M, param.N, param.N);
    [row, col] = find(depth_img ~= 1);
    for i = 1 : size(row, 1)
        depth_vol(max(1, round(depth_img(row(i),col(i)))), row(i), col(i)) = 100;
    end
end
