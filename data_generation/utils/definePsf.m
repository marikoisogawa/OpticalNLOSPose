% -------------------------------------------------------------------------
% Original implementation: 
% M. OÅfToole, D.B. Lindell, G. Wetzstein, 
% ÅgConfocal Non-Line-of-Sight Imaging Based on the Light-Cone TransformÅh, 
% Nature, 2018.
% -------------------------------------------------------------------------

function psf = definePsf(U,V,slope)

    x = linspace(-1,1,2.*U);
    y = linspace(-1,1,2.*U);
    z = linspace(0,2,2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);

    psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
    psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
    psf = psf./sum(psf(:,U,U));
    psf = psf./norm(psf(:));
    psf = circshift(psf,[0 U U]);
    
end