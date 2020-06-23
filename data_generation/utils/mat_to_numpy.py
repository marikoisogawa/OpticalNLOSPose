import numpy as np
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mat_name')
parser.add_argument('--npy_name')
parser.add_argument('--matrix_size', type=int, default=-1)
parser.add_argument('--enable_scale', action='store_true', default=False)
parser.add_argument('--enable_scale1', action='store_true', default=False)
parser.add_argument('--dstdir')
parser.add_argument('--max_val', type=float, default=None)
parser.add_argument('--min_val', type=float, default=None)
parser.add_argument('--start_ind', type=int, default=None)
parser.add_argument('--end_ind', type=int, default=None)
args = parser.parse_args()

print('------------------------')
print('loading... %s' % args.mat_name)
mat_contents = sio.loadmat(args.mat_name)
vol = mat_contents['transient_img']
vol = vol.astype(np.float)
print('numpy shape:')
print(vol.shape)

if args.enable_scale:
    print('scaling...')
    if args.max_val:    # normalization with max/min of whole sequence -----
        print('not in frame normalization')
        max_val = args.max_val
        min_val = args.min_val
    else:               # in frame normalization ---------------------------
        print('in frame normalization')
        max_val = vol[np.unravel_index(np.argmax(vol), vol.shape)]
        min_val = 0
    for xx in range(vol.shape[0]):
        for yy in range(vol.shape[1]):
            for zz in range(vol.shape[2]):
                if vol[xx, yy, zz] > 0:
                    vol[xx, yy, zz] = (vol[xx, yy, zz] - min_val) / (max_val - min_val)
                else:
                    vol[xx, yy, zz] = 0

    if args.matrix_size != -1:  # if it should be resized
        # notice!! never use zoom function for resizing!
        vol = np.resize(vol, (args.matrix_size, args.matrix_size, args.matrix_size))
    print('scaling done. max val:%f' % vol[np.unravel_index(np.argmax(vol), vol.shape)])


if args.enable_scale1:
    print('scaling...')
    max_val = -10000000000000.0
    min_val =  10000000000000.0
    for i in range(args.start_ind, args.end_ind):
        _mat_name = '%s/%08d.mat' % (args.dstdir, i)
        _mat_contents = sio.loadmat(_mat_name)
        _vol = _mat_contents['transient_img']
        _vol = _vol.astype(np.float)
        tmp_max = _vol[np.unravel_index(np.argmax(_vol), _vol.shape)]
        tmp_min = _vol[np.unravel_index(np.argmin(_vol), _vol.shape)]
        max_val = max(max_val, tmp_max)
        min_val = min(min_val, tmp_min)

    print('--------------max: %f' % max_val)
    print('--------------min: %f' % min_val)
    vol = (vol-min_val) / (max_val-min_val)

    print('before scaling. max val:%f' % vol[np.unravel_index(np.argmax(vol), vol.shape)])
    if args.matrix_size != -1:  # if it should be resized
        # notice!! never use zoom function or np.resize for resizing!
        K = 4  # downsample; 1024-->64 dim
        for i in range(K):
            vol = 0.5 * (vol[:, :, ::2] + vol[:, :, 1::2])
    print('scaling done. max val:%f' % vol[np.unravel_index(np.argmax(vol), vol.shape)])


np.save(args.npy_name, vol)
print(vol.shape)
print('numpy file %s saved!' % args.npy_name)
print('------------------------')
