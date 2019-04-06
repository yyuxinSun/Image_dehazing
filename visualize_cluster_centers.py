
import numpy as np
import skimage.io as skio

if __name__ == '__main__':
    patch_X = 10
    patch_Y = 10
    patch_ch = 3

    patches_file = 'clustered_patches.npy'
    outf_im = 'patches.png'

    patches = np.load(patches_file)
    numpatch = patches.shape[0]
    patch_len = patch_X*patch_Y*patch_ch

    # allocate the output image
    w = int(np.ceil(np.sqrt(numpatch)))
    h = int(np.ceil(numpatch / float(w)))
    out_w = w*patch_X
    out_h = h*patch_Y
    out_im = np.zeros((out_h, out_w, patch_ch), dtype='float32')
    for i in xrange(numpatch):
        idx = (i / w)*patch_X
        idy = (i % w)*patch_Y
        print idx, idy
        p = patches[i, :patch_len].reshape((patch_X, patch_Y, patch_ch))
        print out_im[idx:idx+patch_X, idy:idy+patch_Y, :].shape
        out_im[idx:idx+patch_X, idy:idy+patch_Y, :] = p
    
    skio.imsave(outf_im, out_im)
