from math import *
from dct1 import *
from numpy import *

def dct_dft_feat(boxes, fname_annot, len1, n_seg):

    cents_x = (boxes[:, 0:-4:3] + boxes[:, 2:-4:3]) / 2
    cents_y = (boxes[:, 1:-4:3] + boxes[:, 3: -4:3]) / 2

    # read_annotation function
    fr_s,fr_e, score2_vec = read_annotation(fname_annot)

    feats = []

    for i in range(len(fr_s)):
        frs = arange(fr_s(i), fr_e(i))
        cents_x1 = cents_x[frs,:]
        cents_y1 = cents_y[frs,:]

        dist_feat = []
        # not super sure how to replace the bsxfun here
        tmpx = bsxfun(subtract, cents_x1, cents_x1[:, 0])
        tmpy = bsxfun(subtract, cents_y1, cents_y1[:, 0])
        dist_feat = [dist_feat, tmpx, tmpy]

        feat2 = []
        r = len(frs) / n_seg
        for j in range(1, n_seg+1):
            dist_feat1 = dist_feat[round((j - 1) * r + 1): round(j * r),:]

        # feat1 and feat2 having an error here
        feat1 = dct1[dist_feat1, len1]
        feat2 = np.block([[feat2],[abs(feat1)])
    feats[:, i] = feat2.flatten()
