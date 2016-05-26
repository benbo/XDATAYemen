from __future__ import division

import cv2
import numpy as np


def generate_composites(src, backgrounds, poly,
                        randomize_loc=False, randomize_scale=False,
                        global_scale=None, scale_lower=0.7, scale_upper=1.3):
    if isinstance(src, basestring):
        src = cv2.imread(src)
    ys, xs = src.shape[:2]

    src_mask = src.copy()
    src_mask = cv2.cvtColor(src_mask, cv2.COLOR_BGR2GRAY)
    src_mask.fill(0)
    cv2.fillPoly(src_mask, [poly], 255)

    # bounding box around polygon
    y1, x1 = np.min(np.nonzero(src_mask), axis=1)
    y2, x2 = np.max(np.nonzero(src_mask), axis=1)

    x_len = x2 - x1
    y_len = y2 - y1

    # crop mask to region of interest and create inverse mask
    mask = src_mask[y1:y2, x1:x2]
    mask_inv = cv2.bitwise_not(mask)

    # cut out the region of interest from the source image
    src1_cut = src[y1:y2, x1:x2]
    # cut out object
    img2_fg = cv2.bitwise_and(src1_cut, src1_cut, mask=mask)

    for dst in backgrounds:
        yd, xd = dst.shape[:2]
        if global_scale is not None:
            vscale = np.sqrt((x_len * y_len) / (xd * yd * global_scale))
            if randomize_scale:
                vscale = vscale * np.random.uniform(scale_lower, scale_upper) 
        else:
            vscale = np.sqrt((xs * ys) / (xd * yd))
            if randomize_scale:
                vscale = vscale * np.random.uniform(scale_lower, scale_upper) 
            # minimum scale factor, so that object is not less than x percent
            # of image volume (we don't want it to be too small)
            # In this case we set the max scale to a 20% ratio
            vscale = np.min((vscale, (x_len * y_len) / (xd * yd * 0.20)))

        # dst image needs to be big enough for object
        scalefactor = np.max((y_len / yd, x_len / xd, vscale))
        dst_r = cv2.resize(dst, dsize=(int(ceil(scalefactor * xd)),
                                       int(ceil(scalefactor * yd))))
        yd, xd = dst_r.shape[:2]  
        
        ### randomize location
        xtmp = xd - x_len
        ytmp = yd - y_len
        if randomize_loc:
            y_off = 0
            x_off = 0
            if ytmp > 0:
                y_off = np.random.randint(ytmp)
            if xtmp > 0:
                x_off = np.random.randint(xtmp)
        else:
            y_off = y1
            x_off = x1
            if y_off > ytmp:
                y_off = ytmp
            if x_off > xtmp:
                x_off = xtmp
        y1t, y2t = y_off, y_off + y_len
        x1t, x2t = x_off, x_off + x_len

        # create region of interest
        roi = dst_r[y1t:y2t, x1t:x2t]

        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Put object in ROI and modify the main image
        dst_r[y1t:y2t, x1t:x2t] = cv2.add(img1_bg, img2_fg)

        yield dst_r
        # NOTE: this image is in cv2 format (0-255, BGR)
        # To convert to (0-1, RGB): (dst_r / 255)[:, :, (2, 1, 0)]
