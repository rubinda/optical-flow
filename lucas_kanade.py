#!/usr/bin/env python3
#
# Lucas-Kanade optical flow tracker.
#
# With the help of http://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf
#
# @author David Rubin
import cv2
import numpy as np
import argparse
import logging
from scipy import signal, ndimage
from pathlib import Path
from matplotlib import pyplot as plt
from sys import stdout


class LucasKanade:
    def __init__(self, window_size=15, eigen_size=1e-2, eigen_ratio=3, smooth_kernel_size=7, max_corners=100,
                 feature_quality=0.3, min_distance=7, block_size=7):
        self.window_size = window_size  # Lucas-Kanade window (border region to ignore)
        self.kernel_size = smooth_kernel_size  # Size of kernel for Gaussian smoothing

        # Shi-Tomasi corner detector parameters
        self.feature_quality = feature_quality
        self.max_corners = max_corners
        self.min_distance = min_distance
        self.block_size = block_size

        # Both are parameters for Least squares (A^TA matrix check)
        self.eigen_size = eigen_size
        self.eigen_ratio = eigen_ratio

        # Kernels for convolution
        self.kernel_x = np.array([[-1., 1.], [-1., 1.]])
        self.kernel_y = np.array([[-1., -1.], [1., 1.]])
        self.kernel_t = np.ones((2, 2))  # The minus here controls arrow directions (drawing specifics)

        self.smooth_kernel = (self.kernel_size, self.kernel_size)

    def __call__(self, frame, frame_next, features, is_sparse=False):
        """
        Calculate the optical flow estimation from the 2 given frames

        :param frame:  a frame of video (Grayscale)
        :param frame_next:  the next frame of video (Grayscale)
        :param is_sparse:  use features detected by a Shi Tomasi detector instead of all points
        :return:
        """
        # Smooth the image to eliminate high frequency noise
        smoothed_frame1 = cv2.GaussianBlur(frame, self.smooth_kernel, cv2.BORDER_DEFAULT)
        smoothed_frame2 = cv2.GaussianBlur(frame_next, self.smooth_kernel, cv2.BORDER_DEFAULT)

        # Derivatives in x, y (gradient) and time
        mode = 'same'
        # f_x, f_y = np.gradient(smoothed_frame1)
        f_x = signal.convolve2d(smoothed_frame1, self.kernel_x, mode=mode)
        f_y = signal.convolve2d(smoothed_frame1, self.kernel_y, mode=mode)
        f_t = signal.convolve2d(smoothed_frame1, self.kernel_t, mode=mode) + \
              signal.convolve2d(smoothed_frame2, -self.kernel_t, mode=mode)

        optical_flow = np.zeros((*smoothed_frame1.shape, 2), dtype=np.float64)
        # Travese from [-w, w] around pixel
        log.debug(f'Using {len(features)} features')
        # Create a window_size/2 border around image so not out of bound errors
        w = self.window_size // 2  # Shorthand
        features_moved = []
        features_status = []
        for feature in features:
            j, i = feature[0]
            if i + w + 1 >= smoothed_frame1.shape[0] or j + w + 1 >= smoothed_frame1.shape[1]:
                features_moved.append([[i, j]])
                features_status.append(0)
                continue
            # Compute derivatives for pixels around the given one
            I_x = f_x[i - w:i + w, j - w:j + w].flatten()
            I_y = f_y[i - w:i + w, j - w:j + w].flatten()
            I_t = f_t[i - w:i + w, j - w:j + w].flatten()

            # Use a least squares solution:
            # Column vector of I_t (reshape)
            b = np.reshape(-I_t, (I_t.shape[0], 1))
            # Matrix of [ I_x | I_y ] (column vectors I_x & I_y)
            A = np.vstack((I_x, I_y)).T

            # Todo: this were some conditions I've found in the literature,
            #  but the algorithm works ~fine without them
            # Compute the flow only if:
            #   - eigenvalues are not too small
            #   - lamba1 / lamba2 is not too big (lambda1 is the bigger eigenvalue) ... well conditioned
            # AT_A = np.dot(A.T, A)
            # eigenvalues = np.abs((np.linalg.eigvals(AT_A)))
            # not_too_small = np.sum(np.abs(eigenvalues)) >= self.eigen_size
            # well_conditioned = True if np.min(eigenvalues) == 0 else \
            #                         np.max(eigenvalues) / np.min(eigenvalues) < self.eigen_ratio
            # #
            # if not_too_small and well_conditioned:
            # if np.min(abs(np.linalg.eigvals(AT_A))) >= self.eigen_size:
            # Calculate x using least square (same as np.linalg.lstsq(A, b)[0])
            x = np.matmul(np.linalg.pinv(A), b).ravel()
            optical_flow[i, j] = [x[0], x[1]]
            if is_sparse:
                # Store new point positions
                x = x.ravel()
                features_moved.append([[round(j + x[0]), round(i + x[1])]])
                # Also store if the new position has actually changes
                has_flow = np.sum(x) > 1e-14
                features_status.append(1 if has_flow else 0)
            # else:
            #     print('TOO SMALL OR NOT CONDITIONED WELL')

        # Return the flow in x and y direction for given frames
        return optical_flow, np.array(features_moved), np.array(features_status)  # xs, ys, u, v

    def find_features(self, first_frame, shi_tomasi=False):
        """
        Call first to define which features to use. Subsequent calls should return improved features
        :param first_frame: frame of motion video in BGR format
        :param shi_tomasi:  whether to use sparse optical flow (Shi Tomasi features to track)
        :return:    feature points
        """
        frame1_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) if len(first_frame.shape) > 2 else first_frame
        smoothed_frame1 = frame1_gray  # cv2.GaussianBlur(frame1_gray, kernel, cv2.BORDER_DEFAULT)
        # As per https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
        w = self.window_size // 2
        features_to_track = []
        # Should initialize which features to use on first call and reuse features on subsequent calls
        if shi_tomasi:
            shi_tomasi_params = dict(maxCorners=self.max_corners,
                                     qualityLevel=self.feature_quality,
                                     minDistance=self.min_distance,
                                     blockSize=self.block_size)
            features_to_track = cv2.goodFeaturesToTrack(smoothed_frame1, mask=None, **shi_tomasi_params).astype(int)
        else:
            x_range = range(w, smoothed_frame1.shape[0] - w, 5)
            y_range = range(w, smoothed_frame1.shape[1] - w, 5)
            features_to_track = [[[i, j]] for i in y_range for j in x_range]
        return features_to_track


def cv_draw_flow_dense(img, flow, step=8):
    """
    Somewhat based on the openCV method to draw optical flow

    :param img:     image to draw on
    :param flow:    vector with u, v components
    :param step:    grid density
    :return:    image with flow drawn onto
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * 2
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (255, 100, 0), 2)
    for (x1, y1), (_, _) in lines:
        cv2.circle(img, (x1, y1), 1, (54, 255, 64), -1)
    return img


def cv_draw_flow_sparse(frame, mask, features_old, features_new):
    """
    Draws sparse flow (only for certain points)

    :param frame:           current frame snapshot
    :param mask:            mask that we draw previous flow lines onto
    :param features_old:    previous points from flow
    :param features_new:    current points from flow
    :return:    frame with marked features that are tracked and mask with flow lines (use cv.add to combine them)
    """
    for (new, old) in zip(features_new, features_old):
        x, y = new.ravel()
        _x, _y = old.ravel()
        mask = cv2.line(mask, (x, y), (_x, _y), (0, 0, 255), 2)
        frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optical flow estimater. Draws flow onto images.")
    parser.add_argument('video', help='Input video file (tested with MP4)')
    parser.add_argument('-o', '--output', help='Output file name')
    parser.add_argument('-s', '--sparse', help='Use sparse optical flow estimation using Shi-Tomasi features', action="store_true")
    parser.add_argument('-w', '--window-size', help='Lucas-Kanade window size to use (default=31)', default=31, type=int)
    parser.add_argument('-f', '--max-features', help='Max number of features to detect. Sparse flow only (default=100)', default=100, type=int)
    parser.add_argument('-l', '--frame-limit', help='Max number of frames to process', type=int)
    parser.add_argument('-p', '--multi-scale', help='Use Pyramid Lucas-Kanade', action="store_true")

    args = parser.parse_args()

    Log_Format = "%(asctime)s [%(levelname)s] - %(message)s"
    logging.basicConfig(stream=stdout,
                        filemode="w",
                        format=Log_Format,
                        level=logging.DEBUG)

    log = logging.getLogger()

    track = LucasKanade(window_size=args.window_size, max_corners=args.max_features)

    input_path = Path(args.video)
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f'Given input "{args.video}" is not a file');

    cap = cv2.VideoCapture(args.video)

    # Read first frame, needed to initialize the algorithm
    ret, prev_frame = cap.read()
    video_size = (prev_frame.shape[1], prev_frame.shape[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    log.debug(f'Initial size {prev_gray.shape}')
    levels = 3
    # windows = [15, 15, 5]
    # track.window_size = windows[-1]
    prev_gray_scaled = [prev_gray]
    for k in range(1, levels):
        prev_gray_scaled.append(cv2.pyrDown(prev_gray_scaled[k-1]))

    # Mask to draw on
    mask = np.zeros_like(prev_frame)

    output_path = args.output
    if not output_path or output_path == '':
        output_path = f'{input_path.with_suffix("")}_flow{input_path.suffix}'
    # Output file writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 15, video_size)

    # Detect initial features
    features = track.find_features(prev_frame, shi_tomasi=args.sparse)
    if args.multi_scale:
        features = track.find_features(prev_gray_scaled[-1], shi_tomasi=args.sparse)
    # Plot the initial features that will be tracked throughout the video
    xs = [f[0][0] for f in features]
    ys = [f[0][1] for f in features]
    plt.scatter(xs, ys)
    plt.imshow(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
    plt.title('Selected features to track')
    plt.show()

    frame_cap = args.frame_limit
    log.info('Starting flow calculation, this might take a while (especially if using dense flow) so grab a coffee ...')
    while cap.isOpened() and (frame_cap is None or frame_cap > 0):
        if frame_cap is not None:
            log.info(f'Remaining {frame_cap} frames')
            frame_cap -= 1
        ret, frame = cap.read()
        if not ret:
            log.warning("Oops! Can't read from video stream anymore! (Don't worry this happens when the video ends)")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_scaled = [gray]
        if args.multi_scale:
            for k in range(1, levels):
                gray_scaled.append(cv2.pyrDown(gray_scaled[k-1]))
            gray_n = gray_scaled[-1]
            prev_gray_n = prev_gray_scaled[-1]
            # Calculate flow on lowest level
            optical_flow_prev, _, _ = track(prev_gray_n, gray_n, features, is_sparse=args.sparse)
            n = levels - 2
            while n >= 0:
                #track.window_size = windows[n]
                log.debug(f'running on level {n}')
                # Loop backward and estimate + correct flow on higher resolutions
                gray_n = gray_scaled[n]
                prev_gray_n = prev_gray_scaled[n]
                # Upsample flow on level-1 (cv2. resize uses bilinear interpolation by default)
                log.debug(f'reshaping to {gray_n.shape}')
                optical_flow_n = ndimage.zoom(optical_flow_prev, (2, 2, 1), order=1)    # cv2.resize(optical_flow_prev, (gray_n.shape[1], gray_n.shape[0]))
                optical_flow_n *= 2
                new_features = []
                for pt in features:
                    # TODO: x,y and u,v might be reversed here :thinking_face
                    x, y = pt[0]
                    u, v = optical_flow_n[y, x]
                    # For some reason I reversed the components in features ...
                    new_features.append([[round(x+u), round(y+v)]])
                # Calculate (corrections for) flow on level+1 for the moved features
                optical_flow, new_features, st = track(prev_gray_n, gray_n, new_features, is_sparse=args.sparse)
                # The final flow is the estimation from the previous stage + the corrections from this stage
                optical_flow += optical_flow_n
                optical_flow_prev = optical_flow
                n -= 1
        else:
            optical_flow, new_features, st = track(prev_gray, gray, features, is_sparse=args.sparse)

        if args.sparse:
            # Updates mask, and old frame with drawings
            good_new = new_features[st==1]
            good_old = features[st==1]
            frame, mask = cv_draw_flow_sparse(frame, mask, good_old, good_new)
            flow_on_image = cv2.add(frame, mask)
            features = good_new.reshape(-1, 1, 2)
        else:
            flow_on_image = cv_draw_flow_dense(frame, optical_flow, step=15)

        prev_gray = gray.copy()
        prev_gray_scaled = gray_scaled
        # Write video output
        out.write(flow_on_image)

    log.info(f'Written flow video to "{output_path}"')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
