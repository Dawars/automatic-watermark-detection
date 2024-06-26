#!/usr/bin/env python
#SBATCH -p fat
#SBATCH --cpus-per-task=16
#SBATCH --mem=500GB

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import *

num_channels = 1  # or 3

image_dir = Path("/mnt/hdd/datasets/photobank")
device = "cuda"
backup_path = Path("watermark_mainichi_gray_multiple.pkl")

num_images_load = 1000
num_images_estimate_alpha = 1000
num_images_solve = 100
mask_logo = False


def main():
    images, image_paths = load_images(image_dir, num_images=num_images_load, num_channels=num_channels)
    num_images = len(images)
    print(num_images)

    gx, gy, gxlist, gylist = estimate_watermark(images)

    bounds_odd = np.array([[309, 281],
                           [344, 247],
                           [509, 412],
                           [474, 446], ]) - np.array([309, 281]) + np.array([45, 109])
    bounds_even = np.array([[309, 281],  # 45, 109
                            [344, 247],
                            [509, 412],
                            [474, 446], ])
    row_offset = np.array([436, 437])
    column_offset = np.array([89, -89])

    bounds_all = [bounds_odd + column_offset * i for i in range(0, 1)] + \
                 [bounds_odd + column_offset * i + row_offset for i in range(1, 4)] + \
                 [bounds_even + column_offset * i for i in range(-2, 3)]  # + \
                 # [bounds_even + column_offset * i + row_offset for i in range(-3, 4)]
    template_bounds = bounds_even - column_offset  # strong median

    grad = np.sqrt(gx**2 + gy**2)
    cv2.drawContours(grad, bounds_all, -1, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.drawContours(grad, [template_bounds], -1, (255, 255, 255), 1, cv2.LINE_AA)
    plt.imsave("grad_mag.png", grad[..., 0])

    gradx_logo = gx[template_bounds[:, 1].min():template_bounds[:, 1].max(), template_bounds[:, 0].min():template_bounds[:, 0].max()]
    grady_logo = gy[template_bounds[:, 1].min():template_bounds[:, 1].max(), template_bounds[:, 0].min():template_bounds[:, 0].max()]

    # est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
    # cropped_gx, cropped_gy = crop_watermark(gx, gy)
    W_m = poisson_reconstruct(gradx_logo, grady_logo)

    # random photo
    # img = cv2.imread('Resized/image0000.jpg')
    # im, start, end = watermark_detector(img, cropped_gx, cropped_gy)


    # plt.imshow(im)
    # plt.show()
    # We are done with watermark estimation
    # W_m is the cropped watermark

    # J = images[:, bounds[:, 1].min():bounds[:, 1].max(), bounds[:, 0].min():bounds[:, 0].max()]
    # merge multiple watermarks
    J_logos = []
    for bounds in bounds_all:
        J_logos.append(images[:, bounds[:, 1].min():bounds[:, 1].max(), bounds[:, 0].min():bounds[:, 0].max()])
    J = np.concatenate(J_logos)

    # get a random subset of J
    # Wm = (255*PlotImage(W_m))
    Wm = W_m - W_m.min()  # subtract shift

    mask = np.zeros(gx.shape[:2], np.uint8)
    cv2.drawContours(mask, [bounds], -1, (255, 255, 255), -1, cv2.LINE_AA)
    mask = mask[bounds[:, 1].min():bounds[:, 1].max(), bounds[:, 0].min():bounds[:, 0].max()]
    plt.imsave("mask.png", mask, cmap="gray")
    if mask_logo:
        Wm = Wm * (mask > 0)[..., None]

    plt.imsave("W_m.png", Wm[:, :, 0], cmap="gray")

    if backup_path.exists():
        watermark_data = pickle.loads(backup_path.read_bytes())
    else:
        # get threshold of W_m for alpha matte estimate
        alph_est = estimate_normalized_alpha(J, Wm, num_images=num_images_estimate_alpha)
        if num_channels == 3:
            alph = torch.stack([alph_est, alph_est, alph_est], dim=0)
        else:
            alph = alph_est[..., None]
        C, est_Ik = estimate_blend_factor(J, Wm, alph)
        plt.imsave("est_Ik.png", est_Ik[..., 0], cmap="gray")
        alpha = alph.copy()
        for i in range(num_channels):
            alpha[:, :, i] = C[i] * alpha[:, :, i]

        Wm = Wm + alpha * est_Ik

        W = Wm.copy()
        for i in range(num_channels):
            W[:, :, i] /= C[i]

        watermark_data = {
            "W_m": W_m,
            "Wm": Wm,
            "alpha": alpha,
            "W": W,
        }
        backup_path.write_bytes(pickle.dumps(watermark_data))


    Jt = J[:num_images_solve]
    # now we have the values of alpha, Wm, J  # todo Wm or W_m?
    # Solve for all images

    # W_m_cv = W_m.permute(1, 2, 0).cpu().numpy()
    # Wm_cv = W_m.permute(1, 2, 0).cpu().numpy()
    # alpha_cv = alpha.permute(1, 2, 0).cpu().numpy()
    # W_cv = W.permute(1, 2, 0).cpu().numpy()

    plt.subplot(4, 1, 1)
    plt.imshow(W_m, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(4, 1, 2)
    plt.imshow(Wm, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(4, 1, 3)
    plt.imshow(alpha, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(4, 1, 4)
    plt.imshow(W, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"solve_images_input_cv.png")

    Wk, Ik, W, alpha1 = solve_images(Jt,
                                     W_m,
                                     alpha,
                                     W)
    # W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
    # ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)

    for i in range(len(Wk)):
        plt.imsave(f"recon/Jt_{i}.png", Jt[i, :, :, 0], cmap="gray")
        plt.imsave(f"recon/Ik_{i}.png", Ik[i, :, :, 0], cmap="gray")

        plt.subplot(4, 2, 1)
        plt.imshow(Jt[i, :, :, 0], cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.subplot(4, 2, 2)
        plt.imshow(Ik[i,:,:,0], cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.subplot(4, 2, 3)
        plt.imshow(Wk[i], cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.subplot(4, 2, 4)
        plt.imshow(W[..., 0], cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.subplot(4, 2, 5)
        plt.imshow(alpha1[..., 0], cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.savefig(f"recon/output_{i}.png")
        plt.close()


    print()


if __name__ == '__main__':
    main()