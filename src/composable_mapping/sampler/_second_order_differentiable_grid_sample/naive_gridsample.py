"""Grid sample implementation that allows for second order differentiation.

Copyright (c) 2022-2023 Aliaksandr Siarohin

Licensed under the MIT License

Slighly modified from:

https://github.com/AliaksandrSiarohin/cuda-gridsample-grad2
"""

import torch


def grid_sample_3d(image: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
    """Naive grid sample 3D implementation with padding_mode="border"."""
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():

        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.reshape(N, C, ID * IH * IW)

    tnw_val = torch.gather(
        image,
        2,
        (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    tne_val = torch.gather(
        image,
        2,
        (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    tsw_val = torch.gather(
        image,
        2,
        (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    tse_val = torch.gather(
        image,
        2,
        (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    bnw_val = torch.gather(
        image,
        2,
        (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    bne_val = torch.gather(
        image,
        2,
        (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    bsw_val = torch.gather(
        image,
        2,
        (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )
    bse_val = torch.gather(
        image,
        2,
        (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().reshape(N, 1, D * H * W).repeat(1, C, 1),
    )

    out_val = (
        tnw_val.reshape(N, C, D, H, W) * tnw.reshape(N, 1, D, H, W)
        + tne_val.reshape(N, C, D, H, W) * tne.reshape(N, 1, D, H, W)
        + tsw_val.reshape(N, C, D, H, W) * tsw.reshape(N, 1, D, H, W)
        + tse_val.reshape(N, C, D, H, W) * tse.reshape(N, 1, D, H, W)
        + bnw_val.reshape(N, C, D, H, W) * bnw.reshape(N, 1, D, H, W)
        + bne_val.reshape(N, C, D, H, W) * bne.reshape(N, 1, D, H, W)
        + bsw_val.reshape(N, C, D, H, W) * bsw.reshape(N, 1, D, H, W)
        + bse_val.reshape(N, C, D, H, W) * bse.reshape(N, 1, D, H, W)
    )

    return out_val


def grid_sample_2d(image: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
    """Naive grid sample 2D implementation with padding_mode="border"."""
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W)
        + ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W)
        + sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W)
        + se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W)
    )

    return out_val
