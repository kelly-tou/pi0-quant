import math
from typing import Optional

import torch
import torch.nn.functional as F

from fp_formats import AddendSel, OutputFmtSel
from params_and_requests import InnerProductTreeParams, ComputeReq, WeightLoadReq
from inner_product_trees_model import InnerProductTreesModel


# Bit-casting helpers

def torch_float_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float().contiguous()
    u32 = x_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u32 >> 16) & 0xFFFF
    lsb = upper & 1
    rounded = u32 + (0x7FFF + lsb)
    return ((rounded >> 16) & 0xFFFF).to(torch.int32)


def torch_bf16_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    u32 = (bits.to(torch.int32) & 0xFFFF) << 16
    return u32.view(torch.float32)


# scalar quantizers / encoders

def quant_bf16_tensor(x: torch.Tensor) -> torch.Tensor:
    bits = torch_float_to_bf16_bits(x.float())
    return torch_bf16_bits_to_float(bits)


def _float_to_e4m3_byte_scalar(v: float) -> int:
    # Reference encoder for finite values.
    # Subnormals underflow to zero; NaN -> 0; inf -> max finite.
    if math.isnan(v):
        return 0
    if math.isinf(v):
        return 0xFE if v < 0 else 0x7E
    if v == 0.0:
        return 0

    sign = 1 if v < 0 else 0
    a = abs(v)

    exp = math.floor(math.log2(a))
    if exp > 8:
        return 0xFE if sign else 0x7E
    if exp < -6:
        return 0

    mant = a / (2.0 ** exp)  # in [1, 2)
    frac_real = (mant - 1.0) * 8.0
    frac = int(round(frac_real))

    if frac == 8:
        frac = 0
        exp += 1

    if exp > 8:
        return 0xFE if sign else 0x7E
    if exp < -6:
        return 0

    exp_field = exp + 7
    return ((sign & 1) << 7) | ((exp_field & 0xF) << 3) | (frac & 0x7)


def float_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().float().cpu().reshape(-1).tolist()
    out = [_float_to_e4m3_byte_scalar(v) for v in flat]
    return torch.tensor(out, dtype=torch.uint8, device=x.device).reshape(x.shape)


# BF16 container decode from model output
# - OutBF16: 16-bit BF16 bits
# - OutE4M3: low 8 bits contain E4M3, upper bits zero

def decode_model_output_bits(out_bits: torch.Tensor, out_fmt_sel: OutputFmtSel) -> torch.Tensor:
    if out_fmt_sel == OutputFmtSel.OutBF16:
        return torch_bf16_bits_to_float(out_bits.to(torch.int32))
    else:
        # If you want to keep exact E4M3 bytes instead of converting back to float,
        # you can return the raw byte tensor here.
        e4m3_bytes = (out_bits & 0xFF).to(torch.uint8)
        # For pi0 integration, usually easiest is to decode to float.
        return e4m3_bytes_to_float(e4m3_bytes)


def e4m3_bytes_to_float(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    frac = x & 0x7

    out = torch.zeros_like(x, dtype=torch.float32)

    normal = (exp > 0) & (exp < 0xF)
    out[normal] = (1.0 + frac[normal].float() / 8.0) * torch.pow(
        2.0, (exp[normal] - 7).float()
    )

    neg = sign.bool()
    out[neg] = -out[neg]
    return out


# Tile-based adapter: behaves like F.linear numerically
# using the RTL functional model

class AtlasLinearRTLFunction:
    """
    Functional adapter around InnerProductTreesModel.

    Assumptions:
      - input activations / weights are quantized to E4M3 before entering the MXU
      - psum is BF16
      - bias is loaded as E4M3 if used in the first tile
      - output container is BF16-width
      - output shape matches F.linear
    """

    def __init__(
        self,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ):
        self.p = InnerProductTreeParams.withPipelineDepth(
            pipeline_depth,
            InnerProductTreeParams(numLanes=num_lanes, vecLen=vec_len),
        )
        self.out_fmt_sel = out_fmt_sel

    def __call__(
        self,
        x_q: torch.Tensor,                  # [..., in_features], quantized values in float domain
        w_q: torch.Tensor,                  # [out_features, in_features], quantized values in float domain
        b_q: Optional[torch.Tensor] = None, # [out_features], quantized values in float domain
        scale_exp: int = 0,
    ) -> torch.Tensor:
        original_shape = x_q.shape[:-1]
        in_features = x_q.shape[-1]
        out_features = w_q.shape[0]

        x2 = x_q.reshape(-1, in_features).float()
        w2 = w_q.float()
        b2 = b_q.float() if b_q is not None else None

        batch = x2.shape[0]
        device = x_q.device

        vec_len = self.p.vecLen
        num_lanes = self.p.numLanes
        num_k_tiles = math.ceil(in_features / vec_len)

        # Quantize once
        x_e4m3 = float_to_e4m3_bytes(x2)
        w_e4m3 = float_to_e4m3_bytes(w2)
        b_e4m3 = float_to_e4m3_bytes(b2) if b2 is not None else None

        # Convert once to Python lists for the pure-Python RTL model.
        x_e4m3_list = x_e4m3.cpu().tolist()                       # [B][K]
        w_e4m3_list = w_e4m3.cpu().tolist()                       # [O][K]
        b_e4m3_list = b_e4m3.cpu().tolist() if b_e4m3 is not None else None

        # Store model output bits in Python first, then decode once per tile.
        y_bits = torch.zeros(batch, out_features, dtype=torch.int32, device=device)

        zero_vec = [0] * vec_len
        zero_bias = [0] * num_lanes
        scale_list = [scale_exp] * num_lanes

        for out_base in range(0, out_features, num_lanes):
            lane_count = min(num_lanes, out_features - out_base)

            dut = InnerProductTreesModel(self.p)

            # psum storage for this output tile: plain Python ints
            psum_bits = [[0] * num_lanes for _ in range(batch)]

            # Precompute per-lane output index mapping for this tile
            lane_out_idx = [out_base + lane for lane in range(lane_count)]

            for k_tile in range(num_k_tiles):
                k0 = k_tile * vec_len
                k1 = min(k0 + vec_len, in_features)
                tile_width = k1 - k0
                needs_pad = tile_width < vec_len

                # Load weights into writable buffer
                for lane in range(num_lanes):
                    if lane < lane_count:
                        row = w_e4m3_list[out_base + lane][k0:k1]
                        if needs_pad:
                            row = row + zero_vec[tile_width:]
                    else:
                        row = zero_vec

                    dut.load_weights(
                        WeightLoadReq(
                            weightsDma=row,
                            laneIdx=lane,
                            last=(lane == num_lanes - 1),
                        )
                    )

                # Bias behavior for this tile
                if k_tile == 0 and b_e4m3_list is not None:
                    bias_list = zero_bias.copy()
                    for lane in range(lane_count):
                        bias_list[lane] = b_e4m3_list[out_base + lane]
                    addend_sel = AddendSel.UseBias
                elif k_tile == 0:
                    bias_list = zero_bias
                    addend_sel = AddendSel.UseAct
                else:
                    bias_list = zero_bias
                    addend_sel = AddendSel.UsePsum

                # Compute each batch row
                for b_idx in range(batch):
                    act = x_e4m3_list[b_idx][k0:k1]
                    if needs_pad:
                        act = act + zero_vec[tile_width:]

                    req = ComputeReq(
                        act=act,
                        bias=bias_list,
                        psum=psum_bits[b_idx],
                        scaleExp=scale_list,
                        addendSel=addend_sel,
                        outFmtSel=self.out_fmt_sel,
                    )

                    psum_bits[b_idx] = dut.compute_now(req)

            # Final decode for this output tile
            tile_bits = torch.tensor(
                [row[:lane_count] for row in psum_bits],
                dtype=torch.int32,
                device=device,
            )
            y_bits[:, out_base:out_base + lane_count] = tile_bits

        y = decode_model_output_bits(y_bits, self.out_fmt_sel)
        return y.reshape(*original_shape, out_features)
