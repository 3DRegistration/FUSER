import torch
import torch.nn as nn
import MinkowskiEngine as ME
from functools import partial
from torch.utils.checkpoint import checkpoint
from models.layers.attention import FlashAttention
from models.layers.block import Block
from models.layers.camera_head import CameraHead
from models.layers.mlp import MLP
from models.layers.transformer_head import TransformerDecoder
from models.layers.mink_encoder import MinkEncoder
from models.layers.pe import PositionEmbeddingCoordsSine
from utils.seq_manipulation import pad_sequence

class FUSERBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --------------------------------------------
        #     Absolute Geometric Encoder
        # --------------------------------------------
        self.mink_in_feats_dim = cfg.model.mink_in_feats_dim
        self.mink_conv1_kernel_size = cfg.model.mink_conv1_kernel_size
        self.mink_normalize_feature = cfg.model.mink_normalize_feature
        self.mink_bn_momentum = cfg.model.mink_bn_momentum
        self.mink_encoder = MinkEncoder(in_feats_dim=self.mink_in_feats_dim,
                                        conv1_kernel_size=self.mink_conv1_kernel_size,
                                        normalize_feature=self.mink_normalize_feature,
                                        bn_momentum=self.mink_bn_momentum)

        # --------------------------------------------
        #     Geometric Alternating Attention
        # --------------------------------------------
        self.dec_embed_dim = cfg.model.dec_embed_dim
        self.dec_num_heads = cfg.model.dec_num_heads
        self.mlp_ratio = cfg.model.mlp_ratio
        self.dec_depth = cfg.model.dec_depth
        self.pos_emb_scaling = cfg.model.pos_emb_scaling

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.dec_embed_dim, scale=self.pos_emb_scaling)
        self.feat_proj = nn.Linear(self.mink_encoder.CHANNELS[-1], self.dec_embed_dim, bias=True)
        self.decoder = nn.ModuleList([
            Block(
                dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=MLP,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttention
            ) for _ in range(self.dec_depth)])

        # ----------------------
        #     Register Token
        # ----------------------
        self.num_register_tokens = cfg.model.num_register_tokens
        self.patch_start_idx = self.num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, self.num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.cam_dec_num_heads = cfg.model.cam_dec_num_heads
        self.cam_out_dim = cfg.model.cam_out_dim
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=self.dec_embed_dim,
            dec_num_heads=self.cam_dec_num_heads,
            out_dim=self.cam_out_dim,
            use_checkpoint=True
        )
        self.camera_head = CameraHead(dim=self.cam_out_dim)


    def decode(self, feats_c, slens_c, coords_c, B, S):

        token = self.feat_proj(feats_c)
        token = torch.split(token, slens_c, dim=0)
        pos = torch.split(self.pos_embed(coords_c.to(feats_c.dtype)), slens_c, dim=0)

        token, mask, _ = pad_sequence(token, require_padding_mask=True)
        pos, _, _ = pad_sequence(pos)

        token = token.transpose(1, 0).contiguous()
        pos = pos.transpose(1, 0).contiguous()
        BS, P, C = token.shape

        register_token = self.register_token.repeat(B, S, 1, 1).reshape(B*S, *self.register_token.shape[-2:])
        token = torch.cat([register_token, token], dim=1)
        P = token.shape[1]

        pos_prefix = pos.new_zeros((pos.shape[0], self.patch_start_idx, *pos.shape[2:]))
        mask_prefix = mask.new_zeros((mask.shape[0], self.patch_start_idx))
        pos = torch.cat([pos_prefix, pos], dim=1)
        mask = torch.cat([mask_prefix.bool(), mask], dim=1)

        final_output = []
        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            if i % 2 == 0:
                pos = pos.reshape(B * S, P, -1)
                token = token.reshape(B * S, P, -1)
                mask = mask.reshape(B * S, P)
            else:
                pos = pos.reshape(B, S * P, -1)
                token = token.reshape(B, S * P, -1)
                mask = mask.reshape(B, S * P)

            if self.training:
                token = checkpoint(blk, token, pos, mask, use_reentrant=False)
            else:
                token = blk(token, pos=pos, mask=mask)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(token.reshape(B * S, P, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(BS, P, -1), mask.reshape(BS, P)

    def forward_backbone(self, batch):

        B = len(batch["pcd_seqs"])
        S = len(batch["pcd_seqs"][0])
        coord_seqs_mink = batch["coord_seqs_mink"]
        feat_seqs_mink = batch["feat_seqs_mink"]

        # Absolute Geometric Encoding
        sinput = ME.SparseTensor(feat_seqs_mink, coordinates=coord_seqs_mink)
        feats_c, slens_c, coords_c = self.mink_encoder.forward(sinput)

        # Geometric Alternating Attention
        token, pos, mask = self.decode(feats_c, slens_c, coords_c, B, S)

        # Global Pose Prediction
        camera_token = self.camera_decoder(token, pos=pos, mask=mask)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            camera_token = camera_token.float()
            camera_poses = self.camera_head(camera_token[:, self.patch_start_idx:], B, S).reshape(B, S, 4, 4)

        return camera_poses
