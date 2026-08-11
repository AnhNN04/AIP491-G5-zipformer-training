import contextlib
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from icefall.utils import add_sos
from scaling import ScaledLinear
from utils import GradMultiply

class AsrModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 768,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        encoder_feature_layer: int = -1,
    ):
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        self.encoder = encoder
        self.encoder_feature_layer = encoder_feature_layer

        self.use_transducer = use_transducer
        if use_transducer:
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        do_final_down_sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = torch.zeros_like(x, dtype=torch.bool)

        encoder_out, padding_mask, encoder_out_lens = self.encoder.extract_features(
            source=x,
            padding_mask=padding_mask,
            mask=self.encoder.training,
            output_layer=self.encoder_feature_layer,
            do_final_down_sample=do_final_down_sample,
        )
        assert torch.all(encoder_out_lens > 0), encoder_out_lens

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctc_output = self.ctc_output(encoder_out)  

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        decoder_out = self.decoder(sos_y_padded)

        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.amp.autocast("cuda", enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.amp.autocast("cuda", enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        y: k2.RaggedTensor,
        padding_mask: Optional[torch.Tensor] = None,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        freeze_encoder: bool = False,
        encoder_grad_scale: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 3, x.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == y.dim0, (x.shape, y.dim0)

        if freeze_encoder:
            with torch.no_grad():
                encoder_out, encoder_out_lens = self.forward_encoder(x, padding_mask)
        else:
            encoder_out, encoder_out_lens = self.forward_encoder(x, padding_mask)
            if encoder_grad_scale != 1:
                GradMultiply.apply(encoder_out, encoder_grad_scale)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, encoder_out_lens
