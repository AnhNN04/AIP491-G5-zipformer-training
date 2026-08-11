import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
from icefall import ContextGraph, ContextState, NgramLm, NgramLmStateCost
from icefall.decode import Nbest, one_best_decoding
from icefall.lm_wrapper import LmScorer
from icefall.rnn_lm.model import RnnLmModel
from icefall.transformer_lm.model import TransformerLM
from icefall.utils import (
    DecodingResults,
    KeywordResult,
    add_eos,
    add_sos,
    get_texts,
    get_texts_with_timestamp,
)
from torch import nn

def fast_beam_search_one_best(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    temperature: float = 1.0,
    ilme_scale: float = 0.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        ilme_scale=ilme_scale,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
    )

    best_path = one_best_decoding(lattice)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)

def fast_beam_search_nbest_LG(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    nbest_scale: float = 0.5,
    use_double_scores: bool = True,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    ilme_scale: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
        ilme_scale=ilme_scale,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    word_fsa = k2.invert(nbest.fsa)
    if hasattr(lattice, "aux_labels"):
        del word_fsa.aux_labels
    word_fsa.scores.zero_()
    word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
    path_to_utt_map = nbest.shape.row_ids(1)

    if hasattr(lattice, "aux_labels"):
        inv_lattice = k2.invert(lattice)
        inv_lattice = k2.arc_sort(inv_lattice)
    else:
        inv_lattice = k2.arc_sort(lattice)

    if inv_lattice.shape[0] == 1:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=torch.zeros_like(path_to_utt_map),
            sorted_match_a=True,
        )
    else:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=path_to_utt_map,
            sorted_match_a=True,
        )

    path_lattice = k2.top_sort(k2.connect(path_lattice))
    tot_scores = path_lattice.get_tot_scores(
        use_double_scores=use_double_scores,
        log_semiring=True,  
    )

    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
    best_hyp_indexes = ragged_tot_scores.argmax()
    best_path = k2.index_fsa(nbest.fsa, best_hyp_indexes)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)

def fast_beam_search_nbest(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    nbest_scale: float = 0.5,
    use_double_scores: bool = True,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        blank_penalty=blank_penalty,
        temperature=temperature,
        allow_partial=allow_partial,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    nbest = nbest.intersect(lattice)

    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)

def fast_beam_search_nbest_oracle(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    ref_texts: List[List[int]],
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    hyps = nbest.build_levenshtein_graphs()
    refs = k2.levenshtein_graph(ref_texts, device=hyps.device)

    levenshtein_alignment = k2.levenshtein_alignment(
        refs=refs,
        hyps=hyps,
        hyp_to_ref_map=nbest.shape.row_ids(1),
        sorted_match_ref=True,
    )

    tot_scores = levenshtein_alignment.get_tot_scores(
        use_double_scores=False, log_semiring=False
    )
    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)

    max_indexes = ragged_tot_scores.argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)

def fast_beam_search(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    temperature: float = 1.0,
    subtract_ilme: bool = False,
    ilme_scale: float = 0.1,
    allow_partial: bool = False,
    blank_penalty: float = 0.0,
) -> k2.Fsa:
    assert encoder_out.ndim == 3

    context_size = model.decoder.context_size
    vocab_size = model.decoder.vocab_size

    B, T, C = encoder_out.shape

    config = k2.RnntDecodingConfig(
        vocab_size=vocab_size,
        decoder_history_len=context_size,
        beam=beam,
        max_contexts=max_contexts,
        max_states=max_states,
    )
    individual_streams = []
    for i in range(B):
        individual_streams.append(k2.RnntDecodingStream(decoding_graph))
    decoding_streams = k2.RnntDecodingStreams(individual_streams, config)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        shape, contexts = decoding_streams.get_contexts()
        contexts = contexts.to(torch.int64)
        decoder_out = model.decoder(contexts, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        log_probs = (logits / temperature).log_softmax(dim=-1)

        if ilme_scale != 0:
            ilme_logits = model.joiner(
                torch.zeros_like(
                    current_encoder_out, device=current_encoder_out.device
                ).unsqueeze(2),
                decoder_out.unsqueeze(1),
                project_input=False,
            )
            ilme_logits = ilme_logits.squeeze(1).squeeze(1)
            if blank_penalty != 0:
                ilme_logits[:, 0] -= blank_penalty
            ilme_log_probs = (ilme_logits / temperature).log_softmax(dim=-1)
            log_probs -= ilme_scale * ilme_log_probs

        decoding_streams.advance(log_probs)
    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(
        encoder_out_lens.tolist(), allow_partial=allow_partial
    )

    return lattice

def merge_greedy_search_batch(
    model_lis: nn.Module,
    encoder_out_lis: torch.Tensor,
    encoder_out_lens_lis: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:

    new_encoder_out_lis = []
    for model, encoder_out, encoder_out_lens in zip(
        model_lis, encoder_out_lis, encoder_out_lens_lis
    ):
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        device = next(model.parameters()).device

        blank_id = model.decoder.blank_id
        unk_id = getattr(model, "unk_id", blank_id)
        context_size = model.decoder.context_size

        batch_size_list = packed_encoder_out.batch_sizes.tolist()
        N = encoder_out.size(0)
        assert torch.all(encoder_out_lens > 0), encoder_out_lens
        assert N == batch_size_list[0], (N, batch_size_list)

        encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)
        new_encoder_out_lis.append(encoder_out)
    encoder_out_lis = new_encoder_out_lis

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    timestamps = [[] for _ in range(N)]
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  

    decoder_out_lis = []
    for model in model_lis:
        decoder_out = model.decoder(decoder_input, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        decoder_out_lis.append(decoder_out)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size

        logits_lis = []
        for model, encoder_out, decoder_out in zip(
            model_lis, encoder_out_lis, decoder_out_lis
        ):
            current_encoder_out = encoder_out.data[start:end]
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
            offset = end

            decoder_out = decoder_out[:batch_size]

            logits = model.joiner(
                current_encoder_out, decoder_out.unsqueeze(1), project_input=False
            )
            logits_lis.append(logits)

        logits = logits_lis[0]
        for item in logits_lis[1:]:
            logits += item
        logits /= len(logits_lis)

        logits = logits.squeeze(1).squeeze(1)  
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )

            decoder_out_lis = []
            for model in model_lis:
                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
                decoder_out_lis.append(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )

def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    max_sym_per_frame: int,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id)

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], device=device, dtype=torch.int64
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    timestamp = []

    max_sym_per_utt = 1000

    sym_per_frame = 0

    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )

        if blank_penalty != 0:
            logits[:, :, :, 0] -= blank_penalty

        y = logits.argmax().item()
        if y not in (blank_id, unk_id):
            hyp.append(y)
            timestamp.append(t)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )

            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
    hyp = hyp[context_size:]  

    if not return_timestamps:
        return hyp
    else:
        return DecodingResults(
            hyps=[hyp],
            timestamps=[timestamp],
        )

def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    timestamps = [[] for _ in range(N)]
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )

        logits = logits.squeeze(1).squeeze(1)  
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )

@dataclass
class Hypothesis:
    ys: List[int]

    log_prob: torch.Tensor

    ac_probs: Optional[List[float]] = None

    timestamp: List[int] = field(default_factory=list)

    lm_score: Optional[torch.Tensor] = None

    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    state_cost: Optional[NgramLmStateCost] = None

    context_state: Optional[ContextState] = None

    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        return "_".join(map(str, self.ys))

class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)

def get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
    num_hyps = [len(h) for h in hyps]

    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans

def keywords_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    keywords_graph: ContextGraph,
    beam: int = 4,
    num_tailing_blanks: int = 0,
    blank_penalty: float = 0,
) -> List[List[KeywordResult]]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert keywords_graph is not None

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                context_state=keywords_graph.root,
                timestamp=[],
                ac_probs=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    sorted_ans = [[] for _ in range(N)]
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]

        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        probs = logits.softmax(dim=-1)  

        log_probs = probs.log()

        probs = probs.reshape(-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        ragged_probs = k2.RaggedTensor(shape=log_probs_shape, value=probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)
            hyp_probs = ragged_probs[i].tolist()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                new_ac_probs = hyp.ac_probs[:]
                context_score = 0
                new_context_state = hyp.context_state
                new_num_tailing_blanks = hyp.num_tailing_blanks + 1
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    new_ac_probs.append(hyp_probs[topk_indexes[k]])
                    (
                        context_score,
                        new_context_state,
                        _,
                    ) = keywords_graph.forward_one_step(hyp.context_state, new_token)
                    new_num_tailing_blanks = 0
                    if new_context_state.token == -1:  
                        new_ys[-context_size:] = [-1] * (context_size - 1) + [blank_id]

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                    ac_probs=new_ac_probs,
                    context_state=new_context_state,
                    num_tailing_blanks=new_num_tailing_blanks,
                )
                B[i].add(new_hyp)

            top_hyp = B[i].get_most_probable(length_norm=True)
            matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
            if matched:
                ac_prob = (
                    sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
                )
            if (
                matched
                and top_hyp.num_tailing_blanks > num_tailing_blanks
                and ac_prob >= matched_state.ac_threshold
            ):
                keyword = KeywordResult(
                    hyps=top_hyp.ys[-matched_state.level :],
                    timestamps=top_hyp.timestamp[-matched_state.level :],
                    phrase=matched_state.phrase,
                )
                sorted_ans[i].append(keyword)
                B[i] = HypothesisList()
                B[i].add(
                    Hypothesis(
                        ys=[-1] * (context_size - 1) + [blank_id],
                        log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                        context_state=keywords_graph.root,
                        timestamp=[],
                        ac_probs=[],
                    )
                )

    B = B + finalized_B

    for i, hyps in enumerate(B):
        top_hyp = hyps.get_most_probable(length_norm=True)
        matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
        if matched:
            ac_prob = (
                sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
            )
        if matched and ac_prob >= matched_state.ac_threshold:
            keyword = KeywordResult(
                hyps=top_hyp.ys[-matched_state.level :],
                timestamps=top_hyp.timestamp[-matched_state.level :],
                phrase=matched_state.phrase,
            )
            sorted_ans[i].append(keyword)

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans

def merge_modified_beam_search(
    model_lis: nn.Module,
    encoder_out_lis: torch.Tensor,
    encoder_out_lens_lis: torch.Tensor,
    context_graph: Optional[ContextGraph] = None,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:

    new_encoder_out_lis = []
    for model, encoder_out, encoder_out_lens in zip(
        model_lis, encoder_out_lis, encoder_out_lens_lis
    ):
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        blank_id = model.decoder.blank_id
        unk_id = getattr(model, "unk_id", blank_id)
        context_size = model.decoder.context_size
        device = next(model.parameters()).device

        batch_size_list = packed_encoder_out.batch_sizes.tolist()
        N = encoder_out.size(0)
        assert torch.all(encoder_out_lens > 0), encoder_out_lens
        assert N == batch_size_list[0], (N, batch_size_list)
        new_encoder_out_lis.append(model.joiner.encoder_proj(packed_encoder_out.data))

    encoder_out_lis = new_encoder_out_lis

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                context_state=None if context_graph is None else context_graph.root,
                timestamp=[],
            )
        )

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]

        B = [HypothesisList() for _ in range(batch_size)]

        log_probs_lis = []
        for model, encoder_out in zip(model_lis, encoder_out_lis):
            current_encoder_out = encoder_out.data[start:end]
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
            offset = end

            ys_log_probs = torch.cat(
                [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
            )  

            decoder_input = torch.tensor(
                [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
                device=device,
                dtype=torch.int64,
            )  

            decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
            decoder_out = model.joiner.decoder_proj(decoder_out)

            current_encoder_out = torch.index_select(
                current_encoder_out,
                dim=0,
                index=hyps_shape.row_ids(1).to(torch.int64),
            )  

            logits = model.joiner(
                current_encoder_out,
                decoder_out,
                project_input=False,
            )  

            logits = logits.squeeze(1).squeeze(1)  

            if blank_penalty != 0:
                logits[:, 0] -= blank_penalty

            log_probs = (logits / temperature).log_softmax(
                dim=-1
            )  
            log_probs_lis.append(log_probs)

        log_probs = log_probs_lis[0]
        for item in log_probs_lis[1:]:
            log_probs += item
        log_probs /= len(log_probs_lis)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                        ) = context_graph.forward_one_step(hyp.context_state, new_token)

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                    context_state=new_context_state,
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    if context_graph is not None:
        finalized_B = [HypothesisList() for _ in range(len(B))]
        for i, hyps in enumerate(B):
            for hyp in list(hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                finalized_B[i].add(
                    Hypothesis(
                        ys=hyp.ys,
                        log_prob=hyp.log_prob + context_score,
                        timestamp=hyp.timestamp,
                        context_state=new_context_state,
                    )
                )
        B = finalized_B

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )

def modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    context_graph: Optional[ContextGraph] = None,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                context_state=None if context_graph is None else context_graph.root,
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]

        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        log_probs = (logits / temperature).log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                        ) = context_graph.forward_one_step(hyp.context_state, new_token)

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                    context_state=new_context_state,
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    if context_graph is not None:
        finalized_B = [HypothesisList() for _ in range(len(B))]
        for i, hyps in enumerate(B):
            for hyp in list(hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                finalized_B[i].add(
                    Hypothesis(
                        ys=hyp.ys,
                        log_prob=hyp.log_prob + context_score,
                        timestamp=hyp.timestamp,
                        context_state=new_context_state,
                    )
                )
        B = finalized_B

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )

def modified_beam_search_lm_rescore(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    lm_scale_list: List[int],
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        log_probs = (logits / temperature).log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    hyps_shape = get_hyps_shape(B)
    am_scores = torch.tensor([hyp.log_prob.item() for b in B for hyp in b])
    am_scores = k2.RaggedTensor(value=am_scores, shape=hyps_shape).to(device)

    candidate_seqs = [hyp.ys[context_size:] for b in B for hyp in b]
    possible_seqs = k2.RaggedTensor(candidate_seqs)
    row_splits = possible_seqs.shape.row_splits(1)
    sentence_token_lengths = row_splits[1:] - row_splits[:-1]
    possible_seqs_with_sos = add_sos(possible_seqs, sos_id=1)
    possible_seqs_with_eos = add_eos(possible_seqs, eos_id=1)
    sentence_token_lengths += 1

    x = possible_seqs_with_sos.pad(mode="constant", padding_value=blank_id)
    y = possible_seqs_with_eos.pad(mode="constant", padding_value=blank_id)
    x = x.to(device).to(torch.int64)
    y = y.to(device).to(torch.int64)
    sentence_token_lengths = sentence_token_lengths.to(device).to(torch.int64)

    lm_scores = LM.lm(x=x, y=y, lengths=sentence_token_lengths)
    assert lm_scores.ndim == 2
    lm_scores = -1 * lm_scores.sum(dim=1)

    ans = {}
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()

    for lm_scale in lm_scale_list:
        key = f"nnlm_scale_{lm_scale:.2f}"
        tot_scores = am_scores.values + lm_scores * lm_scale
        ragged_tot_scores = k2.RaggedTensor(shape=am_scores.shape, value=tot_scores)
        max_indexes = ragged_tot_scores.argmax().tolist()
        unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
        hyps = []
        for idx in unsorted_indices:
            hyps.append(unsorted_hyps[idx])

        ans[key] = hyps
    return ans

def modified_beam_search_lm_rescore_LODR(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    LODR_lm: NgramLm,
    sp: spm.SentencePieceProcessor,
    lm_scale_list: List[int],
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        log_probs = (logits / temperature).log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    hyps_shape = get_hyps_shape(B)
    am_scores = torch.tensor([hyp.log_prob.item() for b in B for hyp in b])
    am_scores = k2.RaggedTensor(value=am_scores, shape=hyps_shape).to(device)

    candidate_seqs = [hyp.ys[context_size:] for b in B for hyp in b]
    possible_seqs = k2.RaggedTensor(candidate_seqs)
    row_splits = possible_seqs.shape.row_splits(1)
    sentence_token_lengths = row_splits[1:] - row_splits[:-1]
    possible_seqs_with_sos = add_sos(possible_seqs, sos_id=1)
    possible_seqs_with_eos = add_eos(possible_seqs, eos_id=1)
    sentence_token_lengths += 1

    x = possible_seqs_with_sos.pad(mode="constant", padding_value=blank_id)
    y = possible_seqs_with_eos.pad(mode="constant", padding_value=blank_id)
    x = x.to(device).to(torch.int64)
    y = y.to(device).to(torch.int64)
    sentence_token_lengths = sentence_token_lengths.to(device).to(torch.int64)

    lm_scores = LM.lm(x=x, y=y, lengths=sentence_token_lengths)
    assert lm_scores.ndim == 2
    lm_scores = -1 * lm_scores.sum(dim=1)

    import math

    LODR_scores = []
    for seq in candidate_seqs:
        tokens = " ".join(sp.id_to_piece(seq))
        LODR_scores.append(LODR_lm.score(tokens))
    LODR_scores = torch.tensor(LODR_scores).to(device) * math.log(
        10
    )  
    assert lm_scores.shape == LODR_scores.shape

    ans = {}
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()

    LODR_scale_list = [0.05 * i for i in range(1, 20)]
    for lm_scale in lm_scale_list:
        for lodr_scale in LODR_scale_list:
            key = f"nnlm_scale_{lm_scale:.2f}_lodr_scale_{lodr_scale:.2f}"
            tot_scores = (
                am_scores.values / lm_scale + lm_scores - LODR_scores * lodr_scale
            )
            ragged_tot_scores = k2.RaggedTensor(shape=am_scores.shape, value=tot_scores)
            max_indexes = ragged_tot_scores.argmax().tolist()
            unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
            hyps = []
            for idx in unsorted_indices:
                hyps.append(unsorted_hyps[idx])

            ans[key] = hyps
    return ans

def _deprecated_modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:

    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    device = next(model.parameters()).device

    T = encoder_out.size(1)

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[-1] * (context_size - 1) + [blank_id],
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            timestamp=[],
        )
    )
    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        A = list(B)
        B = HypothesisList()

        ys_log_probs = torch.cat([hyp.log_prob.reshape(1, 1) for hyp in A])

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyp in A],
            device=device,
            dtype=torch.int64,
        )

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = current_encoder_out.expand(
            decoder_out.size(0), 1, 1, -1
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)

        log_probs = logits.log_softmax(dim=-1)

        log_probs.add_(ys_log_probs)

        log_probs = log_probs.reshape(-1)
        topk_log_probs, topk_indexes = log_probs.topk(beam)

        topk_hyp_indexes = topk_indexes // logits.size(-1)
        topk_token_indexes = topk_indexes % logits.size(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topk_hyp_indexes = topk_hyp_indexes.tolist()
            topk_token_indexes = topk_token_indexes.tolist()

        for i in range(len(topk_hyp_indexes)):
            hyp = A[topk_hyp_indexes[i]]
            new_ys = hyp.ys[:]
            new_timestamp = hyp.timestamp[:]
            new_token = topk_token_indexes[i]
            if new_token not in (blank_id, unk_id):
                new_ys.append(new_token)
                new_timestamp.append(t)
            new_log_prob = topk_log_probs[i]
            new_hyp = Hypothesis(
                ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
            )
            B.add(new_hyp)

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  

    if not return_timestamps:
        return ys
    else:
        return DecodingResults(hyps=[ys], timestamps=[best_hyp.timestamp])

def beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [blank_id] * context_size,
        device=device,
        dtype=torch.int64,
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[-1] * (context_size - 1) + [blank_id], log_prob=0.0, timestamp=[]
        )
    )

    max_sym_per_utt = 20000

    sym_per_utt = 0

    decoder_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        A = B
        B = HypothesisList()

        joint_cache: Dict[str, torch.Tensor] = {}

        while True:
            y_star = A.get_most_probable()
            A.remove(y_star)

            cached_key = y_star.key

            if cached_key not in decoder_cache:
                decoder_input = torch.tensor(
                    [y_star.ys[-context_size:]],
                    device=device,
                    dtype=torch.int64,
                ).reshape(1, context_size)

                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
                decoder_cache[cached_key] = decoder_out
            else:
                decoder_out = decoder_cache[cached_key]

            cached_key += f"-t-{t}"
            if cached_key not in joint_cache:
                logits = model.joiner(
                    current_encoder_out,
                    decoder_out.unsqueeze(1),
                    project_input=False,
                )

                if blank_penalty != 0:
                    logits[:, :, :, 0] -= blank_penalty

                log_prob = (logits / temperature).log_softmax(dim=-1)
                log_prob = log_prob.squeeze()
                joint_cache[cached_key] = log_prob
            else:
                log_prob = joint_cache[cached_key]

            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob

            B.add(
                Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    timestamp=y_star.timestamp[:],
                )
            )

            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i in (blank_id, unk_id):
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                new_timestamp = y_star.timestamp + [t]
                A.add(
                    Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        timestamp=new_timestamp,
                    )
                )

            A_most_probable = A.get_most_probable()

            kept_B = B.filter(A_most_probable.log_prob)

            if len(kept_B) >= beam:
                B = kept_B.topk(beam)
                break

        t += 1

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  

    if not return_timestamps:
        return ys
    else:
        return DecodingResults(hyps=[ys], timestamps=[best_hyp.timestamp])

def fast_beam_search_with_nbest_rescoring(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    ngram_lm_scale_list: List[float],
    num_paths: int,
    G: k2.Fsa,
    sp: spm.SentencePieceProcessor,
    word_table: k2.SymbolTable,
    oov_word: str = "<UNK>",
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Dict[str, Union[List[List[int]], DecodingResults]]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    nbest = nbest.intersect(lattice)

    am_scores = nbest.tot_scores()

    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)  

    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.labels.contiguous())
    tokens = tokens.remove_values_leq(0)  

    token_list: List[List[int]] = tokens.tolist()
    word_list: List[List[str]] = sp.decode(token_list)

    assert isinstance(oov_word, str), oov_word
    assert oov_word in word_table, oov_word
    oov_word_id = word_table[oov_word]

    word_ids_list: List[List[int]] = []

    for words in word_list:
        this_word_ids = []
        for w in words.split():
            if w in word_table:
                this_word_ids.append(word_table[w])
            else:
                this_word_ids.append(oov_word_id)
        word_ids_list.append(this_word_ids)

    word_fsas = k2.linear_fsa(word_ids_list, device=lattice.device)
    word_fsas_with_self_loops = k2.add_epsilon_self_loops(word_fsas)

    num_unique_paths = len(word_ids_list)

    b_to_a_map = torch.zeros(
        num_unique_paths,
        dtype=torch.int32,
        device=lattice.device,
    )

    rescored_word_fsas = k2.intersect_device(
        a_fsas=G,
        b_fsas=word_fsas_with_self_loops,
        b_to_a_map=b_to_a_map,
        sorted_match_a=True,
        ret_arc_maps=False,
    )

    rescored_word_fsas = k2.remove_epsilon_self_loops(rescored_word_fsas)
    rescored_word_fsas = k2.top_sort(k2.connect(rescored_word_fsas))
    ngram_lm_scores = rescored_word_fsas.get_tot_scores(
        use_double_scores=True,
        log_semiring=False,
    )

    ans: Dict[str, Union[List[List[int]], DecodingResults]] = {}
    for s in ngram_lm_scale_list:
        key = f"ngram_lm_scale_{s}"
        tot_scores = am_scores.values + s * ngram_lm_scores
        ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)

        if not return_timestamps:
            ans[key] = get_texts(best_path)
        else:
            ans[key] = get_texts_with_timestamp(best_path)

    return ans

def fast_beam_search_with_nbest_rnn_rescoring(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    ngram_lm_scale_list: List[float],
    num_paths: int,
    G: k2.Fsa,
    sp: spm.SentencePieceProcessor,
    word_table: k2.SymbolTable,
    rnn_lm_model: torch.nn.Module,
    rnn_lm_scale_list: List[float],
    oov_word: str = "<UNK>",
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Dict[str, Union[List[List[int]], DecodingResults]]:
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    nbest = nbest.intersect(lattice)

    am_scores = nbest.tot_scores()

    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)  

    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.labels.contiguous())
    tokens = tokens.remove_values_leq(0)  

    token_list: List[List[int]] = tokens.tolist()
    word_list: List[List[str]] = sp.decode(token_list)

    assert isinstance(oov_word, str), oov_word
    assert oov_word in word_table, oov_word
    oov_word_id = word_table[oov_word]

    word_ids_list: List[List[int]] = []

    for words in word_list:
        this_word_ids = []
        for w in words.split():
            if w in word_table:
                this_word_ids.append(word_table[w])
            else:
                this_word_ids.append(oov_word_id)
        word_ids_list.append(this_word_ids)

    word_fsas = k2.linear_fsa(word_ids_list, device=lattice.device)
    word_fsas_with_self_loops = k2.add_epsilon_self_loops(word_fsas)

    num_unique_paths = len(word_ids_list)

    b_to_a_map = torch.zeros(
        num_unique_paths,
        dtype=torch.int32,
        device=lattice.device,
    )

    rescored_word_fsas = k2.intersect_device(
        a_fsas=G,
        b_fsas=word_fsas_with_self_loops,
        b_to_a_map=b_to_a_map,
        sorted_match_a=True,
        ret_arc_maps=False,
    )

    rescored_word_fsas = k2.remove_epsilon_self_loops(rescored_word_fsas)
    rescored_word_fsas = k2.top_sort(k2.connect(rescored_word_fsas))
    ngram_lm_scores = rescored_word_fsas.get_tot_scores(
        use_double_scores=True,
        log_semiring=False,
    )

    blank_id = model.decoder.blank_id
    sos_id = sp.piece_to_id("sos_id")
    eos_id = sp.piece_to_id("eos_id")

    sos_tokens = add_sos(tokens, sos_id)
    tokens_eos = add_eos(tokens, eos_id)
    sos_tokens_row_splits = sos_tokens.shape.row_splits(1)
    sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

    x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
    y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

    x_tokens = x_tokens.to(torch.int64)
    y_tokens = y_tokens.to(torch.int64)
    sentence_lengths = sentence_lengths.to(torch.int64)

    rnn_lm_nll = rnn_lm_model(x=x_tokens, y=y_tokens, lengths=sentence_lengths)
    assert rnn_lm_nll.ndim == 2
    assert rnn_lm_nll.shape[0] == len(token_list)
    rnn_lm_scores = -1 * rnn_lm_nll.sum(dim=1)

    ans: Dict[str, List[List[int]]] = {}
    for n_scale in ngram_lm_scale_list:
        for rnn_scale in rnn_lm_scale_list:
            key = f"ngram_lm_scale_{n_scale}_rnn_lm_scale_{rnn_scale}"
            tot_scores = (
                am_scores.values + n_scale * ngram_lm_scores + rnn_scale * rnn_lm_scores
            )
            ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
            max_indexes = ragged_tot_scores.argmax()
            best_path = k2.index_fsa(nbest.fsa, max_indexes)

            if not return_timestamps:
                ans[key] = get_texts(best_path)
            else:
                ans[key] = get_texts_with_timestamp(best_path)

    return ans

def modified_beam_search_ngram_rescoring(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ngram_lm: NgramLm,
    ngram_lm_scale: float,
    beam: int = 4,
    temperature: float = 1.0,
) -> List[List[int]]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    lm_scale = ngram_lm_scale

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state_cost=NgramLmStateCost(ngram_lm),
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [
                hyp.log_prob.reshape(1, 1) + hyp.state_cost.lm_score * lm_scale
                for hyps in A
                for hyp in hyps
            ]
        )  

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        log_probs = (logits / temperature).log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    state_cost = hyp.state_cost.forward_one_step(new_token)
                else:
                    state_cost = hyp.state_cost

                new_log_prob = topk_log_probs[k] - hyp.state_cost.lm_score * lm_scale

                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, state_cost=state_cost
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans

def modified_beam_search_LODR(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LODR_lm: NgramLm,
    LODR_lm_scale: float,
    LM: LmScorer,
    beam: int = 4,
    context_graph: Optional[ContextGraph] = None,
) -> List[List[int]]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert LM is not None
    lm_scale = LM.lm_scale

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    sos_id = getattr(LM, "sos_id", 1)
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
    lens = torch.tensor([1]).to(device)
    init_score, init_states = LM.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state=init_states,  
                lm_score=init_score.reshape(-1),
                state_cost=NgramLmStateCost(
                    LODR_lm
                ),  
                context_state=None if context_graph is None else context_graph.root,
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]  
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        log_probs = logits.log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        token_list = []
        hs = []
        cs = []
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    if LM.lm_type == "rnn":
                        token_list.append([new_token])
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        token_list.append(
                            [sos_id] + hyp.ys[context_size:] + [new_token]
                        )

        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if LM.lm_type == "rnn":
                tokens_to_score = (
                    torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
                )
                hs = torch.cat(hs, dim=1).to(device)
                cs = torch.cat(cs, dim=1).to(device)
                state = (hs, cs)
            else:
                tokens_list = [torch.tensor(tokens) for tokens in token_list]
                tokens_to_score = (
                    torch.nn.utils.rnn.pad_sequence(
                        tokens_list, batch_first=True, padding_value=0.0
                    )
                    .to(device)
                    .to(torch.int64)
                )

                state = None

            scores, lm_states = LM.score_token(tokens_to_score, x_lens, state)

        count = 0  
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]

                lm_score = hyp.lm_score
                state = hyp.state

                hyp_log_prob = topk_log_probs[k]  
                new_token = topk_token_indexes[k]

                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state
                if new_token not in (blank_id, unk_id):
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                        ) = context_graph.forward_one_step(hyp.context_state, new_token)

                    ys.append(new_token)
                    state_cost = hyp.state_cost.forward_one_step(new_token)

                    current_ngram_score = state_cost.lm_score - hyp.state_cost.lm_score

                    assert current_ngram_score <= 0.0, (
                        state_cost.lm_score,
                        hyp.state_cost.lm_score,
                    )
                    hyp_log_prob += (
                        lm_score[new_token] * lm_scale
                        + LODR_lm_scale * current_ngram_score
                        + context_score
                    )  

                    lm_score = scores[count]
                    if LM.lm_type == "rnn":
                        state = (
                            lm_states[0][:, count, :].unsqueeze(1),
                            lm_states[1][:, count, :].unsqueeze(1),
                        )
                    count += 1
                else:
                    state_cost = hyp.state_cost

                new_hyp = Hypothesis(
                    ys=ys,
                    log_prob=hyp_log_prob,
                    state=state,
                    lm_score=lm_score,
                    state_cost=state_cost,
                    context_state=new_context_state,
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    if context_graph is not None:
        finalized_B = [HypothesisList() for _ in range(len(B))]
        for i, hyps in enumerate(B):
            for hyp in list(hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                finalized_B[i].add(
                    Hypothesis(
                        ys=hyp.ys,
                        log_prob=hyp.log_prob + context_score,
                        timestamp=hyp.timestamp,
                        context_state=new_context_state,
                    )
                )
        B = finalized_B

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans

def modified_beam_search_lm_shallow_fusion(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    beam: int = 4,
    return_timestamps: bool = False,
) -> List[List[int]]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert LM is not None
    lm_scale = LM.lm_scale

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    sos_id = getattr(LM, "sos_id", 1)
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
    lens = torch.tensor([1]).to(device)
    init_score, init_states = LM.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state=init_states,
                lm_score=init_score.reshape(-1),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]  
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        lm_scores = torch.cat(
            [hyp.lm_score.reshape(1, -1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  

        logits = logits.squeeze(1).squeeze(1)  

        log_probs = logits.log_softmax(dim=-1)  

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        token_list = []  
        hs = []
        cs = []
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    if LM.lm_type == "rnn":
                        token_list.append([new_token])
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        token_list.append(
                            [sos_id] + hyp.ys[context_size:] + [new_token]
                        )

        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if LM.lm_type == "rnn":
                tokens_to_score = (
                    torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
                )
                hs = torch.cat(hs, dim=1).to(device)
                cs = torch.cat(cs, dim=1).to(device)
                state = (hs, cs)
            else:
                tokens_list = [torch.tensor(tokens) for tokens in token_list]
                tokens_to_score = (
                    torch.nn.utils.rnn.pad_sequence(
                        tokens_list, batch_first=True, padding_value=0.0
                    )
                    .to(device)
                    .to(torch.int64)
                )

                state = None

            scores, lm_states = LM.score_token(tokens_to_score, x_lens, state)

        count = 0  
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]

                lm_score = hyp.lm_score
                state = hyp.state

                hyp_log_prob = topk_log_probs[k]  
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    ys.append(new_token)
                    new_timestamp.append(t)

                    hyp_log_prob += lm_score[new_token] * lm_scale  

                    lm_score = scores[count]
                    if LM.lm_type == "rnn":
                        state = (
                            lm_states[0][:, count, :].unsqueeze(1),
                            lm_states[1][:, count, :].unsqueeze(1),
                        )
                    count += 1

                new_hyp = Hypothesis(
                    ys=ys,
                    log_prob=hyp_log_prob,
                    state=state,
                    lm_score=lm_score,
                    timestamp=new_timestamp,
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )
