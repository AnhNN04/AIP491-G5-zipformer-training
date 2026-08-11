import contextlib
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from lhotse.utils import fix_random_seed
from torch import Tensor, nn
from torch.optim import Optimizer

class BatchedOptimizer(Optimizer):

    def __init__(self, params, defaults):
        super(BatchedOptimizer, self).__init__(params, defaults)

    @contextlib.contextmanager
    def batched_params(self, param_group, group_params_names):
        batches = defaultdict(
            list
        )  
        batches_names = defaultdict(
            list
        )  

        assert len(param_group) == len(group_params_names)
        for p, named_p in zip(param_group, group_params_names):
            key = (str(p.dtype), *p.shape)
            batches[key].append(p)
            batches_names[key].append(named_p)

        batches_names_keys = list(batches_names.keys())
        sorted_idx = sorted(
            range(len(batches_names)), key=lambda i: batches_names_keys[i]
        )
        batches_names = [batches_names[batches_names_keys[idx]] for idx in sorted_idx]
        batches = [batches[batches_names_keys[idx]] for idx in sorted_idx]

        stacked_params_dict = dict()

        tuples = []

        for batch, batch_names in zip(batches, batches_names):
            p = batch[0]
            state = self.state[p]
            p_stacked = torch.stack(batch)
            grad = torch.stack(
                [torch.zeros_like(p) if p.grad is None else p.grad for p in batch]
            )
            p_stacked.grad = grad
            stacked_params_dict[key] = p_stacked
            tuples.append((p_stacked, state, batch_names))

        yield tuples  

        for ((stacked_params, _state, _names), batch) in zip(tuples, batches):
            for i, p in enumerate(batch):  
                p.copy_(stacked_params[i])

class ScaledAdam(BatchedOptimizer):

    def __init__(
        self,
        params,
        lr=3e-02,
        clipping_scale=None,
        betas=(0.9, 0.98),
        scalar_lr_scale=0.1,
        eps=1.0e-08,
        param_min_rms=1.0e-05,
        param_max_rms=3.0,
        scalar_max=10.0,
        size_update_period=4,
        clipping_update_period=100,
    ):

        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            betas=betas,
            scalar_lr_scale=scalar_lr_scale,
            eps=eps,
            param_min_rms=param_min_rms,
            param_max_rms=param_max_rms,
            scalar_max=scalar_max,
            size_update_period=size_update_period,
            clipping_update_period=clipping_update_period,
        )

        self.show_dominant_parameters = True
        param_groups, parameters_names = self._get_names_of_parameters(params)
        super(ScaledAdam, self).__init__(param_groups, defaults)
        assert len(self.param_groups) == len(parameters_names)
        self.parameters_names = parameters_names

    def _get_names_of_parameters(
        self, params_or_named_params
    ) -> Tuple[List[Dict], List[List[str]]]:

        iterable_or_groups = list(params_or_named_params)
        if len(iterable_or_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        param_groups = []

        param_groups_names = []

        if not isinstance(iterable_or_groups[0], dict):
            param_iterable_cur_group = []
            param_names_cur_group = []
            for p_or_np in iterable_or_groups:
                if isinstance(p_or_np, tuple):
                    name, param = p_or_np
                else:
                    assert isinstance(p_or_np, torch.Tensor)
                    param = p_or_np
                    name = "foo"
                    self.show_dominant_parameters = False
                param_iterable_cur_group.append(param)
                param_names_cur_group.append(name)
            param_groups.append({"params": param_iterable_cur_group})
            param_groups_names.append(param_names_cur_group)
        else:
            for cur_group in iterable_or_groups:
                if "named_params" in cur_group:
                    name_list = [x[0] for x in cur_group["named_params"]]
                    p_list = [x[1] for x in cur_group["named_params"]]
                    del cur_group["named_params"]
                    cur_group["params"] = p_list
                else:
                    assert "params" in cur_group
                    name_list = ["foo" for _ in cur_group["params"]]
                param_groups.append(cur_group)
                param_groups_names.append(name_list)

        return param_groups, param_groups_names

    def __setstate__(self, state):
        super(ScaledAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        batch = True

        for group, group_params_names in zip(self.param_groups, self.parameters_names):

            with self.batched_params(group["params"], group_params_names) as batches:

                if (
                    len(batches[0][1]) == 0
                ):  
                    clipping_scale = 1
                else:
                    clipping_scale = self._get_clipping_scale(group, batches)

                for p, state, _ in batches:
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "ScaledAdam optimizer does not support sparse gradients"
                        )
                    if len(state) == 0:
                        self._init_state(group, p, state)

                    self._step_one_batch(group, p, state, clipping_scale)

        return loss

    def _init_state(self, group: dict, p: Tensor, state: dict):
        size_update_period = group["size_update_period"]

        state["step"] = 0

        kwargs = {"device": p.device, "dtype": p.dtype}

        state["delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        batch_size = p.shape[0]
        numel = p.numel() // batch_size

        if numel > 1:
            param_rms = (p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt()
            state["param_rms"] = param_rms

            state["scale_exp_avg_sq"] = torch.zeros_like(param_rms)
            state["scale_grads"] = torch.zeros(
                size_update_period, *param_rms.shape, **kwargs
            )

        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _get_clipping_scale(
        self, group: dict, tuples: List[Tuple[Tensor, dict, List[str]]]
    ) -> float:
        assert len(tuples) >= 1
        clipping_scale = group["clipping_scale"]
        (first_p, first_state, _) = tuples[0]
        step = first_state["step"]
        if clipping_scale is None or step == 0:
            return 1.0
        clipping_update_period = group["clipping_update_period"]
        scalar_lr_scale = group["scalar_lr_scale"]

        tot_sumsq = torch.tensor(0.0, device=first_p.device)
        for (p, state, param_names) in tuples:
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "ScaledAdam optimizer does not support sparse gradients"
                )
            if p.numel() == p.shape[0]:  
                tot_sumsq += (grad**2).sum() * (
                    scalar_lr_scale**2
                )  
            else:
                tot_sumsq += ((grad * state["param_rms"]) ** 2).sum()

        tot_norm = tot_sumsq.sqrt()
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(
                clipping_update_period, device=p.device
            )
        first_state["model_norms"][step % clipping_update_period] = tot_norm

        irregular_estimate_steps = [
            i for i in [10, 20, 40] if i < clipping_update_period
        ]
        if step % clipping_update_period == 0 or step in irregular_estimate_steps:
            sorted_norms = first_state["model_norms"].sort()[0].to("cpu")
            if step in irregular_estimate_steps:
                sorted_norms = sorted_norms[-step:]
            num_norms = sorted_norms.numel()
            quartiles = []
            for n in range(0, 5):
                index = min(num_norms - 1, (num_norms // 4) * n)
                quartiles.append(sorted_norms[index].item())

            median = quartiles[2]
            if median - median != 0:
                raise RuntimeError("Too many grads were not finite")
            threshold = clipping_scale * median
            if step in irregular_estimate_steps:
                threshold = threshold * 2.0
            first_state["model_norm_threshold"] = threshold
            percent_clipped = (
                first_state["num_clipped"] * 100.0 / num_norms
                if "num_clipped" in first_state
                else 0.0
            )
            first_state["num_clipped"] = 0
            quartiles = " ".join(["%.3e" % x for x in quartiles])
            logging.warn(
                f"Clipping_scale={clipping_scale}, grad-norm quartiles {quartiles}, "
                f"threshold={threshold:.3e}, percent-clipped={percent_clipped:.1f}"
            )

        try:
            model_norm_threshold = first_state["model_norm_threshold"]
        except KeyError:
            return 1.0  

        ans = min(1.0, (model_norm_threshold / (tot_norm + 1.0e-20)).item())
        if ans != ans:  
            ans = 0.0
        if ans < 1.0:
            first_state["num_clipped"] += 1
        if ans < 0.1:
            logging.warn(
                f"Scaling gradients by {ans}, model_norm_threshold={model_norm_threshold}"
            )
            if self.show_dominant_parameters:
                assert p.shape[0] == len(param_names)
                self._show_gradient_dominating_parameter(
                    tuples, tot_sumsq, group["scalar_lr_scale"]
                )

        if ans == 0.0:
            for (p, state, param_names) in tuples:
                p.grad.zero_()  

        return ans

    def _show_gradient_dominating_parameter(
        self,
        tuples: List[Tuple[Tensor, dict, List[str]]],
        tot_sumsq: Tensor,
        scalar_lr_scale: float,
    ):
        all_sumsq_orig = {}
        for (p, state, batch_param_names) in tuples:
            batch_grad = p.grad
            if p.numel() == p.shape[0]:  
                batch_rms_orig = torch.full(
                    p.shape, scalar_lr_scale, device=batch_grad.device
                )
            else:
                batch_rms_orig = state["param_rms"]
            batch_sumsq_orig = (batch_grad * batch_rms_orig) ** 2
            if batch_grad.ndim > 1:
                batch_sumsq_orig = batch_sumsq_orig.sum(
                    dim=list(range(1, batch_grad.ndim))
                )
            for name, sumsq_orig, rms, grad in zip(
                batch_param_names, batch_sumsq_orig, batch_rms_orig, batch_grad
            ):

                proportion_orig = sumsq_orig / tot_sumsq
                all_sumsq_orig[name] = (proportion_orig, sumsq_orig, rms, grad)

        sorted_by_proportion = {
            k: v
            for k, v in sorted(
                all_sumsq_orig.items(), key=lambda item: item[1][0], reverse=True
            )
        }
        dominant_param_name = next(iter(sorted_by_proportion))
        (
            dominant_proportion,
            dominant_sumsq,
            dominant_rms,
            dominant_grad,
        ) = sorted_by_proportion[dominant_param_name]
        logging.warn(
            f"Parameter dominating tot_sumsq {dominant_param_name}"
            f" with proportion {dominant_proportion:.2f},"
            f" where dominant_sumsq=(grad_sumsq*orig_rms_sq)"
            f"={dominant_sumsq:.3e},"
            f" grad_sumsq={(dominant_grad**2).sum():.3e},"
            f" orig_rms_sq={(dominant_rms**2).item():.3e}"
        )

    def _step_one_batch(
        self, group: dict, p: Tensor, state: dict, clipping_scale: float
    ):
        lr = group["lr"]
        size_update_period = group["size_update_period"]
        beta1 = group["betas"][0]

        grad = p.grad
        if clipping_scale != 1.0:
            grad *= clipping_scale
        step = state["step"]
        delta = state["delta"]

        delta.mul_(beta1)
        batch_size = p.shape[0]
        numel = p.numel() // batch_size
        if numel > 1:
            scale_grads = state["scale_grads"]
            scale_grads[step % size_update_period] = (p * grad).sum(
                dim=list(range(1, p.ndim)), keepdim=True
            )
            if step % size_update_period == size_update_period - 1:
                param_rms = state["param_rms"]  
                param_rms.copy_(
                    (p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt()
                )
                if step > 0:
                    self._size_update(group, scale_grads, p, state)

        if numel == 1:
            self._step_scalar(group, p, state)
        else:
            self._step(group, p, state)

        state["step"] = step + 1

    def _size_update(
        self, group: dict, scale_grads: Tensor, p: Tensor, state: dict
    ) -> None:

        param_rms = state["param_rms"]
        beta1, beta2 = group["betas"]
        size_lr = group["lr"] * group["scalar_lr_scale"]
        param_min_rms = group["param_min_rms"]
        param_max_rms = group["param_max_rms"]
        eps = group["eps"]
        step = state["step"]
        batch_size = p.shape[0]

        size_update_period = scale_grads.shape[0]
        beta2_corr = beta2**size_update_period

        scale_exp_avg_sq = state["scale_exp_avg_sq"]  
        scale_exp_avg_sq.mul_(beta2_corr).add_(
            (scale_grads**2).mean(dim=0),  
            alpha=1 - beta2_corr,
        )  

        size_step = (step + 1) // size_update_period
        bias_correction2 = 1 - beta2_corr**size_step

        denom = scale_exp_avg_sq.sqrt() + eps

        scale_step = (
            -size_lr * (bias_correction2**0.5) * scale_grads.sum(dim=0) / denom
        )

        is_too_small = param_rms < param_min_rms

        scale_step.masked_fill_(is_too_small, 0.0)

        scale_step = torch.minimum(scale_step, (param_max_rms - param_rms) / param_rms)

        delta = state["delta"]
        delta.add_(p * scale_step, alpha=(1 - beta1))

    def _step(self, group: dict, p: Tensor, state: dict):
        grad = p.grad
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        param_min_rms = group["param_min_rms"]
        step = state["step"]

        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

        this_step = state["step"] - (state["zero_step"] if "zero_step" in state else 0)
        bias_correction2 = 1 - beta2 ** (this_step + 1)
        if bias_correction2 < 0.99:
            exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)

        denom = exp_avg_sq.sqrt()
        denom += eps
        grad = grad / denom

        alpha = -lr * (1 - beta1) * state["param_rms"].clamp(min=param_min_rms)

        delta = state["delta"]
        delta.add_(grad * alpha)
        p.add_(delta)

    def _step_scalar(self, group: dict, p: Tensor, state: dict):
        beta1, beta2 = group["betas"]
        scalar_max = group["scalar_max"]
        eps = group["eps"]
        lr = group["lr"] * group["scalar_lr_scale"]
        grad = p.grad

        exp_avg_sq = state["exp_avg_sq"]  
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction2 = 1 - beta2 ** (state["step"] + 1)
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps

        delta = state["delta"]
        delta.add_(grad / denom, alpha=-lr * (1 - beta1))
        p.clamp_(min=-scalar_max, max=scalar_max)
        p.add_(delta)

class LRScheduler(object):

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        return {
            "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        if is_verbose:
            logging.warn(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )

class Eden(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        lr_epochs: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super(Eden, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.25 * (
            ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]

class Eden2(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.warmup_batches = warmup_batches

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.5
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]

def _test_eden():
    m = torch.nn.Linear(100, 100)
    optim = ScaledAdam(m.parameters(), lr=0.03)

    scheduler = Eden(optim, lr_batches=100, lr_epochs=2, verbose=True)

    for epoch in range(10):
        scheduler.step_epoch(epoch)  

        for step in range(20):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()

    logging.info(f"last lr = {scheduler.get_last_lr()}")
    logging.info(f"state dict = {scheduler.state_dict()}")

class Eve(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-3,
        target_rms=0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0 <= weight_decay <= 0.1:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            target_rms=target_rms,
        )
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() * (bias_correction2**-0.5)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1
                target_rms = group["target_rms"]
                weight_decay = group["weight_decay"]

                if p.numel() > 1:
                    is_above_target_rms = p.norm() > (target_rms * (p.numel() ** 0.5))
                    p.mul_(1 - (weight_decay * is_above_target_rms))

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if random.random() < 0.0005:
                    step = (exp_avg / denom) * step_size
                    logging.info(
                        f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}"
                    )

        return loss

def _test_scaled_adam(hidden_dim: int):
    import timeit

    from scaling import ScaledLinear

    E = 100
    B = 4
    T = 2
    logging.info("in test_eve_cain")
    device = torch.device("cpu")
    dtype = torch.float32

    fix_random_seed(42)
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    for iter in [1, 0]:
        fix_random_seed(42)
        Linear = torch.nn.Linear if iter == 0 else ScaledLinear

        m = torch.nn.Sequential(
            Linear(E, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, E),
        ).to(device)

        train_pairs = [
            (
                100.0
                * torch.randn(B, T, E, device=device, dtype=dtype)
                * input_magnitudes,
                torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes,
            )
            for _ in range(20)
        ]

        if iter == 0:
            optim = Eve(m.parameters(), lr=0.003)
        elif iter == 1:
            optim = ScaledAdam(m.parameters(), lr=0.03, clipping_scale=2.0)
        scheduler = Eden(optim, lr_batches=200, lr_epochs=5, verbose=False)

        start = timeit.default_timer()
        avg_loss = 0.0
        for epoch in range(180):
            scheduler.step_epoch()

            for n, (x, y) in enumerate(train_pairs):
                y_out = m(x)
                loss = ((y_out - y) ** 2).mean() * 100.0
                if epoch == 0 and n == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
                if n == 0 and epoch % 5 == 0:
                    lr = scheduler.get_last_lr()[0]
                    logging.info(
                        f"Iter {iter}, epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}"
                    )  
                loss.log().backward()
                optim.step()
                optim.zero_grad()
                scheduler.step_batch()

        stop = timeit.default_timer()
        logging.info(f"Iter={iter}, Time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger().setLevel(logging.INFO)
    import subprocess

    s = subprocess.check_output(
        "git status -uno .; git log -1; git diff HEAD .", shell=True
    )
    logging.info(s)
    import sys

    if len(sys.argv) > 1:
        hidden_dim = int(sys.argv[1])
    else:
        hidden_dim = 200

    _test_scaled_adam(hidden_dim)
    _test_eden()
