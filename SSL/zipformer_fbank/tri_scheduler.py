import math

from optim import LRScheduler

class TriStageLRSchedule(LRScheduler):

    def __init__(
        self,
        optimizer,
        init_lr_scale=0.01,
        final_lr_scale=0.01,
        verbose: bool = False,
        max_update: int = 0,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        decay_steps: int = 0,
        phase_ratio=None,
    ):
        super(TriStageLRSchedule, self).__init__(optimizer, verbose)
        peak_lr = self.base_lrs[0]
        self.peak_lr = peak_lr
        self.init_lr = init_lr_scale * peak_lr
        self.final_lr = final_lr_scale * peak_lr

        if phase_ratio is not None:
            assert max_update > 0
            assert sum(phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(max_update * phase_ratio[0])
            self.hold_steps = int(max_update * phase_ratio[1])
            self.decay_steps = int(max_update * phase_ratio[2])
        else:
            self.warmup_steps = warmup_steps
            self.hold_steps = hold_steps
            self.decay_steps = decay_steps

        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

    def _decide_stage(self, update_step):
        if update_step < self.warmup_steps:
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            return 2, update_step - offset

        offset += self.decay_steps

        return 3, update_step - offset

    def get_lr(self):
        stage, steps_in_stage = self._decide_stage(self.batch)
        lr = []
        peak_lr = self.base_lrs[0]
        if stage == 0:
            lr.append(self.init_lr + self.warmup_rate * steps_in_stage)
        elif stage == 1:
            lr.append(peak_lr)
        elif stage == 2:
            lr.append(peak_lr * math.exp(-self.decay_factor * steps_in_stage))
        elif stage == 3:
            lr.append(self.final_lr)
        else:
            raise ValueError("Undefined stage")

        return lr
