# -----------------------------
# AdaptiveScheduler
# -----------------------------
#TODO: Für Human Feedback LR höher setzen. -> Automatisieren
from torch import optim


class AdaptiveScheduler:
    def __init__(self, optimizer, params_dict=None):
        if params_dict is None:
            params_dict = {}

        self.optimizer = optimizer

        # Lernraten
        self.lr_normal = params_dict.get('lr_normal', 1e-4)
        self.lr_plateau = params_dict.get('lr_plateau', 5e-5)
        self.lr_precision = params_dict.get('lr_precision', 1e-5)
        self.lr_warmup_start = params_dict.get('lr_warmup_start', 1e-3)  # beginnt hoch
        self.lr_human = params_dict.get('lr_human', 2e-4)

        # Scheduler Instanzen
        self.step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params_dict.get('step_size', 2),
                                                        gamma=params_dict.get('step_gamma', 0.5))
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=params_dict.get('plateau_factor', 0.5),
            patience=params_dict.get('plateau_patience', 1)
        )
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                     T_max=params_dict.get('cosine_Tmax', 5))

        # Tracking
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.overshoot_detected = False
        self.active_scheduler = self.step_scheduler

        # Warmup
        self.warmup_steps = params_dict.get('warmup_steps', 100)
        self.step_num = 0
        self.lr_base = self.lr_normal

        # initiale LR auf lr_warmup_start setzen
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_warmup_start

        # Sprung-Parameter
        self.jump_prob = params_dict.get('jump_prob', 0.1)
        self.jump_factor = params_dict.get('jump_factor', 0.05)

    def step(self, loss: float, human_input=False):
        self.step_num += 1

        # Warmup linear von lr_warmup_start → lr_base
        if self.step_num <= self.warmup_steps:
            lr = self.lr_warmup_start - (self.lr_warmup_start - self.lr_base) * (self.step_num / self.warmup_steps)
            self._set_lr(lr)
        else:
            # Normales adaptives Scheduling
            if loss < self.best_loss - 1e-4:
                self.best_loss = loss
                self.plateau_count = 0
                self.overshoot_detected = False
                self._set_lr(self.lr_normal)
            elif loss > self.best_loss * 1.05:
                self.overshoot_detected = True
                self._set_lr(self.lr_precision)
            else:
                self.plateau_count += 1
                if self.plateau_count > 3:
                    self._set_lr(self.lr_plateau)

            # Human Input override
            if human_input:
                self._set_lr(self.lr_human)

        # Stochastic LR Jump
        self._maybe_jump_lr()

        # Scheduler step
        match type(self.active_scheduler):
            case optim.lr_scheduler.ReduceLROnPlateau:
                # TODO: Was da los? Warum nicht float?
                self.active_scheduler.step(loss)
            case _:
                self.active_scheduler.step()

    def _maybe_jump_lr(self):
        import random
        if random.random() < self.jump_prob:
            direction = 1 if random.random() > 0.5 else -1
            new_lr = self.get_lr() * (1 + direction * self.jump_factor)
            self._set_lr(new_lr)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
