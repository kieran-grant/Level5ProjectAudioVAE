import math


class CyclicAnnealing:
    def __init__(
            self,
            min_beta: float = 0.001,
            max_beta: float = 10.,
            start_epoch: int = 0,
            end_epoch: int = 10,
            cycle_length: int = 20
    ):
        self.beta = min_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.cycle_length = cycle_length

    def step(self, current_epoch: float):
        if current_epoch >= self.end_epoch:
            self.beta = self.max_beta
        elif current_epoch < self.start_epoch:
            self.beta = self.min_beta
        else:
            progress_in_cycle = (current_epoch - self.start_epoch) % self.cycle_length / self.cycle_length
            cosine_decay = 0.5 * (1 + math.cos(progress_in_cycle * math.pi))
            new_beta = self.min_beta + cosine_decay * (self.max_beta - self.min_beta)
            self.beta = new_beta

            if current_epoch % self.cycle_length == 0:
                print(f"\nStarting new cycle: Beta weight updated: {self.beta:.4f}\n")
