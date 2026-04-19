import torch, numpy as np


class DiffusionScheduler(torch.nn.Module):

    def __init__(self, n_diff_steps, mode="cosine"):
        super().__init__()
        self.num_steps = n_diff_steps
        self.mode = mode

        if self.mode == 'linear':
            raise NotImplementedError
        elif self.mode == 'cosine':
            def betas_fn(s):
                T = self.num_steps

                def f(t, T):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

                alphas = []
                f0 = f(0, T)
                for t in range(T + 1):
                    alphas.append(f(t, T) / f0)

                betas = []
                for t in range(1, T + 1):
                    betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
                return betas

            betas = betas_fn(s=0.008)
            betas = torch.FloatTensor(betas)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1 - betas

        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        gamma0 = torch.zeros_like(betas)
        gamma1 = torch.zeros_like(betas)
        gamma2 = torch.zeros_like(betas)
        for t in range(2, self.num_steps + 1):  # 2 to T
            gamma0[t] = betas[t] * torch.sqrt(alpha_bars[t - 1]) / (1. - alpha_bars[t])
            gamma1[t] = (1. - alpha_bars[t - 1]) * torch.sqrt(alphas[t]) / (1. - alpha_bars[t])
            gamma2[t] = 1 + (torch.sqrt(alpha_bars[t]) - 1) * (torch.sqrt(alphas[t]) + torch.sqrt(alpha_bars[t - 1])) / (1. - alpha_bars[t])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("gamma0", gamma0)
        self.register_buffer("gamma1", gamma1)
        self.register_buffer("gamma2", gamma2)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()
