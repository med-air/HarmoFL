import torch


class WPOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.05, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(WPOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def generate_delta(self, zero_grad=False):
        device = self.param_groups[0]["params"][0].device 
        grad_norm = torch.norm(
            torch.stack([
                (1.0 * p.grad).norm(p=2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]),p=2
        )
        for group in self.param_groups:
            scale = group["alpha"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                delta = 1.0 * p.grad * scale.to(p)
                p.add_(delta) 
                self.state[p]["delta"] = delta

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["delta"]) 
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
