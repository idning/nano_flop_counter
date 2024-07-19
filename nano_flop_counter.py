from torch.profiler import profile, ProfilerActivity, record_function
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
import torch

class NanoFlopCounter:
    def __init__(self, model):
        self.model = model
        self.prof = torch.profiler.profile(with_flops=True)

    def __enter__(self):
        self._update_module_forward(self.model, '~')
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_module_forward(self.model)
        self.prof.__exit__(exc_type, exc_val, exc_tb)

    def _update_module_forward(self, module, prefix):
        assert not hasattr(module, "__orig_forward")
        module.__orig_forward = module.forward
        module.forward = torch.autograd.profiler.record_function(prefix)(module.forward)
        for name, sub_module in module._modules.items():
            submodule_prefix = prefix + ("." if prefix else "") + name
            self._update_module_forward(sub_module, submodule_prefix)

    def _restore_module_forward(self, module):
        module.forward = module.__orig_forward
        for name, sub_module in module._modules.items():
            self._restore_module_forward(sub_module)

    def report(self):
        def get_event_path(e):
            if e.name.startswith("~"):
                return e.name
            if e.cpu_parent:
                return f"{get_event_path(e.cpu_parent)}/{e.name}"
            return e.name

        df = pd.DataFrame([(get_event_path(e), e.flops) for e in self.prof.events() if e.flops], columns=["fqn", "flops"],)  # fmt: skip
        df["module"] = df["fqn"].str.split("/", expand=True)[0]

        def get_all_prefixes(name):
            parts = name.split(".")
            prefixes = []

            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                prefixes.append(prefix)

            return prefixes

        d_flops = defaultdict(int)
        for _, row in df.iterrows():
            for prefix in get_all_prefixes(row.module):
                d_flops[prefix] += row.flops

        d_param_size = defaultdict(int)
        for n, p in self.model.named_parameters():
            for prefix in get_all_prefixes("~." + n):
                d_param_size[prefix] += p.numel()

        data = [(k, d_flops[k], d_param_size[k]) for k in d_flops.keys()]
        df = pd.DataFrame(data, columns=["module", "flops", "params"])
        # print(tabulate(df, headers='keys', tablefmt='plain', intfmt=","))
        return df

# Usage
from torchvision import models as torchvision_models
resnet18 = torchvision_models.resnet18()
a = torch.randn((1, 3, 224, 224), requires_grad=True)
with NanoFlopCounter(resnet18) as mode:
    resnet18(a)
print(tabulate(mode.report(), headers='keys', tablefmt='plain', intfmt=","))
