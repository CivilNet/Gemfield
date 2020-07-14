import torch
from torch import optim
import torch.distributed as dist

the_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
paras_only_bn, paras_wo_bn = separate_bn_paras(the_model)

optimizer = optim.SGD([
                {'params': paras_wo_bn, 'weight_decay': 5e-4},
                {'params': paras_only_bn}
            ], lr = 0.0027030, momentum = 0.9)

#extract BN parameters to apply different weight decay
def separate_bn_paras(modules):
    paras_only_bn = []
    paras_wo_bn = []
    memo = set()
    gemfield_set = set()
    gemfield_set.update(set(modules.parameters()))
    print("set len: ",len(gemfield_set))
    named_modules = modules.named_modules(prefix='')
    for module_prefix, module in named_modules:
        if "module" not in module_prefix:
            print("skip ",module_prefix)
            continue

        members = module._parameters.items()
        for k, v in members:
            name = module_prefix + ('.' if module_prefix else '') + k
            if v is None:
                continue
            if v in memo:
                continue
            memo.add(v)
            if "batchnorm" in str(module.__class__):
                paras_only_bn.append(v)
            else:
                paras_wo_bn.append(v)
    print("param len: ",len(paras_wo_bn),len(paras_only_bn))

    for v in paras_wo_bn:
        print("debug wo bn: ",v in gemfield_set)

    for v in paras_only_bn:
        print("debug bn: ",v in gemfield_set)

    return paras_only_bn, paras_wo_bn

