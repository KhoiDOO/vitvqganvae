def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def accum_log(log, new_logs: dict):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def count_parameters(model, requires_grad = True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)