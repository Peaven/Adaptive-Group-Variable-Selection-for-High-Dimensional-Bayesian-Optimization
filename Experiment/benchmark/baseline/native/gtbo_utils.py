def from_unit_cube(x, lb, ub):
    return lb + (ub-lb)*x

def to_unit_cube(x, lb, ub):
    return (x-lb)/(ub-lb)

def logit(x):
    import torch
    return torch.logit(torch.as_tensor(x))
