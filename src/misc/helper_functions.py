import numpy as np


def demand_update(d_ori, p, ph, pf, jitter=0):
    """ 
    Update demand by price. Special cases: if the price and demand are both 0, return 0; 
    if the demand is 0 and price not 0, add jitter.

    d_ori: Original demand
    p: price
    ph: maximum price
    pf: reference price.
    
    """
    if pf == 0 and d_ori == 0:
        return 0
    elif d_ori == 0:
        d_jitter = jitter*(ph-p)/(ph-pf)
        return round(d_jitter)
    else:
        return round(d_ori*(ph-p)/(ph-pf))