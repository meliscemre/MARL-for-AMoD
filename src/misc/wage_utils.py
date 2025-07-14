import numpy as np

def generate_fixed_od_wages(wage_mode, G, std=5, base_mean=25, income_array=None, scaled_region_means=None, seed=42):
    """
    Generates fixed wages per OD pair for a given wage mode.
    
    Args:
        wage_mode (str): One of ['normal25', 'fixed_mean', 'scaled_region']
        G (networkx.DiGraph): Graph with nodes as regions
        std (float): Std deviation for normal sampling
        base_mean (float): Mean for fixed_mean mode
        income_array (np.ndarray): Used in 'scaled_region' mode
        scaled_region_means (dict): Optional override for scaled region means
        seed (int): Random seed for reproducibility

    Returns:
        dict: {(i, j): wage}
    """
    rng = np.random.default_rng(seed)
    region_means = {}

    if wage_mode == "normal25":
        region_means = {i: 25 for i in G.nodes}

    elif wage_mode == "fixed_mean":
        region_means = {i: base_mean for i in G.nodes}

    elif wage_mode == "scaled_region":
        if scaled_region_means is None:
            assert income_array is not None
            target_mean = 25
            scale_factor = target_mean / income_array.mean()
            scaled = (income_array * scale_factor).round(2)
            region_means = {i: m for i, m in enumerate(scaled)}
        else:
            region_means = scaled_region_means

    else:
        raise ValueError(f"Unsupported wage_mode: {wage_mode}")

    od_wages = {}
    for i in G.nodes:
        for j in G.nodes:
            mean = region_means.get(i, 25)  # origin determines wage
            w = rng.normal(loc=mean, scale=std)
            od_wages[i, j] = max(w, 1e-3)

    return od_wages
