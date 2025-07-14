import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_mean_rebalancing_per_region(reb_steps, geojson_path="nyc_zones.geojson", output_path="mean_rebalancing_per_region.png"):
    nyc_zones = gpd.read_file(geojson_path).to_crs(epsg=4326)
    district_codes = [
        "301", "302", "303", "304", "305", "308", "309", 
        "316", "317", "306", "307", "310", "311", "312", 
        "313", "314", "315", "318"
    ]
    bk_zones = nyc_zones[nyc_zones['districtcode'].astype(str).isin(district_codes)].copy()
    region_to_district = {
        0: "301", 1: "302", 2: "303", 3: "304", 4: "305", 5: "308",
        6: "309", 7: "316", 8: "317", 9: "306", 10: "307", 11: "310",
        12: "311", 13: "312", 14: "313", 15: "314", 16: "315", 17: "318"
    }

    reb_counts = defaultdict(float)
    for rebFlow in reb_steps:
        for (_, j), flow_dict in rebFlow.items():
            reb_counts[j] += sum(flow_dict.values())

    num_episodes = len(reb_steps)
    mean_reb = {k: v / num_episodes for k, v in reb_counts.items()}

    bk_zones["districtcode"] = bk_zones["districtcode"].astype(str)
    bk_zones["mean_rebalancing"] = bk_zones["districtcode"].map({
        region_to_district[r]: mean_reb[r]
        for r in mean_reb if r in region_to_district
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10))
    bk_zones.plot(column="mean_rebalancing", cmap="Blues", linewidth=0.8, edgecolor="black", legend=True, ax=ax)
    ax.set_title("Mean Rebalancing Trips per Region", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mean_origin_demand_per_region(origin_steps, geojson_path="data/nyc_zones.geojson", output_path="visualizations/mean_origin_demand_per_region.png"):
    import geopandas as gpd
    from collections import defaultdict
    import matplotlib.pyplot as plt

    nyc_zones = gpd.read_file(geojson_path).to_crs(epsg=4326)
    district_codes = [
        "301", "302", "303", "304", "305", "308", "309", 
        "316", "317", "306", "307", "310", "311", "312", 
        "313", "314", "315", "318"
    ]
    region_to_district = {
        0: "301", 1: "302", 2: "303", 3: "304", 4: "305", 5: "308",
        6: "309", 7: "316", 8: "317", 9: "306", 10: "307", 11: "310",
        12: "311", 13: "312", 14: "313", 15: "314", 16: "315", 17: "318"
    }

    demand_counts = defaultdict(float)
    for step in origin_steps:
        for i, v in step.items():
            demand_counts[i] += v

    num_episodes = len(origin_steps)
    mean_demand = {i: v / num_episodes for i, v in demand_counts.items()}

    bk_zones = nyc_zones[nyc_zones['districtcode'].astype(str).isin(district_codes)].copy()
    bk_zones["districtcode"] = bk_zones["districtcode"].astype(str)
    bk_zones["mean_origin_demand"] = bk_zones["districtcode"].map({
        region_to_district[r]: mean_demand[r]
        for r in mean_demand if r in region_to_district
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10))
    bk_zones.plot(column="mean_origin_demand", cmap="Purples", linewidth=0.8, edgecolor="black", legend=True, ax=ax)
    ax.set_title("Mean Accepted Demand per Origin Region", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_mean_price_scalar_per_region(price_steps, geojson_path="data/nyc_zones.geojson", output_path="visualizations/price_scalar_map.png"):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from collections import defaultdict

    region_to_district = {
        0: "301", 1: "302", 2: "303", 3: "304", 4: "305", 5: "308",
        6: "309", 7: "316", 8: "317", 9: "306", 10: "307", 11: "310",
        12: "311", 13: "312", 14: "313", 15: "314", 16: "315", 17: "318"
    }
    district_codes = list(region_to_district.values())

    # Average price scalar per region
    price_totals = defaultdict(float)
    for step in price_steps:
        for i, v in step.items():
            price_totals[i] += v
    n_episodes = len(price_steps)
    mean_prices = {i: v / n_episodes for i, v in price_totals.items()}

   
    nyc_zones = gpd.read_file(geojson_path).to_crs(epsg=4326)
    bk_zones = nyc_zones[nyc_zones['districtcode'].astype(str).isin(district_codes)].copy()
    bk_zones["districtcode"] = bk_zones["districtcode"].astype(str)

    bk_zones["price_scalar"] = bk_zones["districtcode"].map({
        region_to_district[r]: mean_prices[r]
        for r in mean_prices if r in region_to_district
    }).fillna(0)


    fig, ax = plt.subplots(figsize=(12, 10))
    bk_zones.plot(
        column="price_scalar",
        cmap="Greens",
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax
    )

    for idx, row in bk_zones.iterrows():
        if row["price_scalar"] > 0 and row["geometry"].centroid.is_valid:
            centroid = row["geometry"].centroid
            ax.text(
                centroid.x, centroid.y,
                f"{row['price_scalar']:.2f}",
                ha="center", va="center", fontsize=9, color="black"
            )

    ax.set_title("Mean Price Scalar per Region", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
