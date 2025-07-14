import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Region ↔ District mapping
region_to_district = {
    0: "301", 1: "302", 2: "303", 3: "304", 4: "305", 5: "308",
    6: "309", 7: "316", 8: "317", 9: "306", 10: "307", 11: "310",
    12: "311", 13: "312", 14: "313", 15: "314", 16: "315", 17: "318"
}
district_codes = list(region_to_district.values())


def plot_mean_rebalancing_per_region_agents(
    reb_steps_agent0, reb_steps_agent1, geojson_path, out0, out1
):
    """
    Plot mean rebalancing arrivals per region for both agents with shared color scale.
    """
    # Compute per-agent rebalance totals
    reb_agent_data = {}
    for agent_id, reb_steps in zip([0, 1], [reb_steps_agent0, reb_steps_agent1]):
        region_arrivals = defaultdict(float)
        for reb in reb_steps:
            for (o, d), flows in reb.items():
                region_arrivals[d] += sum(flows.values())
        n_episodes = len(reb_steps)
        reb_agent_data[agent_id] = {k: v / n_episodes for k, v in region_arrivals.items()}

    # Shared vmin/vmax for colormap
    all_vals = list(reb_agent_data[0].values()) + list(reb_agent_data[1].values())
    vmin, vmax = min(all_vals), max(all_vals)

    # Load and prepare map
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    gdf = gdf[gdf["districtcode"].astype(str).isin(district_codes)].copy()
    gdf["districtcode"] = gdf["districtcode"].astype(str)

    for agent_id, output_path in zip([0, 1], [out0, out1]):
        mean_data = reb_agent_data[agent_id]
        gdf["value"] = gdf["districtcode"].map({
            region_to_district[r]: mean_data[r] for r in mean_data if r in region_to_district
        }).fillna(0)

        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column="value", cmap="Blues", linewidth=0.8, edgecolor="black",
            legend=True, ax=ax, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Rebalancing Trips per Region (Agent {agent_id})", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_mean_accepted_demand_per_region_agents(
    demand_agent0, demand_agent1, geojson_path, out0, out1
):
    """
    Plot average accepted demand per origin region for both agents with shared color scale.
    """
    demand_agent_data = {}
    for agent_id, demand_steps in zip([0, 1], [demand_agent0, demand_agent1]):
        region_departures = defaultdict(float)
        for d in demand_steps:
            for (i, j), flow_dict in d.items():
                region_departures[i] += sum(flow_dict.values())
        n_episodes = len(demand_steps)
        demand_agent_data[agent_id] = {k: v / n_episodes for k, v in region_departures.items()}

    # Shared vmin/vmax
    all_vals = list(demand_agent_data[0].values()) + list(demand_agent_data[1].values())
    vmin, vmax = min(all_vals), max(all_vals)

    # Load and prepare map
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    gdf = gdf[gdf["districtcode"].astype(str).isin(district_codes)].copy()
    gdf["districtcode"] = gdf["districtcode"].astype(str)

    for agent_id, output_path in zip([0, 1], [out0, out1]):
        mean_data = demand_agent_data[agent_id]
        gdf["value"] = gdf["districtcode"].map({
            region_to_district[r]: mean_data[r] for r in mean_data if r in region_to_district
        }).fillna(0)

        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column="value", cmap="Purples", linewidth=0.8, edgecolor="black",
            legend=True, ax=ax, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Demand per Origin Region (Agent {agent_id})", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

def plot_mean_price_scalar_per_region_agents(
    price_agent0, price_agent1, geojson_path, out0, out1):
    """
    Plot average price scalar per region for both agents with shared color scale.
    price_agentX: list of dicts {region: price_scalar} over episodes.
    """
    price_data = {}
    for agent_id, price_steps in zip([0, 1], [price_agent0, price_agent1]):
        region_prices = defaultdict(float)
        for p in price_steps:
            for i, v in p.items():
                region_prices[i] += v
        n_episodes = len(price_steps)
        price_data[agent_id] = {k: v / n_episodes for k, v in region_prices.items()}
    all_vals = list(price_data[0].values()) + list(price_data[1].values())
    vmin, vmax = min(all_vals), max(all_vals)
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    gdf = gdf[gdf["districtcode"].astype(str).isin(district_codes)].copy()
    gdf["districtcode"] = gdf["districtcode"].astype(str)
    for agent_id, output_path in zip([0, 1], [out0, out1]):
        mean_data = price_data[agent_id]
        gdf["value"] = gdf["districtcode"].map({
            region_to_district[r]: mean_data[r]
            for r in mean_data if r in region_to_district
        }).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column="value", cmap="Greens", linewidth=0.8, edgecolor="black",
            legend=True, ax=ax, vmin=vmin, vmax=vmax
        )
        
        for idx, row in gdf.iterrows():
            if row["value"] > 0 and row["geometry"].centroid.is_valid:
                centroid = row["geometry"].centroid
                ax.text(
                    centroid.x, centroid.y,
                    f"{row['value']:.2f}",
                    ha="center", va="center", fontsize=9, color="black"
                )
                
        ax.set_title(f"Price Scalar per Region (Agent {agent_id})", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()