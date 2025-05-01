import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as mpatches

def load_geojson(file_path):
    return gpd.read_file(file_path)

terrain_data = load_geojson("data/raw/Sediment_wgs84.geojson")

soil_type_column = 'tsym'
if 'tsym' not in terrain_data.columns:
    print(f"Available columns in sediment data: {terrain_data.columns.tolist()}")
    soil_type_columns = [col for col in terrain_data.columns if 'type' in col.lower() or 'sym' in col.lower() or 'soil' in col.lower()]
    soil_type_column = soil_type_columns[0] if soil_type_columns else terrain_data.columns[0]

top_soil_types = terrain_data[soil_type_column].value_counts().head(5)
grand_total = terrain_data[soil_type_column].count()
top5_total = top_soil_types.sum()

top5_percentages = (top_soil_types / top5_total * 100).round(1)

total_percentages = (top_soil_types / grand_total * 100).round(1)

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(top_soil_types.index, top_soil_types.values, color='skyblue')

for i, (soil, count) in enumerate(top_soil_types.items()):
    ax.text(i, count + (max(top_soil_types) * 0.02), 
         f"{count}", 
         ha='center', va='bottom', fontweight='bold', color='black')
    ax.text(i, count/2, 
         f"{total_percentages[soil]}%\nof all data", 
         ha='center', va='center', fontweight='bold', color='white')

ax.set_title("Top 5 Soil Types Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Soil Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.3)

top5_percent_of_total = (top5_total / grand_total * 100).round(1)
legend_handles = [
    mpatches.Patch(color='skyblue', label=f'Top 5 soil types: {top5_total} samples'),
    mpatches.Patch(color='white', alpha=0.0, 
                   label=f'({top5_percent_of_total}% of all {grand_total} samples)'),
    mpatches.Patch(color='white', alpha=0.0, 
                   label=f'Other types: {grand_total - top5_total} samples'),
    mpatches.Patch(color='white', alpha=0.0, 
                   label=f'({100 - top5_percent_of_total}% of all samples)')
]

ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.savefig("soil_types_legend_right.png", dpi=300, bbox_inches='tight')
plt.show()