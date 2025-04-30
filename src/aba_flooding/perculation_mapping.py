

# # Soil types in Denmark

# percolation_rates = {
#     "BK": 0.1,  # Danian bryozoan og corallian limestone
#     "BY": 0.0,  # Town
#     "DG": 0.3,  # Meltwater gravel
#     "DI": 0.05,  # Meltwater silt
#     "DL": 0.07,  # Meltwater clay
#     "DS": 0.2,  # Meltwater sand
#     "DV": 0.15,  # Alternating thin meltwater beds
#     "ED": 0.02,  # Eocene diatomite
#     "EE": 0.01,  # Eocene volcanic ash
#     "EK": 0.25,  # Aeolian dune sand
#     "ES": 0.2,  # Aeolian sand
#     "FG": 0.3,  # Freshwater gravel
#     "FHG": 0.28,  # Delta gravel
#     "FHL": 0.15,  # Delta clay
#     "FHS": 0.22,  # Delta sand
#     "FI": 0.1,  # Freshwater silt
#     "FJ": 0.05,  # Ocher and bog iron
#     "FK": 0.06,  # Tufa, bog-and lake marl
#     "FL": 0.08,  # Freshwater clay
#     "FP": 0.12,  # Freshwater gyttja
#     "FS": 0.25,  # Freshwater sand
#     "FT": 0.1,  # Freshwater peat
#     "FV": 0.14,  # Alternating thin freshwater beds
#     "G": 0.27,  # Gravel / sand and gravel
#     "GC": 0.02,  # Oligocene/Miocene/Pliocene brown coal
#     "GL": 0.01,  # Oligocene/Miocene/Pliocene mica clay
#     "GS": 0.03,  # Oligocene/Miocene/Pliocene mica sand
#     "GV": 0.04,  # Oligocene/Miocene/Pliocene alternating layers
#     "HG": 0.3,  # Saltwater gravel
#     "HI": 0.05,  # Saltwater silt
#     "HL": 0.1,  # Saltwater clay
#     "HP": 0.12,  # Saltwater gyttja
#     "HS": 0.22,  # Saltwater sand
#     "HSG": 0.18,  # Saltwater shell gravel
#     "HT": 0.1,  # Saltwater peat
#     "HV": 0.15,  # Alternating thin saltwater beds, marsh
#     "IA": 0.0,  # No access
#     "IT": 0.1,  # Freshwater peat
#     "K": 0.05,  # Chalk and limestone
#     "KMG": 0.2,  # Limey till, gravelly
#     "KML": 0.1,  # Limey till, clayey
#     "KMS": 0.15,  # Limey till, sandy
#     "KS": 0.25,  # Miocene quartz sand
#     "LL": 0.04,  # Eocene clay, plastic clay
#     "LRÅ": 0.0,  # Abandoned pit
#     "LSL": 0.02,  # Landslide
#     "MG": 0.2,  # Gravelly till
#     "MI": 0.1,  # Silty till
#     "ML": 0.08,  # Clayey till
#     "MS": 0.12,  # Sandy till
#     "MV": 0.14,  # Alternating thin till beds
#     "O": 0.01,  # Landfill
#     "OL": 0.03,  # Oligocene clay
#     "PKV": 0.02,  # Pre-Quaternary layers
#     "PL": 0.04,  # Selandian clay, Paleocene clay
#     "PS": 0.05,  # Selandian sand, Paleocene greensand
#     "QG": 0.3,  # Saltwater gravel
#     "QL": 0.1,  # Saltwater clay
#     "QS": 0.22,  # Saltwater sand
#     "RÅ": 0.0,  # Pit
#     "RL": 0.02,  # Eocene Røsnæs clay
#     "S": 0.2,  # Sand
#     "SK": 0.05,  # Campanian-Maastrichtian chalk
#     "SL": 0.04,  # Eocene Svind marl
#     "SØ": 0.3,  # Freshwater
#     "TA": 0.0,  # Technical and artificial construction
#     "TG": 0.3,  # Meltwater gravel
#     "TI": 0.05,  # Meltwater silt
#     "TL": 0.07,  # Meltwater clay
#     "TS": 0.2,  # Meltwater sand
#     "TV": 0.15,  # Alternating thin meltwater beds
#     "X": 0.0,  # Bed unknown, no information
#     "YG": 0.3,  # Saltwater gravel
#     "YL": 0.1,  # Saltwater clay
#     "YP": 0.12,  # Saltwater gyttja
#     "YS": 0.22,  # Saltwater sand
#     "ZG": 0.2,  # Glaciolacustrine gravel
#     "ZK": 0.05,  # Danian chalk / chalk and flint
#     "ZL": 0.03,  # Glaciolacustrine clay
#     "ZS": 0.2,  # Glaciolacustrine sand
#     "ZV": 0.14,  # Alternating thin glaciolacustrine beds
# }

# # Example usage:
# print(percolation_rates["DG"])  # Output: 0.3


# New dictionary with min and max percolation rates for Denmark soil types
percolation_rates_updated = {
    "BK": {
        "soil_type": "Danian Bryozoan og Corallian Limestone",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Limestone / Dolomite" from the CSV. The geological composition is equivalent, and the percolation rates align with the reference values of 1e-9 to 6e-6 m/s.
    },
    "BY": {
        "soil_type": "Town",
        "min": 1.00E-06,
        "max": 1.00E-03,
        # Guestimate: Urban areas consist of a mix of impermeable surfaces (e.g., roads, buildings) and permeable areas (e.g., parks, lawns). Permeability varies depending on the proportion of green spaces and the type of construction. A range from 1.00E-06 m/s (similar to compacted soils or partially paved areas) to 1.00E-03 m/s (comparable to sandy, permeable surfaces) reflects this variability.
    },
    "DG": {
        "soil_type": "Meltwater Gravel",
        "min": 1.00E-08,
        "max": 1.00E-06,
        # Direct 1-to-1 mapping: Matches "Sand/Gravelly Sand; Well Graded; Little to No Fines" from the CSV. Meltwater gravel is typically well-sorted with little fine material due to the sorting action of glacial meltwater. Percolation values align with the reference 1e-8 to 1e-6 m/s.
    },
    "DI": {
        "soil_type": "Meltwater Silt",
        "min": 1.00E-11,
        "max": 1.40E-08,
        # Direct 1-to-1 mapping: Matches "Siltstone" from the CSV. Although meltwater silt is not fully lithified like siltstone, it shares similar grain size and permeability characteristics. The percolation rates of 1e-11 to 1.4e-8 m/s match the reference range.
    },
    "DL": {
        "soil_type": "Meltwater Clay",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Assumed mapping: Matches "Inorganic Clay/Silty Clay/Sandy Clay; Low Plasticity" from the CSV. Meltwater clay typically has low plasticity due to its depositional environment and contains varying amounts of silt and sand. The percolation range of 1e-10 to 5e-8 m/s is appropriate.
    },
    "DS": {
        "soil_type": "Meltwater Sand",
        "min": 9.00E-07,
        "max": 5.00E-04,
        # Direct 1-to-1 mapping: Matches "Medium Sand" from the CSV. Meltwater sand typically consists of medium-grained particles due to the sorting action of glacial meltwater. The percolation values of 9e-7 to 5e-4 m/s align with the reference range.
    },
    "DV": {
        "soil_type": "Alternating Thin Meltwater Beds",
        "min": 7.00E-10,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin meltwater beds typically contain fine sand with varying amounts of silt and clay, creating varied but generally low permeability. Values of 7e-10 to 1e-6 m/s represent this range.
    },
    "ED": {
        "soil_type": "Eocene Diatomite",
        "min": 1.00E-11,
        "max": 4.70E-09,
        # Assumed mapping: Matches "Clay" from the CSV. Although diatomite is not clay, its extremely fine particle size and high porosity but low permeability make it functionally similar to clay in water movement. Percolation values of 1e-11 to 4.7e-9 m/s reflect this property.
    },
    "EE": {
        "soil_type": "Eocene Volcanic Ash",
        "min": 1.00E-10,
        "max": 1.00E-07,
        # Assumed mapping: Matches "Inorganic Clay; High Plasticity" from the CSV. Weathered volcanic ash often develops clay-like properties with high plasticity. The percolation range of 1e-10 to 1e-7 m/s is appropriate for this material.
    },
    "EK": {
        "soil_type": "Aeolian Dune Sand",
        "min": 1.00E-05,
        "max": 1.00E-02,
        # Direct 1-to-1 mapping: Matches "Sand; Clean; Good Aquifer" from the CSV. Aeolian dune sand is well-sorted, clean sand with excellent drainage properties, making it a good aquifer material. The percolation values of 1e-5 to 1e-2 m/s match the reference range perfectly.
    },
    "ES": {
        "soil_type": "Aeolian Sand",
        "min": 9.00E-07,
        "max": 6.00E-03,
        # Assumed mapping: Matches "Coarse Sand" from the CSV. While drifting sand can vary in grain size, it's typically coarser than dune sand and has excellent drainage properties. Percolation values of 9e-7 to 6e-3 m/s reflect this characteristic.
    },
    "FG": {
        "soil_type": "Freshwater Gravel",
        "min": 5.00E-04,
        "max": 5.00E-02,
        # Direct 1-to-1 mapping: Matches "Gravel/Sandy Gravel; Poorly Graded; Little to No Fines" from the CSV. Freshwater gravel deposits are typically poorly graded with little fine material due to fluvial sorting. The percolation range of 5e-4 to 5e-2 m/s aligns perfectly with reference values.
    },
    "FHG": {
        "soil_type": "Delta Gravel",
        "min": 3.00E-04,
        "max": 3.00E-02,
        # Direct 1-to-1 mapping: Matches "Gravel" from the CSV. Delta gravel deposits share characteristics with general gravel deposits but may include some stratification. The percolation values of 3e-4 to 3e-2 m/s represent this material well.
    },
    "FHL": {
        "soil_type": "Delta Clay",
        "min": 5.00E-09,
        "max": 1.00E-07,
        # Direct 1-to-1 mapping: Matches "Organic Clay/Silty Clay; Low Plasticity" from the CSV. Delta clay typically contains organic material and varying amounts of silt with generally low plasticity due to its depositional environment. The percolation range of 5e-9 to 1e-7 m/s is appropriate.
    },
    "FHS": {
        "soil_type": "Delta Sand",
        "min": 2.00E-07,
        "max": 2.00E-04,
        # Direct 1-to-1 mapping: Matches "Fine Sand" from the CSV. Delta sand is often fine-grained due to the decreasing energy environment of delta deposition. The percolation values of 2e-7 to 2e-4 m/s align with the reference range for fine sand.
    },
    "FI": {
        "soil_type": "Freshwater Silt",
        "min": 7.00E-10,
        "max": 7.00E-08,
        # Direct 1-to-1 mapping: Matches "Silt; Compacted" from the CSV. Freshwater silt deposits are often somewhat compacted due to overburden pressure and time. The percolation range of 7e-10 to 7e-8 m/s accurately represents this material.
    },
    "FJ": {
        "soil_type": "Ocher and Bog Iron",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Assumed mapping: Matches "Inorganic Silt; High Plasticity" from the CSV. Bog iron forms in waterlogged areas and often contains significant amounts of silt with iron oxide cementation, giving it low permeability similar to high-plasticity silt. Values of 1e-10 to 5e-8 m/s reflect this property.
    },
    "FK": {
        "soil_type": "Tufa, Bog-and Lake Marl",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Assumed mapping: Matches "Limestone / Dolomite" from the CSV. Marl and tufa are calcium carbonate-rich deposits similar to limestone in composition and permeability characteristics. The percolation range of 1e-9 to 6e-6 m/s is appropriate.
    },
    "FL": {
        "soil_type": "Freshwater Clay",
        "min": 8.00E-13,
        "max": 2.00E-09,
        # Assumed mapping: Matches "Marine Clay; Unweathered" from the CSV. While freshwater clay is not marine in origin, unweathered clay deposits share similar very low permeability characteristics regardless of depositional environment. Values of 8e-13 to 2e-9 m/s reflect this very low permeability.
    },
    "FP": {
        "soil_type": "Freshwater Gyttja",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Direct 1-to-1 mapping: Matches "Organic Clay; High Plasticity" from the CSV. Gyttja is organic-rich mud with high plasticity characteristics. The percolation values of 5e-10 to 1e-7 m/s align with the reference range for this material type.
    },
    "FS": {
        "soil_type": "Freshwater Sand",
        "min": 9.00E-07,
        "max": 6.00E-03,
        # Direct 1-to-1 mapping: Matches "Coarse Sand" from the CSV. Freshwater sand in lakes and rivers often contains coarser particles due to higher energy depositional environments. The percolation range of 9e-7 to 6e-3 m/s represents this material well.
    },
    "FT": {
        "soil_type": "Freshwater Peat",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Assumed mapping: Matches "Organic Clay; High Plasticity" from the CSV. While peat is not clay, it shares similar high organic content and generally low permeability characteristics with organic clay. Values of 5e-10 to 1e-7 m/s reflect this property.
    },
    "FV": {
        "soil_type": "Alternating Thin Freshwater Beds",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin freshwater beds typically contain fine sand with varying amounts of silt and clay, creating varied but generally low permeability. Values of 5e-9 to 1e-6 m/s represent this range.
    },
    "G": {
        "soil_type": "Gravel / Sand and Gravel",
        "min": 3.00E-04,
        "max": 3.00E-02,
        # Direct 1-to-1 mapping: Matches "Gravel" from the CSV. This is a straightforward match for general gravel deposits. The percolation values of 3e-4 to 3e-2 m/s align perfectly with the reference range.
    },
    "GC": {
        "soil_type": "Oligocene/Miocene/Pliocene Brown Coal",
        "min": 1.00E-13,
        "max": 2.00E-09,
        # Assumed mapping: Matches "Shale" from the CSV. While brown coal (lignite) is not shale, both are low-permeability sedimentary materials with similar water movement characteristics. The percolation range of 1e-13 to 2e-9 m/s reflects this very low permeability.
    },
    "GL": {
        "soil_type": "Oligocene/Miocene/Pliocene Mica Clay",
        "min": 1.00E-11,
        "max": 4.70E-09,
        # Direct 1-to-1 mapping: Matches "Clay" from the CSV. Mica clay is a specific type of clay, but its permeability characteristics align with general clay properties. The percolation values of 1e-11 to 4.7e-9 m/s match the reference range.
    },
    "GS": {
        "soil_type": "Oligocene/Miocene/Pliocene Mica Sand",
        "min": 3.00E-10,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Sandstone" from the CSV. While mica sand is not fully lithified like sandstone, older Tertiary sands often have sandstone-like permeability due to compaction and cementation. Values of 3e-10 to 6e-6 m/s reflect this range.
    },
    "GV": {
        "soil_type": "Oligocene/Miocene/Pliocene Alternating Layers",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Assumed mapping: Matches "Inorganic Silt; High Plasticity" from the CSV. Alternating layers from these periods typically include significant amounts of silt and clay with generally low permeability. The percolation range of 1e-10 to 5e-8 m/s represents this material.
    },
    "HG": {
        "soil_type": "Saltwater Gravel",
        "min": 4.00E-04,
        "max": 4.00E-03,
        # Direct 1-to-1 mapping: Matches "Alluvial Gravel/Sand" from the CSV. Marine gravel shares many characteristics with alluvial deposits, though it's formed in a different environment. The percolation values of 4e-4 to 4e-3 m/s align with the reference range.
    },
    "HI": {
        "soil_type": "Saltwater Silt",
        "min": 7.00E-10,
        "max": 7.00E-08,
        # Direct 1-to-1 mapping: Matches "Silt; Compacted" from the CSV. Marine silt is often somewhat compacted due to overburden pressure. The percolation range of 7e-10 to 7e-8 m/s accurately represents this material.
    },
    "HL": {
        "soil_type": "Saltwater Clay",
        "min": 8.00E-13,
        "max": 2.00E-09,
        # Direct 1-to-1 mapping: Matches "Marine Clay; Unweathered" from the CSV. This is a perfect match as the soil type directly corresponds to the reference material. The percolation values of 8e-13 to 2e-9 m/s align with the reference range.
    },
    "HP": {
        "soil_type": "Saltwater Gyttja",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Direct 1-to-1 mapping: Matches "Organic Clay; High Plasticity" from the CSV. Marine gyttja is organic-rich mud with high plasticity characteristics. The percolation values of 5e-10 to 1e-7 m/s align with the reference range for this material type.
    },
    "HS": {
        "soil_type": "Saltwater Sand",
        "min": 2.00E-07,
        "max": 2.00E-04,
        # Direct 1-to-1 mapping: Matches "Fine Sand" from the CSV. Marine sand is often fine-grained due to sorting in the marine environment. The percolation values of 2e-7 to 2e-4 m/s align with the reference range for fine sand.
    },
    "HSG": {
        "soil_type": "Saltwater Shell Gravel",
        "min": 5.00E-04,
        "max": 5.00E-02,
        # Assumed mapping: Matches "Gravel/Sandy Gravel; Poorly Graded; Little to No Fines" from the CSV. Shell gravel consists primarily of shell fragments that create a poorly graded gravel-like material with little fine content. Values of 5e-4 to 5e-2 m/s reflect this high permeability.
    },
    "HT": {
        "soil_type": "Saltwater Peat",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Assumed mapping: Matches "Organic Clay; High Plasticity" from the CSV. While peat is not clay, it shares similar high organic content and generally low permeability characteristics with organic clay. Values of 5e-10 to 1e-7 m/s reflect this property.
    },
    "HV": {
        "soil_type": "Alternating Thin Saltwater Beds, Marsh",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin marine beds typically contain fine sand with varying amounts of silt and clay, creating varied but generally low permeability. Values of 5e-9 to 1e-6 m/s represent this range.
    },
    "IA": {
        "soil_type": "No Access",
        "min": 0.0,
        "max": 0.0,
        # No match: "No Access" is not a soil type but an access classification. No appropriate match in the CSV as this is not a physical soil property.
    },
    "IT": {
        "soil_type": "Freshwater Peat",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Assumed mapping: Matches "Organic Clay; High Plasticity" from the CSV. While peat is not clay, it shares similar high organic content and generally low permeability characteristics with organic clay. Values of 5e-10 to 1e-7 m/s reflect this property.
    },
    "K": {
        "soil_type": "Chalk and Limestone",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Limestone / Dolomite" from the CSV. This is a straightforward match for chalk and limestone deposits. The percolation values of 1e-9 to 6e-6 m/s align with the reference range.
    },
    "KMG": {
        "soil_type": "Limey Till, Gravelly",
        "min": 2.55E-05,
        "max": 5.35E-04,
        # Direct 1-to-1 mapping: Matches "Sand/Gravelly Sand; Poorly Graded; Little to No Fines" from the CSV. Limey gravelly till has significant gravel content with a carbonate matrix, exhibiting permeability similar to poorly graded gravelly sand. Values of 2.55e-5 to 5.35e-4 m/s align with reference range.
    },
    "KML": {
        "soil_type": "Limey Till, Clayey",
        "min": 1.00E-10,
        "max": 1.00E-09,
        # Direct 1-to-1 mapping: Matches "Clay; Compacted" from the CSV. Limey clayey till is essentially compacted clay with carbonate content. The percolation values of 1e-10 to 1e-9 m/s match the reference range for compacted clay.
    },
    "KMS": {
        "soil_type": "Limey Till, Sandy",
        "min": 5.50E-09,
        "max": 5.50E-06,
        # Direct 1-to-1 mapping: Matches "Clayey Sand" from the CSV. Limey sandy till consists of sand with significant clay content (from the limestone), exhibiting permeability similar to clayey sand. Values of 5.5e-9 to 5.5e-6 m/s align with the reference range.
    },
    "KS": {
        "soil_type": "Miocene Quartz Sand",
        "min": 3.00E-10,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Sandstone" from the CSV. Older Miocene quartz sand often exhibits sandstone-like permeability due to age, compaction, and potential cementation. The percolation values of 3e-10 to 6e-6 m/s align with the reference range.
    },
    "LL": {
        "soil_type": "Eocene Clay, Plastic Clay",
        "min": 1.00E-10,
        "max": 1.00E-07,
        # Direct 1-to-1 mapping: Matches "Inorganic Clay; High Plasticity" from the CSV. Eocene plastic clay is known for its high plasticity characteristics. The percolation values of 1e-10 to 1e-7 m/s align perfectly with the reference range.
    },
    "LRÅ": {
        "soil_type": "Abandoned Pit",
        "min": 0.0,
        "max": 0.0,
        # No match: "Abandoned Pit" is not a soil type but a land use classification. No appropriate match in the CSV as this is not a physical soil property but a human-altered landscape feature.
    },
    "LSL": {
        "soil_type": "Landslide",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Landslide material is typically a mixed, disturbed material with varied composition but generally contains significant fine material. Values of 5e-9 to 1e-6 m/s represent this mixed material.
    },
    "MG": {
        "soil_type": "Gravelly Till",
        "min": 2.55E-05,
        "max": 5.35E-04,
        # Direct 1-to-1 mapping: Matches "Sand/Gravelly Sand; Poorly Graded; Little to No Fines" from the CSV. Gravelly till contains significant gravel in a mixed matrix, exhibiting permeability similar to poorly graded gravelly sand. Values of 2.55e-5 to 5.35e-4 m/s align with reference range.
    },
    "MI": {
        "soil_type": "Silty Till",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Direct 1-to-1 mapping: Matches "Inorganic Silt; High Plasticity" from the CSV. Silty till is dominated by silt particles and typically exhibits high plasticity. The percolation values of 1e-10 to 5e-8 m/s match the reference range.
    },
    "ML": {
        "soil_type": "Clayey Till",
        "min": 5.00E-10,
        "max": 5.00E-08,
        # Direct 1-to-1 mapping: Matches "Inorganic Clay/Silty Clay/Sandy Clay; Low Plasticity" from the CSV. Clayey till is a mixture of clay with varying amounts of silt and sand, typically with low plasticity. Values of 5e-10 to 5e-8 m/s reflect this range.
    },
    "MS": {
        "soil_type": "Sandy Till",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Direct 1-to-1 mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Sandy till typically contains fine sand with varying amounts of silt and clay, exhibiting slight plasticity. Values of 5e-9 to 1e-6 m/s align with reference range.
    },
    "MV": {
        "soil_type": "Alternating Thin Till Beds",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin till beds contain varying materials but typically include fine sand with silt and clay, creating varied but generally low permeability. Values of 5e-9 to 1e-6 m/s represent this range.
    },
    "O": {
        "soil_type": "Landfill",
        "min": 5.00E-08,
        "max": 1.00E-05,
        # Assumed mapping: Matches "Silty Sand" from the CSV. While landfill material is highly variable, it often develops permeability characteristics similar to silty sand due to the mixed nature of materials and compaction. Values of 5e-8 to 1e-5 m/s represent this material.
    },
    "OL": {
        "soil_type": "Oligocene Clay",
        "min": 1.00E-11,
        "max": 4.70E-09,
        # Direct 1-to-1 mapping: Matches "Clay" from the CSV. Oligocene clay is a specific type of clay, but its permeability characteristics align with general clay properties. The percolation values of 1e-11 to 4.7e-9 m/s match the reference range.
    },
    "PKV": {
        "soil_type": "Pre-Quaternary Layers",
        "min": 3.00E-10,
        "max": 6.00E-06,
        # Assumed mapping: Matches "Sandstone" from the CSV. Pre-Quaternary layers are typically older, consolidated sediments with permeability characteristics similar to sandstone. The percolation range of 3e-10 to 6e-6 m/s represents these materials.
    },
    "PL": {
        "soil_type": "Selandian Clay, Paleocene Clay",
        "min": 1.00E-11,
        "max": 4.70E-09,
        # Direct 1-to-1 mapping: Matches "Clay" from the CSV. Paleocene clay is a specific type of clay, but its permeability characteristics align with general clay properties. The percolation values of 1e-11 to 4.7e-9 m/s match the reference range.
    },
    "PS": {
        "soil_type": "Selandian Sand, Paleocene Greensand",
        "min": 3.00E-10,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Sandstone" from the CSV. Paleocene greensand often exhibits sandstone-like permeability due to age, compaction, and potential cementation. The percolation values of 3e-10 to 6e-6 m/s align with the reference range.
    },
    "QG": {
        "soil_type": "Saltwater Gravel",
        "min": 4.00E-04,
        "max": 4.00E-03,
        # Direct 1-to-1 mapping: Matches "Alluvial Gravel/Sand" from the CSV. Marine gravel shares many characteristics with alluvial deposits, though it's formed in a different environment. The percolation values of 4e-4 to 4e-3 m/s align with the reference range.
    },
    "QL": {
        "soil_type": "Saltwater Clay",
        "min": 8.00E-13,
        "max": 2.00E-09,
        # Direct 1-to-1 mapping: Matches "Marine Clay; Unweathered" from the CSV. This is a perfect match as the soil type directly corresponds to the reference material. The percolation values of 8e-13 to 2e-9 m/s align with the reference range.
    },
    "QS": {
        "soil_type": "Saltwater Sand",
        "min": 2.00E-07,
        "max": 2.00E-04,
        # Direct 1-to-1 mapping: Matches "Fine Sand" from the CSV. Marine sand is often fine-grained due to sorting in the marine environment. The percolation values of 2e-7 to 2e-4 m/s align with the reference range for fine sand.
    },
    "RÅ": {
        "soil_type": "Pit",
        "min": 0.0,
        "max": 0.0,
        # No match: "Pit" is not a soil type but a land use classification. No appropriate match in the CSV as this is not a physical soil property but a human-altered landscape feature.
    },
    "RL": {
        "soil_type": "Eocene Røsnæs Clay",
        "min": 1.00E-13,
        "max": 2.00E-09,
        # Direct 1-to-1 mapping: Matches "Shale" from the CSV. Røsnæs clay is a highly consolidated clay that behaves more like shale in terms of permeability. The percolation values of 1e-13 to 2e-9 m/s align with the reference range for shale.
    },
    "S": {
        "soil_type": "Sand",
        "min": 2.00E-07,
        "max": 2.00E-04,
        # Direct 1-to-1 mapping: Matches "Fine Sand" from the CSV. Generic sand in Denmark often consists primarily of fine sand. The percolation values of 2e-7 to 2e-4 m/s align perfectly with the reference range.
    },
    "SK": {
        "soil_type": "Campanian-Maastrichtian Chalk",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Limestone / Dolomite" from the CSV. Chalk is a type of limestone, making this a straightforward match. The percolation values of 1e-9 to 6e-6 m/s align with the reference range.
    },
    "SL": {
        "soil_type": "Eocene Svind Marl",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Assumed mapping: Matches "Limestone / Dolomite" from the CSV. Marl is a calcareous clay material with similarities to limestone in terms of composition and permeability. The percolation range of 1e-9 to 6e-6 m/s is appropriate for this material.
    },
    "SØ": {
        "soil_type": "Freshwater",
        "min": 0.0,
        "max": 0.0,
        # No match: "Freshwater" is not a soil type but a water body classification. No appropriate match in the CSV as this represents open water rather than a soil or sediment type.
    },
    "TA": {
        "soil_type": "Technical and Artificial Construction",
        "min": 0.0,
        "max": 0.0,
        # No match: "Technical and Artificial Construction" is not a natural soil type but represents human-made structures. Permeability would vary widely depending on construction materials and design. No appropriate match in the CSV.
    },
    "TG": {
        "soil_type": "Meltwater Gravel",
        "min": 4.00E-05,
        "max": 4.00E-03,
        # Direct 1-to-1 mapping: Matches "Sand/Gravel; Well Graded; No Fines" from the CSV. Meltwater gravel is typically well-graded with minimal fines due to the sorting action of glacial meltwater. The percolation values of 4e-5 to 4e-3 m/s align with the reference range.
    },
    "TI": {
        "soil_type": "Meltwater Silt",
        "min": 7.00E-10,
        "max": 7.00E-08,
        # Direct 1-to-1 mapping: Matches "Silt; Compacted" from the CSV. Meltwater silt is often somewhat compacted due to overburden pressure and time. The percolation range of 7e-10 to 7e-8 m/s accurately represents this material.
    },
    "TL": {
        "soil_type": "Meltwater Clay",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Assumed mapping: Matches "Inorganic Clay/Silty Clay/Sandy Clay; Low Plasticity" from the CSV. Meltwater clay typically has low plasticity due to its depositional environment and contains varying amounts of silt and sand. The percolation range of 1e-10 to 5e-8 m/s is appropriate.
    },
    "TS": {
        "soil_type": "Meltwater Sand",
        "min": 9.00E-07,
        "max": 5.00E-04,
        # Direct 1-to-1 mapping: Matches "Medium Sand" from the CSV. Meltwater sand typically consists of medium-grained particles due to the sorting action of glacial meltwater. The percolation values of 9e-7 to 5e-4 m/s align with the reference range.
    },
    "TV": {
        "soil_type": "Alternating Thin Meltwater Beds",
        "min": 7.00E-10,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin meltwater beds typically contain fine sand with varying amounts of silt and clay, creating varied but generally low permeability. Values of 7e-10 to 1e-6 m/s represent this range.
    },
    "X": {
        "soil_type": "Bed Unknown, No Information",
        "min": 0.0,
        "max": 0.0,
        # No match: "Bed Unknown" is not a soil type but an information classification. No appropriate match in the CSV as this represents a lack of data rather than a physical soil property.
    },
    "YG": {
        "soil_type": "Saltwater Gravel",
        "min": 5.00E-04,
        "max": 5.00E-02,
        # Direct 1-to-1 mapping: Matches "Gravel/Sandy Gravel; Well Graded; Little to No Fines" from the CSV. Marine gravel is often well-graded with little fine material due to wave action sorting. The percolation values of 5e-4 to 5e-2 m/s align with the reference range.
    },
    "YL": {
        "soil_type": "Saltwater Clay",
        "min": 8.00E-13,
        "max": 2.00E-09,
        # Direct 1-to-1 mapping: Matches "Marine Clay; Unweathered" from the CSV. This is a perfect match as the soil type directly corresponds to the reference material. The percolation values of 8e-13 to 2e-9 m/s align with the reference range.
    },
    "YP": {
        "soil_type": "Saltwater Gyttja",
        "min": 5.00E-10,
        "max": 1.00E-07,
        # Direct 1-to-1 mapping: Matches "Organic Clay; High Plasticity" from the CSV. Marine gyttja is organic-rich mud with high plasticity characteristics. The percolation values of 5e-10 to 1e-7 m/s align with the reference range for this material type.
    },
    "YS": {
        "soil_type": "Saltwater Sand",
        "min": 2.00E-07,
        "max": 2.00E-04,
        # Direct 1-to-1 mapping: Matches "Fine Sand" from the CSV. Marine sand is often fine-grained due to sorting in the marine environment. The percolation values of 2e-7 to 2e-4 m/s align with the reference range for fine sand.
    },
    "ZG": {
        "soil_type": "Glaciolacustrine Gravel",
        "min": 4.00E-05,
        "max": 4.00E-03,
        # Assumed mapping: Matches "Sand/Gravel; Well Graded; No Fines" from the CSV. Glaciolacustrine gravel is typically well-graded with minimal fines due to the sorting in glacial lake environments. Values of 4e-5 to 4e-3 m/s represent this material.
    },
    "ZK": {
        "soil_type": "Danian Chalk / Chalk and Flint",
        "min": 1.00E-09,
        "max": 6.00E-06,
        # Direct 1-to-1 mapping: Matches "Limestone / Dolomite" from the CSV. Danian chalk is a type of limestone, making this a straightforward match. The percolation values of 1e-9 to 6e-6 m/s align with the reference range.
    },
    "ZL": {
        "soil_type": "Glaciolacustrine Clay",
        "min": 1.00E-10,
        "max": 5.00E-08,
        # Assumed mapping: Matches "Inorganic Clay/Silty Clay/Sandy Clay; Low Plasticity" from the CSV. Glaciolacustrine clay typically has low plasticity due to its depositional environment and contains varying amounts of silt and sand. Values of 1e-10 to 5e-8 m/s represent this material.
    },
    "ZS": {
        "soil_type": "Glaciolacustrine Sand",
        "min": 1.00E-08,
        "max": 5.00E-06,
        # Direct 1-to-1 mapping: Matches "Silty Sand" from the CSV. Glaciolacustrine sand typically contains significant silt due to the low-energy depositional environment of glacial lakes. The percolation values of 1e-8 to 5e-6 m/s align with the reference range.
    },
    "ZV": {
        "soil_type": "Alternating Thin Glaciolacustrine Beds",
        "min": 5.00E-09,
        "max": 1.00E-06,
        # Assumed mapping: Matches "Inorganic Silty Fine Sand/Clayey Fine Sand; Slight Plasticity" from the CSV. Alternating thin glaciolacustrine beds typically contain fine sand with varying amounts of silt and clay. Values of 5e-9 to 1e-6 m/s represent this mixed material.
    },
}

# Example usage:
print(percolation_rates_updated["DG"])  # Output: {'soil_type': 'Meltwater Gravel', 'min': 1e-08, 'max': 1e-06}