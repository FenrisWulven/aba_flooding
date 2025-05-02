import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def calculate_events_for_threshold(wog_array, threshold):
    """Calculate number of events above threshold"""
    observed = (wog_array > threshold).astype(int)
    # Count transitions from 0 to 1 to identify unique events
    event_starts = np.diff(np.concatenate([[0], observed]))
    return np.sum(event_starts > 0)

def analyze_wog_sensitivity(station_id=None):
    # Create output directory
    os.makedirs("outputs/sensitivity_analysis", exist_ok=True)
    
    # Find all survival data files
    if station_id:
        data_files = [f"data/processed/survival_data_{station_id}.parquet"]
    else:
        data_files = glob.glob("data/processed/survival_data_*.parquet")
    
    print(f"Found {len(data_files)} station data files to analyze")
    
    # Dictionary to store results
    all_results = {}
    all_proportions = {}
    
    # Thresholds to test
    thresholds = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    
    for file_path in data_files:
        station = file_path.split('_')[-1].replace('.parquet', '')
        print(f"Processing station {station}...")
        
        try:
            # Load data
            df = pd.read_parquet(file_path)
            
            # Find WOG columns for this station
            wog_columns = [col for col in df.columns if col.startswith(f'{station}_WOG_')]
            
            if not wog_columns:
                print(f"  No WOG columns found for station {station}, skipping...")
                continue
                
            # Process each soil type
            for wog_col in wog_columns:
                soil_type = wog_col.split('_WOG_')[1]
                print(f"  Analyzing soil type: {soil_type}")
                
                # Get WOG values
                wog_array = df[wog_col].fillna(0).values
                total_timepoints = len(wog_array)
                
                # Calculate events and proportions for each threshold
                events_by_threshold = {}
                proportions_by_threshold = {}
                
                for threshold in thresholds:
                    # Calculate raw event count
                    events = calculate_events_for_threshold(wog_array, threshold)
                    events_by_threshold[threshold] = events
                    
                    # Calculate proportion of time points with events
                    time_above_threshold = np.sum(wog_array > threshold)
                    proportion = time_above_threshold / total_timepoints if total_timepoints > 0 else 0
                    proportions_by_threshold[threshold] = proportion
                    
                    print(f"    Threshold {threshold}mm: {events} events ({proportion:.4f} proportion)")
                
                # Store results
                key = f"{station}_{soil_type}"
                all_results[key] = events_by_threshold
                all_proportions[key] = proportions_by_threshold
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process results and create visualizations
    plot_sensitivity_results(all_results, all_proportions, thresholds)
    
    return all_results, all_proportions

def plot_sensitivity_results(all_results, all_proportions, thresholds):
    """Create visualizations of sensitivity analysis results"""
    # Plot 1: Proportion of time above threshold for all soil types
    plt.figure(figsize=(12, 8))
    
    for station_soil, proportions in all_proportions.items():
        # Extract values for plotting
        prop_values = [proportions.get(t, 0) for t in thresholds]
        plt.plot(thresholds, prop_values, marker='o', label=station_soil)
    
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel('Proportion of Time Above Threshold')
    plt.title('Sensitivity of Flooding Proportion to WOG Threshold')
    plt.grid(True, alpha=0.3)
    
    # Add log scale option if range is large
    if (max([max(props.values()) for props in all_proportions.values()]) / 
        max(0.001, min([min(props.values()) for props in all_proportions.values() if min(props.values()) > 0])) > 20):
        plt.yscale('log')
        plt.title('Sensitivity of Flooding Proportion to WOG Threshold (Log Scale)')
    
    # Don't show legend if too many lines
    if len(all_proportions) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_proportion_all.png", dpi=300)
    
    # Plot 2: Original raw event count (kept for comparison)
    plt.figure(figsize=(12, 8))
    
    for station_soil, events in all_results.items():
        # Extract values for plotting
        event_counts = [events.get(t, 0) for t in thresholds]
        plt.plot(thresholds, event_counts, marker='o', label=station_soil)
    
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel('Number of Events')
    plt.title('Sensitivity of Flooding Events to WOG Threshold')
    plt.grid(True, alpha=0.3)
    
    # Add log scale option if range is large
    if max([max(events.values()) for events in all_results.values()]) / max(1, min([min(events.values()) for events in all_results.values() if min(events.values()) > 0])) > 20:
        plt.yscale('log')
        plt.title('Sensitivity of Flooding Events to WOG Threshold (Log Scale)')
    
    # Don't show legend if too many lines  
    if len(all_results) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_events_all.png", dpi=300)
    
    # Calculate average proportional reduction
    reference_threshold = 5
    avg_proportion_change = {}
    avg_event_change = {}
    
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
            
        # Calculate changes in proportions
        proportion_changes = []
        for station_soil, props in all_proportions.items():
            if props.get(reference_threshold, 0) > 0:
                pct_change = (props.get(threshold, 0) - props.get(reference_threshold, 0)) / props.get(reference_threshold, 0) * 100
                proportion_changes.append(pct_change)
        
        if proportion_changes:
            avg_proportion_change[threshold] = np.mean(proportion_changes)
            
        # Calculate changes in event counts (kept for comparison)
        event_changes = []
        for station_soil, events in all_results.items():
            if events.get(reference_threshold, 0) > 0:
                pct_change = (events.get(threshold, 0) - events.get(reference_threshold, 0)) / events.get(reference_threshold, 0) * 100
                event_changes.append(pct_change)
        
        if event_changes:
            avg_event_change[threshold] = np.mean(event_changes)
    
    # Plot 3: Average percent change in proportions
    plt.figure(figsize=(10, 6))
    thresholds_without_ref = [t for t in thresholds if t != reference_threshold]
    proportion_pct_changes = [avg_proportion_change.get(t, 0) for t in thresholds_without_ref]
    
    plt.bar(thresholds_without_ref, proportion_pct_changes)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel(f'Average % Change in Proportion (vs {reference_threshold}mm threshold)')
    plt.title('Average Effect of Changing WOG Threshold on Flooding Proportion')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_proportion_avg_change.png", dpi=300)
    
    # Plot 4: Average percent change in events (kept for comparison)
    plt.figure(figsize=(10, 6))
    event_pct_changes = [avg_event_change.get(t, 0) for t in thresholds_without_ref]
    
    plt.bar(thresholds_without_ref, event_pct_changes)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel(f'Average % Change in Events (vs {reference_threshold}mm threshold)')
    plt.title('Average Effect of Changing WOG Threshold on Event Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_events_avg_change.png", dpi=300)
    
    # Create summary tables
    # Table 1: Raw event counts (kept for reference)
    events_df = pd.DataFrame(all_results).T
    events_df.index.name = 'Station_SoilType'
    events_df.columns = [f'{t}mm' for t in thresholds]
    
    # Add percent change columns for events
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
        col_name = f'Events_%Change_{threshold}mm'
        events_df[col_name] = ((events_df[f'{threshold}mm'] - events_df[f'{reference_threshold}mm']) / 
                              events_df[f'{reference_threshold}mm'] * 100).round(1)
    
    # Table 2: Proportions
    proportions_df = pd.DataFrame(all_proportions).T
    proportions_df.index.name = 'Station_SoilType'
    proportions_df.columns = [f'{t}mm' for t in thresholds]
    
    # Add percent change columns for proportions
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
        col_name = f'Prop_%Change_{threshold}mm'
        proportions_df[col_name] = ((proportions_df[f'{threshold}mm'] - proportions_df[f'{reference_threshold}mm']) / 
                                   proportions_df[f'{reference_threshold}mm'] * 100).round(1)
    
    # Save summary tables
    events_df.to_csv("outputs/sensitivity_analysis/wog_threshold_events_summary.csv")
    proportions_df.to_csv("outputs/sensitivity_analysis/wog_threshold_proportions_summary.csv")
    
    # Create combined summary with both metrics
    combined_df = pd.DataFrame(index=events_df.index)
    
    # Add essential columns from both dataframes
    for threshold in thresholds:
        combined_df[f'Events_{threshold}mm'] = events_df[f'{threshold}mm']
        combined_df[f'Prop_{threshold}mm'] = proportions_df[f'{threshold}mm'].round(4)
    
    # Add percent changes for reference comparisons
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
        combined_df[f'Events_%Change_{threshold}mm'] = events_df[f'Events_%Change_{threshold}mm']
        combined_df[f'Prop_%Change_{threshold}mm'] = proportions_df[f'Prop_%Change_{threshold}mm']
    
    combined_df.to_csv("outputs/sensitivity_analysis/wog_threshold_combined_summary.csv")
    
    print(f"Results saved to outputs/sensitivity_analysis/")
    
    return events_df, proportions_df, combined_df

if __name__ == "__main__":
    results, proportions = analyze_wog_sensitivity()
    
    # Print summary statistics
    print("\nSummary of proportion metrics by threshold:")
    proportion_summary = {threshold: [] for threshold in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]}
    
    for station_soil, props in proportions.items():
        for threshold, proportion in props.items():
            proportion_summary[threshold].append(proportion)
    
    for threshold, props in sorted(proportion_summary.items()):
        if props:
            print(f"Threshold {threshold}mm: " +
                  f"Avg Proportion={np.mean(props):.4f}, " +
                  f"Min={np.min(props):.4f}, " +
                  f"Max={np.max(props):.4f}, " +
                  f"Median={np.median(props):.4f}")
            
    print("\nSummary of event counts by threshold (for reference):")
    event_summary = {threshold: [] for threshold in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]}
    
    for station_soil, events in results.items():
        for threshold, count in events.items():
            event_summary[threshold].append(count)
    
    for threshold, counts in sorted(event_summary.items()):
        if counts:
            print(f"Threshold {threshold}mm: " +
                  f"Avg={np.mean(counts):.1f}, " +
                  f"Min={np.min(counts)}, " +
                  f"Max={np.max(counts)}, " +
                  f"Total={np.sum(counts)}")