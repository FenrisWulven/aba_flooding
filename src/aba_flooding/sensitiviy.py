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
                
                # Calculate events for each threshold
                events_by_threshold = {}
                for threshold in thresholds:
                    events = calculate_events_for_threshold(wog_array, threshold)
                    events_by_threshold[threshold] = events
                    print(f"    Threshold {threshold}mm: {events} events")
                
                # Store results
                key = f"{station}_{soil_type}"
                all_results[key] = events_by_threshold
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process results and create visualizations
    plot_sensitivity_results(all_results, thresholds)
    
    return all_results

def plot_sensitivity_results(all_results, thresholds):
    """Create visualizations of sensitivity analysis results"""
    # Overall plot with all soil types
    plt.figure(figsize=(12, 8))
    
    for station_soil, events in all_results.items():
        # Extract values for plotting
        event_counts = [events.get(t, 0) for t in thresholds]
        plt.plot(thresholds, event_counts, marker='o', label=station_soil)
    
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel('Number of Events')
    plt.title('Sensitivity of Flooding Events to WOG Threshold')
    plt.grid(True, alpha=0.3)
    
    # Add log scale option if range is large (it will be :) )
    if max([max(events.values()) for events in all_results.values()]) / min([min(events.values()) for events in all_results.values() if min(events.values()) > 0]) > 20:
        plt.yscale('log')
        plt.title('Sensitivity of Flooding Events to WOG Threshold (Log Scale)')
    
    # Dont show legend
    if len(all_results) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_sensitivity_all.png", dpi=300)
    
    # Calculate average event reduction
    avg_reduction = {}
    reference_threshold = 5
    
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
            
        percent_changes = []
        for station_soil, events in all_results.items():
            if events.get(reference_threshold, 0) > 0:
                pct_change = (events.get(threshold, 0) - events.get(reference_threshold, 0)) / events.get(reference_threshold, 0) * 100
                percent_changes.append(pct_change)
        
        if percent_changes:
            avg_reduction[threshold] = np.mean(percent_changes)
    
    # Plot average percent change
    plt.figure(figsize=(10, 6))
    thresholds_without_ref = [t for t in thresholds if t != reference_threshold]
    pct_changes = [avg_reduction.get(t, 0) for t in thresholds_without_ref]
    
    plt.bar(thresholds_without_ref, pct_changes)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('WOG Threshold (mm)')
    plt.ylabel(f'Average % Change in Events (vs {reference_threshold}mm threshold)')
    plt.title('Average Effect of Changing WOG Threshold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis/wog_threshold_avg_change.png", dpi=300)
    
    # Create summary table
    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = 'Station_SoilType'
    summary_df.columns = [f'{t}mm' for t in thresholds]
    
    # Add percent change columns
    for threshold in thresholds:
        if threshold == reference_threshold:
            continue
        col_name = f'%Change_{threshold}mm'
        summary_df[col_name] = ((summary_df[f'{threshold}mm'] - summary_df[f'{reference_threshold}mm']) / summary_df[f'{reference_threshold}mm'] * 100).round(1)
    
    # Save summary table
    summary_df.to_csv("outputs/sensitivity_analysis/wog_threshold_summary.csv")
    print(f"Results saved to outputs/sensitivity_analysis/")
    
    return summary_df

if __name__ == "__main__":
    results = analyze_wog_sensitivity()
    
    # Print summary statistics
    print("\nSummary of event counts by threshold:")
    summary = {threshold: [] for threshold in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]}
    
    for station_soil, events in results.items():
        for threshold, count in events.items():
            summary[threshold].append(count)
    
    for threshold, counts in sorted(summary.items()):
        if counts:
            print(f"Threshold {threshold}mm: Avg={np.mean(counts):.1f}, Min={np.min(counts)}, Max={np.max(counts)}, Total={np.sum(counts)}")