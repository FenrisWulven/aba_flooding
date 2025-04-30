import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

def load_and_clean_data(file_path):
    """Load and clean the station data from the specified file."""
    # Try reading with semicolon delimiter
    df = pd.read_csv(file_path, sep=';')
    
    # The first column contains multiple column names separated by semicolons
    # Split the first column into multiple columns
    first_col = df.columns[0]
    
    # If we have a single column with semicolon-separated values
    if len(df.columns) == 1 and ';' in first_col:
        # Split the header and rename columns
        new_cols = first_col.split(';')
        
        # Split each row's values and create a new dataframe
        split_data = []
        for _, row in df.iterrows():
            # Split the single string value into multiple columns
            values = row[first_col].split(';')
            # Make sure we have the right number of values
            if len(values) == len(new_cols):
                split_data.append(values)
        
        # Create a new DataFrame with the split data
        df = pd.DataFrame(split_data, columns=new_cols)
    
    # Keep only rows with valid year values in precip_past1h column
    numeric_df = df[df['precip_past1h'] != '-'].copy()
    
    # Convert to numeric values
    numeric_df['year'] = pd.to_numeric(numeric_df['precip_past1h'], errors='coerce')
    
    # Drop any rows where conversion failed
    numeric_df = numeric_df.dropna(subset=['year'])
    
    # Sort by year
    numeric_df = numeric_df.sort_values('year')
    
    return numeric_df

def create_gantt_chart(df, output_file='station_gantt_chart.png'):
    """Create a Gantt chart showing station establishment over time."""
    # Group stations by year
    year_counts = df['year'].value_counts().sort_index()
    
    # Get unique years and counts
    years = year_counts.index.tolist()
    counts = year_counts.values.tolist()
    
    # Create year groups - combine years with few stations
    if len(years) > 15:  # If more than 15 years, group some together
        # Find years with small counts
        threshold = np.percentile(counts, 50)  # Years with counts below median
        small_years = [y for y, c in zip(years, counts) if c < threshold]
        
        # Group adjacent small years
        year_groups = []
        current_group = []
        
        for year in sorted(years):
            if year in small_years and (not current_group or year <= current_group[-1] + 2):
                current_group.append(year)
            else:
                if current_group:
                    year_groups.append(current_group)
                    current_group = []
                if year in small_years:
                    current_group.append(year)
                else:
                    year_groups.append([year])
        
        if current_group:
            year_groups.append(current_group)
        
        # Create new year labels and counts
        new_years = []
        new_counts = []
        
        for group in year_groups:
            if len(group) == 1:
                new_years.append(str(group[0]))
                new_counts.append(year_counts[group[0]])
            else:
                label = f"{min(group)}-{max(group)}"
                new_years.append(label)
                new_counts.append(sum(year_counts[y] for y in group))
        
        years = new_years
        counts = new_counts
    else:
        years = [str(y) for y in years]
    
    # Create figure and plot
    plt.figure(figsize=(14, 8))
    
    # Define bar height and spacing
    bar_height = 0.6
    
    # Create the horizontal bars with custom positions
    y_positions = np.arange(len(years))
    
    # Plot horizontal bars (reversed to have chronological order from top to bottom)
    bars = plt.barh(y_positions[::-1], counts, height=bar_height, color='#d95f5f', alpha=0.8)
    
    # Set labels
    plt.yticks(y_positions[::-1], years[::-1])
    plt.xlabel('Number of Weather Stations Established', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.title('Weather Station Establishment Timeline', fontsize=16)
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{int(width)}", ha='left', va='center')
    
    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Set x-axis to start at 0
    plt.xlim(0, max(counts) * 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def main():
    """Main function to run the visualization script."""
    # Set the style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load and process the data
    file_path = '/Users/maks/Documents/GitHub/aba_flooding/data/raw/Station_availability.csv'  # Use the file you provided
    station_data = load_and_clean_data(file_path)
    
    print(f"Total stations with year data: {len(station_data)}")
    
    # Create the Gantt chart visualization
    gantt_chart = create_gantt_chart(station_data)
    
    print(f"Created Gantt chart visualization: {gantt_chart}")

if __name__ == "__main__":
    main()