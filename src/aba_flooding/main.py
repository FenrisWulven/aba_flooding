# filepath: /home/rasmus/Desktop/School Local/42578 Advanced Business Analytics/aba_flooding/src/aba_flooding/main.py
import geopandas as gpd
import pydeck as pdk
import shapely.geometry as sg
import os
import streamlit as st
from src.aba_flooding import geo_utils as gu
from src.aba_flooding import vis_geo as vg
from src.aba_flooding import model as md

def main():
    """
    Main function to run the flood visualization application.
    Loads data, initializes the model, creates the visualization,
    and sets up interactive UI components.
    """
    # Setup page configuration
    st.set_page_config(page_title="Flood Visualization", layout="wide")
    st.title("Flood Prediction and Visualization Tool")
    
    # Load the stations
    try:
        stations = vg.load_terrain_data("stations.geojson")
        station_points = [gu.create_point(station.x, station.y) for station in stations.geometry]
        
    except Exception as e:
        st.error(f"Error loading station data: {e}")
        return

    # Load the model
    try:
        model = md.load_model("Survival")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create visualization components
    try:
        # Create the main map deck
        deck = vg.visualize_terrain("terrain.gpkg")
        
        # Create UI elements
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Add the slider for time control
            slider_time = vg.create_slider(deck)
            
        with col2:
            # Add checkbox for terrain visibility
            check_box_terrain = vg.create_check_box(deck)
            
        with col3:
            # Add search functionality
            search_bar = vg.create_search(deck)
        
        # Add the callback functions for interactive elements
        vg.add_slider_callback(deck, slider_time)
        vg.add_checkbox_callback(deck, check_box_terrain)
        vg.add_search_callback(deck, search_bar)
        
        # Display the map
        st.pydeck_chart(deck)
        
        # Add statistics or additional information panel
        with st.expander("Flood Prediction Details"):
            st.write("This section displays detailed prediction information.")
            # Add more detailed statistics or model outputs here
            
    except Exception as e:
        st.error(f"Error in visualization: {e}")

if __name__ == "__main__":
    main()