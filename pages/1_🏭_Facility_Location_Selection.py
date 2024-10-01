# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 2024

@author: Nishank
"""

import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Function to calculate spherical distance matrix using the Haversine formula
def spherical_dist_matrix(positions, r=3958.75):  # r is the Earth's radius in miles
    positions = np.radians(positions)  # Convert degrees to radians

    # Extract latitude and longitude
    lat1, lon1 = np.meshgrid(positions[:, 0], positions[:, 0], indexing='ij')
    lat2, lon2 = np.meshgrid(positions[:, 1], positions[:, 1], indexing='ij')

    # Calculate deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Return distance matrix (multiply by Earth's radius in miles)
    return r * c

# Load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Calculate distance matrix
def calculate_distances(data):
    coordinates = data[['latitude', 'longitude']].to_numpy()
    dist_mat = spherical_dist_matrix(coordinates)
    return pd.DataFrame(dist_mat, index=data['zip_code'], columns=data['zip_code'])

def create_model(distances, population, P, objective_type, r_max=None):    
    num_locations = len(distances)
    model = pyo.ConcreteModel(name="Support Center Optimization")
    
    model.I = pyo.Set(initialize=range(1, num_locations + 1))
    model.J = pyo.Set(initialize=range(1, num_locations + 1))
    
    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.I, within=pyo.Binary)
    model.d = pyo.Param(model.I, model.J, initialize=lambda model, i, j: distances.iloc[i-1, j-1])
    
    # Introduce z_max for K-Center
    model.z_max = pyo.Var(within=pyo.NonNegativeReals)  # Maximum distance to be minimized
    
    model.population = pyo.Param(model.I, initialize=lambda model, i: population[i-1]) 
    
    # Objective functions
    if objective_type == 'P-Median':
        model.objective = pyo.Objective(
            expr=sum(model.x[i, j] * model.d[i, j] for i in model.I for j in model.J),
            sense=pyo.minimize
        )
    elif objective_type == 'K-Center':
        model.objective = pyo.Objective(expr=model.z_max, sense=pyo.minimize)
        
    elif objective_type == 'MCLP':
        model.objective = pyo.Objective(
            expr=sum(model.x[i, j] * model.population[j] for i in model.I for j in model.J),
            sense=pyo.maximize
        )
    
    # Constraints
    model.zip_code_assignment = pyo.Constraint(
        model.J,
        rule=lambda model, j: sum(model.x[i, j] for i in model.I) == 1
    )
    
    model.number_support_center = pyo.Constraint(
        expr=sum(model.y[i] for i in model.I) == P
    )
    
    model.restrict_zipcode_assignment = pyo.Constraint(
        model.I, model.J,
        rule=lambda model, i, j: model.x[i, j] <= model.y[i]
    )

    # Additional constraints for K-Center:
    if objective_type == 'K-Center':
        model.max_distance_constraint = pyo.Constraint(
            model.I, model.J,
            rule=lambda model, i, j: model.z_max >= model.x[i, j] * model.d[i, j]
        )

    # Additional constraints for MCLP:
    if objective_type == 'MCLP':
        model.coverage_constraint = pyo.Constraint(
            model.I,  
            rule=lambda model, i: sum(model.x[i, j] for j in model.J if model.d[i, j] <= r_max) >= model.y[i]
        )

    return model

def main():
    st.title("🏭 Facility Location Selection")

    # Load default data
    file = 'Database.csv'
    data = load_data(file)

    # Check column names for compatibility
    required_columns = ['zip_code', 'population', 'latitude', 'longitude', 'state_name']
    if not all(col in data.columns for col in required_columns):
        st.error("CSV file must contain 'zip_code', 'population', 'latitude', 'longitude', and 'state_name' columns.")
        return

    # Display loaded data
    st.write("Loaded data:")
    st.dataframe(data)

    # Filter data by state
    state_names = data['state_name'].unique()
    selected_state = st.selectbox("Select a state", state_names)
    filtered_data = data[data['state_name'] == selected_state]

    # Show filtered data
    st.write(f"Input data for state: {selected_state}")
    st.dataframe(filtered_data)

    # Convert filtered DataFrame to lists for optimization model
    Zipcode1 = filtered_data['zip_code'].tolist()
    population = filtered_data['population'].tolist()

    # Calculate distance matrix for filtered data
    dist_mat = calculate_distances(filtered_data)
    st.write(f"Calculated distances for state: {selected_state}")
    st.dataframe(dist_mat)
    
    # Select number of support centers and objective function
    P = st.number_input("Select number of support centers to be built", min_value=1, max_value=len(dist_mat), value=3)
    objective_type = st.selectbox(
        "Select objective function",
        options=['P-Median', 'K-Center', 'MCLP']
    )

    # Allow the user to set the maximum coverage radius for MCLP
    if objective_type == 'MCLP':
        r_max = st.number_input("Select maximum coverage radius (miles)", min_value=1, max_value=int(dist_mat.max().max()), value=5)
    else:
        r_max = None  # No need for r_max in other objective types

    # Run the model
    if st.button("Run Model"):
        model = create_model(dist_mat, population, P, objective_type, r_max)
        solver = pyo.SolverFactory('glpk')

        # Ensure the solver is correctly applied
        results = solver.solve(model, tee=True)

        # Check solver status
        if results.solver.status != pyo.SolverStatus.ok:
            st.error(f"Solver status: {results.solver.status}. Model may not be solved correctly.")
            return

        # Extract results
        support_centers = [Zipcode1[i - 1] for i in model.I if pyo.value(model.y[i]) == 1]
        assignments = {zipcode: [] for zipcode in support_centers}

        # Display support centers
        st.write("Selected support center locations:")
        st.dataframe(support_centers, column_config={"value": "Support Center Locations"})

        for i in model.I:
            for j in model.J:
                if pyo.value(model.x[i, j]) == 1:
                    assignments[Zipcode1[i - 1]].append(Zipcode1[j - 1])

        # Display allocations
        st.write("Allocations:")
        assignment_df = pd.DataFrame([(k, v) for k, vals in assignments.items() for v in vals], columns=['Support Center', 'Locations served'])
        st.dataframe(assignment_df)

        # Prepare data for visualization
        result_data = filtered_data[filtered_data['zip_code'].isin(support_centers)]
        
        fig = go.Figure()

        # Add scatter points for support centers
        fig.add_trace(go.Scattermapbox(
            lat=result_data['latitude'],
            lon=result_data['longitude'],
            mode='markers',
            marker=dict(size=10, color='blue'),
            hovertext=result_data['zip_code'].astype(str),
            hoverinfo='text',
            name='Support Centers'
        ))

        # Add scatter points for other locations
        other_data = filtered_data[~filtered_data['zip_code'].isin(support_centers)]
        
        fig.add_trace(go.Scattermapbox(
            lat=other_data['latitude'],
            lon=other_data['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            hovertext=other_data['zip_code'],
            hoverinfo='text',
            name='Served Locations'
        ))

        # Add arcs connecting locations to their assigned support centers
        for sc, locs in assignments.items():
            sc_lat, sc_lon = filtered_data[filtered_data['zip_code'] == sc][['latitude', 'longitude']].values[0]
            for loc in locs:
                loc_lat, loc_lon = filtered_data[filtered_data['zip_code'] == loc][['latitude', 'longitude']].values[0]
                distance = spherical_dist_matrix(np.array([[sc_lat, sc_lon], [loc_lat, loc_lon]]))
                fig.add_trace(go.Scattermapbox(
                    lat=[sc_lat, loc_lat],
                    lon=[sc_lon, loc_lon],
                    mode='lines',
                    line=dict(width=2, color='green'),
                    name=f'Distance: {distance[0, 1]:.2f} miles'
                ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=5,
            mapbox_center={"lat": filtered_data['latitude'].mean(), "lon": filtered_data['longitude'].mean()},
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
