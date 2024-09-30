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

# Function to calculate spherical distance matrix using vectorization
def spherical_dist_matrix(positions, r=3958.75):
    positions = np.radians(positions)  # Convert to radians
    lat1, lon1 = np.meshgrid(positions[:, 0], positions[:, 0], indexing='ij')
    lat2, lon2 = np.meshgrid(positions[:, 1], positions[:, 1], indexing='ij')

    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)
    cos_lat_d = np.cos(lat1 - lat2)
    cos_lon_d = np.cos(lon1 - lon2)

    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

# Load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    return data

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
    required_columns = ['zip_code', 'population', 'latitude', 'longitude']
    if not all(col in data.columns for col in required_columns):
        st.error("CSV file must contain 'zip_code', 'population', 'latitude' and 'longitude' columns.")
        return

    # Convert filtered DataFrame to lists for optimization model
    Zipcode1 = data['zip_code'].tolist()
    population = data['population'].tolist()

    # Calculate distance matrix
    dist_mat = calculate_distances(data)

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
        result_data = data[data['zip_code'].isin(support_centers)]
        
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
        other_data = data[~data['zip_code'].isin(support_centers)]
        
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
            sc_lat, sc_lon = data[data['zip_code'] == sc][['latitude', 'longitude']].values[0]
            for loc in locs:
                loc_lat, loc_lon = data[data['zip_code'] == loc][['latitude', 'longitude']].values[0]
                distance = spherical_dist_matrix(np.array([[sc_lat, sc_lon], [loc_lat, loc_lon]]))
                fig.add_trace(go.Scattermapbox(
                    lat=[sc_lat, loc_lat],
                    lon=[sc_lon, loc_lon],
                    mode='lines',
                    line=dict(width=1, color='black'),
                    name=f'Distance: {distance[0, 1]:.2f} miles'
                ))

        # Set the map center and layout
        average_lat = data['latitude'].mean()
        average_lon = data['longitude'].mean()
        fig.update_layout(
            mapbox_style="open-street-map",
            height=800,
            title="Location Assignments and Costs",
            mapbox=dict(center=dict(lat=average_lat, lon=average_lon), zoom=10),
            showlegend=True
        )

        # Show map
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
