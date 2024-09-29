
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


# Function to calculate spherical distance
def spherical_dist(pos1, pos2, r=3958.75):
    pos1 = np.array(pos1, dtype=float)
    pos2 = np.array(pos2, dtype=float)
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

# Load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    return data

def calculate_distances(data):
    num_locations = len(data)
    dist_mat = pd.DataFrame(0, index=data['zip_code'], columns=data['zip_code'])
    for i in range(num_locations):
        for j in range(num_locations):
            dist_mat.iloc[i, j] = spherical_dist(
                [data.iloc[i, 2], data.iloc[i, 3]],  # latitude, longitude
                [data.iloc[j, 2], data.iloc[j, 3]]
            )
    return dist_mat

# Create model with different objectives
def create_model(distances, P, objective_type):
    num_locations = len(distances)
    model = pyo.ConcreteModel(name="Support Center Optimization")
    
    model.I = pyo.Set(initialize=range(1, num_locations + 1))
    model.J = pyo.Set(initialize=range(1, num_locations + 1))
    
    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.I, within=pyo.Binary)
    model.d = pyo.Param(model.I, model.J, initialize=lambda model, i, j: distances.iloc[i-1, j-1])
    
    # Objective functions
    if objective_type == 'P-Median':
        model.objective = pyo.Objective(
            expr=sum(model.x[i, j] * model.d[i, j] for i in model.I for j in model.J),
            sense=pyo.minimize
        )
    elif objective_type == 'K-Center':
        model.objective = pyo.Objective(
            expr=sum(model.x[i, j] * model.d[i, j] for i in model.I for j in model.J),
            sense=pyo.minimize
        )
    elif objective_type == 'MCLP':
        model.objective = pyo.Objective(
            expr=sum(model.x[i, j] * model.d[i, j] for i in model.I for j in model.J),
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
    
    return model

def main():
    st.title("🏭 Facility Location Selection")

    # Load default data
    file = 'Database.csv'
    data = load_data(file)

    # Allow users to edit the data
    st.write("Input data:")
    edited_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")



    # Check column names for compatibility
    if 'zip_code' not in edited_data.columns or 'latitude' not in edited_data.columns or 'longitude' not in edited_data.columns:
        st.error("CSV file must contain 'zip_code', 'latitude', and 'longitude' columns.")
        return

    # Convert edited DataFrame to lists for optimization model
    Zipcode1 = edited_data['zip_code'].tolist()
    latitude1 = edited_data['latitude'].tolist()
    longitude1 = edited_data['longitude'].tolist()

    # Calculate distance matrix
    dist_mat = calculate_distances(edited_data)
    st.write("Calculated distance matrix:")
    st.dataframe(dist_mat)

    # Select number of support centers and objective function
    P = st.slider("Select number of support centers to be built", min_value=1, max_value=len(dist_mat) , value=3)
    objective_type = st.selectbox(
        "Select objective function",
        options=['P-Median', 'K-Center', 'MCLP']
    )

    # Run the model
    if st.button("Run Model"):
        model = create_model(dist_mat, P, objective_type)
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

        #Display support centers
        st.write("Selected support center locations:")
        st.dataframe(support_centers,column_config={"value": "Support Center Locations"})
        

        for i in model.I:
            for j in model.J:
                if pyo.value(model.x[i, j]) == 1:
                    assignments[Zipcode1[i - 1]].append(Zipcode1[j - 1])

        # Display allocations
        st.write("Allocations:")
        assignment_df = pd.DataFrame([(k, v) for k, vals in assignments.items() for v in vals], columns=['Support Center', 'Locations served'])
        st.dataframe(assignment_df)

        # Prepare data for visualization
        result_data = edited_data[edited_data['zip_code'].isin(support_centers)]
        st.dataframe(result_data)
        fig = go.Figure()

        # Add scatter points for support centers
        fig.add_trace(go.Scattermapbox(
            lat=result_data['latitude'],
            lon=result_data['longitude'],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=result_data['zip_code'],
            name='Support Centers'
        ))

        # Add scatter points for other locations
        other_data = edited_data[~edited_data['zip_code'].isin(support_centers)]
        st.dataframe(other_data)
        fig.add_trace(go.Scattermapbox(
            lat=other_data['latitude'],
            lon=other_data['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            text=other_data['zip_code'],
            name='Served Locations'
        ))

        # Add arcs connecting locations to their assigned support centers
        for sc, locs in assignments.items():
            sc_lat, sc_lon = edited_data[edited_data['zip_code'] == sc][['latitude', 'longitude']].values[0]
            for loc in locs:
                loc_lat, loc_lon = edited_data[edited_data['zip_code'] == loc][['latitude', 'longitude']].values[0]
                distance = spherical_dist([sc_lat, sc_lon], [loc_lat, loc_lon])
                fig.add_trace(go.Scattermapbox(
                    lat=[sc_lat, loc_lat],
                    lon=[sc_lon, loc_lon],
                    mode='lines',
                    line=dict(width=1, color='black'),
                    name=f'Distance: {distance:.2f} miles'
                ))

        average_lat = edited_data['latitude'].mean()
        average_lon = edited_data['longitude'].mean()
        #fig.update_layout(mapbox_style="open-street-map", height=800, title="Location Assignments and Costs")
        fig.update_layout(mapbox_style="open-street-map", height=800, title="Location Assignments and Costs", mapbox=dict(center=dict(lat=edited_data['latitude'].mean(), lon=edited_data['longitude'].mean()), zoom=10))
        st.plotly_chart(fig)

        st.write(f"```\n{results}\n```")
        

if __name__ == "__main__":
    main()

