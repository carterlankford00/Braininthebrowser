import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tqdm import tqdm
from scipy import io
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# Where electrodes are located in 3D space
# Identify which electrode is which and what area of the brain it is located in
# Make the brain more translucent/opacity
# Remove linear graph and input polar graph with clockish data circadian rhythms

# alter heatmap to display ECOG pattern
# 

data_coord = io.loadmat('Channel_Coordinates (2).mat')
new_coordinates = data_coord['coords'] 
def generate_fig(GEOM, color=(0.8, 0.8, 0.8), target_time=0.0376, motif_index=0):    
            
    # Create mesh
    meshes = []
    for hemisphere in ['lh', 'rh']:
        meshes.append(
                go.Mesh3d(
                        # x, y, and z give the positions of the vertices
                        x=GEOM[hemisphere]['vert'][:,0],
                        y=GEOM[hemisphere]['vert'][:,1],
                        z=GEOM[hemisphere]['vert'][:,2],

                        # i, j and k give the vertex indices of the triangles
                        i=GEOM[hemisphere]['tri'][:,0],
                        j=GEOM[hemisphere]['tri'][:,1],
                        k=GEOM[hemisphere]['tri'][:,2], 

                        # Intensity of each vertex, which will be interpolated and color-coded
                        #vertexcolor=brain_color_rgba[0],
                        showscale=True,
                        color='rgb(0.8,0.8,0.8)',
                        opacity=0.05, #previously 1.0

                        hoverinfo='skip',

                        lighting={
                            'ambient': 0.4225,
                            'diffuse': 0.6995,
                            'specular': 0.333,
                        },
                    )
        )
    
    axis = {'showbackground': False,
            'showline': False,
            'zeroline': False,
            'showgrid': False,
            'showticklabels': False,
            'title': ''}

    layout = {
        'title': 'Subject with ECoG',
        'width': 1600,
        'height': 900,
        'showlegend': True,
        'scene': {'xaxis': axis,
                  'yaxis': axis,
                  'zaxis': axis},
        'hovermode': 'closest',
    }                                             
    
   
       
    
    #define all_motifs and plot data
    #ensure axis depict time and channels
    #work with motif 0 first
    data_motifs=io.loadmat('Motifs.mat')
    all_motifs=data_motifs['motif']
    heatmap_data = all_motifs[0,:,:]
    time_intervals = np.arange(0, 0.2304, 0.0009)
    
    motif_channel_average = np.mean(all_motifs, axis=2)
    motif_channel_average_min = np.min(motif_channel_average,axis=1)[:,None]
    motif_channel_average_max = np.max(motif_channel_average,axis=1)[:,None]
    normalized_colors = (motif_channel_average-motif_channel_average_min) / (motif_channel_average_max - motif_channel_average_min)


    text_labels=[str(i+0) for i in range(len(new_coordinates))]
    
    
    # MAKE 2 SUBPLOTS, LEFT FOR BRAIN AND RIGHT FOR FHEAT MAP
       
    fig_3d = make_subplots(rows=1, cols=2, 
                        specs=[[{'type':'scene'}, {"type":"heatmap"}]],
                        horizontal_spacing = 0.1,
                        vertical_spacing = 0.0227,
                        column_width=[0.7,0.3],
                        print_grid=True)
    
    matrices_variable = all_motifs
    centers_of_mass_all_matrices = []
    means_of_all_matrices = []

    for matrix in matrices_variable:
        centers_of_mass = []
        means = []
        for row_values in matrix:
            # Calculate the center of mass
            center_of_mass = np.average(range(len(row_values)), weights=row_values)
            centers_of_mass.append(center_of_mass)
            row_mean = np.mean(row_values)
            means.append(row_mean)

        centers_of_mass_all_matrices.append(centers_of_mass)
        means_of_all_matrices.append(means)

    means_of_matrices_np = [np.array(matrix) for matrix in means_of_all_matrices]
    
    rows_of_top_values_all_matrices = []

    for matrix in means_of_matrices_np:
        # Flatten the matrix into a 1D array
        flat_matrix = matrix.flatten()

        # Sort the values and their indices based on the values
        sorted_indices = np.argsort(flat_matrix)
        sorted_values = flat_matrix[sorted_indices]

        # Calculate the number of elements representing the top 10%
        top_10_percent = int(0.1 * len(sorted_values))

        # Get the indices of the top 10% values
        top_indices = sorted_indices[-top_10_percent:]

        # Reshape the top indices to match the original matrix shape
        original_shape = matrix.shape
        top_indices_2d = np.unravel_index(top_indices, original_shape)

        # Get the column indices of the top values
        rows_of_top_values = top_indices_2d[0]

        # Append the column indices to the list
        rows_of_top_values_all_matrices.append(rows_of_top_values)

      
    
    # PLOT BRAIN MESH INTO LEFT SUBPLOT
    
    for mesh in meshes:
      
        fig_3d.add_trace(mesh,row=1, col=1,)
    
    #MAKE SCATTER PLOT WITH CORRESPONDING COLORS
        
    for rows_of_top_values in rows_of_top_values_all_matrices:
        num_points = new_coordinates.shape[0]
        all_colors=np.zeros((new_coordinates.shape[0],4))
        all_colors[:,0:3] = [255,0,0]
        all_colors[:,3] = 0.2
        all_colors[rows_of_top_values,3] = 1
        all_sizes = np.ones(num_points) * 10  # Default size for all markers
        top_values_count = len(rows_of_top_values)
        # Define marker sizes based on order in the list
        for i, row_index in enumerate(rows_of_top_values):
            all_sizes[row_index] = 10 + (1.3*i)  # Adjust size based on index in rows_of_top_values
            all_rgba = ['rgba({},{},{},{})'.format(int(r), int(g), int(b), alpha) for r, g, b, alpha in all_colors]    
            scatter_trace = go.Scatter3d(
                visible=False,
                x=new_coordinates[:,0],
                y=new_coordinates[:,1],
                z=new_coordinates[:,2],
                mode='markers+text',
                text=text_labels,
                textposition='bottom center',
                marker=dict(
                    size=all_sizes,
                    color= all_rgba,
                    #colorscale='magma',
                    cmin=0,
                    cmax=1,
                    colorbar=None

                ),
            )
        '''
        print(scatter_trace.marker.color)
        print(scatter_trace.marker.color.shape)
        scatter_trace.marker.color[rows_of_top_values] = "midnightblue"  # Set color for rows of top values
        scatter_trace.marker.color[rows_of_top_values] = np.array(scatter_trace.marker.color[rows_of_top_values])[:, :-1] + (1,)  # Set full opacity for rows of top values
        '''

        
     # ADD THE SCATTER PLOT TO THE LEFT SUBPLOT ON TOP OF THE BRAIN
    
        fig_3d.add_trace(scatter_trace, row=1, col=1)
    '''
    dict(
    title=None,
    tickvals=[0, 1],
    ticktext=['Min Value', 'Max Value'],
    lenmode='fraction',
    len=0.2,  # Adjust the length as needed
    thicknessmode='pixels',
    thickness=25  # Adjust the thickness as needed
    '''
    
    # CREATE HEATMAP FOR ALL MOTIFS
    for i in range(all_motifs.shape[0]):
        heatmap_trace = go.Heatmap(z=all_motifs[i], x=time_intervals, showscale=True, colorscale='magma', visible= False,
                                colorbar=dict(
                                   title="Channel Intensity (a.u.)",
                                   lenmode='fraction', 
                                   len=0.5,  # Adjust the length as needed
                                   thicknessmode='pixels',  # Use pixels for thickness
                                   thickness=20  # Adjust the thickness as needed
                                 
                                ))
        
     # ADD HEATMAP TO RIGHT SUBPLOT, THIS PLOTS ALL OF THE SUBPLOTS ON TOP OF EACH OTHER, BUT STARTS WITH A BLANK HEATMAP AS THE START AND MAKES ALL OF THE OTHERS NOT VISIBLE  
        fig_3d.add_trace(heatmap_trace, row=1, col=2) 
    
    fig_3d.update_xaxes(range=[0,.2304], title_text='Time (seconds)', row=1, col=2)  # Update the x-axis title
    fig_3d.update_yaxes(title_text='Channels', row=1, col=2)   
    fig_3d.update_layout(layout)

    
    # CREATE DROPDOWN MENU, ENSURES THAT BOTH TRACES OF THE BRAIN AND THE SCATTER PLOT REMAIN ON THE PAGE WHEN SWITCHING BETWEEN MOTIFS   
    layout['updatemenus'] = [
        dict(
            buttons=[dict(args=[{"visible": [True, True] + [(True if i == j else False) for j in (range(all_motifs.shape[0]))] + [(True if i == j else False) for j in (range(all_motifs.shape[0]))]}],
                          label=f"Motif {i}",
                          method="update")
                     for i in range(all_motifs.shape[0])],
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]    
    
    # Update the annotation for the dropdown
    layout['annotations'] = [
        dict(text="Select Motif:", showarrow=False, x=0, y=1.085, yref="paper", align="left")
    ]  
 
        
    fig_3d.update_layout(layout, showlegend = False)
    
    
    return fig_3d
   