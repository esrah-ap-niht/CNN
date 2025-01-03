# Import packages 
import numpy as np
from skimage import io
from dash import Dash, html, Input, Output, callback, dcc, State, ctx, dash_table
from dash.exceptions import PreventUpdate
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash_canvas.utils.io_utils import array_to_data_url

import dash_bootstrap_components as dbc
import plotly.express as px
import os
from tkinter import filedialog
from tkinter import *
import gc
from glob import glob
from os.path import join
from ipywidgets import widgets
from PIL import Image
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import dash_daq as daq
import json
from skimage import draw, morphology
from scipy import ndimage
from PIL import ImageColor
import cv2

##############################################################################################################
#### Garbage collection and tkinter settings 
# tkinter is Python's built in GUI for selecting local files, folders, etc. 
##############################################################################################################
gc.enable()

try: 
    root = Tk()
    root.withdraw()
    root.attributes('-topmost',1)
except:
    pass 


# This function was copied from dash_canvas.utils because I was unable to import the function. 
def _indices_of_path(path, scale=1):
    """
    Retrieve pixel indices (integer values). 

    Parameters
    ----------

    path: SVG-like path formatted as JSON string
        The path is formatted like
        ['M', x0, y0],
        ['Q', xc1, yc1, xe1, ye1],
        ['Q', xc2, yc2, xe2, ye2],
        ...
        ['L', xn, yn]
        where (xc, yc) are for control points and (xe, ye) for end points.

    Notes
    -----

    I took a weight of 1 and it seems fine from visual inspection.
    """
    rr, cc = [], []
    for (Q1, Q2) in zip(path[:-2], path[1:-1]):
        # int(round()) is for Python 2 compatibility
        inds = draw.bezier_curve(int(round(Q1[-1] / scale)), 
                                 int(round(Q1[-2] / scale)), 
                                 int(round(Q2[2] / scale)), 
                                 int(round(Q2[1] / scale)), 
                                 int(round(Q2[4] / scale)), 
                                 int(round(Q2[3] / scale)), 1)
        rr += list(inds[0])
        cc += list(inds[1])
    return rr, cc

# Initialize the app 
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Get user to select folder of images 
image_path = filedialog.askdirectory()

# Get user to select folder for annotations 
annotation_path = filedialog.askdirectory()

filelist = []
for ext in ('*.gif', '*.png', '*.jpg'):
   filelist.extend(glob(join(image_path, ext)))
 
annotation_colormap = px.colors.qualitative.Light24

# Specify possible brush sizes, contrast and annotation opacity ranges. 
brush_sizes = list(range(1,30)) # brush sizes of 1 to 30 pixels 
contrasts = list(range(0,100)) # contrast enhancement of 0 to 100 percent 
opacity_range = list(range(0,101)) # opacity of annotations in percent 

annotation_types = ["A", "B"]

# Specify annotation options for overwriting pixels 
overwrite_options = ["Only Blank Pixels",
                     "Only Annotated Pixels",
                     "All Pixels"]

# Specify default options 
DEFAULT_ATYPE = annotation_types[0]
DEFAULT_OVERWRITE = overwrite_options[0]
DEFAULT_BSIZE = 5
DEFAULT_CONTRAST = 0
DEFAULT_OPACITY = 100

# Specify layout settings 
canvas_width = 900

app.layout = html.Div(
    [
        # Main title   
        dcc.Markdown(children='# Popcorn - Simplified CNN Training',
                     style={'text-align':'center'}
                     ), 
        
        # Setup the three main tabs in the interface - setup, data labeling, and training. 
        dcc.Tabs([
            #### Setup tab - beginning 
            dcc.Tab(label='Setup / Select Project', children = [
                dbc.Row(
                    [
                        dcc.Input(
                            id="project_title_input",
                            type='text',
                            placeholder="Your Project Title Here..",
                            style = {'backgroundColor': 'green', 
                                     'text-align':'center',
                                     'margin-bottom': '5px',
                                     'margin-top': '5px'
                                     },
                            ), 
                            
                        dbc.Button(
                            "Create New Project", 
                            id="create-new-project-button", 
                            outline=False, 
                            color = 'success',
                            style = {
                                     'margin-bottom': '5px'
                                     },
                            ),
                        
                        dbc.Button(
                            "Load Previous Project", 
                            id="load-previous-project-button", 
                            outline=False, 
                            color = 'success',
                            style = {
                                     'margin-bottom': '5px'
                                     },
                            ),
                        
                        dcc.Markdown(children='Select Task', 
                                     style={'backgroundColor': 'green', 
                                            'text-align':'center',
                                            'margin-bottom': '5px'},
                                     ),
                        
                        dcc.RadioItems(
                            [
                            "Image Classification", 
                            "Image Labeling",
                            "Semantic Segmentation",
                            "Instance Segmentation"
                            ],
                            inline=False,
                            style={'backgroundColor': 'green', 
                                   'margin-bottom': '50px', 
                                   'text-align':'center'},
                            ),
                        
                        dbc.Button(
                            "Import Labels from CSV or TXT File", 
                            id="import-labels-button", 
                            outline=False, 
                            color = 'primary',
                            style={'margin-bottom': '5px'},
                            ),
                        
                        dcc.Input(
                            id="new-label-input",
                            type='text',
                            placeholder="New Label Here..",
                            style = {'backgroundColor': 'lightblue', 
                                     'text-align':'center',
                                     'margin-bottom': '5px'},
                            ), 
                                        
                        dbc.Button(
                            "Add New Label", 
                            id="add-new-labels-button", 
                            outline=False, 
                            color = 'primary',
                            style={'margin-bottom': '5px'},
                            ),
                        
                        dcc.Checklist(
                            [],
                            id = 'labels_checklist',
                            inline = False,
                            style = {'backgroundColor': 'lightblue', 
                                     'text-align':'center',
                                     'margin-bottom': '5px'},
                            ),
                        
                        dbc.Button(
                            'Remove Selected Label(s)', 
                            id = "remove-labels-button", 
                            outline = False, 
                            color = "primary",
                            style={'margin-bottom': '50px'}
                            ),
                        
                        dbc.Button(
                            'Save/Update Project', 
                            id = "save-project-button", 
                            outline = False, 
                            color = "danger",
                            style={'margin-bottom': '50px'}
                            ),
                        ])
                ]),
            #### Setup tab - end 
            
            #### Labeling tab - beginning 
            dcc.Tab(label='Label Data', children=[
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div([
                                        DashCanvas(
                                            id='canvas',
                                            lineWidth=20,
                                            width=canvas_width
                                        ),
                                        ], className="five columns"),
                                    html.Div(html.Img(id='my-iimage', width=canvas_width), className="five columns"),

                                ]
                            )
                        ),
                        
                        dbc.Col(
                            [
                                dcc.Markdown(children='Current Image Index'),

                                html.Br(),
                                
                                dcc.Slider(0, len(filelist)-1, 1,
                                value=0,
                                id='progress-slider'
                                ),
                                
                                html.Br(),
                                
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Row(
                                                dbc.Button(
                                                "Save and Previous Image", id="previous-image-button", outline=False, color = 'warning'
                                                ),
                                            align="center"
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Row(
                                                dbc.Button(
                                                    "Save and Next Image", id="next-image-button", outline=False, color = 'success'
                                                ),
                                            align="center"
                                            )
                                        ),
                                    ]
                                ),
                                
                                dcc.Markdown(children='Annotation Label'),
                                dcc.Dropdown(
                                    id="annotation-type-dropdown",
                                    options=[
                                        {"label": t, "value": t} for t in annotation_types
                                    ],
                                    value=DEFAULT_ATYPE,
                                    clearable=False,
                                ),
                                
                                dcc.Markdown(children='Annotation Opacity'),
                                
                                dcc.Slider(0, 255, 1,
                                marks=None,
                                value=255,
                                id='opacity-slider'
                                ),
                                                      
                                dcc.Markdown(children='Brush Size (px)'),
                                dcc.Slider(1, 30, 1,
                                value=5,
                                id='brush-size-slider'
                                ),
                              
                                dcc.Markdown(children='Image Contrast'),
                                dcc.Slider(-3, 3, 0.1, 
                                marks=None,
                                value=1,
                                id='contrast-slider'
                                ),
                                
                            ],
                        ),
                    ]
                ),
                
                
                
                
                
                
                
                
                
                
                
                
                
                ]),
            #### Labeling tab - end 
            
            #### Training tab - beginning  
            dcc.Tab(label='Train CNN', children=[])
            #### Training tab - end 
            ]),
        
            
        
        
        
        
        
    
    
    
    
    
    
    ]
)


@callback(
    Output('labels_checklist', 'options'),
    Input('add-new-labels-button', 'n_clicks'),
    Input('remove-labels-button', 'n_clicks'),
    State('new-label-input', 'value'),
    State('labels_checklist', 'value'),
    State('labels_checklist', 'options'),
    prevent_initial_call=True
    )
def change_labels( n_clicks_add, n_clicks_remove, new_label, labels_to_remove_list, labels_list):
    # Note: Plotly does not allow the same output to be affected by multiple callbacks. 
    # One work around is to have multiple inputs trigger the same callback, and then 
    # execute different pieces of code depending on which input triggered the callback. 
    # https://dash.plotly.com/duplicate-callback-outputs
    triggered_id = ctx.triggered_id

    if triggered_id == 'add-new-labels-button':
        
        try:
            labels_list.append(new_label)    
        except:
            labels_list = [new_label]

    elif triggered_id == 'remove-labels-button':
            
        for label in labels_to_remove_list:
            try:
                labels_list.remove(label)
            except:
                pass 
            
    return labels_list




@callback(
    Output('canvas', 'image_content'), 
    Input('progress-slider','value'),
    Input('contrast-slider', 'value')
    )
def update_canvas( index, alpha ):
    index = int(index)
    alpha = float(alpha) 
    
    # Note, you can pass images from a local drive to plotly in multiple ways as discussed in this thread. 
    # https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839
    # However, if you are passing filepaths, THEY MUST BE THE RELATIVE PATH and not absolute filepath 
    filename = os.path.relpath(filelist[index])
    image = cv2.imread( filename )
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=None)
    new_image = array_to_data_url(new_image)
    return new_image
    

@callback(
    Output('canvas', 'lineWidth'), 
    Input('brush-size-slider', 'value')
    )
def update_brush_size(size): 
    return size


@callback(
    Output('canvas', 'lineColor'), 
    Input('annotation-type-dropdown', 'value'),
    Input('opacity-slider', 'value'),
    )
def update_canvas_linecolor(label, alpha):
    index = annotation_types.index(label)
    RGBA = list(   ImageColor.getrgb(annotation_colormap[index])   )
    RGBA.append(alpha)
    RGBA = np.array(RGBA) / 255
    RGBA = mpl.colors.rgb2hex(RGBA, keep_alpha=True)

    return RGBA
    
    
@callback(
    Output('progress-slider', 'value'),
    Input('previous-image-button', 'n_clicks'),
    Input('next-image-button', 'n_clicks'),
    State('progress-slider', 'value'), 
    prevent_initial_call=True
    )
def update_image_index_slider(b1, b2, index):
    # Note: Plotly does not allow the same output to be affected by multiple callbacks. 
    # One work around is to have multiple inputs trigger the same callback, and then 
    # execute different pieces of code depending on which input triggered the callback. 
    # https://dash.plotly.com/duplicate-callback-outputs
    triggered_id = ctx.triggered_id

    if triggered_id == 'previous-image-button':
         index = index - 1 
         return index
     
    elif triggered_id == 'next-image-button':
         index = index + 1 
         return index
     

@callback(
    Output('my-iimage', 'src'),
    Input('canvas', 'json_data'),
    Input('progress-slider','value'),
    State('annotation-type-dropdown', 'value'),
    prevent_initial_call=True
    )
def update_annotation_data(string, image_index, label):
    
    filename = os.path.relpath(filelist[image_index])
    index = annotation_types.index(label)
  
    if string:
        # The following is a heavily edited version of dash_canvas.utils.parse_jsonstring
        # It was copied and edited because the original version was restricted to 
        # creating a binary mask of annotations. However, for image annotation I 
        # need a full RGB color mask. 
        
        # Get the XY shape of the image being annotated 
        shape = io.imread(filename, as_gray=True).shape
        
        # Make a 3 color mask of said image 
        mask = np.zeros( (shape[0], shape[1], 3) , dtype = np.int16)
        
        # Get the JSON string from the canvas component 
        data = json.loads(string)
        
        # For each seperate annotation...
        for obj in data['objects']:
            if obj['type'] == 'image':
                scale = obj['scaleX']
            elif obj['type'] == 'path':
                scale_obj = obj['scaleX']
                # Get the array indices of the path traveled 
                inds = _indices_of_path(obj['path'], scale=scale / scale_obj)
                radius = round(obj['strokeWidth'] / 2. / scale)
                
                # First create a binary mask of the annotation 
                mask_tmp = np.zeros( shape , dtype = bool)
                # Set the array indices affected to true
                mask_tmp[inds[0], inds[1]] = 1
                # Use the dilation tool to account for the brush size selected 
                mask_tmp = ndimage.binary_dilation(mask_tmp, morphology.disk(radius))
                # Now create a three color empty mask 
                mask_tmp_2 = np.zeros( (shape[0], shape[1], 3) , dtype = np.int16)
                # And assign the correct RGB values to the XY indices indicated in the binary mask 
                mask_tmp_2[mask_tmp == 1] = ImageColor.getrgb(annotation_colormap[index])  # hex color to RGB
                # Add this annotation to the overall mask 
                mask += mask_tmp_2
        
    else:
        raise PreventUpdate
       
    data = array_to_data_url((mask).astype(np.uint8))
    np.save(filename, mask, allow_pickle=True)

    return data


if __name__ == '__main__':
    app.run_server(port=8050)
    
