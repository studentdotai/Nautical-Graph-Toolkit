import streamlit as st
from streamlit import session_state

from Data import Data
from MARITIME_MODULE import Miscellaneous, VISUALIZATION
import plotly.graph_objects as go
import pandas as pd

# Initialize Data class
data = Data()
mm_plot = VISUALIZATION()
mapbox_token = st.secrets.mapbox.access_token
mm_misc = Miscellaneous()
port = data.ports_msi_shp
port_df = mm_misc.shp_to_gdf(port)
port_acronym = data.ports_msi_acronyms
port_acronym_df = pd.read_csv(port_acronym)

def port_str(name):
	return (f"  {name['PORT_NAME']}, {name['COUNTRY']} (LAT: {int(name['LAT_DEG'])} \N{DEGREE SIGN} {name['LAT_MIN']}' {name['LAT_HEMI']}  LONG: {int(name['LONG_DEG'])} \N{DEGREE SIGN} {name['LONG_MIN']}' {name['LONG_HEMI']})  ")

def name_index(port_df, port_name):
	try:
		return port_df[port_df['PORT_NAME'] == port_name.upper()].index[0]
	except IndexError:
		return 0


# Get coordinates for selected ports
def get_port_coords(port_name):
	port_info = port_df[port_df['PORT_NAME'] == port_name].iloc[0]
	return port_info
	# 	{
	# 	'name': port_name,
	# 	'lat': port_info.geometry.y,
	# 	'lon': port_info.geometry.x,
	# 	'geometry': port_info.geometry,
	# 	'country' : port_info["COUNTRY"]
	# }

def update_port_traces(port_fig, dep_port, arr_port, port_df = port_df):
	# Get Dataframe Row of selected ports using boolean indexing
	dep_df = port_df[port_df['PORT_NAME'] == dep_port]
	arr_df = port_df[port_df['PORT_NAME'] == arr_port]
	# Get index of selected ports
	arr_port_idx = mm_plot.get_trace_index(port_fig, arr_port)
	dep_port_idx = mm_plot.get_trace_index(port_fig, dep_port)
	st.write(arr_port_idx, dep_port_idx)

	# Verify we have valid data before proceeding
	if not dep_df.empty and not arr_df.empty:

		# Store in session state (DF=>Series)
		st.session_state['departure_port'] = dep_df.iloc[0]
		st.session_state['arrival_port'] = arr_df.iloc[0]

		# Reset color to black
		port_fig.data[0].marker.color = colors = ['black'] * len(port_fig.data[0].text)

		active_color = []
		# Update specific port traces by finding their index
		for trace in enumerate(port_fig.data[0].text):
			if dep_port in trace[1]:
				active_color.append('blue')
			elif arr_port in trace[1]:
				active_color.append('red')
			else:
				active_color.append('black')
		port_fig.data[0].marker.color = active_color


		# Get coordinates of selected ports
		dep_coords = st.session_state['departure_port'] = get_port_coords(dep_port)
		arr_coords = st.session_state['arrival_port'] = get_port_coords(arr_port)
		latitude = arr_coords.geometry.y
		longitude = arr_coords.geometry.x


		# Update layout to center map on selected ports
		port_fig.update_layout(
			mapbox=dict(
				center=dict(
					lat= dep_coords.geometry.y,
					lon= dep_coords.geometry.x
				),
				zoom = 7,
				bearing = 0
			)

		)

		plotly_config = {
			'displayModeBar': 'hover',  # Show mode bar on hover
			'responsive': True,  # Make the chart responsive
			'scrollZoom': True,  # Enable scroll to zoom
			'displaylogo': False,
			'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'pan', 'select', 'lassoSelect'],
			'modeBarButtonsToAdd': ['autoScale', 'hoverCompareCartesian']
		}

		chart_container.plotly_chart(port_fig, config = plotly_config, use_container_width=True)



def convert_port_df(port_df, port_acronym_df):
	# Convert Series to DataFrame if needed
	df = port_df.copy()

	# List of columns to drop
	indices_to_drop = ['INDEX_NO', 'REGION_NO', 'LAT_DEG', 'LAT_MIN', 'LAT_HEMI', 'LONG_DEG', 'LONG_MIN', 'LONG_HEMI']
	df = df.drop(index=indices_to_drop, errors='ignore')

	# Drop columns with N, 0, None or empty values
	df = df.replace(['N', 0, '', None], pd.NA)
	df = df.dropna()

	# Convert acronyms in index names using port_acronym_df
	acronym_dict = dict(zip(port_acronym_df['Acronym'], port_acronym_df['Meaning']))
	df = df.map(lambda x: acronym_dict.get(x, x))

	return df

# PORT INFO PAGE
top_container = st.container()
top_container.title('Port Information')
top_container.write(st.session_state)

# Port Dataframe
port_df_container = st.container()
with port_df_container:
	with st.expander("Port Dataframe"):
		st.dataframe(port_df)


# Create list of port names
port_names = port_df['PORT_NAME'].tolist()

# CHART
if 'port_fig' not in st.session_state:
	st.session_state.port_fig = None

port_fig = st.session_state.port_fig = mm_plot.create_base_map(title = 'Global Port Locations', mapbox_token=mapbox_token)
mm_plot.add_ports_trace(port_fig, port_df)

chart_container = st.container()
with st.empty():
	if "departure_port" not in st.session_state:
		# Display the Plotly figure in Streamlit
		chart_container.plotly_chart(port_fig, use_container_width=True)
# END CHART

with st.container(key = 'sidebar_port'):
	# Sidebar Controls
	if 'departure_port' not in st.session_state:
		dep_idx = 0
		arr_idx = 0
	else:
		dep_idx = int(name_index(port_df, st.session_state.departure_port["PORT_NAME"]))
		arr_idx = int(name_index(port_df, st.session_state.arrival_port["PORT_NAME"]))
	dep_port =  st.sidebar.selectbox(label = "Departure Port New", options = port_names, index = dep_idx)
	arr_port =  st.sidebar.selectbox(label = "Arrival Port", options = port_names, index =  arr_idx)

	# Save button
	if st.sidebar.button("Save Selection", on_click=update_port_traces, args=(port_fig, dep_port, arr_port)):
		st.sidebar.success("Ports saved!")

# Visualize the selected ports
if 'departure_port' in st.session_state and 'arrival_port' in st.session_state:
	dep_port_coords = st.session_state['departure_port']
	arr_port_coords = st.session_state['arrival_port']

	# Display the selected ports
	col1, col2, col3, col4= st.columns(4)
	with col2:
		dep_string = "Departure Port:" + port_str(dep_port_coords)
		with st.expander(dep_string):
			st.dataframe(convert_port_df(dep_port_coords, port_acronym_df),
			             use_container_width=True)

	with col3:
		arr_string = "Arrival Port:" + port_str(arr_port_coords)
		with st.expander(arr_string):
			st.dataframe(convert_port_df(arr_port_coords, port_acronym_df),
			             use_container_width=True
			             )

