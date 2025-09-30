import streamlit as st
from streamlit import session_state

from Data import Data, NOAA_DB, GPKG, PostGIS
from MARITIME_MODULE import ENC, VISUALIZATION, Miscellaneous

schema_name = 's57_US11'
path = "G:\Python Projects\GIS\RouteAssistant\GIS_files\CGD11_ENCs"
token = st.secrets.mapbox.access_token
mm_enc = ENC(path)
mm_plot = VISUALIZATION()
mm_misc = Miscellaneous()
plotly_config = {
		'displayModeBar': 'hover',  # Show mode bar on hover
		'responsive': True,  # Make the chart responsive
		'scrollZoom': True,  # Enable scroll to zoom
		'displaylogo': False,
		'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'pan', 'select', 'lassoSelect'],
		'modeBarButtonsToAdd': ['autoScale', 'hoverCompareCartesian']}


def init_gpkg(folder_path):
	gpkg = GPKG(folder_path)
	st.session_state['gpkg'] = gpkg

def init_postgis():
	"""Initialize PostGIS connection and store in session state"""
	postgis = PostGIS()
	postgis.connect()
	st.session_state.postgis = postgis

def create_map(title="ENC Analysis"):
	"""Create a map using the data from the session state"""
	fig = mm_plot.create_base_map(title=title, mapbox_token=token)
	mm_plot.add_single_port_trace(figure=fig, port_series=st.session_state.departure_port, name="Departure", color='blue')
	mm_plot.add_single_port_trace(figure=fig, port_series=st.session_state.arrival_port, name="Arrival", color='red')

	fig.update_layout(
		mapbox=dict(
			center=dict(
				lat=st.session_state.departure_port.geometry.y,
				lon=st.session_state.departure_port.geometry.x
			),
			zoom=7,
			bearing=0,

		),
		legend=dict(
			traceorder='normal',
			tracegroupgap=1,
			title=dict(
				text='Legend',
				font=dict(
					family='Arial, sans-serif',
					size=16,
					color='black'
				)
			),
			font=dict(
				family="Calibri",
				size=16,
				color="black"
			),
			indentation=5,
			yanchor="top",
			xanchor="left",
			y=0.99,
			x=0.005,
			bgcolor = "#889ec8",
			bordercolor = "#ededed",
			borderwidth = 1
		)
	)


	if "enc_analysis_map" not in st.session_state:
		st.session_state.enc_analysis_map = fig
	st.plotly_chart(fig, config = plotly_config, use_container_width=True)


def create_enc_boundaries(figure, bbox_df):
	mm_plot.add_enc_bbox_trace(figure, bbox_df, usage_bands=[2,3])
	mm_plot.update_legend(figure, traceorder='normal', legend_text="Legend", font_size=16, font_color="black")
	st.plotly_chart(figure, config = plotly_config,  use_container_width=True)

@st.fragment
def boundary_map():
	with st.container():
		control_col, map_col = st.columns([2,8], vertical_alignment='center')


		with map_col:
			with st.empty():
				def display_map():
					if 'enc_analysis_map' in st.session_state:
						fig = st.session_state.enc_analysis_map
						mm_plot.update_legend(fig, traceorder='normal', legend_text="Legend", font_size=16, font_color="black")
						st.plotly_chart(
							fig,
							config=plotly_config,
							use_container_width=True
						)
					else:
						create_map(title="ENC Analysis")

				display_map()

		with control_col:
			st.subheader("Map Controls")
			# Add control elements

			if st.checkbox("Show ENC Boundaries"):
				mm_plot.add_enc_bbox_trace(st.session_state.enc_analysis_map, pg_bbox, usage_bands=[3, 4, 5])

			if st.button("Reset Map"):
				st.session_state.enc_analysis_map.empty()


# MAIN BODY
st.title("ENC Analysis")
st.write(st.session_state)

# Sidebar
sidebar1 = st.container()
with sidebar1:
	folder_path = st.sidebar.text_input("Input folder path", value = path)

	stbut1, stbut2 = st.sidebar.columns(2, vertical_alignment="top")
	with stbut1:
		gpkgB = st.button("Geopackage", icon = ":material/drive_folder_upload:",use_container_width=True, on_click=init_gpkg(folder_path))
	with stbut2:
		pgB = st.button("PostGIS", icon = ":material/database:", use_container_width=True, on_click=init_postgis())

	# Source connedtion
	with st.sidebar:
		if gpkgB:
			st.success("Geopackage connection established!")
			st.session_state.workflow = 'gpkg'
		if pgB:
			st.success("PostGIS connection established!")
			st.session_state.workflow = 'postgis'


	# PostgIS Workflow
if 'workflow' in st.session_state:
	if st.session_state.workflow == 'postgis':

		postgis = st.session_state.postgis
		db_summary = postgis.enc_db_summary(schema_name)
		with st.expander("PostGIS Database Summary"):
			st.write(db_summary, use_container_width=True)
		list = db_summary['dsid_dsnm'].to_list()
		pg_bbox = postgis.enc_bbox(list)
		with st.expander("PostGIS Bounding Box Data"):
			st.write(pg_bbox, use_container_width=True)
		map1_cont = st.container()

		boundary_map()



	# GeoPackage Workflow
	if st.session_state.workflow == 'gpkg':

		st.write("GPKG")
		gpkg = st.session_state['gpkg']
		foldr_info = gpkg.enc_folder_summary()
		st.dataframe(foldr_info)
		st.write("text 2")

else:
	st.warning("Please select a Data source.")




