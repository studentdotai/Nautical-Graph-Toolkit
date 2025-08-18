import os
import logging
from collections import defaultdict
from typing import Union, List, Dict, Any
import re

import fiona
from shapely.geometry import shape, mapping
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import pyogrio
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
import requests
from bs4 import BeautifulSoup


class Data:
	def __init__(self):
		self.ports_msi_shp = 'GIS_files/World Port Index_2019Shapefile/WPI.shp'
		self.ports_msi_acronyms = 'GIS_files/WorldPortIndex_2019.csv'
		self.land_shp = "GIS_files/ne_10m_land/ne_10m_land.shp"
		self.grid_shp = "GIS_files/ne_10m_graticules_1/ne_10m_graticules_1.shp"
		self.coast_10_shp = "GIS_files/ne_10m_coastline/ne_10m_coastline.shp"
		self.coast_110_shp = "GIS_files/ne_110m_coastline/ne_110m_coastline.shp"
		self.ocean_shp = "GIS_files/ne_110m_ocean"

		self.enc_folder = "GIS_files/CGD11_ENCs"
		self.all_enc_folder = "GIS_export/ALL_ENCs"

		self.input_folder = 'GIS_files/'
		self.output_folder = 'GIS_export/'

		self.s57_attributes_csv = 'GIS_files/s57attributes.csv'
		self.s57_objects_csv = 'GIS_files/s57objectclasses.csv'
		self.s57_expectedInput_csv = 'GIS_files/s57expectedinput.csv'

	@staticmethod
	def parse_string(s: str) -> str:
		"""
	    Extracts string content after colon if present, removing parentheses.

	    Args:
	        s (str): Input string, e.g., '(1:6)' or '(2:3,6)'
	    Returns:
	        str: Content after colon with parentheses removed
	    """
		if ':' in s:
			s = s.strip('()')
			return s.split(':')[1].strip()
		return s

	def clean_enc_name(enc_name: str) -> str:
		"""
		Cleans ENC filename by removing common file extensions.
		Args:
			enc_name: ENC filename to clean
		Returns:
			str: Cleaned ENC name without extensions
		Examples:
			clean_enc_name("US5VA51M.000") -> "US5VA51M"
			clean_enc_name("US5VA51M.gpkg") -> "US5VA51M"
			clean_enc_name("US5VA51M") -> "US5VA51M"
		"""
		# Split on first period and take base name
		return enc_name.split('.')[0]

	def clean_enc_names_column(self, df, column_name: str) -> pd.DataFrame:
		"""
		Cleans ENC names in specified DataFrame column by removing file extensions.
		Args:
			df: Input DataFrame containing ENC names
			column_name: Name of column containing ENC filenames
		Returns:
			pd.DataFrame: DataFrame with cleaned ENC names
		Examples:
			# Clean 'ENC_NAME' column
			df = clean_enc_names_column(df, 'ENC_NAME')
		"""
		df = df.copy()
		df[column_name] = df[column_name].str.split('.').str[0]
		return df

	def s57_attributes_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_attributes_csv or not os.path.exists(self.s57_attributes_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_attributes_csv}")

		try:
			df = pd.read_csv(self.s57_attributes_csv)
			df.set_index('Code', inplace=True)
			return df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_objects_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_objects_csv or not os.path.exists(self.s57_objects_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_objects_csv}")

		try:
			df = pd.read_csv(self.s57_objects_csv)
			df.set_index('Code', inplace=True)
			return df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_properties_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_expectedInput_csv or not os.path.exists(self.s57_expectedInput_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_expectedInput_csv}")

		attr_df = self.s57_attributes_df()

		try:
			df = pd.read_csv(self.s57_expectedInput_csv)
			df.set_index('Code', inplace=True)

			# Filter attr_df to only include codes present in df
			attr_df = attr_df[attr_df.index.isin(df.index)]

			prop_df = pd.merge(df, attr_df, on='Code', how='outer')
			prop_df.insert(1, 'Acronym', prop_df.pop('Acronym'))
			prop_df.insert(2, 'Attribute', prop_df.pop('Attribute'))
			# Clean all ID rows with value NaN
			prop_df = prop_df.dropna(subset=['ID'])
			return prop_df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_attributes_convert(self, acronym_str):
		"""
		Convert an acronym string into a attribute string.
		Note: make sure acronym_str is Uppercase. Add .upprer() before passing string to this function
		:param acronym_str: String containing acronyms separated by semicolons.
		:return: String of attributes or None if acronyms not found
		"""

		attribute_df = self.s57_attributes_df()
		if acronym_str in attribute_df['Acronym'].values:
			return attribute_df.loc[attribute_df['Acronym'] == acronym_str, 'Attribute'].iloc[0]
		return None

	def s57_objects_convert(self, acronym_str):
		"""
		Convert an acronym string into a attribute string.
		Note: make sure acronym_str is Uppercase. Add .upprer() before passing string to this function
		:param acronym_str: String containing acronyms separated by semicolons
		:return: String of attributes or None if acronyms not found
		"""
		object_df = self.s57_objects_df()
		if acronym_str in object_df['Acronym'].values:
			return object_df.loc[object_df['Acronym'] == acronym_str, 'ObjectClass'].iloc[0]
		return None

	def s57_properties_convert(self, acronym_str: str, property_value: Any, prop_mixed: bool = False, debug = False) -> Union[ str, List[str] ]:
		"""
	    Convert an S-57 layer property value to meaningful names.

	    Args:
	        acronym_str (str): S-57 property acronym (e.g., 'NATSUR').
	        property_value (Any): Value or ID associated with the property.
	        prop_mixed (bool, optional):
	            If True, returns "Name (Code)". If False, returns only "Name". Defaults to False.
	        debug (bool, optional):
	            If True, prints debug information. Defaults to False.

	    Returns:
	        Union[str, List[str]]:
	            - A single string if `property_value` corresponds to one entry.
	            - A list of strings if multiple values are processed.
	            - Original `property_value` if conversion isn't applicable.

	    Examples:
	        data.s57_properties_convert('NATSUR', '(1:4,1)')
	        ['sand', 'mud']

	        data.s57_properties_convert('SOUND', '4,6')
	        ["Sound Level 4 (4)", "Sound Level 6 (6)"]
	    """
		prop_df = self.s57_properties_df()

		# Early validation
		if property_value is None or pd.isna(property_value):
			return None

		if isinstance(property_value, (int, float)):
			if property_value == -2147483648:
				return None
			if property_value < 0:
				return property_value



		# Determin if Property neme is in the propertytable
		if acronym_str in prop_df['Acronym'].values and property_value is not None:
			if debug:
				print(f"Acronym found: {acronym_str} ")
			attrType = prop_df.loc[prop_df['Acronym'] == acronym_str, 'Attribute'].head(1).iloc[0]
			# Check if property is Sting and need to be parsed to check IDs with property_df
			if isinstance(property_value, str):
				if debug:
					print(f"String value found: {property_value}")
				# Filters Free Text Strings
				if attrType == ("S"):
					return property_value
				# If property has multiple values they separated by ":", that needs to be cleaned
				if ":" in property_value:
					parsed_properties = []
					property_value = self.parse_string(property_value)
					if debug:
						print(f"Parsed string value: {property_value} ({type(property_value)})")
					if all(num.isdigit() for num in property_value.split(',')):
						numbers = [int(x) for x in property_value.split(',')]
						for number in numbers:
							matching_rows = prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
							print(f"Matching rows: {matching_rows}")
							if prop_mixed:
								matching_rows = f"{matching_rows} ({number})"


							parsed_properties.append(matching_rows)
						return parsed_properties

					else:
						return property_value
					# Handle simple comma-separated values without colons
				elif "," in property_value and all(num.strip().isdigit() for num in property_value.split(',')):
					parsed_properties = []
					numbers = [int(x.strip()) for x in property_value.split(',')]
					for number in numbers:
						try:
							matching_rows = prop_df.loc[
								(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
							if debug:
								print(f"Matching rows: {matching_rows}")
							if prop_mixed:
								matching_rows = f"{matching_rows} ({number})"
							parsed_properties.append(matching_rows)
						except IndexError:
							if debug:
								print(f"No matching meaning found for {acronym_str} with ID {number}")
							parsed_properties.append(str(number))
					return parsed_properties
					# Handle single numeric strings (e.g., "1")
				elif property_value.isdigit():
					try:
						number = int(property_value)
						matching_rows = \
						prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
						if debug:
							print(f"Matching rows: {matching_rows}")
						if prop_mixed:
							matching_rows = f"{matching_rows} ({number})"
						return matching_rows
					except IndexError:
						if debug:
							print(f"No matching meaning found for {acronym_str} with ID {property_value}")
						return property_value

			if isinstance(property_value, list):
				if debug:
					print(f"Processing list: {property_value}")

				parsed_properties = []

				# Handle nested lists like [['1'], ['2']]
				if all(isinstance(x, list) for x in property_value):
					if debug:
						print(f"Processing list as list of lists")
					for item in property_value:
						if item and isinstance(item[0], str) and item[0].isdigit():
							number = int(item[0])
							try:
								matching_rows = prop_df.loc[
									(prop_df['Acronym'].astype(str) == acronym_str) &
									(prop_df['ID'].astype(int) == number), 'Meaning']

								if not matching_rows.empty:
									meaning = matching_rows.iloc[0]
									if prop_mixed:
										meaning = f"{meaning} ({number})"
									parsed_properties.append(meaning)
								else:
									parsed_properties.append(str(number))
							except (IndexError, KeyError):
								parsed_properties.append(str(number))

				# Handle simple lists like ['1', '3', '5'] or ['3'] or ['31,33']
				else:
					for item in property_value:
						if isinstance(item, str):
							# Handle comma-separated values within the string
							if ',' in item and all(num.strip().isdigit() for num in item.split(',')):
								# Split the comma-separated string and process each number
								numbers = [int(x.strip()) for x in item.split(',')]
								for number in numbers:
									try:
										matching_rows = prop_df.loc[
											(prop_df['Acronym'].astype(str) == acronym_str) &
											(prop_df['ID'].astype(int) == number), 'Meaning']

										if not matching_rows.empty:
											meaning = matching_rows.iloc[0]
											if prop_mixed:
												meaning = f"{meaning} ({number}) "
											parsed_properties.append(meaning)
										else:
											parsed_properties.append(str(number))
									except (IndexError, KeyError):
										parsed_properties.append(str(number))
							# Handle single digits
							elif item.isdigit():
								number = int(item)
								try:
									matching_rows = prop_df.loc[
										(prop_df['Acronym'].astype(str) == acronym_str) &
										(prop_df['ID'].astype(int) == number), 'Meaning']

									if not matching_rows.empty:
										meaning = matching_rows.iloc[0]
										if prop_mixed:
											meaning = f"{meaning} ({number})"
										parsed_properties.append(meaning)
									else:
										parsed_properties.append(str(number))
								except (IndexError, KeyError):
									parsed_properties.append(str(number))
				if debug:
					print(f"Return: {parsed_properties}")
				return parsed_properties

			# Filters integer values (e.g. "SCAMIN", "Compilation Scale", "Soundingdistance") that has no String property
			if attrType == "I":
				return property_value
			else:
				matching_rows = prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == property_value), 'Meaning'].iloc[0]
				print(f"Matching rows: {matching_rows}")
				if prop_mixed:
					matching_rows = f"{matching_rows} ({property_value})"
				return matching_rows

		elif isinstance(property_value, str) and ":" in property_value:
			property_values = self.parse_string(property_value)
		else:
			return property_value











class NOAA_DB:
	"""
	A class that scrapes and manages NOAA Electronic Navigational Charts (ENC) data from the official website.

	Attributes:
		url (str): The URL of the NOAA ENC website (https://www.charts.noaa.gov/ENCs/ENCsIndv.shtml)
		df (pandas.DataFrame): DataFrame containing the scraped ENC data
		session (requests.Session): HTTP session for making requests

	Methods:
		get_dataframe():
			Returns the scraped data as a pandas DataFrame.
			Returns: pandas.DataFrame containing ENC information

		save_to_csv(filename="ENC_DB.csv"):
			Saves the DataFrame to a CSV file.
			Args:
				filename (str): Name of output CSV file

	Usage Example:
		noaa_db = NOAA_DB()
		enc_data = noaa_db.get_dataframe()
		noaa_db.save_to_csv("my_enc_data.csv")
	"""

	def __init__(self):
		self.url = "https://www.charts.noaa.gov/ENCs/ENCsIndv.shtml" # URL of the NOAA ENC website
		self.df = None
		self.session = requests.Session()

	@staticmethod
	def create_dataframe(headers, rows):
		"""Create a pandas DataFrame from the parsed table data."""
		df = pd.DataFrame(rows, columns=headers)
		df = df[df['#'] != '#']  # Remove rows which duplicate header names
		df = df.set_index('#')  # Set index to the "#" column
		df.columns = (df.columns
		              .str.replace('\xa0', ' ')
		              .str.strip()
		              .str.replace(' ', '_'))
		return df

	@staticmethod
	def parse_table(soup):
		"""Extract and parse the table data from the HTML content."""
		table = soup.find('table')
		if not table:
			raise ValueError("No table found in the HTML content")

		inner_tbody = table.find('tr').find('td').find_all('tr')  # .find_all('td')

		columns = []
		for row in inner_tbody:
			row_list = row.find_all('td')
			for i in range(9):
				column = row_list[i].text
				columns.append(column)

		# Separate headers and data rows
		headers = columns[:9]
		rows = [columns[i:i + 9] for i in range(0, len(columns), 9)]
		return headers, rows

	def _get_data(self):
		"""Fetch the HTML content from the NOAA ENC website."""
		try:
			response = self.session.get(self.url, timeout=10)
			response.raise_for_status()
			return BeautifulSoup(response.content, 'html.parser')
		except requests.RequestException as e:
			raise ConnectionError(f"Failed to fetch data from {self.url}: {str(e)}")

	def _scrape_enc_data(self):
		"""Main method to scrape the ENC data and create a DataFrame."""
		try:
			soup = self._get_data()
			headers, rows = self.parse_table(soup)
			self.df = self.create_dataframe(headers, rows)
		except Exception as e:
			raise RuntimeError(f"Failed to import ENC data: {str(e)}")

	def get_dataframe(self):
		"""Return the scraped data as a pandas DataFrame."""
		if self.df is None:
			self._scrape_enc_data()
		return self.df

	def save_to_csv(self, filename="ENC_DB.csv"):
		"""Save the DataFrame to a CSV file."""
		if self.df is None:
			self._scrape_enc_data()
		try:
			self.df.to_csv(filename)
		except Exception as e:
			raise IOError(f"Failed to save CSV file: {str(e)}")

class PostGIS:
	def __init__(self):
		self.data = Data()
		self.noaa_db = NOAA_DB()
		self.data.s57_attributes_df()
		self.data.s57_objects_df()
		self.data.s57_properties_df()

		self.engine = None
		self.session = None
		self.connection = None

		load_dotenv("SECRET.env")

		self.DB_CONFIG = {
			'host': os.getenv('DB_HOST'),
			'database': os.getenv('DB_NAME'),
			'user': os.getenv('DB_USER'),
			'password': os.getenv('DB_PASSWORD'),
			'port': os.getenv('DB_PORT')
		}


	def connect(self):
		"""Establish connection to PostGIS database using SQLAlchemy"""
		try:
			connection_string = f"postgresql://{self.DB_CONFIG['user']}:{self.DB_CONFIG['password']}@{self.DB_CONFIG['host']}:{self.DB_CONFIG['port']}/{self.DB_CONFIG['database']}"
			self.engine = create_engine(connection_string)
			self.session = sessionmaker(bind=self.engine)
			self.connection = self.engine.connect()
			return self.connection
		except Exception as e:
			print(f"Connection error: {e}")
			return False

	def is_connection_alive(self):
		try:
			result = self.connection.execute("SELECT 1;")
			return result.fetchone() is not None
		except Exception as e:
			print("Connection check failed:", e)
			return False

	def connection_test(self):
		if self.connection is None or not self.is_connection_alive():
			try:
				self.connection = self.connect()  # or however you reconnect
			except Exception as e:
				print("Reconnection attempt failed:", e)
				return False
		return True

	def _tables_exist(self, schema = 'public'):
		"""Check if tables exist in the database"""
		try:
			inspector = inspect(self.engine)
			table_names = inspector.get_table_names(schema=schema)
			print(f"Tables in the database: {table_names}")
			return True
		except Exception as e:
			print(f"Error checking tables: {e}")
			return False

	def _format_enc_names(self, enc_names: Union[str, List[str]]) -> List[str]:
		"""
		Formats ENC names to S-57 standard with .000 extension.

		Args:
			enc_names: Single ENC name or list of ENC names

		Returns:
			List[str]: ENC names formatted as NAME.000

		Examples:
			format_enc_names('US5CA51M') -> ['US5CA51M.000']
			format_enc_names(['US5CA51M', 'US5CA52M.000']) -> ['US5CA51M.000', 'US5CA52M.000']
		"""
		# Convert single string to list
		if isinstance(enc_names, str):
			enc_names = [enc_names]

		formatted_names = []
		for name in enc_names:
			# Remove any existing extensions (.000, .gpkg)
			base_name = name.split('.')[0]
			formatted_names.append(f"{base_name}.000")
		return formatted_names

	def enc_db_summary(self, schema_name = 'public', detailed: bool = False, show_outdated: bool = False, noaa_data: bool = False):
		"""
		Retrieves comprehensive ENC summary information from PostGIS database.

	    Args:
	    	schema_name (str): Database schema name. Defaults to 'public'.
	        detailed (bool): If True, provides detailed information unified column Names
	        show_outdated (bool): If True, includes outdated ENC status.

	    Returns:
            pd.DataFrame: A DataFrame summarizing ENC information.
		"""
		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		table_name = 'dsid'
		columns = ['dsid_dsnm', 'dsid_edtn', 'dsid_updn']

		raw_sql = f"""
		SELECT {', '.join(columns)}
		FROM "{schema_name}".{table_name};
		"""

		# Convert query result to GeoPandas DataFrame
		df = pd.read_sql(raw_sql, self.engine)
		cleaned_df = self.data.clean_enc_names_column(df, 'dsid_dsnm')

		noaa_df = self.noaa_db.get_dataframe()
		if show_outdated:
			# Check for outdated ENCs against NOAA database
			for idx, row in cleaned_df.iterrows():
				db_entry = noaa_df[noaa_df['ENC_Name'] == row['dsid_dsnm']]
				if not db_entry.empty:
					db_edition = db_entry['Edition'].iloc[0]
					db_update = db_entry['Update'].iloc[0]
					cleaned_df.loc[idx, 'OUTDATED'] = (
							(row['dsid_edtn'] < db_edition) or
							(row['dsid_edtn'] == db_edition and row['dsid_updn'] < db_update)
					)
					if noaa_data:
						get_string = f"Ed: {db_edition}, Upd: {db_update}, Date: {db_entry['Update_Application_Date'].iloc[0]}"
						cleaned_df.loc[idx, 'NOAA_DATA'] = get_string
				else:
					cleaned_df.loc[idx, 'OUTDATED'] = False


		if detailed:
			cleaned_df = cleaned_df.rename(columns={'dsid_dsnm': 'ENC_NAME', 'dsid_edtn': 'ENC_EDITION', 'dsid_updn': 'ENC_UPDATE'})


		return cleaned_df



	def layers_summary(self, schema_name: str = 'public', cleaned: bool = False):
		"""
		Retrieves layer information from the specified database schema.

		Args:
			schema_name (str): The name of the schema to inspect. Defaults to 'public'.
			cleaned (bool): If True, filters out layers with zero entries and missing names.

		Returns:
			pd.DataFrame: A DataFrame containing layer names, acronyms, and entry counts.
		"""
		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		# Get table names from the database
		inspector = inspect(self.engine)
		table_names = inspector.get_table_names(schema=schema_name)

		data = Data()
		layers = []
		acronyms = []
		entries = []
		for table in table_names:
			count_query = text(f""" SELECT COUNT(*) FROM "{schema_name}"."{table}" """)

			full_name = data.s57_objects_convert(table.upper())
			with self.engine.connect() as connection:
				result = connection.execute(count_query)
				count = result.scalar()
			layers.append(full_name)
			acronyms.append(table)
			entries.append(count)

		df = pd.DataFrame({'Layer name': layers,
							 'Acronym': acronyms,
							 'Entries': entries})

		if cleaned:
			df = df[(df['Entries'] > 0) & (df['Layer name'].notna())]
			return df
		else:
			return df

	def sort_enc(self, sort_input: str, sort_by: str = "usage band",  schema_name = "public" , output: str = "list", ascending = True):
		"""
		Sort and filter Electronic Navigational Charts (ENC) data from PostGIS database.

		Usage Bands:
		1: Overview
		2: General
		3: Coastal
		4: Approach
		5: Harbour
		6: Berthing

		Args:
		sort_by (str):
			Attribute to sort by. Options:
				- 'usage band'
				- 'code'
				- 'number'
		sort_input (str):
			Filter value corresponding to `sort_by`.
		output (str, optional):
			Format of the returned data. Choices:
				- 'list' (default)
				- 'dataframe'
		ascending (bool, optional):
			Sort order. `True` for ascending (default), `False` for descending.

		Returns:
			Union[List[str], pd.DataFrame, Dict[str, List[str]]]:
            - List of sorted `dsid_dsnm` values if `output` is 'list'
            - DataFrame sorted by `dsid_dsnm` if `output` is 'dataframe'
            - Dictionary with usage bands as keys if `output` is 'dict'
            - `None` if database connection fails

		 Examples:

		# Get all Usage Band 1 ENCs as DataFrame
		sort_enc('Usage Band', '1', True, 'dataframe')

		# Get US ENCs as list
		sort_enc('Code', 'US', True, 'list')

		# Search specific ENC number or part on the number
		sort_enc('Number', '11', True, 'dataframe')
		"""

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return


		table_name = 'dsid'
		columns = 'dsid_dsnm'

		pattern =   {'pattern': f'__{sort_input}%'} if sort_by.lower() == 'usage band' else \
					{'pattern': f'{sort_input}%'}   if sort_by.lower() == 'code' else \
					{'pattern': f'%{sort_input}%'}  if sort_by.lower() == 'number' else \
					{'pattern': '%'}


		raw_sql = f"""
		SELECT {columns}
		FROM "{schema_name}"."{table_name}"
		WHERE dsid_dsnm LIKE %(pattern)s
		"""

		df  = pd.read_sql(raw_sql, self.engine, params=pattern)
		sorted_df = df.sort_values(by='dsid_dsnm', ascending=ascending)
		return sorted_df['dsid_dsnm'].tolist() if output.lower() == 'list' else sorted_df

	def enc_bbox(self, enc_names: list[str] = None, schema_name: str = 'public'):
		"""
		Retrieves bounding geometries for specified ENC names from the database.

		Parameters:
		- enc_names (list): A list of ENC names to query.
		- schema_name (str): The name of the schema to query.
		Returns:
		- GeoDataFrame: A GeoDataFrame containing the geometries sorted by 'dsid_dsnm'.
		"""

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return
		if enc_names is None:
			print("Please provide a list of ENC names.")
			return
		# Format names to S-57 standard
		formatted_names = self._format_enc_names(enc_names)


		table_name = 'm_covr'
		columns = ['dsid_dsnm', 'wkb_geometry']

		# Verify the table exists and has data:
		verify_sql = f"""
						SELECT COUNT(*)
						FROM "{schema_name}"."{table_name}"
						"""
		verify_table =  pd.read_sql(verify_sql, self.engine )

		if not verify_table.empty:
			print(f"Table has: {verify_table['count'][0]} entries")
		else:
			print (f"Table {schema_name}.{table_name} is empty")

		# Generate the SQL query with placeholders

		placeholders = ', '.join(['%s'] * len(enc_names))
		raw_sql = f"""
				SELECT {', '.join(columns)}
				FROM "{schema_name}"."{table_name}"
				WHERE dsid_dsnm IN ({placeholders})
				"""
		# Convert enc_names directly to tuple
		pattern = tuple(formatted_names)

		gdf = gpd.read_postgis(raw_sql, self.engine, params=pattern, geom_col='wkb_geometry')
		gdf = self.data.clean_enc_names_column(gdf, 'dsid_dsnm')


		return gdf.sort_values(by='dsid_dsnm')


	def get_layer(self, schema_name = 'public', layer_name: str = "dsid", filter_by_enc: list[str] = "ALL", progress_callback = None):

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		if filter_by_enc != "ALL":
			filter_by_enc = self._format_enc_names(filter_by_enc)
			placeholders = ', '.join(['%s'] * len(filter_by_enc))
			raw_sql = f"""
					SELECT *
					FROM "{schema_name}"."{layer_name}"
					WHERE dsid_dsnm IN ({placeholders})
					"""
			pattern = tuple(filter_by_enc)
			gdf = gpd.read_postgis(raw_sql, self.engine, params=pattern,  geom_col='wkb_geometry')

		else:
			raw_sql = f"""
					SELECT *
					FROM "{schema_name}"."{layer_name}"
					"""
			gdf = gpd.read_postgis(raw_sql, self.engine, geom_col='wkb_geometry')
		print(len(gdf))

		return gdf.sort_values(by='dsid_dsnm')



class GPKG_Old:
	def __init__(self, folder_path):
		self.folder_path: str = folder_path
		self.output_dir = os.path.join(self.folder_path, 'export_folder')
		self.enc_files = []
		self.sorted_encs: Dict[str, List[str]] = defaultdict(list)
		self.usage_bands: Dict[str, str] = {
			'1': 'Overview',
			'2': 'General',
			'3': 'Coastal',
			'4': 'Approach',
			'5': 'Harbor',
			'6': 'Berthing'
		}
		self.enc_db = NOAA_DB().get_dataframe()
		self.data = Data()

		# Check if folder exists
		if not os.path.isdir(self.folder_path):
			raise FileNotFoundError(f"The folder path {self.folder_path} does not exist.")

		# Check if output directory exists
		if not os.path.isdir(self.output_dir):
			os.makedirs(self.output_dir, exist_ok=True)

		# Collect all ENC files
		self.enc_files = [file for file in os.listdir(self.folder_path) if file.endswith('.gpkg')]
		if not self.enc_files:
			print("No ENC files found in the folder.")

	def _is_gpkg(self, gpkg_name: str, debug = False) -> str:
		"""
	    Ensures filename has .gpkg extension.

	    Args:
	        gpkg_name: Filename to check/modify
	    Returns:
	        str: Filename with .gpkg extension
	    Raises:
	        TypeError: If input is None or not string
	    """
		if debug:
			print(f"Input: {gpkg_name} \nType: {type(gpkg_name)}")
		if not isinstance(gpkg_name, str) or not gpkg_name:
			raise TypeError("Filename must be a non-empty string")
		converted_name = f"{gpkg_name}.gpkg" if not gpkg_name.endswith('.gpkg') else gpkg_name
		if debug:
			print(f"Output: {converted_name} \nType: {type(converted_name)}")
		return converted_name


	def _is_str_list(self, str_list: Union[str, List[str]], debug = False) -> List[str]:
		"""
	    Validates and converts input to list of strings.

	    Args:
	        str_list: Single string or list of strings to validate
	    Returns:
	        List[str]: Original list or single string converted to list
	    Raises:
	        TypeError: If input is None or contains non-string elements
	    Examples:
	        _is_str_list("test")
	        ["test"]
	        _is_str_list(["a", "b"])
	        ["a", "b"]
	        _is_str_list(["a", 1])  # Raises TypeError
	    """
		if str_list is None:
			raise TypeError("Input cannot be None")
		if debug:
			print(f"Input type: {type(str_list)}")
			print(f"Input value: {str_list}")
		# If str_list is a string, convert it to a list
		confirmed_str_list = [str_list] if isinstance(str_list, str) else str_list
		if debug:
			print(f"Converted value: {confirmed_str_list}")

		if all(isinstance(item, str) for item in confirmed_str_list):
			return confirmed_str_list
		else:
			raise TypeError("Input must be a string or a list of strings.")


	def _fiona_stamp(self, enc_path):

		with fiona.open(enc_path, 'r', layer='DSID') as dsid_layer:
			dsid_info = next(iter(dsid_layer))['properties']

		enc_name = dsid_info.get('DSID_DSNM', 'Unknown')
		# Remove .gpkg or .000 ending from ENC_NAME
		enc_name = enc_name.rsplit('.', 1)[0]

		return {
			'ENC_NAME': enc_name,
			'ENC_EDITION': dsid_info.get('DSID_EDTN', 'Unknown'),
			'ENC_UPDATE': dsid_info.get('DSID_UPDN', 'Unknown')
		}

	def _gpd_stamp(self, enc_path):

		dsid = gpd.read_file(enc_path, layer='DSID')
		return {
			'ENC_NAME': dsid['DSID_DSNM'].iloc[0],
			'ENC_EDITION': dsid['DSID_EDTN'].iloc[0],
			'ENC_UPDATE': dsid['DSID_UPDN'].iloc[0]
		}

	def usage_band(self):
		"""Sort ENC files by usage band."""
		"""Return: A dictionary with usage bands as keys and a list of filenames as values."""
		for filename in self.enc_files:
			usage_band = filename[2]
			if usage_band in self.usage_bands:
				self.sorted_encs[self.usage_bands[usage_band]].append(filename)

		# Add count to each usage band
		enc_by_udage_band = {band: (files, len(files)) for band, files in self.sorted_encs.items()}
		return enc_by_udage_band

	def enc_folder_summary(self,detailed: bool = False,
								show_outdated: bool = False,
								save_to_csv: bool = False,
								csv_name: str = "ENC_Summary"):
		"""
			Summarizes ENC files with .gpkg extension in the specified folder..

			Args:
				detailed (bool): If True, provides detailed information about ENC files.
				show_outdated (bool): If True, includes outdated ENC files in the summary.
				save_to_csv (bool): If True, saves the summary to a CSV file.
				csv_name (str): Name of the CSV file for saving the summary (if enabled).

			Returns:
				pd.DataFrame: A DataFrame summarizing ENC files.
			"""
		# Check if folder exists
		if not os.path.isdir(self.folder_path):
			raise FileNotFoundError(f"The folder path {self.folder_path} does not exist.")

		# Collect all ENC files
		self.enc_files = [file for file in os.listdir(self.folder_path) if file.endswith('.gpkg')]
		if not self.enc_files:
			print("No ENC files found in the folder.")
			return pd.DataFrame()  # Return an empty DataFrame

		# Initialize DataFrame for detailed summary
		detailed_data = []

		for file in self.enc_files:
			enc_path = os.path.join(self.folder_path, file)
			try:
				# Extract metadata using fiona_stamp
				stamp = self._fiona_stamp(enc_path)

				if detailed:
					# Check for outdated ENCs if required
					if show_outdated:
						db_entry = self.enc_db[self.enc_db['ENC_Name'] == stamp['ENC_NAME']]
						if not db_entry.empty:
							db_edition = db_entry['Edition'].iloc[0]
							db_update = db_entry['Update'].iloc[0]
							if (
									(stamp['ENC_EDITION'] < db_edition)
									or (stamp['ENC_EDITION'] == db_edition and stamp['ENC_UPDATE'] < db_update)
							):
								stamp['OUTDATED'] = True
							else:
								stamp['OUTDATED'] = False
						else:
							stamp['OUTDATED'] = False

					detailed_data.append(stamp)

			except Exception as e:
				print(f"Error processing file {file}: {e}")

		if detailed:
			# Create DataFrame from detailed data
			df = pd.DataFrame(detailed_data)

			# Save to Datafreame if required
			if save_to_csv:
				if not csv_name.endswith('.csv'):
					csv_name += '.csv'
				output_path = os.path.join(self.folder_path, csv_name)
				try:
					df.to_csv(output_path)
					print(f"Summary saved to GeoPackage: {output_path}")
				except Exception as e:
					print(f"Error saving GeoPackage: {e}")

			return df
		else:
			return pd.DataFrame({'ENC_Files': self.enc_files})

	def bbox(self, enc_files):
		boundary_data = []

		for enc_file in enc_files:
			enc_path = os.path.join(self.folder_path, enc_file)
			stamp = self._fiona_stamp(enc_path)

			with fiona.open(enc_path, layer='M_COVR') as layer:
				for feature in layer:
					if feature['properties'].get('CATCOV') == 1:  # 1 indicates coverage area
						boundary = {
							'ENC_NAME': stamp['ENC_NAME'],
							'ENC_EDITION': stamp['ENC_EDITION'],
							'ENC_UPDATE': stamp['ENC_UPDATE'],
							'geometry': feature['geometry']
						}
						boundary_data.append(boundary)

		return gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

	def layers_summary(self, enc_files: Union[str, List[str]], unique: bool = False, acronym_convert: bool = False, verbose: bool = False) -> Union[Dict[str, List[str]], List[str]]:
		"""
	    Summarizes layers from one or more ENC (Electronic Navigational Chart) GeoPackage files.

	    Args:
	        enc_files (Union[str, List[str]]): Single ENC filename or list of ENC filenames.
	        unique (bool, optional): If True, returns only unique layers across all ENCs. Defaults to False.
	        acronym_convert (bool, optional): If True, converts ENC GPKG layer acronyms to full names. Defaults to False.
	        verbose (bool, optional): If True, prints detailed processing information. Defaults to False.

	    Returns:
	        Union[Dict[str, List[str]], List[str]]:
	            - If `unique=False`, returns a dictionary mapping ENC filenames to their list of layers.
	            - If `unique=True`, returns a sorted list of unique layer names across all ENCs.
	    """
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if verbose else logging.INFO)

		enc_files = self._is_str_list(enc_files)
		if verbose:
			logger.info(f"Reading {len(enc_files)} ENC files...")

		all_layers = set()  # Use a set to store unique layers
		enc_layer_dict = {} # Store ENC name with layer names
		unique_dict = {}  # Use a set to store unique layers

		for enc in enc_files:
			# Check enc file format ending with .GPKG
			if verbose:
				logger.info(f"Reading {enc}...")
			enc = self._is_gpkg(enc)
			enc_layer_dict[enc] = {}

			# List all layers in the GeoPackage
			full_path = os.path.join(self.folder_path, enc)
			# Check if ENC file exists
			if not os.path.exists(full_path):
				logger.warning(f"ENC file '{enc}' not found at path '{full_path}'. Skipping.")
				continue

			try:
				# List all layers in the GeoPackage
				fiona_layers = fiona.listlayers(full_path)
				logger.debug(f"Layers in '{enc}': {fiona_layers}")
			except Exception as e:
				logger.error(f"Failed to list layers in ENC '{enc}': {e}")
				continue

			# Add layers to set
			all_layers.update(fiona_layers)

			if not unique:
				# Sort layers alphabetically (case-insensitive)
				sorted_layers = sorted(fiona_layers, key=lambda x: (x[0] or "").lower())
				if acronym_convert:
					for layer in sorted_layers:
						obj_name = self.data.s57_objects_convert(layer)
						enc_layer_dict[enc][layer] = obj_name
				else:
					enc_layer_dict[enc] = sorted_layers

		if unique:
		# Sort unique layers alphabetically (case-insensitive)
			sorted_unique_layers = sorted(all_layers, key=lambda x: (x[0] or "").lower())
			if verbose:
				logger.info("Compiled unique layers across all ENC files")

			if acronym_convert:
				# Convert acronyms to full names
				for layer in sorted_unique_layers:
					obj_name = self.data.s57_objects_convert(layer)
					unique_dict.update({layer : obj_name})
				return unique_dict
			else:
				return sorted_unique_layers
		else:
			return enc_layer_dict


	def explore_enc_layer(self, enc_name: str, layer_name: str, acronym_convert: bool = False, property_convert: bool = False, prop_mixed: bool = False, debug: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
		"""
		Opens and reads an ENC layer, returning GeoDataFrame if geometry present, else DataFrame

		Args:
			enc_name: Name of ENC file
			layer_name: Name of layer to read
			acronym_convert: Convert S-57 acronyms to full names
			property_convert: Convert property values
			debug: Enable debug logging
			check_type: Include type information in property names

		Returns:
			GeoDataFrame or DataFrame depending on geometry presence
		"""
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if debug else logging.INFO)

		# Check if user provided ENC Name and its Valid
		if enc_name is None:
			logger.warning("Please provide a valid ENC name.")
			return None
		enc_name = self._is_gpkg(enc_name)
		# Check if Layer Name or List of names provided
		if layer_name is None:
			logger.warning("Please provide a valid layer name.")
			return None
		layer_name = self._is_str_list(layer_name)
		if len(layer_name) > 1:
			logger.warning("Please provide a single layer name.")
			return None

		if enc_name:
			full_path = os.path.join(self.folder_path, enc_name)
			if not os.path.exists(full_path):
				logger.warning(f"ENC file '{enc_name}' not found at path '{full_path}'. Skipping.")
				return None

		# List all layers in the GeoPackage ENC file
		fiona_layers = fiona.listlayers(full_path)
		# Check if Input layers match with ENC layers
		verified_layers = [layer for layer in layer_name if layer in fiona_layers]
		if not verified_layers:
			logger.warning(f"Layer not found in {enc_name} ENC file.")
			logger.info(f"Available layers in {enc_name} ENC file: {fiona_layers}")
			return None

		for layer_name in verified_layers:
			# Initialize empty lists for features and geometries
			features = []
			geometries = []

			with fiona.open(full_path, layer=layer_name) as layer:
				# Iterate through features
				for feature in layer:
					# Store properties
					properties = dict(feature['properties'])

					if acronym_convert:
						if property_convert:
							converted_properties = {}
							for key, value in properties.items():
								k = f"{self.data.s57_attributes_convert(key) or key} ({key})"
								v = self.data.s57_properties_convert(key, value, prop_mixed=prop_mixed, debug=debug)
								converted_properties[k] = v

							properties = converted_properties


						else:
							properties = {f"{self.data.s57_attributes_convert(key) or key} ({key})": value
										  for key, value in properties.items()}

					features.append(properties)

					if 'geometry' in feature and feature['geometry']:
						geometries.append(feature['geometry'])

				if debug:
					logger.info(f"Features in layer '{layer_name}':")
					logger.info(f"Total features: {len(layer)}")
					if geometries:
						logger.info(f"CRS: {layer.crs}")
						logger.info(f"Bounds: {layer.bounds}")

			# Create appropriate dataframe type
			if geometries:
				gdf = gpd.GeoDataFrame(
					features,
					geometry=[shape(geom) for geom in geometries],
					crs=layer.crs
				)
				return gdf
			else:
				pdf = pd.DataFrame(features)
				return pdf


	def enc_to_layers(self,
	                  enc_names: Union[str, List[str]],
	                  layer_names: Union[str, List[str]],
	                  export_folder: str = None,
	                  create_df: bool = False,
	                  debug:bool = False) -> Union[Dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame, None]:
		"""
		 Process and extract layers from multiple ENC (Electronic Navigational Chart) files efficiently.

	    Args:
	        enc_names (Union[str, List[str]]):
	            Single ENC filename or list of ENC filenames.
	        layer_names (Union[str, List[str]]):
	            Layer name or list of layer names to process.
	        export_folder (str, optional):
	            Directory to save GeoPackage files. If None, uses `self.output_dir`.
	        create_df (bool, optional):
	            If True, returns a DataFrame/GeoDataFrame for the requested layer(s).
	            If False, saves layers to GeoPackage files. Defaults to False.

	    Returns:
	        Union[Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame], None]:
	            - If `create_df=True` and a single layer is requested: Returns a single GeoDataFrame/DataFrame.
	            - If `create_df=True` and multiple layers are requested: Returns a dictionary mapping layer names to GeoDataFrames/DataFrames.
	            - If `create_df=False`: Returns `None` after saving layers to GeoPackage files.

	    Raises:
	        ValueError:
	            If `create_df=True` and multiple layers are requested.
	        FileNotFoundError:
	            If an ENC file does not exist at the specified path.

	    Examples:
	        # Get DataFrame for a single layer
	        df = processor.enc_to_layers('US5VA51M.GPKG', 'Layer1', create_df=True)

	        # Export multiple layers to GeoPackage
	        processor.enc_to_layers(['US5VA51M.GPKG', 'US5VA50M.GPKG'], ['Layer1', 'Layer2'], export_folder='output')

	    Notes:
	        - ENC files must be in GeoPackage (.gpkg) format.
	        - Layers not found in ENC files are skipped with a warning.
	        - GeoPackage files are named after their respective layer names.
	    """
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.INFO)

		# Validate inputs
		enc_names = self._is_str_list(enc_names)
		layer_names = self._is_str_list(layer_names)


		if create_df and len(layer_names) > 1:
			raise ValueError("DataFrame creation is only supported for single layer name")

		# Ensure output directory exists
		if not create_df:
			if export_folder:
				export_path = os.path.join(self.folder_path, export_folder)
				if not os.path.isdir(export_path):
					os.makedirs(export_path, exist_ok=True)
			else:
				export_path = self.output_dir
				if not os.path.isdir(export_path):
					os.makedirs(export_path, exist_ok=True)

		# Process each layer
		layer_dict = {}
		total_layers = len(layer_names)

		for layer_idx, layer_name in enumerate(layer_names, 1):
			logger.info(f"Processing layer {layer_idx}/{total_layers}: {layer_name}")
			features = []
			geometries = []
			layer_schema = None
			layer_crs = None

			# Process each ENC for current layer
			for enc in enc_names:
				# Ensure ENC has .gpkg ending
				enc = self._is_gpkg(enc)
				enc_path = os.path.join(self.folder_path, enc)

				if not os.path.exists(enc_path):
					logger.warning(f"ENC file not found: {enc}")
					continue

				try:
					# Check if layer exists in current ENC
					available_layers = fiona.listlayers(enc_path)
					if layer_name not in available_layers:
						logger.debug(f"Layer {layer_name} not found in {enc}")
						continue

					# Get ENC metadata
					stamp = self._fiona_stamp(enc_path)
					logger.info(f"Stamp created: {stamp}")

					# Process features
					with fiona.open(enc_path, layer=layer_name) as layer:
						if layer_schema is None:
							layer_schema = layer.schema.copy()
							layer_schema['properties'].update({
								'ENC_NAME': 'str',
								'ENC_EDITION': 'str',
								'ENC_UPDATE': 'str'
							})
							layer_crs = layer.crs

						logger.info(f"Reading features from {enc}")
						for feature in layer:
							# Clean properties during extraction
							properties = {}
							for k, v in feature['properties'].items():
								if isinstance(v, (int, float)) and v == -2147483648:
									properties[k] = None
								elif pd.isna(v):
									properties[k] = None
								else:
									properties[k] = v

							if debug:
								logger.info(f"Properties: {properties}")

							# Add ENC metadata
							properties.update({
								'ENC_NAME': stamp['ENC_NAME'],
								'ENC_EDITION': stamp['ENC_EDITION'],
								'ENC_UPDATE': stamp['ENC_UPDATE']
							})
							features.append(properties)

							if 'geometry' in feature and feature['geometry']:
								geometries.append(feature['geometry'])

				except Exception as e:
					logger.error(f"Error processing {enc}: {str(e)}")
					continue

			try:
				# GeoDataFrame creation
				if geometries:
					df = gpd.GeoDataFrame(
						features,
						geometry=[shape(geom) for geom in geometries],
						crs=layer_crs
					)
				else:
					df = pd.DataFrame(features)

			except Exception as e:
				logger.error(f"Error creating DataFrame for layer '{layer_name}': {e}")
				continue

			# Handle output based on mode
			if create_df:
				return df
			else:
				output_path = os.path.join(export_path, f"{layer_name}.gpkg")

			if not create_df:
				if geometries:
					with fiona.open(output_path, 'w',
					                driver='GPKG',
					                schema=layer_schema,
					                crs=layer_crs) as dst:
						for _, row in df.iterrows():
							# Ensure properties match schema order and handle NaN value
							ordered_properties = {
								k: None if pd.isna(row.get(k)) else row.get(k)
								for k in layer_schema['properties'].keys()
							}
							if debug:
								logger.info(f"Adding properties: {ordered_properties}")
							dst.write({
								'geometry': mapping(row.geometry) if row.geometry else None,
								'properties': ordered_properties
							})
				else:
					with fiona.open(output_path, 'w',
					                driver='GPKG',
					                schema=layer_schema,
					                options=['ASPATIAL_VARIANT=GPKG_ATTRIBUTES']) as dst:
						for _, row in df.iterrows():
							# Ensure properties match schema order and handle NaN value
							ordered_properties = {
								k: None if pd.isna(row.get(k)) else row.get(k)
								for k in layer_schema['properties'].keys()
							}
							if debug:
								logger.info(f"Adding properties: {ordered_properties}")
							dst.write({
								'geometry': None,
								'properties': ordered_properties
							})

					logger.info(f"Saved layer to: {output_path}")
					layer_dict[layer_name] = df

			return None if not create_df else layer_dict


	def read_enc_layers(self, enc_name: list[str], layer_name: list[str],
	                    save_to_gpkg: bool = False, output_path: str = None,
	                    verbose: bool = False) -> Union[Dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
		"""
		Read layers from ENC files with option to save as separate GeoPackages

		Args:
			enc_name: Single ENC filename or list of ENC files
			layer_name: Single layer name or list of layer names
			save_to_gpkg: If True, saves each layer as separate GeoPackage
			output_path: Custom output path for saved GeoPackages
			verbose: Print progress information

		Returns:
			If save_to_gpkg=True: Dictionary mapping layer names to GeoDataFrames
			If save_to_gpkg=False: Single concatenated GeoDataFrame
		"""
		# Convert inputs to lists
		enc_names = [enc_name] if isinstance(enc_name, str) else enc_name
		layer_names = [layer_name] if isinstance(layer_name, str) else layer_name
		layer_dict = {}

		for layer in layer_names:
			layer_dict[layer] = gpd.GeoDataFrame()

			for enc in enc_names:
				if not enc.endswith('.gpkg'):
					enc += '.gpkg'

				enc_path = os.path.join(self.folder_path, enc)
				try:
					gdf = gpd.read_file(enc_path, layer=layer)
					stamp = self._fiona_stamp(enc_path)

					gdf['ENC_NAME'] = stamp['ENC_NAME']
					gdf['ENC_EDITION'] = stamp['ENC_EDITION']
					gdf['ENC_UPDATE'] = stamp['ENC_UPDATE']


					layer_dict[layer] = pd.concat([layer_dict[layer], gdf], ignore_index=True)
				except Exception as e:
					print(f"Error reading layer {layer} from {enc}: {str(e)}")

		return layer_dict

	def explore_gpkg_layers(self, gpkg_path: str, acronym_convert: bool = False,
	                        property_convert: bool = False, prop_mixed: bool = False,
	                        debug: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
		"""
		Opens and reads a layer-specific GeoPackage file, converting S-57 acronyms and properties.

		Args:
			gpkg_path: Path to the layer GeoPackage file
			acronym_convert: Convert S-57 acronyms to full names
			property_convert: Convert property values to meaningful names
			prop_mixed: Include original codes with converted property values
			debug: Enable debug logging

		Returns:
			GeoDataFrame if geometry present, DataFrame otherwise

		Examples:
			# Basic usage
			df = gpkg.explore_gpkg_layers("output/LIGHTS.gpkg")

			# With full conversion
			df = gpkg.explore_gpkg_layers("output/LIGHTS.gpkg",
										acronym_convert=True,
										property_convert=True)
		"""
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if debug else logging.INFO)

		if not os.path.exists(gpkg_path):
			logger.error(f"GeoPackage file not found: {gpkg_path}")
			return None

		try:
			# Check if file has geometry
			with fiona.open(gpkg_path) as src:
				has_geometry = src.schema.get('geometry') is not None

			if has_geometry:
				df = gpd.read_file(gpkg_path)
				if debug:
					logger.info(f"Loaded GeoPackage with geometry. CRS: {df.crs}")
			else:
				df = pd.read_file(gpkg_path)
				if debug:
					logger.info("Loaded GeoPackage without geometry")

			if acronym_convert:
				converted_columns = {}
				for col in df.columns:
					if col not in ['geometry', 'ENC_NAME', 'ENC_EDITION', 'ENC_UPDATE']:
						if property_convert:
							converted_properties = {}
							for idx, value in df[col].items():
								converted_value = self.data.s57_properties_convert(
									col, value, prop_mixed=prop_mixed, debug=debug
								)
								converted_properties[idx] = converted_value
							df[col] = pd.Series(converted_properties)

						converted_name = f"{self.data.s57_attributes_convert(col) or col} ({col})"
						converted_columns[col] = converted_name

				df = df.rename(columns=converted_columns)

			return df

		except Exception as e:
			logger.error(f"Error processing GeoPackage: {str(e)}")
			return None

class GPKG:
	def __init__(self, input_source: Union[str, List[str]], find_all: bool = False):
		"""
		Initialize GPKG handler that works with folder path, single ENC file, or list of file paths

		Args:
			input_source: Either folder path, single .gpkg file path, or list of .gpkg file paths
			find_all: If True, find all ENC files in folder, otherwise use provided list
		"""
		self.enc_files = []
		self.sorted_encs: Dict[str, List[str]] = defaultdict(list)
		self.usage_bands: Dict[str, str] = {
			'1': 'Overview',
			'2': 'General',
			'3': 'Coastal',
			'4': 'Approach',
			'5': 'Harbor',
			'6': 'Berthing'
		}

		self.enc_db = NOAA_DB().get_dataframe()
		self.data = Data()

		# Handle input source type
		if isinstance(input_source, str):
			if input_source.endswith('.gpkg'):
				# Input is a single ENC file
				if not os.path.isfile(input_source):
					raise FileNotFoundError(f"The file {input_source} does not exist.")
				self.folder_path = os.path.dirname(input_source)
				self.enc_files = [os.path.basename(input_source)]
			else:
				# Input is a folder path
				self.folder_path = input_source
				if not os.path.isdir(self.folder_path):
					raise FileNotFoundError(f"The folder path {self.folder_path} does not exist.")
				self.enc_files = [file for file in os.listdir(self.folder_path)
				                  if file.endswith('.gpkg')]


		elif isinstance(input_source, list):
			# Input is a list of file paths
			self.enc_files = [os.path.basename(path) for path in input_source
			                  if path.endswith('.gpkg')]

			# Set folder path as common parent directory
			if self.enc_files:
				self.folder_path = os.path.dirname(input_source[0])
			else:
				raise ValueError("No valid .gpkg files found in provided paths")
		else:
			raise TypeError("Input must be either file path, folder path, or list of file paths")

		if find_all:
			pattern = r'\.\d+\.gpkg$'
			all_encs = [file for file in os.listdir(self.folder_path) if file.endswith('.gpkg')]
			# Filter out versioned files
			filtered_files = [f for f in all_encs if not re.search(pattern, f)]
			self.enc_files = filtered_files

		# Set up output directory
		self.output_dir = os.path.join(self.folder_path, 'export_folder')
		os.makedirs(self.output_dir, exist_ok=True)

		if not self.enc_files:
			print("No ENC files found in the provided source.")

	def get_raw_enc_list(self) -> list[str]:
		"""
		Returns a list of ENC file names.
		Returns:
			List[str]: List of ENC file names
		"""
		return self.enc_files

	def get_folder_path(self) -> str:
		"""
		Returns the folder path where ENC files are located.
		Returns:
			str: Folder path where ENC files are located
		"""
		return self.folder_path

	def _is_gpkg(self, gpkg_name: str, debug = False) -> str:
		"""
	    Ensures filename has .gpkg extension.

	    Args:
	        gpkg_name: Filename to check/modify
	    Returns:
	        str: Filename with .gpkg extension
	    Raises:
	        TypeError: If input is None or not string
	    """
		if debug:
			print(f"Input: {gpkg_name} \nType: {type(gpkg_name)}")
		if not isinstance(gpkg_name, str) or not gpkg_name:
			raise TypeError("Filename must be a non-empty string")
		converted_name = f"{gpkg_name}.gpkg" if not gpkg_name.endswith('.gpkg') else gpkg_name
		if debug:
			print(f"Output: {converted_name} \nType: {type(converted_name)}")
		return converted_name


	def _is_str_list(self, str_list: Union[str, List[str]], debug = False) -> List[str]:
		"""
	    Validates and converts input to list of strings.

	    Args:
	        str_list: Single string or list of strings to validate
	    Returns:
	        List[str]: Original list or single string converted to list
	    Raises:
	        TypeError: If input is None or contains non-string elements
	    Examples:
	        _is_str_list("test")
	        ["test"]
	        _is_str_list(["a", "b"])
	        ["a", "b"]
	        _is_str_list(["a", 1])  # Raises TypeError
	    """
		if str_list is None:
			raise TypeError("Input cannot be None")
		if debug:
			print(f"Input type: {type(str_list)}")
			print(f"Input value: {str_list}")
		# If str_list is a string, convert it to a list
		confirmed_str_list = [str_list] if isinstance(str_list, str) else str_list
		if debug:
			print(f"Converted value: {confirmed_str_list}")

		if all(isinstance(item, str) for item in confirmed_str_list):
			return confirmed_str_list
		else:
			raise TypeError("Input must be a string or a list of strings.")


	def _fiona_stamp(self, enc_path):

		with fiona.open(enc_path, 'r', layer='DSID') as dsid_layer:
			dsid_info = next(iter(dsid_layer))['properties']

		enc_name = dsid_info.get('DSID_DSNM', 'Unknown')
		# Remove .gpkg or .000 ending from ENC_NAME
		enc_name = enc_name.rsplit('.', 1)[0]

		return {
			'ENC_NAME': enc_name,
			'ENC_EDITION': dsid_info.get('DSID_EDTN', 'Unknown'),
			'ENC_UPDATE': dsid_info.get('DSID_UPDN', 'Unknown')
		}

	def _gpd_stamp(self, enc_path):

		dsid = gpd.read_file(enc_path, layer='DSID')
		return {
			'ENC_NAME': dsid['DSID_DSNM'].iloc[0],
			'ENC_EDITION': dsid['DSID_EDTN'].iloc[0],
			'ENC_UPDATE': dsid['DSID_UPDN'].iloc[0]
		}



	def clean_enc_name(self, enc_files: Union[str, list[str]]) -> Union[str, list[str]]:
		"""
		Removes .gpkg extension from ENC filenames in enc_files list.
		Returns:
			List[str]: List of cleaned ENC names without .gpkg extension
		Examples:
			# Before: ['US5VA51M.gpkg', 'US5VA52M.gpkg']
			# After: ['US5VA51M', 'US5VA52M']
		"""
		if isinstance(enc_files, str):
			return enc_files.rsplit('.', 1)[0]
		else:
			return [enc.rsplit('.', 1)[0] for enc in enc_files]


	def filter_versioned_files(self, enc_files: List[str], exclude_versions: bool = True) -> List[str]:
		"""
		Filters ENC files by handling versioned files (e.g., US5CA9BM.1.gpkg).

		Args:
			enc_files: List of ENC filenames
			exclude_versions: If True, excludes versioned files. If False, keeps only latest version
		Returns:
			List[str]: Filtered list of ENC files
		Examples:
			Input: ['US5CA9BM.1.gpkg', 'US5CA11M.0.gpkg', 'US5CA12M.gpkg']
			Output with exclude_versions=True: ['US5CA12M.gpkg']
			Output with exclude_versions=False: ['US5CA9BM.1.gpkg', 'US5CA11M.0.gpkg', 'US5CA12M.gpkg']
		"""
		if not exclude_versions:
			return enc_files

		# Regular expression to match versioned files
		pattern = r'\.\d+\.gpkg$'

		# Filter out versioned files
		filtered_files = [f for f in enc_files if not re.search(pattern, f)]
		return filtered_files

	def normalize_enc_files(self, enc_files: List[str]) -> List[str]:
		"""
		Normalizes versioned ENC filenames by removing version numbers.

		Args:
			enc_files: List of ENC filenames
		Returns:
			List[str]: List with normalized filenames
		Examples:
			Input: ['US5CA9BM.1.gpkg', 'US5CA11M.0.gpkg', 'US5CA12M.gpkg']
			Output: ['US5CA9BM.gpkg', 'US5CA11M.gpkg', 'US5CA12M.gpkg']
		"""
		# Regular expression to match version numbers before .gpkg
		pattern = r'(\.\d+)(\.gpkg)$'

		# Normalize filenames and get unique values using set
		normalized_files = list(set(re.sub(pattern, r'\2', f) for f in enc_files))

		# Sort for consistent output
		normalized_files.sort()
		return normalized_files



	def usage_band(self, clean_name: bool = True):
		"""Sort ENC files by usage band."""
		"""Return: A dictionary with usage bands as keys and a list of filenames as values."""
		self.sorted_encs.clear()
		working_files = self.normalize_enc_files(self.enc_files)
		print(f" After: {working_files}")

		if clean_name:
			working_files = self.clean_enc_name(working_files)
			print(f" After: {working_files}")

		for filename in working_files:
			usage_band = filename[2]
			if usage_band in self.usage_bands:
				self.sorted_encs[self.usage_bands[usage_band]].append(filename)

		# Add count to each usage band
		enc_by_udage_band = {band: (files, len(files)) for band, files in self.sorted_encs.items()}
		return enc_by_udage_band

	def enc_folder_summary(self,detailed: bool = False,
								show_outdated: bool = False,
	                            noaa_data: bool = False,
								save_to_csv: bool = False,
								csv_name: str = "ENC_Summary",
	                            filter_versioned: bool = True,
								clean_name: bool = True,
	                       ):
		"""
			Summarizes ENC files with .gpkg extension in the specified folder..

			Args:
				detailed (bool): If True, provides detailed information about ENC files.
				show_outdated (bool): If True, includes outdated ENC files in the summary.
				noaa_data (bool): If True, includes NOAA data in the summary.
				save_to_csv (bool): If True, saves the summary to a CSV file.
				csv_name (str): Name of the CSV file for saving the summary (if enabled).
				filter_versioned (bool): If True, filters out versioned ENC files, like ['US5CA9BM.1.gpkg', 'US5CA11M.0.gpkg', 'US5CA12M.gpkg']
				clean_name (bool): If True, removes .gpkg extension from ENC names.

			Returns:
				pd.DataFrame: A DataFrame summarizing ENC files.


			"""
		# Check if folder exists
		if not os.path.isdir(self.folder_path):
			raise FileNotFoundError(f"The folder path {self.folder_path} does not exist.")

		# Collect all ENC files
		all_encs = [file for file in os.listdir(self.folder_path) if file.endswith('.gpkg')]
		if not all_encs:
			print("No ENC files found in the folder.")
			return pd.DataFrame()  # Return an empty DataFrame

		# Initialize DataFrame for detailed summary
		detailed_data = []
		filtered_ver_files = []
		if filter_versioned:
			filtered_ver_files = self.filter_versioned_files(all_encs)
		else:
			filtered_ver_files = all_encs


		for file in filtered_ver_files:
			enc_path = os.path.join(self.folder_path, file)
			try:
				# Extract metadata using fiona_stamp
				stamp = self._fiona_stamp(enc_path)

				if detailed:
					# Check for outdated ENCs if required
					if show_outdated:
						db_entry = self.enc_db[self.enc_db['ENC_Name'] == stamp['ENC_NAME']]
						if not db_entry.empty:
							db_edition = db_entry['Edition'].iloc[0]
							db_update = db_entry['Update'].iloc[0]
							if (
									(stamp['ENC_EDITION'] < db_edition)
									or (stamp['ENC_EDITION'] == db_edition and stamp['ENC_UPDATE'] < db_update)
							):
								stamp['OUTDATED'] = True
							else:
								stamp['OUTDATED'] = False
							if noaa_data:
								get_string = f"Ed: {db_edition},  Upd: {db_update}, Date: {db_entry['Update_Application_Date'].iloc[0]}  "
								stamp['NOAA_DATA'] = get_string
						else:
							stamp['OUTDATED'] = False

					detailed_data.append(stamp)

			except Exception as e:
				print(f"Error processing file {file}: {e}")

		if detailed:
			# Create DataFrame from detailed data
			df = pd.DataFrame(detailed_data)

			# Save to Datafreame if required
			if save_to_csv:
				if not csv_name.endswith('.csv'):
					csv_name += '.csv'
				output_path = os.path.join(self.folder_path, csv_name)
				try:
					df.to_csv(output_path)
					print(f"Summary saved to GeoPackage: {output_path}")
				except Exception as e:
					print(f"Error saving GeoPackage: {e}")

			return df
		else:
			if clean_name:
				enc_files = self.clean_enc_name(filtered_ver_files)
			else:
				enc_files = self.enc_files
			return pd.DataFrame({'ENC_Files': enc_files})

	def bbox(self, enc_files):
		boundary_data = []

		for enc_file in enc_files:
			enc_file = self._is_gpkg(enc_file)
			enc_path = os.path.join(self.folder_path, enc_file)
			stamp = self._fiona_stamp(enc_path)

			with fiona.open(enc_path, layer='M_COVR') as layer:
				for feature in layer:
					if feature['properties'].get('CATCOV') == 1:  # 1 indicates coverage area
						boundary = {
							'ENC_NAME': stamp['ENC_NAME'],
							'ENC_EDITION': stamp['ENC_EDITION'],
							'ENC_UPDATE': stamp['ENC_UPDATE'],
							'geometry': feature['geometry']
						}
						boundary_data.append(boundary)

		return gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

	def layers_summary(self, enc_files: Union[str, List[str]], unique: bool = False, acronym_convert: bool = False, show_log: bool = False) -> Union[Dict[str, List[str]], List[str]]:
		"""
	    Summarizes layers from one or more ENC (Electronic Navigational Chart) GeoPackage files.

	    Args:
	        enc_files (Union[str, List[str]]): Single ENC filename or list of ENC filenames.
	        unique (bool, optional): If True, returns only unique layers across all ENCs. Defaults to False.
	        acronym_convert (bool, optional): If True, converts ENC GPKG layer acronyms to full names. Defaults to False.
	        show_log (bool, optional): If True, prints detailed processing information. Defaults to False.

	    Returns:
	        Union[Dict[str, List[str]], List[str]]:
	            - If `unique=False`, returns a dictionary mapping ENC filenames to their list of layers.
	            - If `unique=True`, returns a sorted list of unique layer names across all ENCs.
	    """
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if show_log else logging.INFO)

		enc_files = self._is_str_list(enc_files)
		if show_log:
			logger.info(f"Reading {len(enc_files)} ENC files...")

		all_layers = set()  # Use a set to store unique layers
		enc_layer_dict = {} # Store ENC name with layer names
		unique_dict = {}  # Use a set to store unique layers

		for enc in enc_files:
			# Check enc file format ending with .GPKG
			if show_log:
				logger.info(f"Reading {enc}...")
			enc = self._is_gpkg(enc)
			enc_layer_dict[enc] = {}

			# List all layers in the GeoPackage
			full_path = os.path.join(self.folder_path, enc)
			# Check if ENC file exists
			if not os.path.exists(full_path):
				logger.warning(f"ENC file '{enc}' not found at path '{full_path}'. Skipping.")
				continue

			try:
				# List all layers in the GeoPackage
				fiona_layers = fiona.listlayers(full_path)
				logger.debug(f"Layers in '{enc}': {fiona_layers}")
			except Exception as e:
				logger.error(f"Failed to list layers in ENC '{enc}': {e}")
				continue

			# Add layers to set
			all_layers.update(fiona_layers)

			if not unique:
				# Sort layers alphabetically (case-insensitive)
				sorted_layers = sorted(fiona_layers, key=lambda x: (x[0] or "").lower())
				if acronym_convert:
					for layer in sorted_layers:
						obj_name = self.data.s57_objects_convert(layer)
						enc_layer_dict[enc][layer] = obj_name
				else:
					enc_layer_dict[enc] = sorted_layers

		if unique:
		# Sort unique layers alphabetically (case-insensitive)
			sorted_unique_layers = sorted(all_layers, key=lambda x: (x[0] or "").lower())
			if show_log:
				logger.info("Compiled unique layers across all ENC files")

			if acronym_convert:
				# Convert acronyms to full names
				for layer in sorted_unique_layers:
					obj_name = self.data.s57_objects_convert(layer)
					unique_dict.update({layer : obj_name})
				return unique_dict
			else:
				return sorted_unique_layers
		else:
			return enc_layer_dict


	def explore_enc_layer(self, enc_name: str, layer_name: str, acronym_convert: bool = False,
	                      property_convert: bool = False, prop_mixed: bool = False, show_log: bool = False) -> \
						(Union)[pd.DataFrame, gpd.GeoDataFrame]:
		"""
		Opens and reads an ENC layer, returning GeoDataFrame if geometry present, else DataFrame

		Args:
			enc_name: Name of ENC file
			layer_name: Name of layer to read
			acronym_convert: Convert S-57 acronyms to full names
			property_convert: Convert property values
			show_log: Enable debug logging
			check_type: Include type information in property names

		Returns:
			GeoDataFrame or DataFrame depending on geometry presence
		"""
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if show_log else logging.INFO)

		# Check if user provided ENC Name and its Valid
		if enc_name is None:
			logger.warning("Please provide a valid ENC name.")
			return None
		enc_name = self._is_gpkg(enc_name)
		# Check if Layer Name or List of names provided
		if layer_name is None:
			logger.warning("Please provide a valid layer name.")
			return None
		layer_name = self._is_str_list(layer_name)
		if len(layer_name) > 1:
			logger.warning("Please provide a single layer name.")
			return None

		if enc_name:
			full_path = os.path.join(self.folder_path, enc_name)
			print(full_path)
			if not os.path.exists(full_path):
				logger.warning(f"ENC file '{enc_name}' not found at path '{full_path}'. Skipping.")
				return None

		# List all layers in the GeoPackage ENC file
		fiona_layers = fiona.listlayers(full_path)
		# Check if Input layers match with ENC layers
		verified_layers = [layer for layer in layer_name if layer in fiona_layers]
		if not verified_layers:
			logger.warning(f"Layer not found in {enc_name} ENC file.")
			logger.info(f"Available layers in {enc_name} ENC file: {fiona_layers}")
			return None

		for layer_name in verified_layers:
			# Initialize empty lists for features and geometries
			features = []
			geometries = []

			with fiona.open(full_path, layer=layer_name) as layer:
				# Iterate through features
				for feature in layer:
					# Store properties
					properties = dict(feature['properties'])

					if acronym_convert:
						if property_convert:
							converted_properties = {}
							for key, value in properties.items():
								k = f"{self.data.s57_attributes_convert(key) or key} ({key})"
								v = self.data.s57_properties_convert(key, value, prop_mixed=prop_mixed, debug=show_log)
								converted_properties[k] = v

							properties = converted_properties


						else:
							properties = {f"{self.data.s57_attributes_convert(key) or key} ({key})": value
										  for key, value in properties.items()}

					features.append(properties)

					if 'geometry' in feature and feature['geometry']:
						geometries.append(feature['geometry'])

				if show_log:
					logger.info(f"Features in layer '{layer_name}':")
					logger.info(f"Total features: {len(layer)}")
					if geometries:
						logger.info(f"CRS: {layer.crs}")
						logger.info(f"Bounds: {layer.bounds}")

			# Create appropriate dataframe type
			if geometries:
				gdf = gpd.GeoDataFrame(
					features,
					geometry=[shape(geom) for geom in geometries],
					crs=layer.crs
				)
				return gdf
			else:
				pdf = pd.DataFrame(features)
				return pdf


	def enc_to_layers(self,
	                  enc_names: Union[str, List[str]],
	                  layer_names: Union[str, List[str]],
	                  export_folder: str = None,
	                  create_df: bool = False,
	                  debug:bool = False) -> Union[Dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame, None]:
		"""
		 Process and extract layers from multiple ENC (Electronic Navigational Chart) files efficiently.

	    Args:
	        enc_names (Union[str, List[str]]):
	            Single ENC filename or list of ENC filenames.
	        layer_names (Union[str, List[str]]):
	            Layer name or list of layer names to process.
	        export_folder (str, optional):
	            Directory to save GeoPackage files. If None, uses `self.output_dir`.
	        create_df (bool, optional):
	            If True, returns a DataFrame/GeoDataFrame for the requested layer(s).
	            If False, saves layers to GeoPackage files. Defaults to False.

	    Returns:
	        Union[Dict[str, Union[gpd.GeoDataFrame, pd.DataFrame], None]:
	            - If `create_df=True` and a single layer is requested: Returns a single GeoDataFrame/DataFrame.
	            - If `create_df=True` and multiple layers are requested: Returns a dictionary mapping layer names to GeoDataFrames/DataFrames.
	            - If `create_df=False`: Returns `None` after saving layers to GeoPackage files.

	    Raises:
	        ValueError:
	            If `create_df=True` and multiple layers are requested.
	        FileNotFoundError:
	            If an ENC file does not exist at the specified path.

	    Examples:
	        # Get DataFrame for a single layer
	        df = processor.enc_to_layers('US5VA51M.GPKG', 'Layer1', create_df=True)

	        # Export multiple layers to GeoPackage
	        processor.enc_to_layers(['US5VA51M.GPKG', 'US5VA50M.GPKG'], ['Layer1', 'Layer2'], export_folder='output')

	    Notes:
	        - ENC files must be in GeoPackage (.gpkg) format.
	        - Layers not found in ENC files are skipped with a warning.
	        - GeoPackage files are named after their respective layer names.
	    """
		# Initialize logger
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.INFO)

		# Validate inputs
		enc_names = self._is_str_list(enc_names)
		layer_names = self._is_str_list(layer_names)


		if create_df and len(layer_names) > 1:
			raise ValueError("DataFrame creation is only supported for single layer name")

		# Ensure output directory exists
		if not create_df:
			if export_folder:
				export_path = os.path.join(self.folder_path, export_folder)
				if not os.path.isdir(export_path):
					os.makedirs(export_path, exist_ok=True)
			else:
				export_path = self.output_dir
				if not os.path.isdir(export_path):
					os.makedirs(export_path, exist_ok=True)

		# Process each layer
		layer_dict = {}
		total_layers = len(layer_names)

		for layer_idx, layer_name in enumerate(layer_names, 1):
			logger.info(f"Processing layer {layer_idx}/{total_layers}: {layer_name}")
			features = []
			geometries = []
			layer_schema = None
			layer_crs = None

			# Process each ENC for current layer
			for enc in enc_names:
				# Ensure ENC has .gpkg ending
				enc = self._is_gpkg(enc)
				enc_path = os.path.join(self.folder_path, enc)

				if not os.path.exists(enc_path):
					logger.warning(f"ENC file not found: {enc}")
					continue

				try:
					# Check if layer exists in current ENC
					available_layers = fiona.listlayers(enc_path)
					if layer_name not in available_layers:
						logger.debug(f"Layer {layer_name} not found in {enc}")
						continue

					# Get ENC metadata
					stamp = self._fiona_stamp(enc_path)
					logger.info(f"Stamp created: {stamp}")

					# Process features
					with fiona.open(enc_path, layer=layer_name) as layer:
						if layer_schema is None:
							layer_schema = layer.schema.copy()
							layer_schema['properties'].update({
								'ENC_NAME': 'str',
								'ENC_EDITION': 'str',
								'ENC_UPDATE': 'str'
							})
							layer_crs = layer.crs

						logger.info(f"Reading features from {enc}")
						for feature in layer:
							# Clean properties during extraction
							properties = {}
							for k, v in feature['properties'].items():
								if isinstance(v, (int, float)) and v == -2147483648:
									properties[k] = None
								elif pd.isna(v):
									properties[k] = None
								else:
									properties[k] = v

							if debug:
								logger.info(f"Properties: {properties}")

							# Add ENC metadata
							properties.update({
								'ENC_NAME': stamp['ENC_NAME'],
								'ENC_EDITION': stamp['ENC_EDITION'],
								'ENC_UPDATE': stamp['ENC_UPDATE']
							})
							features.append(properties)

							if 'geometry' in feature and feature['geometry']:
								geometries.append(feature['geometry'])

				except Exception as e:
					logger.error(f"Error processing {enc}: {str(e)}")
					continue

			try:
				# GeoDataFrame creation
				if geometries:
					df = gpd.GeoDataFrame(
						features,
						geometry=[shape(geom) for geom in geometries],
						crs=layer_crs
					)
				else:
					df = pd.DataFrame(features)

			except Exception as e:
				logger.error(f"Error creating DataFrame for layer '{layer_name}': {e}")
				continue

			# Handle output based on mode
			if create_df:
				return df
			else:
				output_path = os.path.join(export_path, f"{layer_name}.gpkg")

			if not create_df:
				if geometries:
					with fiona.open(output_path, 'w',
					                driver='GPKG',
					                schema=layer_schema,
					                crs=layer_crs) as dst:
						for _, row in df.iterrows():
							# Ensure properties match schema order and handle NaN value
							ordered_properties = {
								k: None if pd.isna(row.get(k)) else row.get(k)
								for k in layer_schema['properties'].keys()
							}
							if debug:
								logger.info(f"Adding properties: {ordered_properties}")
							dst.write({
								'geometry': mapping(row.geometry) if row.geometry else None,
								'properties': ordered_properties
							})
				else:
					with fiona.open(output_path, 'w',
					                driver='GPKG',
					                schema=layer_schema,
					                options=['ASPATIAL_VARIANT=GPKG_ATTRIBUTES']) as dst:
						for _, row in df.iterrows():
							# Ensure properties match schema order and handle NaN value
							ordered_properties = {
								k: None if pd.isna(row.get(k)) else row.get(k)
								for k in layer_schema['properties'].keys()
							}
							if debug:
								logger.info(f"Adding properties: {ordered_properties}")
							dst.write({
								'geometry': None,
								'properties': ordered_properties
							})

					logger.info(f"Saved layer to: {output_path}")
					layer_dict[layer_name] = df

			return None if not create_df else layer_dict


	def read_enc_layers(self, enc_name: list[str], layer_name: list[str],
	                    save_to_gpkg: bool = False, output_path: str = None,
	                    verbose: bool = False) -> Union[Dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
		"""
		Read layers from ENC files with option to save as separate GeoPackages
		Provides Quick access to layer data.
		For production use ( enc_to_layers ) function for Layer table creation.

		Args:
			enc_name: Single ENC filename or list of ENC files
			layer_name: Single layer name or list of layer names
			save_to_gpkg: If True, saves each layer as separate GeoPackage
			output_path: Custom output path for saved GeoPackages
			verbose: Print progress information

		Returns:
			If save_to_gpkg=True: Dictionary mapping layer names to GeoDataFrames
			If save_to_gpkg=False: Single concatenated GeoDataFrame
		"""
		# Convert inputs to lists
		enc_names = [enc_name] if isinstance(enc_name, str) else enc_name
		layer_names = [layer_name] if isinstance(layer_name, str) else layer_name
		layer_dict = {}

		for layer in layer_names:
			layer_dict[layer] = gpd.GeoDataFrame()

			for enc in enc_names:
				if not enc.endswith('.gpkg'):
					enc += '.gpkg'

				enc_path = os.path.join(self.folder_path, enc)
				try:
					gdf = gpd.read_file(enc_path, layer=layer)
					stamp = self._fiona_stamp(enc_path)

					gdf['ENC_NAME'] = stamp['ENC_NAME']
					gdf['ENC_EDITION'] = stamp['ENC_EDITION']
					gdf['ENC_UPDATE'] = stamp['ENC_UPDATE']


					layer_dict[layer] = pd.concat([layer_dict[layer], gdf], ignore_index=True)
				except Exception as e:
					print(f"Error reading layer {layer} from {enc}: {str(e)}")

		return layer_dict


	# --------- Functions for GPKG Layer Files --------- #
	def explore_layer_file(self, gpkg_path: str, acronym_convert: bool = False,
	                        property_convert: bool = False, prop_mixed: bool = False,
	                        show_log: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
		"""
		Opens and reads a layer-specific GeoPackage file, converting S-57 acronyms and properties.
		GPKG Layer file is similar to PostGIS table.

		Args:
			gpkg_path: Path to the layer GeoPackage file
			acronym_convert: Convert S-57 acronyms to full names
			property_convert: Convert property values to meaningful names
			prop_mixed: Include original codes with converted property values
			show_log: Enable debug logging

		Returns:
			GeoDataFrame if geometry present, DataFrame otherwise

		Examples:
			# Basic usage
			df = gpkg.explore_gpkg_layers("output/LIGHTS.gpkg")

			# With full conversion
			df = gpkg.explore_gpkg_layers("output/LIGHTS.gpkg",
										acronym_convert=True,
										property_convert=True)
		"""
		logger = logging.getLogger(__name__)
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		logger.setLevel(logging.DEBUG if show_log else logging.INFO)

		if not os.path.exists(gpkg_path):
			logger.error(f"GeoPackage file not found: {gpkg_path}")
			return None

		try:
			# Check if file has geometry
			with fiona.open(gpkg_path) as src:
				has_geometry = src.schema.get('geometry') is not None

			if has_geometry:
				df = gpd.read_file(gpkg_path)
				if show_log:
					logger.info(f"Loaded GeoPackage with geometry. CRS: {df.crs}")
			else:
				df = pd.read_file(gpkg_path)
				if show_log:
					logger.info("Loaded GeoPackage without geometry")

			if acronym_convert:
				converted_columns = {}
				for col in df.columns:
					if col not in ['geometry', 'ENC_NAME', 'ENC_EDITION', 'ENC_UPDATE']:
						if property_convert:
							converted_properties = {}
							for idx, value in df[col].items():
								converted_value = self.data.s57_properties_convert(
									col, value, prop_mixed=prop_mixed, debug=show_log
								)
								converted_properties[idx] = converted_value
							df[col] = pd.Series(converted_properties)

						converted_name = f"{self.data.s57_attributes_convert(col) or col} ({col})"
						converted_columns[col] = converted_name

				df = df.rename(columns=converted_columns)

			return df

		except Exception as e:
			logger.error(f"Error processing GeoPackage: {str(e)}")
			return None