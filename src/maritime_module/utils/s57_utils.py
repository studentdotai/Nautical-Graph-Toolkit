#!/usr/bin/env python3
"""
s57_utils.py

Utility classes for the maritime module.
- S57_Utils: Handles S-57 attribute and object class definitions from CSVs.
- NOAA_DB: Scrapes and manages NOAA Electronic Navigational Charts (ENC) data.
"""

import logging
from pathlib import Path
from typing import Optional, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationError

try:
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    # This allows the module to be imported for inspection even if dependencies are missing.
    print(f"Warning: A library required by s57_utils is missing: {e}")

logger = logging.getLogger(__name__)

class NoaaChart(BaseModel):
    """A Pydantic model for a single NOAA Electronic Navigational Chart entry."""
    enc_name: str = Field(..., alias='ENC_Name')
    edition: int = Field(..., alias='Edition')
    update: int = Field(..., alias='Update')
    update_application_date: datetime = Field(..., alias='Update_Application_Date')
    issue_date: datetime = Field(..., alias='Issue_Date')
    cgds: Optional[str] = Field(None, alias='CGDs')

    # The `alias` in Field allows us to map from the DataFrame's column names
    # to our preferred Python attribute names.

    @field_validator('update_application_date', 'issue_date', mode='before')
    @classmethod
    def parse_date(cls, value: Any) -> Optional[datetime]:
        """Custom validator to parse date strings into datetime objects."""
        if isinstance(value, str):
            try:
                # Handles the 'MM/DD/YYYY' format from the website
                return datetime.strptime(value, '%m/%d/%Y')
            except ValueError:
                logger.warning(f"Could not parse date: {value}")
                return None
        return value

class S57Utils:
    """
    Utility class for handling S-57 attribute and object class definitions.
    Loads S-57 metadata from CSV files for easy lookup and conversion.
    """
    _s57_attributes_df = None
    _s57_objects_df = None
    _s57_properties_df = None

    def __init__(self):
        """
        Initializes the S57Utils instance and ensures that the necessary
        S-57 definition files are loaded and cached for performance.
        """
        # The data directory is one level up from the 'utils' directory.
        self._data_dir = Path(__file__).resolve().parent.parent / 'data'
        self._load_attributes_df()
        self._load_objects_df()
        self._load_properties_df()

    def get_attributes_df(self) -> Optional['pd.DataFrame']:
        """Returns the cached S-57 attributes DataFrame."""
        return self.__class__._s57_attributes_df

    def get_objects_df(self) -> Optional['pd.DataFrame']:
        """Returns the cached S-57 object classes DataFrame."""
        return self.__class__._s57_objects_df

    def get_properties_df(self) -> Optional['pd.DataFrame']:
        """Returns the cached S-57 properties DataFrame."""
        return self.__class__._s57_properties_df

    def _load_attributes_df(self, csv_filename: str = 's57attributes.csv'):
        """Loads the attributes CSV, using a path relative to the package structure."""
        if self.__class__._s57_attributes_df is None:
            csv_path = self._data_dir / csv_filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"S-57 attributes CSV not found at: {csv_path}")

            # Index by 'Acronym' for faster, more consistent lookups.
            df = pd.read_csv(csv_path)
            # Standardize column names and set index for quick lookups
            df.columns = [col.lower().strip() for col in df.columns]
            # Convert acronym values to lowercase to match lookup logic
            df['acronym'] = df['acronym'].str.lower()
            # Remove duplicates that may occur after lowercase conversion
            # Keep the first occurrence of each acronym
            df = df.drop_duplicates(subset=['acronym'], keep='first')
            self.__class__._s57_attributes_df = df.set_index('acronym')

    def _load_objects_df(self, csv_filename: str = 's57objectclasses.csv'):
        """Loads the object classes CSV, using a path relative to the package structure."""
        if self.__class__._s57_objects_df is None:
            csv_path = self._data_dir / csv_filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"S-57 object classes CSV not found at: {csv_path}")

            df = pd.read_csv(csv_path)
            df.columns = [col.lower().strip() for col in df.columns]
            # Convert acronym values to lowercase to match lookup logic
            df['acronym'] = df['acronym'].str.lower()
            # Remove duplicates that may occur after lowercase conversion
            # Keep the first occurrence of each acronym
            df = df.drop_duplicates(subset=['acronym'], keep='first')
            self.__class__._s57_objects_df = df.set_index('acronym')

    def _load_properties_df(self):
        """
        Loads and merges the attributes and expected input CSVs to create a master
        lookup table for property conversions. Caches the result for performance.
        """
        if self.__class__._s57_properties_df is None:
            try:
                # Load attributes, keeping 'Acronym' and 'Code' for the merge
                attr_csv_path = self._data_dir / 's57attributes.csv'
                attr_df = pd.read_csv(attr_csv_path, usecols=['Code', 'Attribute', 'Acronym', 'Attributetype'])

                # Load expected inputs
                expected_csv_path = self._data_dir / 's57expectedinput.csv'
                expected_df = pd.read_csv(expected_csv_path)

                # Merge the two dataframes on the 'Code' column
                prop_df = pd.merge(attr_df, expected_df,  on='Code', how='left')

                # Clean up data for reliable lookups
                prop_df.dropna(subset=['ID', 'Acronym'], inplace=True)
                self.__class__._s57_properties_df = prop_df
                logger.debug("Successfully loaded and merged S-57 properties lookup table.")
            except FileNotFoundError as e:
                logger.error(f"Could not load S-57 definition files: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing S-57 definition files: {e}")
                raise

    def get_attribute_name(self, acronym: str) -> Optional[str]:
        """Converts an S-57 attribute acronym (e.g., 'NATSUR') to its full name."""
        try:
            # Use .loc for a fast, direct lookup on the index.
            return self.__class__._s57_attributes_df.loc[acronym.lower()]['attribute']
        except (FileNotFoundError, KeyError):
            logger.warning(f"Attribute acronym '{acronym}' not found.")
            return None

    def get_attribute_type(self, acronym: str) -> Optional[str]:
        """
        Gets the S-57 attribute type (e.g., 'I', 'F', 'S', 'L') for a given acronym.
        This is crucial for data type casting and validation.
        """
        if self.__class__._s57_attributes_df is None:
            return None
        try:
            # Acronyms are stored in lowercase in the index
            return self.__class__._s57_attributes_df.loc[acronym.lower(), 'attributetype']
        except KeyError:
            logger.debug(f"Attribute type for acronym '{acronym}' not found.")
            return None

    def get_attributes_by_type(self, attr_type: str) -> List[str]:
        """
        Gets a list of all attribute acronyms of a specific type.

        Args:
            attr_type (str): The attribute type to filter by (e.g., 'L', 'I', 'S').

        Returns:
            List[str]: A list of attribute acronyms.
        """
        if self.__class__._s57_attributes_df is None:
            return []
        try:
            return self.__class__._s57_attributes_df[self.__class__._s57_attributes_df['attributetype'] == attr_type].index.tolist()
        except KeyError:
            return []

    def get_object_class_name(self, acronym: str) -> Optional[str]:
        """Converts an S-57 object class acronym (e.g., 'SOUNDG') to its full name."""
        try:
            return self.__class__._s57_objects_df.loc[acronym.lower()]['objectclass']
        except (FileNotFoundError, KeyError):
            logger.warning(f"Object Class acronym '{acronym}' not found.")
            return None

    @staticmethod
    def _parse_s57_string_value(s: str) -> str:
        """
        Extracts the value part from a complex S-57 string like '(1:4,1)',
        returning just '4,1'. If no colon is present, returns the original string.
        """
        if ':' in s:
            return s.strip('()').split(':', 1)[1].strip()
        return s

    def s57_properties_convert(self, acronym_str: str, property_value: Any, prop_mixed: bool = False,
                             debug: bool = False) -> Union[str, List[str], None]:
        """
        Converts an S-57 layer property value to its human-readable meaning.

        This function is designed to handle the various data formats found in
        S-57 data, including single IDs, comma-separated lists, and complex
        string formats like '(1:4,1)'.

        Args:
            acronym_str (str): The 6-character S-57 attribute acronym (e.g., 'NATSUR').
            property_value (Any): The raw value from the ENC data (e.g., 6, '4,1', '(1:4,1)').
            prop_mixed (bool, optional): If True, returns "Meaning (ID)" format. Defaults to False.
            debug (bool, optional): If True, logs detailed processing steps. Defaults to False.

        Returns:
            Union[str, List[str], None]: The converted, human-readable value(s).
                                         - A single string for a single ID.
                                         - A list of strings for multiple IDs.
                                         - The original value if no conversion is applicable.
                                         - None for null/empty input.
        """
        # --- 1. Initial Data Validation and Cleanup ---
        # FIX: Handle empty lists explicitly before calling pd.isna() to avoid ValueError.
        # An empty list is a valid, non-null value that should be handled.
        if property_value is None:
            return None
        if isinstance(property_value, list) and not property_value:
            return None
        # Handle the specific integer used for null in some GDAL versions
        if isinstance(property_value, (int, float)) and property_value == -2147483648:
            return None

        # --- 2. Prepare Lookup Table ---
        prop_df = self.__class__._s57_properties_df

        # Filter to only the relevant acronym for much faster lookups
        attr_lookup = prop_df[prop_df['Acronym'] == acronym_str.upper()]

        if attr_lookup.empty:
            if debug: logger.debug(f"Acronym '{acronym_str}' not in properties table. Returning original value.")
            return property_value

        # Get the attribute type (S=String, I=Integer, E=Enumerated, L=List, etc.)
        attr_type = attr_lookup['Attributetype'].iloc[0]
        if attr_type == 'S':  # Free text string, no conversion needed
            if debug: logger.debug(f"Acronym '{acronym_str}' is free text (Type S). Returning original value.")
            return property_value

        # --- 3. Define Internal Helper for ID-to-Meaning Lookup ---
        def _lookup_id(value_id: Any) -> str:
            try:
                num_id = int(value_id)
                match = attr_lookup[attr_lookup['ID'] == num_id]
                if not match.empty:
                    meaning = match['Meaning'].iloc[0]
                    return f"{meaning} ({num_id})" if prop_mixed else meaning
                else:
                    if debug: logger.debug(f"No meaning found for {acronym_str} ID {num_id}. Returning ID.")
                    return str(num_id)
            except (ValueError, TypeError, IndexError):
                if debug: logger.debug(f"Value '{value_id}' is not a valid ID. Returning as is.")
                return str(value_id)

        # --- 4. Process Input Based on Type ---
        value_list = []
        if isinstance(property_value, str):
            cleaned_value = S57Utils._parse_s57_string_value(property_value)
            value_list = [val.strip() for val in cleaned_value.split(',')]
        elif isinstance(property_value, list):
            for item in property_value:
                if isinstance(item, str):
                    value_list.extend([val.strip() for val in item.split(',')])
                else:
                    value_list.append(item)
        elif isinstance(property_value, (int, float)):
            # For enumerated types, lookup the meaning. For plain integers, return as is.
            return _lookup_id(property_value) if attr_type in ['E', 'L'] else property_value
        else:
            return property_value  # Return unhandled types as is

        # --- 5. Perform Lookup and Format Output ---
        results = [_lookup_id(v) for v in value_list if v != '']
        if len(results) == 1:
            return results[0]
        return results if results else property_value

class NoaaDatabase:
    """
    A class that scrapes and manages NOAA Electronic Navigational Charts (ENC) data.

    This class is adapted to the new website structure as of mid-2024, which
    uses a nested table layout.

    Attributes:
        url (str): The URL of the NOAA ENC website.
        df (pd.DataFrame): DataFrame containing the scraped ENC data. Lazily loaded.
        session (requests.Session): The HTTP session for making requests.

    Methods:
        get_dataframe(): Returns the scraped data as a pandas DataFrame.
        save_to_csv(filename="ENC_DB.csv"): Saves the DataFrame to a CSV file.
    """

    def __init__(self):
        self.url = "https://www.charts.noaa.gov/ENCs/ENCsIndv.shtml"
        self.df: Optional[pd.DataFrame] = None
        self.session = requests.Session()
        # It's good practice to identify your scraper with a User-Agent.
        self.session.headers.update({
            'User-Agent': 'MaritimeModule/1.0 (Python Scraper; +https://vectornautical.com)'
        })

    def get_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns the scraped data as a pandas DataFrame.
        The data is scraped only on the first call and cached for subsequent calls.

        Args:
            force_refresh (bool): If True, bypasses the cache and re-scrapes the data.
        """
        if self.df is None or force_refresh:
            if force_refresh:
                logger.info("Forcing a refresh of NOAA ENC data.")
            self._scrape_enc_data()
        return self.df

    def get_charts(self, force_refresh: bool = False, as_dataframe: bool = False) -> Union[List[NoaaChart], pd.DataFrame]:
        """
        Scrapes NOAA data, validates it, and returns it as Pydantic models or a DataFrame.

        This method provides a much more robust and predictable output than a raw
        DataFrame, making it ideal for application and ML pipelines.

        Args:
            force_refresh (bool): If True, bypasses the cache and re-scrapes the data.
            as_dataframe (bool): If True, returns a pandas DataFrame. If False (default),
                                 returns a list of Pydantic NoaaChart models.

        Returns:
            Union[List[NoaaChart], pd.DataFrame]: A list of validated chart objects or a
                                                   DataFrame containing the validated data.
        """
        dataframe = self.get_dataframe(force_refresh=force_refresh)
        if dataframe is None:
            # If as_dataframe is requested, return an empty DataFrame, otherwise an empty list.
            return pd.DataFrame() if as_dataframe else []

        # Convert the DataFrame to a list of dictionaries to feed into Pydantic
        chart_records = dataframe.reset_index().to_dict('records')

        validated_charts = []
        for record in chart_records:
            try:
                # Pydantic performs runtime validation and type coercion here
                validated_charts.append(NoaaChart.model_validate(record))
            except ValidationError as e:
                logger.error(f"Skipping record due to validation error: {record.get('ENC_Name', 'N/A')}\n{e}")

        if as_dataframe:
            if not validated_charts:
                return pd.DataFrame()  # Return empty DataFrame if no charts
            # Convert the list of Pydantic models to a DataFrame
            return pd.DataFrame([chart.model_dump() for chart in validated_charts])

        return validated_charts

    def save_to_csv(self, filename: str = "noaa_enc_data"):
        """
        Saves the DataFrame to a CSV file.
        Automatically appends the .csv extension if it's not already present.

        Args:
            filename (str): The base name for the output file. Defaults to "noaa_enc_data".
        """
        # Ensure the dataframe is loaded before saving.
        if self.df is None:
            self.get_dataframe()

        # --- IMPROVEMENT ---
        # Automatically add the .csv extension if the user hasn't provided it.
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        try:
            self.df.to_csv(filename)
            logger.info(f"Successfully saved NOAA ENC data to {filename}")
        except IOError as e:
            logger.error(f"Failed to save CSV file '{filename}': {e}")
            raise

    def _scrape_enc_data(self):
        """
        Private method to perform the web scraping and data processing.
        This logic is updated to handle the current NOAA nested table structure.
        """
        logger.info(f"Fetching ENC data from {self.url}")
        try:
            response = self.session.get(self.url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            layout_table = soup.find('table')
            if not layout_table:
                raise ValueError("Could not find any <table> element in the HTML content.")

            data_table = layout_table.find('table')
            if not data_table:
                data_table = layout_table

            all_rows = data_table.find_all('tr')
            if not all_rows:
                raise ValueError("No rows (<tr>) found in the data table.")

            header_cells = all_rows[0].find_all('td')
            headers = [cell.text.replace('\xa0', ' ').strip() for cell in header_cells]

            data_rows = []
            for row in all_rows[1:]:
                cells = [cell.text.strip() for cell in row.find_all('td')]
                if cells and len(cells) == len(headers):
                    data_rows.append(cells)

            df = pd.DataFrame(data_rows, columns=headers)

            # --- THIS IS THE FIX ---
            # Remove rows that are just repeated headers from the middle of the data.
            if '#' in df.columns:
                df = df[df['#'] != '#'].copy()

            # Clean up column names for easier use in pandas.
            df.columns = df.columns.str.replace(' ', '_').str.replace('#', 'Num').str.replace('*', '', regex=False)

            if 'Num' in df.columns:
                df.set_index('Num', inplace=True)

            self.df = df
            logger.info("Successfully scraped and parsed NOAA ENC data.")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from NOAA: {e}")
            raise ConnectionError(f"Failed to fetch data from {self.url}: {e}")
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse the NOAA data table. The website structure may have changed. Error: {e}")
            raise RuntimeError(f"Failed to parse ENC data: {e}")
