#!/usr/bin/env python3
# Copyright (C) 2024-2025 Viktor Kolbasov <contact@studentdotai.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
s57_classification.py

Provides a comprehensive classification system for S-57 maritime objects,
encapsulating navigation risk, cost, and other attributes for pathfinding.
"""

import csv
import logging
from enum import Enum
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class NavClass(Enum):
    """Defines the navigation classification levels, from safest to most dangerous."""
    INFORMATIONAL = 0  # Not relevant to navigation, e.g., cartographic text
    SAFE = 1  # Preferred for navigation, e.g., fairways
    CAUTION = 2  # Requires caution, e.g., restricted areas, traffic schemes
    DANGEROUS = 3  # Represents a direct hazard, e.g., wrecks, obstructions


class S57Classifier:
    """
    A service class to provide navigation classification for S-57 objects.

    This class centralizes the logic for determining the risk, cost, and
    traversability of maritime features based on their S-57 acronym.

    Can be initialized with a custom CSV database file, or use the default
    built-in classification database.
    """

    # The default S-57 object classification database.
    # Format: 'ACRONYM': (NavClass, Category, RiskMultiplier, BufferMeters, Description, ImportantAttributes)
    # Output Columns:
        # Acronym - S57 object code
        # NavClass - 0/1/2/3 classification
        # Category - Type (Route, Aid, Depth, Obstruction, etc.)
        # RiskMultiplier - Multiplicative weight factor (0.5=preferred, 1.0=neutral, >1.0=avoid, 100=extreme danger)
        # BufferMeters - Safety buffer distance (0 to 500m)
        # Traversable - Yes/No
        # Description - Human-readable name
        # ImportantAttributes - List of key S-57 attributes for feature extraction

    _DEFAULT_CLASSIFICATION_DB: Dict[str, Tuple[Any, ...]] = {
            # ==================== INFORMATIONAL (0) ====================
            # Meta/Cartographic Objects
            'C_AGGR': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Aggregation - meta object'),
            'C_ASSO': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Association - meta object'),
            'C_STAC': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Stacked on/under'),
            '$AREAS': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Cartographic area'),
            '$LINES': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Cartographic line'),
            '$CSYMB': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Cartographic symbol'),
            '$COMPS': (NavClass.INFORMATIONAL, 'Meta', 0, 0, 'Compass rose'),

            # Terrestrial/Land Features
            'AIRARE': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Airport/airfield'),
            'BUISGL': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Building single'),
            'BUIREL': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Building religious'),
            'CANALS': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Canal', ['horclr']),
            'CANBNK': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Canal bank'),
            'LAKARE': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Lake area'),
            'LAKSHR': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Lake shore'),

            'LNDELV': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Land elevation'),
            'LNDRGN': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Land region'),
            'LNDMRK': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Landmark'),
            'RAILWY': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Railway'),
            'RIVERS': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'River'),
            'RIVBNK': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'River bank'),
            'ROADWY': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Road'),
            'RUNWAY': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Runway'),
            'SPLARE': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Seaplane landing area', ['restrn']),
            'VEGATN': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Vegetation'),
            'MONUMT': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Monument'),
            'SQUARE': (NavClass.INFORMATIONAL, 'Terrestrial', 0, 0, 'Square'),

            # Administrative/Reference Only
            'CHKPNT': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Checkpoint'),
            'CGUSTA': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Coastguard station'),
            'CUSZNE': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Customs zone'),
            'RSCSTA': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Rescue station'),
            'RDOSTA': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Radio station'),
            'RDOCAL': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Radio calling-in point', ['orient', 'trafic']),
            'LOCMAG': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Local magnetic anomaly'),
            'MAGVAR': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Magnetic variation'),
            'DISMAR': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'Distance mark'),
            'NEWOBJ': (NavClass.INFORMATIONAL, 'Administrative', 0, 0, 'New object', ['restrn']),

            # Static Reference Marks
            'DAYMAR': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Daymark'),
            'FOGSIG': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Fog signal'),
            'RADRFL': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Radar reflector'),
            'RADSTA': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Radar station'),
            'RADLNE': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Radar line'),
            'RADRNG': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Radar range'),
            'RTPBCN': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Radar transponder beacon'),
            'RETRFL': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Retro-reflector'),
            'TOPMAR': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Topmark'),
            'SISTAT': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Signal station traffic'),
            'SISTAW': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Signal station warning'),
            'CTRPNT': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Control point'),
            'GENNAV': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Generic NavAid'),
            'notmrk': (NavClass.INFORMATIONAL, 'Reference', 0, 0, 'Notice mark'),

            # Environmental/Oceanographic Data
            'CURENT': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Current data'),
            'TS_PRH': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tidal stream harmonic'),
            'TS_PNH': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tidal stream non-harmonic'),
            'TS_PAD': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tidal stream panel data'),
            'TS_TIS': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tidal stream time series'),
            'TS_FEB': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tidal stream flood/ebb'),
            'T_HMON': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tide harmonic prediction'),
            'T_NHMN': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tide non-harmonic prediction'),
            'T_TIMS': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Tide time series'),
            'WATTUR': (NavClass.INFORMATIONAL, 'Environmental', 0, 0, 'Water turbulence info'),

            # Industrial/Production
            'OSPARE': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Offshore production area', ['restrn']),
            'PRDARE': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Production/storage area'),
            'SILTNK': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Silo/tank'),
            'CONVYR': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Conveyor', ['verclr']),
            'LOGPON': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Log pond'),
            'bunsta': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Bunker station'),
            'refdmp': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Refuse dump'),
            'termnl': (NavClass.INFORMATIONAL, 'Industrial', 0, 0, 'Terminal area'),

            # ==================== SAFE (1) ====================
            # Preferred Routes
            'FAIRWY': (NavClass.SAFE, 'Route', 0.5, 0, 'Fairway - preferred route', ['drval1', 'orient', 'trafic', 'restrn']),
            'NAVLNE': (NavClass.SAFE, 'Route', 0.5, 0, 'Navigation line', ['orient']),
            'RCRTCL': (NavClass.SAFE, 'Route', 0.5, 0, 'Recommended route centerline', ['drval1', 'orient', 'trafic']),
            'RECTRC': (NavClass.SAFE, 'Route', 0.5, 0, 'Recommended track', ['drval1', 'orient', 'trafic']),
            'RCTLPT': (NavClass.SAFE, 'Route', 0.5, 0, 'Recommended traffic lane part', ['orient']),
            'DWRTCL': (NavClass.SAFE, 'Route', 0.5, 0, 'Deep water route centerline', ['drval1', 'orient', 'trafic']),
            'DWRTPT': (NavClass.SAFE, 'Route', 0.5, 0, 'Deep water route part', ['drval1', 'orient', 'trafic', 'restrn']),
            'FERYRT': (NavClass.SAFE, 'Route', 0.8, 50, 'Ferry route'),
            'SUBTLN': (NavClass.SAFE, 'Route', 0.6, 50, 'Submarine transit lane', ['restrn']),
            'TRFLNE': (NavClass.SAFE, 'Route', 0.7, 0, 'Traffic line'),
            'TRNBSN': (NavClass.SAFE, 'Route', 0.8, 0, 'Turning basin'),
            'wtwaxs': (NavClass.SAFE, 'Route', 0.6, 0, 'Waterway axis'),
            'PRCARE': (NavClass.SAFE, 'Route', 0.9, 0, 'Precautionary area', ['restrn']),
            'TSSLPT': (NavClass.SAFE, 'Route', 0.7, 0, 'TSS lane part', ['orient', 'restrn']),

            # Safe Water Marks
            'BCNSAW': (NavClass.SAFE, 'Aid', 1.0, 50, 'Beacon safe water'),
            'BCNCAR': (NavClass.SAFE, 'Aid', 1.0, 50, 'Beacon cardinal'),
            'BOYSAW': (NavClass.SAFE, 'Aid', 1.0, 50, 'Buoy safe water'),
            'BOYCAR': (NavClass.SAFE, 'Aid', 1.0, 50, 'Buoy cardinal'),
            'LIGHTS': (NavClass.SAFE, 'Aid', 1.0, 25, 'Light'),
            'LITFLT': (NavClass.SAFE, 'Aid', 1.0, 50, 'Light float'),
            'LITVES': (NavClass.SAFE, 'Aid', 1.2, 100, 'Light vessel'),
            'bcnwtw': (NavClass.SAFE, 'Aid', 1.0, 50, 'Beacon waterway'),
            'boywtw': (NavClass.SAFE, 'Aid', 1.0, 50, 'Buoy waterway'),

            # Deep Water Areas
            'DEPARE': (NavClass.SAFE, 'Depth', 1.0, 0, 'Depth area - check vessel draft', ['drval1']),
            'SOUNDG': (NavClass.SAFE, 'Depth', 1, 0, 'Sounding - isolated danger', ['depth']),
            'SWPARE': (NavClass.SAFE, 'Depth', 0.8, 0, 'Swept area', ['drval1']),
            'DRGARE': (NavClass.SAFE, 'Depth', 0.9, 0, 'Dredged area', ['drval1', 'restrn']),
            'SEAARE': (NavClass.SAFE, 'Area', 1.0, 0, 'Sea area/named water'),
            'wtwprf': (NavClass.SAFE, 'Depth', 0.9, 0, 'Waterway profile'),

            # Anchorage & Berths
            'ACHARE': (NavClass.SAFE, 'Anchorage', 1.5, 100, 'Anchorage area', ['restrn']),
            'ACHBRT': (NavClass.SAFE, 'Anchorage', 1.3, 75, 'Anchor berth'),
            'BERTHS': (NavClass.SAFE, 'Port', 1.5, 50, 'Berth', ['drval1']),

            # Port Areas
            'HRBARE': (NavClass.SAFE, 'Port', 1.2, 0, 'Harbour area administrative'),
            'HRBFAC': (NavClass.SAFE, 'Port', 1.3, 25, 'Harbour facility'),
            'DOCARE': (NavClass.SAFE, 'Port', 1.4, 50, 'Dock area', ['horclr']),
            'FRPARE': (NavClass.SAFE, 'Port', 1.2, 0, 'Free port area'),
            'SMCFAC': (NavClass.SAFE, 'Port', 1.2, 50, 'Small craft facility'),
            'prtare': (NavClass.SAFE, 'Port', 1.2, 0, 'Port area'),
            'hrbbsn': (NavClass.SAFE, 'Port', 1.3, 50, 'Harbour basin'),
            'wtware': (NavClass.SAFE, 'Area', 1.0, 0, 'Waterway area'),

            # ==================== CAUTION (2) ====================
            # Restricted Areas
            'RESARE': (NavClass.CAUTION, 'Restricted', 5.0, 200, 'Restricted area', ['restrn']),
            'CTNARE': (NavClass.CAUTION, 'Restricted', 4.0, 150, 'Caution area'),

            'MIPARE': (NavClass.CAUTION, 'Restricted', 8.0, 300, 'Military practice area', ['restrn']),
            'ICNARE': (NavClass.CAUTION, 'Restricted', 6.0, 200, 'Incineration area', ['restrn']),
            'DMPGRD': (NavClass.CAUTION, 'Restricted', 5.0, 150, 'Dumping ground', ['restrn']),
            'FSHZNE': (NavClass.CAUTION, 'Restricted', 3.0, 100, 'Fishery zone'),
            'FSHFAC': (NavClass.CAUTION, 'Restricted', 4.0, 150, 'Fishing facility'),
            'FSHGRD': (NavClass.CAUTION, 'Restricted', 3.0, 100, 'Fishing ground'),
            'MARCUL': (NavClass.CAUTION, 'Restricted', 4.0, 150, 'Marine farm/culture', ['valsou', 'restrn']),
            'excnst': (NavClass.CAUTION, 'Restricted', 4.5, 150, 'Exceptional navigation structure'),

            # Traffic Separation
            'TSELNE': (NavClass.CAUTION, 'Traffic', 3.0, 100, 'Traffic separation line'),
            'TSSBND': (NavClass.CAUTION, 'Traffic', 3.5, 100, 'TSS boundary'),
            'TSSCRS': (NavClass.CAUTION, 'Traffic', 4.0, 150, 'TSS crossing', ['restrn']),
            'TSSRON': (NavClass.CAUTION, 'Traffic', 4.5, 200, 'TSS roundabout', ['restrn']),
            'TSEZNE': (NavClass.CAUTION, 'Traffic', 3.0, 100, 'Traffic separation zone'),
            'ISTZNE': (NavClass.CAUTION, 'Traffic', 3.5, 100, 'Inshore traffic zone', ['restrn']),
            'TWRTPT': (NavClass.CAUTION, 'Traffic', 2.5, 50, 'Two-way route part', ['drval1', 'orient', 'trafic']),
            'rtplpt': (NavClass.CAUTION, 'Traffic', 3.0, 100, 'Route planning point'),
            'lg_sdm': (NavClass.CAUTION, 'Traffic', 4.0, 150, 'Max permitted ship dimensions'),
            'lg_vsp': (NavClass.CAUTION, 'Traffic', 3.5, 100, 'Max permitted vessel speed'),
            'tisdge': (NavClass.CAUTION, 'Traffic', 2.5, 0, 'Time schedule'),
            'wtwgag': (NavClass.CAUTION, 'Traffic', 3.0, 50, 'Waterway gauge'),

            # Shallow/Variable Depth
            'SBDARE': (NavClass.CAUTION, 'Depth', 3.0, 100, 'Seabed area - variable depth'),
            'UNSARE': (NavClass.CAUTION, 'Depth', 6.0, 200, 'Unsurveyed area'),
            'SNDWAV': (NavClass.CAUTION, 'Depth', 2.5, 100, 'Sand waves'),
            'SLOTOP': (NavClass.CAUTION, 'Depth', 3.0, 100, 'Slope topline'),
            'SLOGRD': (NavClass.CAUTION, 'Depth', 3.0, 100, 'Sloping ground'),
            'DEPCNT': (NavClass.CAUTION, 'Depth', 2.0, 50, 'Depth contour'),
            'WEDKLP': (NavClass.CAUTION, 'Depth', 3.5, 100, 'Weed/Kelp'),
            'SPRING': (NavClass.CAUTION, 'Environmental', 2.5, 75, 'Spring'),

            # Lateral Marks & Warnings
            'BCNLAT': (NavClass.CAUTION, 'Aid', 2.0, 75, 'Beacon lateral'),
            'BCNSPP': (NavClass.CAUTION, 'Aid', 1.8, 75, 'Beacon special purpose'),
            'BOYLAT': (NavClass.CAUTION, 'Aid', 2.0, 75, 'Buoy lateral'),
            'BOYSPP': (NavClass.CAUTION, 'Aid', 1.8, 75, 'Buoy special purpose'),
            'BOYINB': (NavClass.CAUTION, 'Aid', 2.5, 100, 'Buoy installation'),
            'OILBAR': (NavClass.CAUTION, 'Infrastructure', 4.0, 150, 'Oil barrier'),
            'ICEARE': (NavClass.CAUTION, 'Environmental', 6.0, 200, 'Ice area'),
            '_extgn': (NavClass.CAUTION, 'Aid', 3.0, 100, 'Light extinguished'),

            # Cables & Pipelines
            'CBLARE': (NavClass.CAUTION, 'Cable', 3.0, 150, 'Cable area', ['restrn']),
            'CBLSUB': (NavClass.CAUTION, 'Cable', 4.0, 200, 'Cable submarine', ['drval1']),
            'CBLOHD': (NavClass.CAUTION, 'Cable', 5.0, 100, 'Cable overhead', ['verclr', 'vercsa']),
            'PIPARE': (NavClass.CAUTION, 'Pipeline', 3.0, 150, 'Pipeline area', ['restrn']),
            'PIPSOL': (NavClass.CAUTION, 'Pipeline', 4.0, 200, 'Pipeline submarine/on land', ['drval1']),
            'PIPOHD': (NavClass.CAUTION, 'Pipeline', 5.0, 100, 'Pipeline overhead', ['verclr']),
            'CHNWIR': (NavClass.CAUTION, 'Infrastructure', 3.5, 100, 'Chain/Wire'),

            # Infrastructure - Moderate Risk
            'PILBOP': (NavClass.CAUTION, 'Port', 2.5, 100, 'Pilot boarding place'),
            'MORFAC': (NavClass.CAUTION, 'Port', 3.0, 100, 'Mooring/warping facility'),
            'HULKES': (NavClass.CAUTION, 'Infrastructure', 3.5, 150, 'Hulk'),
            'PONTON': (NavClass.CAUTION, 'Infrastructure', 3.0, 100, 'Pontoon'),
            'FLODOC': (NavClass.CAUTION, 'Infrastructure', 3.5, 150, 'Floating dock', ['drval1', 'horclr']),
            'CRANES': (NavClass.CAUTION, 'Infrastructure', 2.5, 100, 'Crane', ['verclr']),
            'OFSPLF': (NavClass.CAUTION, 'Infrastructure', 4.0, 200, 'Offshore platform'),
            'BUAARE': (NavClass.CAUTION, 'Area', 2.0, 100, 'Built-up area'),
            'CTSARE': (NavClass.CAUTION, 'Port', 3.0, 100, 'Cargo transshipment area'),
            'LOKBSN': (NavClass.CAUTION, 'Infrastructure', 3.5, 150, 'Lock basin', ['horclr']),
            'lokare': (NavClass.CAUTION, 'Infrastructure', 3.5, 150, 'Lock area'),
            'lkbspt': (NavClass.CAUTION, 'Infrastructure', 3.5, 150, 'Lock basin part'),

            # Boundaries
            'TESARE': (NavClass.CAUTION, 'Boundary', 2.0, 0, 'Territorial sea area', ['restrn']),
            'CONZNE': (NavClass.CAUTION, 'Boundary', 1.8, 0, 'Contiguous zone'),
            'COSARE': (NavClass.CAUTION, 'Boundary', 1.8, 0, 'Continental shelf area'),
            'EXEZNE': (NavClass.CAUTION, 'Boundary', 1.8, 0, 'Exclusive Economic Zone'),
            'comare': (NavClass.CAUTION, 'Port', 2.0, 100, 'Communication area'),
            'vehtrf': (NavClass.CAUTION, 'Port', 2.5, 100, 'Vehicle transfer'),

            # ==================== DANGEROUS (3) ====================
            # Critical Obstructions
            'UWTROC': (NavClass.DANGEROUS, 'Obstruction', 100.0, 500, 'Underwater rock/awash rock', ['valsou']),
            'OBSTRN': (NavClass.DANGEROUS, 'Obstruction', 100.0, 500, 'Obstruction', ['valsou', 'catobs']),
            'WRECKS': (NavClass.DANGEROUS, 'Obstruction', 100.0, 500, 'Wreck', ['valsou', 'catwrk']),
            'FOULAR': (NavClass.DANGEROUS, 'Obstruction', 80.0, 400, 'Foul area'),
            'ACHPNT': (NavClass.DANGEROUS, 'Obstruction', 60.0, 300, 'Anchor on seabed'),
            'PILPNT': (NavClass.DANGEROUS, 'Obstruction', 70.0, 400, 'Pile'),

            'ZEMCNT': (NavClass.DANGEROUS, 'Depth', 100.0, 500, 'Zero meter contour'),

            # Isolated Danger Marks
            'BCNISD': (NavClass.DANGEROUS, 'Aid', 90.0, 500, 'Beacon isolated danger'),
            'BOYISD': (NavClass.DANGEROUS, 'Aid', 90.0, 500, 'Buoy isolated danger'),

            # Bridges & Overhead Structures
            'BRIDGE': (NavClass.DANGEROUS, 'Structure', 100.0, 300, 'Bridge', ['verclr', 'horclr']),
            'PYLONS': (NavClass.DANGEROUS, 'Structure', 80.0, 300, 'Pylon/bridge support'),
            'GATCON': (NavClass.DANGEROUS, 'Structure', 80.0, 300, 'Gate', ['verclr', 'drval1', 'horclr']),
            'TUNNEL': (NavClass.DANGEROUS, 'Structure', 70.0, 200, 'Tunnel', ['verclr', 'horclr']),
            'brgare': (NavClass.DANGEROUS, 'Structure', 100.0, 300, 'Bridge area'),

            # Coastline & Shoreline
            'LNDARE': (NavClass.DANGEROUS, 'Coastline', 100, 0, 'Land area'),
            'COALNE': (NavClass.DANGEROUS, 'Coastline', 100.0, 500, 'Coastline'),
            'SLCONS': (NavClass.DANGEROUS, 'Coastline', 90.0, 400, 'Shoreline construction', ['horclr']),
            'CAUSWY': (NavClass.DANGEROUS, 'Structure', 100.0, 500, 'Causeway'),
            'DAMCON': (NavClass.DANGEROUS, 'Structure', 100.0, 500, 'Dam'),
            'DYKCON': (NavClass.DANGEROUS, 'Structure', 90.0, 400, 'Dyke'),
            'FNCLNE': (NavClass.DANGEROUS, 'Structure', 70.0, 300, 'Fence/wall'),
            'FORSTC': (NavClass.DANGEROUS, 'Structure', 80.0, 400, 'Fortified structure'),

            # Environmental Dangers
            'WATFAL': (NavClass.DANGEROUS, 'Environmental', 100.0, 500, 'Waterfall'),
            'TIDEWY': (NavClass.DANGEROUS, 'Environmental', 60.0, 300, 'Tideway'),
            'RAPIDS': (NavClass.DANGEROUS, 'Environmental', 80.0, 400, 'Rapids'),

            # Port Infrastructure - Hard Obstructions
            'DRYDOC': (NavClass.DANGEROUS, 'Infrastructure', 100.0, 300, 'Dry dock', ['drval1', 'horclr']),
            'GRIDRN': (NavClass.DANGEROUS, 'Infrastructure', 90.0, 300, 'Gridiron'),
            'TOWERS': (NavClass.DANGEROUS, 'Structure', 100.0, 400, 'Tower'),
            'STSLNE': (NavClass.DANGEROUS, 'Coastline', 100.0, 500, 'Straight territorial sea baseline'),

        }

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the S57Classifier with either a custom CSV database or the default database.

        Args:
            csv_path (Optional[str]): Path to a custom CSV classification database.
                                     If None, uses the built-in default database.
        """
        if csv_path:
            self._classification_db = self._load_from_csv(csv_path)
        else:
            self._classification_db = self._DEFAULT_CLASSIFICATION_DB.copy()

    @staticmethod
    def _load_from_csv(csv_path: str) -> Dict[str, Tuple[Any, ...]]:
        """
        Load classification database from a CSV file.

        Expected CSV format (with header):
        Acronym,NavClassValue,NavClassName,Category,RiskMultiplier,BufferMeters,IsTraversable,Description

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            Dict[str, Tuple[Any, ...]]: Classification database dictionary.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the CSV file has invalid format or data.
        """
        import os
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        classification_db = {}

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Validate header
            required_fields = ['Acronym', 'NavClassValue', 'Category',
                             'RiskMultiplier', 'BufferMeters', 'Description']
            if not all(field in reader.fieldnames for field in required_fields):
                raise ValueError(f"CSV file missing required fields. Expected: {required_fields}")

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    acronym = row['Acronym'].strip().upper()
                    nav_class_value = int(row['NavClassValue'])
                    category = row['Category'].strip()
                    risk_multiplier = float(row['RiskMultiplier'])
                    buffer_meters = float(row['BufferMeters'])
                    description = row['Description'].strip()

                    # Convert NavClassValue to NavClass enum
                    try:
                        nav_class = NavClass(nav_class_value)
                    except ValueError:
                        raise ValueError(f"Invalid NavClassValue '{nav_class_value}' at row {row_num}")

                    classification_db[acronym] = (
                        nav_class, category, risk_multiplier, buffer_meters, description
                    )

                except (KeyError, ValueError) as e:
                    raise ValueError(f"Error parsing CSV at row {row_num}: {e}")

        if not classification_db:
            raise ValueError("CSV file is empty or contains no valid data")

        logger.info(f" Loaded {len(classification_db)} classifications from {csv_path}")
        return classification_db

    def get_classification(self, acronym: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full classification details for a given S-57 object acronym.

        Args:
            acronym (str): The 6-character S-57 object acronym (case-insensitive).

        Returns:
            Optional[Dict[str, Any]]: A dictionary with classification details, or None if not found.
        """
        acronym = acronym.upper()
        if acronym not in self._classification_db:
            return None

        # Unpack based on new structure: (NavClass, Category, RiskMultiplier, BufferMeters, Description, [ImportantAttributes])
        entry = self._classification_db[acronym]
        nav_class = entry[0]
        category = entry[1]
        risk_mult = entry[2]
        buffer = entry[3]
        desc = entry[4]

        return {
            'acronym': acronym,
            'nav_class': nav_class,
            'category': category,
            'risk_multiplier': risk_mult,
            'buffer_meters': buffer,
            'is_traversable': nav_class != NavClass.DANGEROUS,
            'description': desc
        }

    def get_nav_class(self, acronym: str) -> NavClass:
        """Returns the NavClass enum for a given acronym."""
        classification = self.get_classification(acronym)
        return classification['nav_class'] if classification else NavClass.INFORMATIONAL

    def is_traversable(self, acronym: str) -> bool:
        """Checks if an object is considered traversable."""
        classification = self.get_classification(acronym)
        return classification['is_traversable'] if classification else True

    def get_cost_factor(self, acronym: str) -> float:
        """
        Calculates a multiplicative cost factor for pathfinding based on the object's classification.
        This factor is multiplied with the edge's base weight.

        - SAFE objects have a factor <= 1.0 (preferred, reduce cost)
        - CAUTION objects have a factor > 1.0 (avoided, increase cost)
        - DANGEROUS objects have an infinite factor (impassable)

        Returns:
            float: The multiplicative risk factor (RiskMultiplier value)
        """
        classification = self.get_classification(acronym)
        if not classification:
            return 1.0  # Neutral factor for unknown objects

        if not classification['is_traversable']:
            return float('inf')

        # Use RiskMultiplier directly as the multiplicative factor
        return classification['risk_multiplier']

    @classmethod
    def generate_static_files(cls, csv_path: str = 's57_classification.csv', summary_path: str = 's57_summary.txt'):
        """
        Generates static CSV and summary text files from the classification database.
        This is useful for documentation or for use in other systems.
        """
        cls._generate_csv(csv_path)
        cls._generate_layer_summary(summary_path)

    @classmethod
    def _generate_csv(cls, output_filename: str):
        """Generates a CSV file from the default classification database."""
        fieldnames = [
            'Acronym', 'NavClassValue', 'NavClassName', 'Category',
            'RiskMultiplier', 'BufferMeters', 'IsTraversable', 'Description'
        ]
        sorted_objects = sorted(cls._DEFAULT_CLASSIFICATION_DB.items(), key=lambda x: (x[1][0].value, x[0]))

        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for acronym, entry in sorted_objects:
                nav_class = entry[0]
                cat = entry[1]
                risk = entry[2]
                buf = entry[3]
                desc = entry[4]

                writer.writerow({
                    'Acronym': acronym,
                    'NavClassValue': nav_class.value,
                    'NavClassName': nav_class.name,
                    'Category': cat,
                    'RiskMultiplier': risk,
                    'BufferMeters': buf,
                    'IsTraversable': nav_class != NavClass.DANGEROUS,
                    'Description': desc
                })
        logger.info(f" Successfully generated classification CSV: {output_filename}")

    @classmethod
    def _generate_layer_summary(cls, output_filename: str):
        """Generates a human-readable summary text file from the default database."""
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("S-57 Navigation Classification Summary\n")
            f.write("=" * 60 + "\n")

            for nav_class_enum in NavClass:
                objects = []
                for acronym, entry in cls._DEFAULT_CLASSIFICATION_DB.items():
                    nc = entry[0]
                    cat = entry[1]
                    desc = entry[4]
                    if nc == nav_class_enum:
                        objects.append((acronym, cat, desc))

                if not objects:
                    continue

                f.write(f"\n{nav_class_enum.name} (Class {nav_class_enum.value})\n")
                f.write("-" * 40 + "\n")

                by_category = {}
                for acronym, cat, desc in objects:
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append((acronym, desc))

                for category in sorted(by_category.keys()):
                    f.write(f"  {category}:\n")
                    for acronym, desc in sorted(by_category[category]):
                        f.write(f"    - {acronym:8s}: {desc}\n")

        logger.info(f" Successfully generated summary file: {output_filename}")


if __name__ == "__main__":
    # This allows you to generate the static files from the command line
    # python -m maritime_module.utils.s57_classification
    logger.info("Generating static classification files...")
    S57Classifier.generate_static_files()
    logger.info("Generation complete.")