import argparse
import itertools
import logging
import os
import sys

from datetime import datetime, timedelta
from networkx.readwrite import json_graph

from flightdatautilities.filesystem_tools import copy_file

from hdfaccess.file import hdf_file

from analysis_engine import hooks, settings, __version__
from analysis_engine.dependency_graph import dependency_order
from analysis_engine.json_tools import json_to_process_flight, process_flight_to_nodes
from analysis_engine.library import np_ma_masked_zeros, repair_mask
from analysis_engine.node import (ApproachNode, Attribute,
                                  derived_param_from_hdf,
                                  DerivedParameterNode,
                                  FlightAttributeNode,
                                  FlightPhaseNode,
                                  KeyPointValueNode,
                                  KeyTimeInstanceNode,
                                  NodeManager, P, Section, SectionNode,
                                  NODE_SUBCLASSES)
from analysis_engine.utils import get_aircraft_info, get_derived_nodes


logger = logging.getLogger(__name__)



def geo_locate(hdf, items):
    '''
    Translate KeyTimeInstance into GeoKeyTimeInstance namedtuples
    '''
    if 'Latitude Smoothed' not in hdf.valid_param_names() \
       or 'Longitude Smoothed' not in hdf.valid_param_names():
        logger.warning("Could not geo-locate as either 'Latitude Smoothed' or "
                       "'Longitude Smoothed' were not found within the hdf.")
        return items
    
    lat_hdf = hdf['Latitude Smoothed']
    lon_hdf = hdf['Longitude Smoothed']
    
    if (not lat_hdf.array.count()) or (not lon_hdf.array.count()):
        logger.warning("Could not geo-locate as either 'Latitude Smoothed' or "
                       "'Longitude Smoothed' have no unmasked values.")
        return items
    
    lat_pos = derived_param_from_hdf(lat_hdf)
    lon_pos = derived_param_from_hdf(lon_hdf)
    
    # We want to place start of flight and end of flight markers at the ends
    # of the data which may extend more than REPAIR_DURATION seconds beyond
    # the end of the valid data. Hence by setting this to None and
    # extrapolate=True we achieve this goal.
    lat_pos.array = repair_mask(lat_pos.array, repair_duration=None, extrapolate=True)
    lon_pos.array = repair_mask(lon_pos.array, repair_duration=None, extrapolate=True)
    
    for item in itertools.chain.from_iterable(items.itervalues()):
        item.latitude = lat_pos.at(item.index) or None
        item.longitude = lon_pos.at(item.index) or None
    return items


def _timestamp(start_datetime, items):
    '''
    Adds item.datetime (from timedelta of item.index + start_datetime)

    :param start_datetime: Origin timestamp used as a base to the index
    :type start_datetime: datetime
    :param item_list: list of objects with a .index attribute
    :type item_list: list
    '''
    for item in itertools.chain.from_iterable(items.itervalues()):
        item.datetime = start_datetime + timedelta(seconds=float(item.index))
    return items


def get_node_type(node, node_subclasses):
    '''
    Return node type string, for logging.
    '''
    # OPT: Looking up bases from a set is much faster than issubclass (250x speedup).
    for base_class in node.__class__.__bases__:
        if base_class in node_subclasses:
            return base_class.__name__
    return node.__class__.__name__


def derive_parameters(hdf, node_mgr, process_order, params={}, force=False):
    '''
    Derives parameters in process_order. Dependencies are sourced via the
    node_mgr.

    :param hdf: Data file accessor used to get and save parameter data and
        attributes
    :type hdf: hdf_file
    :param node_mgr: Used to determine the type of node in the process_order
    :type node_mgr: NodeManager
    :param process_order: Parameter / Node class names in the required order to
        be processed
    :type process_order: list of strings
    '''
    # OPT: local lookup is faster than module-level (small).
    node_subclasses = NODE_SUBCLASSES
    
    # store all derived params that aren't masked arrays
    approaches = {}
    # duplicate storage, but maintaining types
    kpvs = {}
    ktis = {}
    # 'Node Name' : node()  pass in node.get_accessor()
    sections = {}
    flight_attrs = {}
    duration = hdf.duration

    for param_name in process_order:
        if param_name in node_mgr.hdf_keys:
            continue
        
        elif param_name in params:
            node = params[param_name]
            # populate output already at 1Hz
            if node.node_type is KeyPointValueNode:
                kpvs[param_name] = list(node)
            elif node.node_type is KeyTimeInstanceNode:
                ktis[param_name] = list(node)
            elif node.node_type is FlightAttributeNode:
                flight_attrs[param_name] = [Attribute(node.name, node.value)]
            elif node.node_type is SectionNode:
                sections[param_name] = list(node)
            # DerivedParameterNodes are not supported in initial data.
            continue

        elif node_mgr.get_attribute(param_name) is not None:
            # add attribute to dictionary of available params
            ###params[param_name] = node_mgr.get_attribute(param_name)
            #TODO: optimise with only one call to get_attribute
            continue

        #NB raises KeyError if Node is "unknown"
        node_class = node_mgr.derived_nodes[param_name]

        # build ordered dependencies
        deps = []
        node_deps = node_class.get_dependency_names()
        for dep_name in node_deps:
            if dep_name in params:  # already calculated KPV/KTI/Phase
                deps.append(params[dep_name])
            elif node_mgr.get_attribute(dep_name) is not None:
                deps.append(node_mgr.get_attribute(dep_name))
            elif dep_name in node_mgr.hdf_keys:
                # LFL/Derived parameter
                # all parameters (LFL or other) need get_aligned which is
                # available on DerivedParameterNode
                try:
                    dp = derived_param_from_hdf(hdf.get_param(
                        dep_name, valid_only=True))
                except KeyError:
                    # Parameter is invalid.
                    dp = None
                deps.append(dp)
            else:  # dependency not available
                deps.append(None)
        if all([d is None for d in deps]):
            raise RuntimeError(
                "No dependencies available - Nodes cannot "
                "operate without ANY dependencies available! "
                "Node: %s" % node_class.__name__)

        # initialise node
        node = node_class()
        # shhh, secret accessors for developing nodes in debug mode
        node._p = params
        node._h = hdf
        node._n = node_mgr
        logger.info("Processing %s `%s`", get_node_type(node, node_subclasses), param_name)
        # Derive the resulting value
        
        try:
            node = node.get_derived(deps)
        except:
            if not force:
                raise
        
        del node._p
        del node._h
        del node._n

        if node.node_type is KeyPointValueNode:
            params[param_name] = node
            
            aligned_kpvs = []
            for one_hz in node.get_aligned(P(frequency=1, offset=0)):
                if not (0 <= one_hz.index <= duration+4):
                    raise IndexError(
                        "KPV '%s' index %.2f is not between 0 and %d" %
                        (one_hz.name, one_hz.index, duration))
                aligned_kpvs.append(one_hz)
            kpvs[param_name] = aligned_kpvs
        elif node.node_type is KeyTimeInstanceNode:
            params[param_name] = node
            
            aligned_ktis = []
            for one_hz in node.get_aligned(P(frequency=1, offset=0)):
                if not (0 <= one_hz.index <= duration+4):
                    raise IndexError(
                        "KTI '%s' index %.2f is not between 0 and %d" %
                        (one_hz.name, one_hz.index, duration))
                aligned_ktis.append(one_hz)
            ktis[param_name] = aligned_ktis
        elif node.node_type is FlightAttributeNode:
            params[param_name] = node
            try:
                # only has one Attribute node, store as a list for consistency
                flight_attrs[param_name] = [Attribute(node.name, node.value)]
            except:
                logger.warning("Flight Attribute Node '%s' returned empty "
                               "handed.", param_name)
        elif issubclass(node.node_type, SectionNode):
            aligned_section = node.get_aligned(P(frequency=1, offset=0))
            for index, one_hz in enumerate(aligned_section):
                # SectionNodes allow slice starts and stops being None which
                # signifies the beginning and end of the data. To avoid
                # TypeErrors in subsequent derive methods which perform
                # arithmetic on section slice start and stops, replace with 0
                # or hdf.duration.
                fallback = lambda x, y: x if x is not None else y

                duration = fallback(duration, 0)

                start = fallback(one_hz.slice.start, 0)
                stop = fallback(one_hz.slice.stop, duration)
                start_edge = fallback(one_hz.start_edge, 0)
                stop_edge = fallback(one_hz.stop_edge, duration)

                slice_ = slice(start, stop)
                one_hz = Section(one_hz.name, slice_, start_edge, stop_edge)
                aligned_section[index] = one_hz

                if not (0 <= start <= duration and 0 <= stop <= duration + 4):
                    msg = "Section '%s' (%.2f, %.2f) not between 0 and %d"
                    raise IndexError(
                        msg % (one_hz.name, start, stop, duration))
                if not 0 <= start_edge <= duration:
                    msg = "Section '%s' start_edge (%.2f) not between 0 and %d"
                    raise IndexError(msg % (one_hz.name, start_edge, duration))
                if not 0 <= stop_edge <= duration + 4:
                    msg = "Section '%s' stop_edge (%.2f) not between 0 and %d"
                    raise IndexError(msg % (one_hz.name, stop_edge, duration))
                #section_list.append(one_hz)
            params[param_name] = aligned_section
            sections[param_name] = list(aligned_section)
        elif issubclass(node.node_type, DerivedParameterNode):
            if duration:
                # check that the right number of nodes were returned Allow a
                # small tolerance. For example if duration in seconds is 2822,
                # then there will be an array length of  1411 at 0.5Hz and 706
                # at 0.25Hz (rounded upwards). If we combine two 0.25Hz
                # parameters then we will have an array length of 1412.
                expected_length = duration * node.frequency
                if node.array is None or (force and len(node.array) == 0):
                    logger.warning("No array set; creating a fully masked "
                                   "array for %s", param_name)
                    array_length = expected_length
                    # Where a parameter is wholly masked, we fill the HDF
                    # file with masked zeros to maintain structure.
                    node.array = \
                        np_ma_masked_zeros(expected_length)
                else:
                    array_length = len(node.array)
                length_diff = array_length - expected_length
                if length_diff == 0:
                    pass
                elif 0 < length_diff < 5:
                    logger.warning("Cutting excess data for parameter '%s'. "
                                   "Expected length was '%s' while resulting "
                                   "array length was '%s'.", param_name,
                                   expected_length, len(node.array))
                    node.array = node.array[:expected_length]
                else:
                    raise ValueError("Array length mismatch for parameter "
                                     "'%s'. Expected '%s', resulting array "
                                     "length '%s'." % (param_name,
                                                       expected_length,
                                                       array_length))

            hdf.set_param(node)
            # Keep hdf_keys up to date.
            node_mgr.hdf_keys.append(param_name)
        elif issubclass(node.node_type, ApproachNode):
            aligned_approach = node.get_aligned(P(frequency=1, offset=0))
            for approach in aligned_approach:
                # Does not allow slice start or stops to be None.
                valid_turnoff = (not approach.turnoff or
                                 (0 <= approach.turnoff <= duration))
                valid_slice = ((0 <= approach.slice.start <= duration) and
                               (0 <= approach.slice.stop <= duration))
                valid_gs_est = (not approach.gs_est or
                                ((0 <= approach.gs_est.start <= duration) and
                                 (0 <= approach.gs_est.stop <= duration)))
                valid_loc_est = (not approach.loc_est or
                                 ((0 <= approach.loc_est.start <= duration) and
                                  (0 <= approach.loc_est.stop <= duration)))
                if not all([valid_turnoff, valid_slice, valid_gs_est,
                            valid_loc_est]):
                    raise ValueError('ApproachItem contains index outside of '
                                     'flight data: %s' % approach)
            params[param_name] = aligned_approach
            approaches[param_name] = list(aligned_approach)
        else:
            raise NotImplementedError("Unknown Type %s" % node.__class__)
        continue
    return ktis, kpvs, sections, approaches, flight_attrs


def parse_analyser_profiles(analyser_profiles, filter_modules=None):
    '''
    Parse analyser profiles into additional_modules and required nodes as
    expected by process_flight.

    :param analyser_profiles: A list of analyser profile tuples containing
        semicolon separated module paths and whether or not the nodes are
        required e.g. [('package.module_one;package.module_two', True), ]
    :type analyser_profiles: [[str, bool], ]
    :param filter_paths: Optional list of analyser profiles to keep.
    :type filter_paths: [str] or None
    :returns: A list of additional module paths and a list of required node
        names.
    :rtype: [str], [str]
    '''
    additional_modules = []
    required_nodes = []
    for import_paths, is_required in analyser_profiles:
        for import_path in import_paths.split(';'):
            if filter_modules is not None and import_path not in filter_modules:
                continue
            import_path = import_path.strip()
            if not import_path:
                continue
            additional_modules.append(import_path)
            if is_required:
                required_nodes.extend(get_derived_nodes([import_path]))
    return additional_modules, required_nodes


def process_flight(segment_info, tail_number, aircraft_info={}, achieved_flight_record={},
                   requested=[], required=[], include_flight_attributes=True,
                   additional_modules=[], pre_flight_kwargs={}, force=False,
                   initial={}, reprocess=False):
    '''
    Processes the HDF file (segment_info['File']) to derive the required_params (Nodes)
    within python modules (settings.NODE_MODULES).

    Note: For Flight Data Services, the definitive API is located here:
        "PolarisTaskManagement.test.tasks_mask.process_flight"

    :param segment_info: Details of the segment to process
    :type segment_info: dict
    :param aircraft: Aircraft specific attributes
    :type aircraft: dict
    :param achieved_flight_record: See API Below
    :type achieved_flight_record: Dict
    :param requested: Derived nodes to process (dependencies will also be
        evaluated).
    :type requested: List of Strings
    :param required: Nodes which are required, otherwise an exception will be
        raised.
    :type required: List of Strings
    :param include_flight_attributes: Whether to include all flight attributes
    :type include_flight_attributes: Boolean
    :param additional_modules: List of module paths to import.
    :type additional_modules: List of Strings
    :param pre_flight_kwargs: Keyword arguments for the pre-flight analysis hook.
    :type pre_flight_kwargs: dict
    :param force: Ignore errors raised while deriving nodes.
    :type force: bool
    :param initial: Initial content for nodes to avoid reprocessing (excluding parameter nodes which are saved to the hdf).
    :type initial: dict
    :param reprocess: Force reprocessing of all Nodes (including derived Nodes already saved to the HDF file).

    :returns: See below:
    :rtype: Dict

    Sample segment_info
    --------------------
    {
        'File':  # Path to HDF5 file to process
        'Start Datetime':  # Datetime of the origin of the data (at index 0)
        'Segment Type': # segment type obtained from split segments e.g. START_AND_STOP
    }

    Sample aircraft_info
    --------------------
    {
        'Tail Number':  # Aircraft Registration
        'Identifier':  # Aircraft Ident
        'Manufacturer': # e.g. Boeing
        'Manufacturer Serial Number': #MSN
        'Model': # e.g. 737-808-ER
        'Series': # e.g. 737-800
        'Family': # e.g. 737
        'Frame': # e.g. 737-3C
        'Main Gear To Altitude Radio': # Distance in metres
        'Wing Span': # Distance in metres
    }

    Sample achieved_flight_record
    -----------------------------
    {
        # Simple values first, e.g. string, int, float, etc.
        'AFR Flight ID': # e.g. 1
        'AFR Flight Number': # e.g. 1234
        'AFR Type': # 'POSITIONING'
        'AFR Off Blocks Datetime': # datetime(2015,01,01,13,00)
        'AFR Takeoff Datetime': # datetime(2015,01,01,13,15)
        'AFR Takeoff Pilot': # 'Joe Bloggs'
        'AFR Takeoff Gross Weight': # weight in kg
        'AFR Takeoff Fuel': # fuel in kg
        'AFR Landing Datetime': # datetime(2015,01,01,18,45)
        'AFR Landing Pilot': # 'Joe Bloggs'
        'AFR Landing Gross Weight': # weight in kg
        'AFR Landing Fuel': # weight in kg
        'AFR On Blocks Datetime': # datetime(2015,01,01,19,00)
        'AFR V2': # V2 used at takeoff in kts
        'AFR Vapp': # Vapp used in kts
        'AFR Vref': # Vref used in kts
        # More complex data that needs to be looked up next:
        'AFR Takeoff Airport':  {
            'id': 4904, # unique id
            'name': 'Athens Intl Airport Elefterios Venizel',
            'code': {'iata': 'ATH', 'icao': 'LGAV'},
            'latitude': 37.9364,
            'longitude': 23.9445,
            'location': {'city': u'Athens', 'country': u'Greece'},
            'elevation': 266, # ft
            'magnetic_variation': 'E003186 0106',
            }
           },
        'AFR Landing Aiport': {
            'id': 1, # unique id
            'name': 'Athens Intl Airport Elefterios Venizel',
            'code': {'iata': 'ATH', 'icao': 'LGAV'},
            'latitude': 37.9364,
            'longitude': 23.9445,
            'location': {'city': u'Athens', 'country': u'Greece'},
            'elevation': 266, # ft
            'magnetic_variation': 'E003186 0106',
            }
           },
        'AFR Destination Airport': None, # if not required, or exclude this key
        'AFR Takeoff Runway': {
            'id': 1,
            'identifier': '21L',
            'magnetic_heading': 212.6,
            'strip': {
                'id': 1,
                'length': 13123,
                'surface': 'ASP',
                'width': 147},
            'start': {
                'elevation': 308,
                'latitude': 37.952425,
                'longitude': 23.970422},
            'end': {
                'elevation': 279,
                'latitude': 37.923511,
                'longitude': 23.943261},
            'glideslope': {
                'angle': 3.0,
                'elevation': 282,
                'latitude': 37.9473,
                'longitude': 23.9676,
                'threshold_distance': 999},
            'localizer': {
                'beam_width': 4.5,
                'elevation': 256,
                'frequency': 111100,
                'heading': 213,
                'latitude': 37.919281,
                'longitude': 23.939294},
            },
        'AFR Landing Runway': {
            'id': 1,
            'identifier': '21L',
            'magnetic_heading': 212.6,
            'strip': {
                'id': 1,
                'length': 13123,
                'surface': 'ASP',
                'width': 147},
            'start': {
                'elevation': 308,
                'latitude': 37.952425,
                'longitude': 23.970422},
            'end': {
                'elevation': 279,
                'latitude': 37.923511,
                'longitude': 23.943261},
            'glideslope': {
                'angle': 3.0,
                'elevation': 282,
                'latitude': 37.9473,
                'longitude': 23.9676,
                'threshold_distance': 999},
            'localizer': {
                'beam_width': 4.5,
                'elevation': 256,
                'frequency': 111100,
                'heading': 213,
                'latitude': 37.919281,
                'longitude': 23.939294},
            },
    }

    Sample Return
    -------------
    {
        'flight':[Attribute('name value')],
        'kti':[GeoKeyTimeInstance('index name latitude longitude')]
            if lat/long available
            else [KeyTimeInstance('index name')],
        'kpv':[KeyPointValue('index value name slice')]
    }

    sample flight Attributes:

    [
        Attribute('Takeoff Airport', {'id':1234, 'name':'Int. Airport'},
        Attribute('Approaches', [4567,7890]),
        ...
    ],

    '''
    
    hdf_path = segment_info['File']
    if 'Start Datetime' not in segment_info:
        import pytz
        segment_info['Start Datetime'] = datetime.utcnow().replace(tzinfo=pytz.utc)
    logger.info("Processing: %s", hdf_path)

    if aircraft_info:
        # Aircraft info has already been provided.
        logger.info(
            "Using aircraft_info dictionary passed into process_flight '%s'." %
            aircraft_info)
    else:
        aircraft_info = get_aircraft_info(tail_number)

    aircraft_info['Tail Number'] = tail_number

    # go through modules to get derived nodes
    node_modules = additional_modules + settings.NODE_MODULES
    derived_nodes = get_derived_nodes(node_modules)

    if requested:
        requested = \
            list(set(requested).intersection(set(derived_nodes)))
    else:
        # if requested isn't set, try using ALL derived_nodes!
        logger.info("No requested nodes declared, using all derived nodes")
        requested = derived_nodes.keys()

    # include all flight attributes as requested
    if include_flight_attributes:
        requested = list(set(
            requested + get_derived_nodes(
                ['analysis_engine.flight_attribute']).keys()))
    
    initial = process_flight_to_nodes(initial)
    for node_name in requested:
        initial.pop(node_name, None)

    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        hdf.start_datetime = segment_info['Start Datetime']
        if hooks.PRE_FLIGHT_ANALYSIS:
            logger.info("Performing PRE_FLIGHT_ANALYSIS action '%s' with options: %s",
                        hooks.PRE_FLIGHT_ANALYSIS.func_name, pre_flight_kwargs)
            hooks.PRE_FLIGHT_ANALYSIS(hdf, aircraft_info, **pre_flight_kwargs)
        else:
            logger.info("No PRE_FLIGHT_ANALYSIS actions to perform")
        # Track nodes.
        param_names = hdf.valid_lfl_param_names() if reprocess else hdf.valid_param_names()
        node_mgr = NodeManager(
            segment_info, hdf.duration, param_names,
            requested, required, derived_nodes, aircraft_info,
            achieved_flight_record)
        # calculate dependency tree
        process_order, gr_st = dependency_order(node_mgr, draw=False)
        if settings.CACHE_PARAMETER_MIN_USAGE:
            # find params used more than
            for node in gr_st.nodes():
                if node in node_mgr.derived_nodes:
                    # this includes KPV/KTIs but they'll be ignored by HDF
                    qty = len(gr_st.predecessors(node))
                    if qty > settings.CACHE_PARAMETER_MIN_USAGE:
                        hdf.cache_param_list.append(node)
            logging.info("HDF set to cache parameters: %s",
                         hdf.cache_param_list)

        # derive parameters
        ktis, kpvs, sections, approaches, flight_attrs = \
            derive_parameters(hdf, node_mgr, process_order, params=initial, force=force)

        # geo locate KTIs
        ktis = geo_locate(hdf, ktis)
        ktis = _timestamp(segment_info['Start Datetime'], ktis)

        # geo locate KPVs
        kpvs = geo_locate(hdf, kpvs)
        kpvs = _timestamp(segment_info['Start Datetime'], kpvs)

        # Store version of FlightDataAnalyser
        hdf.analysis_version = __version__
        # Store dependency tree
        hdf.dependency_tree = json_graph.dumps(gr_st)
        # Store aircraft info
        hdf.set_attr('aircraft_info', aircraft_info)
        hdf.set_attr('achieved_flight_record', achieved_flight_record)

    return {
        'flight': flight_attrs,
        'kti': ktis,
        'kpv': kpvs,
        'approach': approaches,
        'phases': sections,
    }


def main():
    print 'FlightDataAnalyzer (c) Copyright 2013 Flight Data Services, Ltd.'
    print '  - Powered by POLARIS'
    print '  - http://www.flightdatacommunity.com'
    print ''
    from analysis_engine.plot_flight import csv_flight_details, track_to_kml
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    parser = argparse.ArgumentParser(description="Process a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    help = 'Disable writing a CSV of the processing results.'
    parser.add_argument('-disable-csv', dest='disable_csv',
                        action='store_true', help=help)
    help = 'Disable writing a KML of the flight track.'
    parser.add_argument('-disable-kml', dest='disable_kml',
                        action='store_true', help=help)
    parser.add_argument('-r', '--requested', type=str, nargs='+',
                        dest='requested', default=[], help='Requested nodes.')
    parser.add_argument('-R', '--required', type=str, nargs='+', dest='required',
                        default=[], help='Required nodes.')
    parser.add_argument('-tail', '--tail', dest='tail_number',
                        default='G-FDSL',  # as per flightdatacommunity file
                        help='Aircraft tail number.')
    parser.add_argument('-segment-type', dest='segment_type',
                        default='START_AND_STOP',  # as per flightdatacommunity file
                        help='Type of segment.')
    parser.add_argument('--strip', default=False, action='store_true',
                        help='Strip the HDF5 file to only the LFL parameters')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Verbose logging')

    # Aircraft info
    parser.add_argument('-aircraft-family', dest='aircraft_family', type=str,
                        help='Aircraft family.')
    parser.add_argument('-aircraft-series', dest='aircraft_series', type=str,
                        help='Aircraft series.')
    parser.add_argument('-aircraft-model', dest='aircraft_model', type=str,
                        help='Aircraft model.')
    parser.add_argument('-aircraft-manufacturer', dest='aircraft_manufacturer',
                        type=str, help='Aircraft manufacturer.')
    help = 'Whether or not the aircraft records precise positioning ' \
        'parameters.'
    parser.add_argument('-precise-positioning', dest='precise_positioning',
                        type=str, help=help)
    parser.add_argument('-frame', dest='frame', type=str,
                        help='Data frame name.')
    parser.add_argument('-frame-qualifier', dest='frame_qualifier', type=str,
                        help='Data frame qualifier.')
    parser.add_argument('-identifier', dest='identifier', type=str,
                        help='Aircraft identifier.')
    parser.add_argument('-manufacturer-serial-number',
                        dest='manufacturer_serial_number', type=str,
                        help="Manufacturer's serial number of the aircraft.")
    parser.add_argument('-qar-serial-number', dest='qar_serial_number',
                        type=str, help='QAR serial number.')
    help = 'Main gear to radio altimeter antenna in metres.'
    parser.add_argument('-main-gear-to-radio-altimeter-antenna',
                        dest='main_gear_to_alt_rad',
                        type=float, help=help)
    help = 'Main gear to lowest point of tail in metres.'
    parser.add_argument('-main-gear-to-lowest-point-of-tail',
                        dest='main_gear_to_tail',
                        type=float, help=help)
    help = 'Ground to lowest point of tail in metres.'
    parser.add_argument('-ground-to-lowest-point-of-tail',
                        dest='ground_to_tail',
                        type=float, help=help)
    parser.add_argument('-engine-count', dest='engine_count',
                        type=int, help='Number of engines.')
    parser.add_argument('-engine-manufacturer', dest='engine_manufacturer',
                        type=str, help='Engine manufacturer.')
    parser.add_argument('-engine-series', dest='engine_series', type=str,
                        help='Engine series.')
    parser.add_argument('-engine-type', dest='engine_type', type=str,
                        help='Engine type.')
    
    parser.add_argument('-initial', dest='initial', type=str,
                        help='Path to initial nodes in json format.')
    

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    aircraft_info = {}
    if args.aircraft_model:
        aircraft_info['Model'] = args.aircraft_model
    if args.aircraft_family:
        aircraft_info['Family'] = args.aircraft_family
    if args.aircraft_series:
        aircraft_info['Series'] = args.aircraft_series
    if args.aircraft_manufacturer:
        aircraft_info['Manufacturer'] = args.aircraft_manufacturer
    if args.precise_positioning:
        aircraft_info['Precise Positioning'] = args.precise_positioning
    if args.frame:
        aircraft_info['Frame'] = args.frame
    if args.frame_qualifier:
        aircraft_info['Frame Qualifier'] = args.frame_qualifier
    if args.identifier:
        aircraft_info['Identifier'] = args.identifier
    if args.manufacturer_serial_number:
        aircraft_info['Manufacturer Serial Number'] = \
            args.manufacturer_serial_number
    if args.qar_serial_number:
        aircraft_info['QAR Serial Number'] = args.qar_serial_number
    if args.main_gear_to_alt_rad:
        aircraft_info['Main Gear To Radio Altimeter Antenna'] = \
            args.main_gear_to_alt_rad
    if args.main_gear_to_tail:
        aircraft_info['Main Gear To Lowest Point Of Tail'] = \
            args.main_gear_to_tail
    if args.ground_to_tail:
        aircraft_info['Ground To Lowest Point Of Tail'] = args.ground_to_tail
    if args.engine_count:
        aircraft_info['Engine Count'] = args.engine_count
    if args.engine_series:
        aircraft_info['Engine Series'] = args.engine_series
    if args.engine_manufacturer:
        aircraft_info['Engine Manufacturer'] = args.engine_manufacturer
    if args.engine_series:
        aircraft_info['Engine Series'] = args.engine_series
    if args.engine_type:
        aircraft_info['Engine Type'] = args.engine_type

    # Derive parameters to new HDF
    hdf_copy = copy_file(args.file, postfix='_process')
    if args.strip:
        with hdf_file(hdf_copy) as hdf:
            hdf.delete_params(hdf.derived_keys())
    
    if args.initial:
        if not os.path.exists(args.initial):
            parser.error('Path for initial json data not found: %s' % args.initial)
        initial = json_to_process_flight(open(args.initial, 'rb').read())
    else:
        initial = {}

    segment_info = {
        'File': hdf_copy,
        'Segment Type': args.segment_type,
    }
    res = process_flight(
        segment_info, args.tail_number, aircraft_info=aircraft_info,
        requested=args.requested, required=args.required,
        additional_modules=['flightdataprofiles.fcp.kpvs'],
        initial=initial,
    )
    # Flatten results.
    res = {k: list(itertools.chain.from_iterable(v.itervalues()))
           for k, v in res.iteritems()}
    
    logger.info("Derived parameters stored in hdf: %s", hdf_copy)
    # Write CSV file
    if not args.disable_csv:
        csv_dest = os.path.splitext(hdf_copy)[0] + '.csv'
        csv_flight_details(hdf_copy, res['kti'], res['kpv'], res['phases'],
                           dest_path=csv_dest)
        logger.info("KPV, KTI and Phases writen to csv: %s", csv_dest)
    # Write KML file
    if not args.disable_kml:
        kml_dest = os.path.splitext(hdf_copy)[0] + '.kml'
        dest = track_to_kml(
            hdf_copy, res['kti'], res['kpv'], res['approach'],
            dest_path=kml_dest)
        if dest:
            logger.info("Flight Track with attributes writen to kml: %s", dest)

    # - END -


if __name__ == '__main__':
    main()
