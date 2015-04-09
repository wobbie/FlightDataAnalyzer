import argparse
import logging
import os
import re
import simplejson
import zipfile

from collections import defaultdict
from datetime import datetime
from inspect import getargspec, isclass

from hdfaccess.file import hdf_file
from hdfaccess.utils import strip_hdf

from analysis_engine.api_handler import APIError, get_api_handler
from analysis_engine.dependency_graph import dependencies3, graph_nodes
# node classes required for unpickling
from analysis_engine.node import (
    loads, save, Node, NodeManager,
    DerivedParameterNode,
    KeyPointValueNode,
    KeyTimeInstanceNode,
    FlightPhaseNode,
    FlightAttributeNode,
    ApproachNode,
    NODE_SUBCLASSES,
)
from analysis_engine import settings


logger = logging.getLogger(__name__)


def save_test_data(node, locals):
    '''
    Saves derive method arguments to node files within test_data and returns
    code for loading node files within a test case.
    
    Example usage:
    
    class MyKeyPointValue(KeyPointValueNode):
        def derive(self, airspeed=P('Airspeed'), alt_aal=P('Altitude AAL')):
            from analysis_engine.utils import save_test_data
            save_test_data(self, locals())
            ...
    
    Creates:
    
     - tests/test_data/MyKeyPointValue_airspeed_01.nod
     - tests/test_data/MyKeyPointValue_alt_aal_01.nod
    
    :param node: Node to create test data for.
    :type node: Node
    :param locals: locals() from within the derive method.
    :type locals: dict
    :returns: Code for importing test data.
    :rtype: str
    '''
    # work out test_data location
    package_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = os.path.join(
        os.path.dirname(package_dir),
        'tests',
        'test_data',
    )
    
    code = []
    code.append("test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')")
    
    code.append("node = %s()" % node.__class__.__name__)
    for var_name in getargspec(node.derive).args[1:]:
        # generate unique filename
        counter = 1
        while True:
            filename = '%s_%s_%02d.nod' % (node.__class__.__name__, var_name, counter)
            file_path = os.path.join(test_data_dir, filename)
            if not os.path.exists(file_path):
                break
            counter += 1
        save(locals[var_name], file_path)
        
        code.append("%s = load(os.path.join(test_data_path, '%s'))" % (var_name, filename))
    code.append("node.derive(%s)" % ', '.join(getargspec(node.derive).args[1:]))
    return '\n'.join(code)


def open_node_container(zip_path):
    '''
    Opens a zip file containing nodes.
    
    TODO: Do not compress to the current directory.
    '''
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        filenames = set(zip_file.namelist())
        
        flight_filenames = defaultdict(dict)
        
        for filename in filenames:
            match = re.match('^(?P<flight_pk>\d+) - (?P<node_name>[\w\d\s\*()]+).nod$', filename)
            if not match:
                if not re.match('^(?P<flight_pk>\d+)\.json$', filename):
                    print "Skipping invalid filename '%s'" % filename
                continue
            
            groupdict = match.groupdict()
            flight_filenames[groupdict['flight_pk']][groupdict['node_name']] = filename
        
        for flight_pk, node_filenames in flight_filenames.iteritems():
            nodes = {}
            for node_name, filename in node_filenames.iteritems():
                nodes[node_name] = loads(zip_file.read(filename))
            
            json_filename = '%s.json' % flight_pk
            attrs = simplejson.loads(zip_file.read(json_filename)) if json_filename in filenames else {}
            
            yield flight_pk, nodes, attrs


def get_aircraft_info(tail_number):
    '''
    Fetch aircraft info from settings.API_HANDLER or from LOCAL_API_HANDLER
    if there is an API_ERROR raised.
    
    :param tail_number: Aircraft tail registration
    :type tail_number: string
    :returns: Aircraft information key:value pairs
    :rtype: dict
    '''
    # Fetch aircraft info through the API.
    api_handler = get_api_handler(settings.API_HANDLER)
    
    try:
        aircraft_info = api_handler.get_aircraft(tail_number)
    except APIError:
        if settings.API_HANDLER == settings.LOCAL_API_HANDLER:
            raise
        # Fallback to the local API handler.
        logger.info(
            "Aircraft '%s' could not be found with '%s' API handler. "
            "Falling back to '%s'.", tail_number, settings.API_HANDLER,
            settings.LOCAL_API_HANDLER)
        api_handler = get_api_handler(settings.LOCAL_API_HANDLER)
        aircraft_info = api_handler.get_aircraft(tail_number)
    logger.info("Using aircraft_info provided by '%s' '%s'.",
                api_handler.__class__.__name__, aircraft_info)
    return aircraft_info


def get_derived_nodes(module_names):
    '''
    Create a key:value pair of each node_name to Node class for all Nodes
    within modules provided.
    
    sample module_names = ['path_to.module', 'analysis_engine.flight_phase',..]
    
    :param module_names: Module names to import as locations on PYTHON PATH
    :type module_names: List of Strings
    :returns: Module name to Classes
    :rtype: Dict
    '''
    # OPT: local variable to avoid module-level lookup.
    node_subclasses = NODE_SUBCLASSES
    
    def isclassandsubclass(value, classes, superclass):
        if not isclass(value):
            return False
        
        # OPT: Lookup from set instead of issubclass (200x speedup).
        for base_class in value.__bases__:
            if base_class in classes:
                return True
        return issubclass(value, superclass)
    
    if isinstance(module_names, basestring):
        # This has been done too often!
        module_names = [module_names]
    nodes = {}
    for name in module_names:
        #Ref:
        #http://code.activestate.com/recipes/223972-import-package-modules-at-runtime/
        # You may notice something odd about the call to __import__(): why is
        # the last parameter a list whose only member is an empty string? This
        # hack stems from a quirk about __import__(): if the last parameter is
        # empty, loading class "A.B.C.D" actually only loads "A". If the last
        # parameter is defined, regardless of what its value is, we end up
        # loading "A.B.C"
        ##abstract_nodes = ['Node', 'Derived Parameter Node', 'Key Point Value Node', 'Flight Phase Node'
        ##print 'importing', name
        module = __import__(name, globals(), locals(), [''])
        for c in vars(module).values():
            if isclassandsubclass(c, node_subclasses, Node) \
                    and c.__module__ != 'analysis_engine.node':
                try:
                    #TODO: Alert when dupe node_name found which overrides previous
                    ##name = c.get_name()
                    ##if name in nodes:
                        ### alert about overide happening or raise out?
                    ##nodes[name] = c
                    nodes[c.get_name()] = c
                except TypeError:
                    #TODO: Handle the expected error of top level classes
                    # Can't instantiate abstract class DerivedParameterNode
                    # - but don't know how to detect if we're at that level without resorting to 'if c.get_name() in 'derived parameter node',..
                    logger.exception('Failed to import class: %s' % c.get_name())
    return nodes


def derived_trimmer(hdf_path, node_names, dest):
    '''
    Trims an HDF file of parameters which are not dependencies of nodes in
    node_names.
    
    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param node_names: A list of Node names which are required.
    :type node_names: list of str
    :param dest: destination path for trimmed output file
    :type dest: str
    :return: parameters in stripped hdf file
    :rtype: [str]
    '''
    params = []
    with hdf_file(hdf_path) as hdf:
        derived_nodes = get_derived_nodes(settings.NODE_MODULES)
        node_mgr = NodeManager(
            datetime.now(), hdf.duration, hdf.valid_param_names(), [], [],
            derived_nodes, {}, {})
        _graph = graph_nodes(node_mgr)
        for node_name in node_names:
            deps = dependencies3(_graph, node_name, node_mgr)
            params.extend(filter(lambda d: d in node_mgr.hdf_keys, deps))
    return strip_hdf(hdf_path, params, dest) 


def _get_names(module_locations, fetch_names=True, fetch_dependencies=False,
               filter_nodes=None):
    '''
    Get the names of Nodes and dependencies.
    
    :param module_locations: list of locations to fetch modules from
    :type module_locations: list of strings
    :param fetch_names: Return name of parameters etc. created by class
    :type fetch_names: Bool
    :param fetch_dependencies: Return names of the arguments in derive methods
    :type fetch_dependencies: Bool
    '''
    nodes = get_derived_nodes(module_locations)
    names = []
    for name, node in nodes.iteritems():
        if filter_nodes and name not in filter_nodes:
            continue
        if fetch_names:
            if hasattr(node, 'names'):
                # FormattedNameNode (KPV/KTI) can have many names
                names.extend(node.names())
            else:
                names.append(node.get_name())
        if fetch_dependencies:
            names.extend(node.get_dependency_names())
    return sorted(names)


def list_parameters():
    '''
    Return an ordered list of parameters.
    '''
    # Exclude all KPV, KTI, Section, Attribute, etc:
    exclude = _get_names([
        'analysis_engine.approaches',
        'analysis_engine.flight_attribute',
        'analysis_engine.flight_phase',
        'analysis_engine.key_point_values',
        'analysis_engine.key_time_instances',
    ], fetch_names=True, fetch_dependencies=False)
    # Remove excluded names leaving parameters:
    parameters = set(list_everything()) - set(exclude)
    return sorted(parameters)


def list_derived_parameters():
    '''
    Return an ordered list of the derived parameters which have been coded.
    '''
    return _get_names([
        'analysis_engine.derived_parameters',
        'analysis_engine.multistate_parameters',
    ])


def list_lfl_parameter_dependencies():
    '''
    Return an ordered list of the non-derived parameters.
    
    This should be mostly LFL parameters.
    
    Note: A few Attributes will be in here too!
    '''
    parameters = set(list_parameters()) - set(list_derived_parameters())
    return sorted(parameters)


def list_everything():
    '''
    Return an ordered list of all parameters both derived and required.
    '''
    return sorted(set(_get_names(settings.NODE_MODULES, True, True)))


def list_kpvs():
    '''
    Return an ordered list of the key point values which have been coded.
    '''
    return _get_names(['analysis_engine.key_point_values'])


def list_ktis():
    '''
    Return an ordered list of the key time instances which have been coded.
    '''
    return _get_names(['analysis_engine.key_time_instances'])


def list_flight_attributes():
    '''
    Return an ordered list of the flight attributes which have been coded.
    '''
    return _get_names(['analysis_engine.flight_attribute'])


def list_flight_phases():
    '''
    Return an ordered list of the flight phases which have been coded.
    '''
    return _get_names(['analysis_engine.flight_phase'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command',
                                      description="Utility command, currently "
                                      "'trimmer' and 'list' are supported",
                                      help='Additional help')
    trimmer_parser = subparser.add_parser('trimmer')
    trimmer_parser.add_argument('input_file_path', help='Input hdf filename.')  
    trimmer_parser.add_argument('output_file_path', help='Output hdf filename.')
    trimmer_parser.add_argument('nodes', nargs='+',
                                help='Keep dependencies of the specified nodes '
                                'within the output hdf file. All other '
                                'parameters will be stripped.')
    
    list_parser = subparser.add_parser('list')
    list_parser.add_argument('--filter-nodes', nargs='+', help='Node names')
    list_parser.add_argument('--additional-modules', nargs='+',
                             help='Additional modules')
    
    #list_parser.add_argument('--list', action='store_true',
                             #help='Output as Python list')
    
    args = parser.parse_args()
    if args.command == 'trimmer':
        if not os.path.isfile(args.input_file_path):
            parser.error("Input file path '%s' does not exist." %
                         args.input_file_path)
        if os.path.exists(args.output_file_path):
            parser.error("Output file path '%s' already exists." %
                         args.output_file_path)
        output_parameters = derived_trimmer(args.input_file_path, args.nodes,
                                            args.output_file_path)
        if output_parameters:
            print 'The following parameters are in the output hdf file:'
            for name in output_parameters:
                print ' * %s' % name
        else:
            print 'No matching parameters were found in the hdf file.'
    elif args.command == 'list':
        kwargs = {}
        if args.filter_nodes:
            kwargs['filter_nodes'] = args.filter_nodes
        modules = list(settings.NODE_MODULES)
        if args.additional_modules:
            modules += args.additional_modules
        print _get_names(modules, **kwargs)
    else:
        parser.error("'%s' is not a known command." % args.command)
