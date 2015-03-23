import collections
import pytz
import simplejson as json
import dateutil.parser

from datetime import datetime

from analysis_engine import settings
from analysis_engine.utils import get_derived_nodes


# VERSION will be included in json output. Only json matching the current VERSION number will be loaded.
VERSION = '0.1'

PROCESS_FLIGHT_RESULT_KEYS = (
    'flight',
    'kti',
    'kpv',
    'approach',
    'phases',
)


def sort_dict(d):
    """
    Return OrderedDict with sorted keys.
    """
    res = collections.OrderedDict()
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            v = sort_dict(v)
        res[k] = v
    return res


def node_to_jsondict(node):
    """
    Convert analysis_engine.node.Node into a dictionary convertable into JSON
    object.

    The returned dictionary is a SortedDict because we want to use the JSON to
    compare values.
    """
    if hasattr(node, 'todict'):
        # recordtypes and Attributes
        d = node.todict()
    elif hasattr(node, '_asdict'):
        # nametuples
        d = node._asdict()
    else:
        raise TypeError('Object does not support conversion to dictionary')

    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, datetime):
            d[k] = {
                'type': datetime.__name__,
                'value': v.isoformat(),
            }

        elif isinstance(v, slice):
            d[k] = {
                'type': slice.__name__,
                'value': (v.start, v.stop, v.step),
            }

        elif isinstance(v, dict):
            d[k] = {
                'type': dict.__name__,
                'value': v,
            }

        else:
            d[k] = v

    d['__class__'] = type(node).__name__

    return sort_dict(d)


def node_to_json(node):
    """
    Convert analysis_engine.node.Node into JSON object.
    """
    return json.dumps(node_to_jsondict(node))


def get_node_class(name):
    """
    Return a node class object for given name.

    This logic is needed mainly because our named_tuples exist in multiple
    modules, we need to find the correct class we are after.
    """
    from analysis_engine import node
    from analysis_engine import datastructures

    modules = (node, datastructures)
    for module in modules:
        if getattr(module, name, None):
            return getattr(module, name)


def jsondict_to_node(d):
    """
    Convert a dictionary as returnd from node_to_jsondict back into
    analysis_engine.node.Node.
    """
    node_cls_name = d.pop('__class__')
    cls = get_node_class(node_cls_name)
    kw = {}
    for k, n in d.items():
        if isinstance(n, dict) and 'value' in n and 'type' in n:
            val = n['value']
            if n['type'] == 'datetime':
                # TODO: do we need to force tzinfo to pytz?
                val = dateutil.parser.parse(val)
                if val.tzinfo is None:
                    val.replace(tzinfo=pytz.utc)
            elif n['type'] == 'slice':
                val = slice(*val)
        else:
            val = n

        kw[k] = val

    return cls(**kw)


def json_to_node(txt):
    """
    Convert a JSON string into analysis_engine.node.Node object.
    """
    d = json.loads(txt)
    return jsondict_to_node(d)


def process_flight_to_json(pf_results, indent=2):
    """
    Convert `process_flight` results into JSON object.
    """
    d = collections.OrderedDict()
    for key in PROCESS_FLIGHT_RESULT_KEYS:
        d[key] = {}
        for name, items in pf_results[key].items():
            d[key][name] = [node_to_jsondict(i) for i in items]
        
        # Q: Should we sort?
        #d[key] = sorted(d[key])
    
    d['version'] = VERSION
    
    return json.dumps(sort_dict(d), indent=indent)


def json_to_process_flight(txt):
    """
    Convert JSON to a data structure as returned by `process_flight`.
    
    :rtype: dict
    """
    d = json.loads(txt)
    
    if not d.get('version') == VERSION:
        return {}

    res = {}
    for key in PROCESS_FLIGHT_RESULT_KEYS:
        res[key] = {}
        for name, items in d[key].items():
            res[key][name] = [jsondict_to_node(i) for i in items]

    return res


def process_flight_to_nodes(pf_results):
    '''
    Load process flight results into Node objects.
    '''
    from analysis_engine import node
    
    derived_nodes = get_derived_nodes(settings.NODE_MODULES)
    
    params = {}
    
    for node_type, nodes in pf_results.iteritems():
        
        for node_name, items in nodes.iteritems():
            node_cls = derived_nodes[node_name]
            if node_type == 'flight' and items:
                flight_attr = node.FlightAttributeNode(node_name)
                flight_attr.set_flight_attr(items[0].value)
                params[node_name] = flight_attr
                continue
            
            params[node_name] = node_cls(node_name, items=items)
    
    return params