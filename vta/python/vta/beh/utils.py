'''utils used for bsim'''

# Modified by contributors from Intel Labs

from collections import OrderedDict
import numpy as np
import heterocl as hcl
from . import datatype

def hcldtype2np(hcldtype):
    width = hcl.get_bitwidth(hcldtype)
    dct = {8: np.uint8,
           16: np.uint16,
           32: np.uint32,
           64: np.uint64
          }
    assert width in dct, f'unsupported dtype with width{width}'
    return dct[width]

def proc_instruction_fmt(fmt):
    '''process instruction format, popolate a dict'''
    assert isinstance(fmt, tuple), 'expect tuple type for instruction format'
    assert len(fmt) > 0 and isinstance(fmt[0], tuple), \
        'expect tuple type for the first element in instruction format'
    prolog, epilog = [], []
    props = OrderedDict()
    for elem in fmt:
        if isinstance(elem, tuple):
            target = prolog if len(props) == 0 else epilog
            target.append(elem)
        elif isinstance(elem, dict):
            for k, v in elem.items():
                # TODO: make sure each one is a tuple of tuples
                props[k] = prolog + list(v)
        else:
            raise Exception('unsupported data type in instruction format')
    for k, v in props.items():
        props[k].extend(epilog)

    classes = OrderedDict()
    for k, v in props.items():
        # TODO: make sure there is no spaces in the fields
        offset = 0
        offset_dict = {}
        bwidth_dict = {}
        for name, bitwidth in v:
            bwidth_dict[name] = bitwidth
            offset_dict[name] = offset
            # d[name] = {'bitwidth': bitwidth, 'offset': offset, 'value': None, 'handler': {}}
            offset += bitwidth
        dct = {'_type': k,
               '_defined_fields': [x for x, _ in v],
               '_bitwidth': bwidth_dict,
               '_offset': offset_dict,
               '_width': offset               # total width of the instruction
              }
        classes[k] = dct
    return classes

CLASSES = proc_instruction_fmt(datatype.INSTRUCTION_FMT)

# ts = hcl.Struct({"fa": hcl.Int(8), "fb": hcl.Fixed(8, 2), "fc": hcl.Float()})
def make_struct(encoding):
    '''return a hcl.struct from a given'''
    assert 'type' in encoding
    superclass_name = encoding['type']
    # name, top = encoding['name'], encoding['top']
    assert superclass_name in CLASSES, f'undefined instruction type `{superclass_name}`'
    superclass = CLASSES[superclass_name]
    fields = OrderedDict()
    for f in superclass['_defined_fields']:
        bitwidth = superclass['_bitwidth'][f]
        offset = superclass['_offset'][f]
        fields[f] = hcl.UInt(bitwidth)
        print(f'{f} = hcl.scalar(instr[{offset+bitwidth}:{offset}], name="{f}")')
        # fields.append((f, hcl.UInt(bitwidth)))
    return hcl.Struct(fields)
