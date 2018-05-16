#! /usr/bin/env python
# coding: utf-8

import configparser
import numpy as np
import re,sys,os
from graph import MyGraph
from collections import OrderedDict

def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    from collections import defaultdict
    import io
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

def getFilters(mydict, name):
    #print('find filters for ', name)
    if hasattr(mydict[name], 'filters'):
        return mydict[name].filters
    else:
        assert len(mydict[name].input) >= 1
        return getFilters(mydict, mydict[name].input[0])


def readfile(f, len, msg):
    print("  %s read %d bytes" % (msg, len))
    return f.read(len)


def buildGraph(config_path, weights_path):

    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    weights_file = open(weights_path, 'rb')
    # read out major, minor, revision, net.seen
    readfile(weights_file, (4*4), 'head')

    mydict = OrderedDict()
    # record the output of the original layer
    mylist = []

    count = 4
    import queue
    for _section in cfg_parser.sections():

      sec_q = queue.Queue(0)
      sec_q.put(cfg_parser[_section])

      while not sec_q.empty():
        sec = sec_q.get()
        section = sec.name
        print('Parsing section {}'.format(section))

        # this section will can be a subsection
        if section.startswith('activation') or section.endswith('activation'):
            activation = sec.get('activation', fallback = 'logistic')
            if activation == 'linear':
                pass
            elif activation == 'linear' or activation == 'leaky' or activation == 'relu':
                node = MyGraph.MyNode()
                node.name = section
                node.op = 'Leaky'

                if activation == 'linear':
                    node.slope = 1
                elif  activation == 'leaky':
                    node.slope = 0.1
                elif activation == 'relu':
                    node.slope = 0
                    
                node.input = [prev_output]
                node.input_norm = node.input
                #node.attr = []
                mydict[node.name] = node
                prev_output = node.name
                # prev_layer_filters no change
            else:
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                    activation, section))
            if section.startswith('activation'):
                mylist.append(section)

        elif re.match(r'^(convolutional|depthwise|groupwise)_\d+$', section):
            if section.startswith('convolutional'):
                conv = 'conv'
                filters = sec.getint('filters', fallback = 1)
                groups = 1
                op = 'Conv2D'
            elif section.startswith('depthwise'):
                conv = 'dconv'
                filters = prev_layer_filters
                multiplier = sec.getint('multiplier', fallback = 1)
                assert multiplier == 1
                groups = filters
                op = 'DepthwiseConv2dNative'
            elif section.startswith('groupwise'):
                conv = 'gconv'
                filters = sec.getint('filters', fallback=1)
                groups = sec.getint('groups', fallback = 1)
                op = 'DepthwiseConv2dNative'

            size = sec.getint('size', fallback = 1)
            stride = sec.getint('stride', fallback = 1)
            pad = sec.getint('pad', fallback = 0)
            padding = sec.getint('padding', fallback = 0)
            activation = sec.get('activation', fallback = 'logistic')
            batch_normalize = sec.getint('batch_normalize', 0)

            # padding='same' is equivalent to Darknet pad=1
            # padding = 'same' if pad == 1 else 'valid'
            if pad:
                padding = size//2

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            #prev_layer_shape = prev_layer.shape

            # TODO: This assumes channel last dim_ordering.
            if conv == 'conv':
                weights_shape = (size, size, prev_layer_filters, filters)
                idx_tf2darknet = [0, 1, 2, 3]

            elif conv == 'dconv':
                weights_shape = (size, size, filters)
                idx_tf2darknet = [0, 1, 2]

            elif conv == 'gconv':
                weights_shape = (size, size, prev_layer_filters//groups, filters//groups, groups)
                idx_tf2darknet = [0, 1, 2, 3, 4]

            idxmap = {x: i for i, x in enumerate(idx_tf2darknet)}
            idx_dartnet2tf = [idxmap[i] for i in range(len(idxmap))]
            weights_size = np.product(weights_shape)

            print('  ' + conv, 'bn' if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype=np.float32,
                buffer=readfile(weights_file, (filters * 4), section+'-bias'))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype=np.float32,
                    buffer=readfile(weights_file, (filters * 12), section+'-batchnorm'))
                count += 3 * filters

                # TODO: Keras BatchNormalization mistakenly refers to var
                # as std.
                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=[weights_shape[i] for i in idx_tf2darknet],
                dtype=np.float32,
                buffer=readfile(weights_file, (weights_size * 4), section+'-weights'))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            # TODO: Add check for Theano dim ordering.
            #print("the darknet shape is ", conv_weights.shape)
            conv_weights = np.transpose(conv_weights, idx_dartnet2tf)
            #print("the tf shape is ", conv_weights.shape)
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Create nodes
            #conv_layer = np.zeros([1, 1, filters], dtype = np.float32)
            node = MyGraph.MyNode()
            node.name = section
            node.op = op
            node.input = [prev_output]
            node.input_norm = node.input
            node.kernel = conv_weights[0]
            node.padding = padding
            node.strides = [1,stride,stride,1]
            node.groups = groups
            node.filters = filters
            mydict[node.name] = node
            prev_output = node.name
            prev_layer_filters = filters

            if batch_normalize:
                node = MyGraph.MyNode()
                node.name = section + '_batch_normalize'
                node.op = 'FusedBatchNorm'
                node.input = [prev_output]
                node.input_norm = node.input
                #node.attr = []
                node.gamma = bn_weights[0]
                node.beta = conv_bias
                node.mean = bn_weights[1]
                node.variance = bn_weights[2]
                mydict[node.name] = node
                prev_output = node.name
                # prev_layer_filters no change
            else:
                node = MyGraph.MyNode()
                node.name = section + '_bias'
                node.op = 'BiasAdd'
                node.input = [prev_output]
                node.input_norm = node.input
                #node.attr = []
                node.bias = conv_bias
                mydict[node.name] = node
                prev_output = node.name

            if activation == 'linear':
                mylist.append(prev_output)
            else:
                tmp_parser = configparser.ConfigParser()
                name = section + '_activation'
                tmp_parser.add_section(name)
                tmp_parser.set(name, 'activation', activation)
                sec_q.put(tmp_parser[name])
                mylist.append(name)

        elif section.startswith('shuffle'):
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'Shuffle'
            node.input = [prev_output]
            node.input_norm = node.input

            node.groups = int(cfg_parser[section]['groups'])
            mydict[node.name] = node
            prev_output = node.name
            mylist.append(section)


        elif re.match(r'^(pooling|maxpool|avgpool)_\d+$', section):
            node = MyGraph.MyNode()
            node.stride = sec.getint('stride', fallback = 1)
            node.size = sec.getint('size', node.stride)
            node.padding = sec.getint('padding', fallback = (node.size-1)//2)

            if section.startswith('pooling'):
                node.mode = str(cfg_parser[section]['mode'])
                node.global_pooling = 0
            elif section.startswith('maxpool'):
                node.mode = 'max'
                node.global_pooling = 0
            elif section.startswith('avgpool'):
                node.mode = 'avg'
                node.global_pooling = 1
           
            node.name = section
            node.op = 'Pooling'
            node.input = [prev_output]
            node.input_norm = node.input
            mydict[node.name] = node
            prev_output = node.name
            #print('pooling ', vars(node))
            mylist.append(section)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'NCNNConcat'
            node.input = [mylist[i] for i in ids]
            #print('mylist is ', mylist, 'the ids is ', ids, 'node input is ', node.input)
            node.input_norm = node.input
            node.axis = 0
            node.filters = sum([getFilters(mydict, mylist[i]) for i in ids])
            mydict[node.name] = node
            prev_output = node.name
            mylist.append(section)
            prev_layer_filters = node.filters

        elif section.startswith('reorg'):
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'DarknetReorg'
            node.input = [prev_output]
            node.stride = sec.getint('stride', fallback = 1)
            node.input_norm = node.input
            node.filters = getFilters(mydict, node.input[0]) * node.stride * node.stride
            mydict[node.name] = node
            prev_output = node.name
            mylist.append(section)
            prev_layer_filters = node.filters

        elif re.match(r'^(shortcut)_\d+$', section):
            activation = sec.get('activation', fallback = 'logistic')
            from_ = sec.getint('from')


            node = MyGraph.MyNode()
            node.name = section
            node.op = 'BinaryOp'
            node.op_type = 0
            node.input = [prev_output, mylist[from_]]
            #print('mylist is ', mylist, 'the from_ is ', from_, 'node input is ', node.input)
            node.input_norm = node.input
            mydict[node.name] = node
            prev_output = node.name

            if activation == 'linear':
                mylist.append(prev_output)
            else:
                tmp_parser = configparser.ConfigParser()
                name = section + '_activation'
                tmp_parser.add_section(name)
                tmp_parser.set(name, 'activation', activation)
                sec_q.put(tmp_parser[name])
                # NOTE: this section has relative reference
                mylist.append(name)


        elif section.startswith('connected'):
            activation = sec.get('activation', fallback='linear')
            filters = sec.getint('output', 2)

            bias_data = np.ndarray(
                shape=[filters],
                dtype=np.float32,
                buffer=readfile(weights_file, (filters * 4), section+'-bias'))

            fc_data = np.ndarray(
                shape=[prev_layer_filters, filters],
                dtype=np.float32,
                buffer=readfile(weights_file, (prev_layer_filters * filters * 4), section+'-weight'))

            node = MyGraph.MyNode()
            node.name = section
            node.op = 'MatMul'
            node.input = [prev_output]
            node.input_norm = node.input
            node.multiplier = fc_data
            mydict[node.name] = node
            prev_output = node.name
            prev_layer_filters = filters

            node = MyGraph.MyNode()
            node.name = section + '_bias'
            node.op = 'BiasAdd'
            node.input = [prev_output]
            node.input_norm = node.input
            # node.attr = []
            node.bias = bias_data
            mydict[node.name] = node
            prev_output = node.name


            if activation == 'linear':
                mylist.append(prev_output)
            else:
                tmp_parser = configparser.ConfigParser()
                name = section + '_activation'
                tmp_parser.add_section(name)
                tmp_parser.set(name, 'activation', activation)
                sec_q.put(tmp_parser[name])
                mylist.append(name)

        elif section.startswith('net'):
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'DarknetNet'
            node.input = []
            node.input_norm = []

            node.width = int(cfg_parser['net_0']['width'])
            node.height = int(cfg_parser['net_0']['height'])
            node.channels = int(cfg_parser['net_0']['channels'])
            node.filters = node.channels
            # print(vars(node))
            # node.attr = []
            mydict[node.name] = node
            # start here
            prev_output = node.name
            prev_layer_filters = node.channels
            mylist.append(section)

        elif section.startswith('region'):
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'DarknetRegion'
            node.input = [prev_output]
            node.input_norm = node.input

            node.classes = int(cfg_parser[section]['classes'])
            node.num = int(cfg_parser[section]['num'])
            node.softmax = int(cfg_parser[section]['softmax'])
            node.anchors = [float(i) for i in re.split(r',', cfg_parser[section]['anchors'])]

            #print(vars(node))
            #node.attr = []
            mydict[node.name] = node
            prev_output = node.name
            mylist.append(section)

        elif section.startswith('softmax'):
            node = MyGraph.MyNode()
            node.name = section
            node.op = 'Softmax'
            node.input = [prev_output]
            node.input_norm = node.input

            mydict[node.name] = node
            prev_output = node.name
            mylist.append(section)
            pass

        elif section.startswith('cost'):
            pass  # Configs not currently handled during model definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))
        print('  out filters ', prev_layer_filters)
    print('loaded {} bytes in weights file'.format(count*4))

    mygraph = MyGraph(mydict)
    mygraph.type = 'darknet'
    return mygraph

if __name__ == '__main__':
    config_path = sys.argv[1]
    weights_path = sys.argv[2]
    mygraph = buildGraph(config_path, weights_path)

    # 定义子图所需要的输出节点，输入节点，终止节点
    outputNodes = ['region_0', 'softmax_0']
    stopNodes = []
    inputNodes = ['darknet_0']
    mygraph.extractSubGraph(inputNodes, outputNodes, stopNodes)
    mygraph.generateDot('YoloV2.dot')

    # 生成子图对应的代码
    mygraph.generateSource('YoloV2', os.path.split(config_path)[1]+'.ncnn', os.path.split(weights_path)[1] + '.ncnn')

