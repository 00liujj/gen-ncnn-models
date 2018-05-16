#! /usr/bin/env python
# coding: utf-8

from graph import MyGraph
import tensorflow as tf
import re,sys,os
from collections import OrderedDict

def buildGraph(pbfile, opmap):
    gdef = tf.GraphDef()
    gdef.ParseFromString(open(pbfile, 'rb').read())
    open('frozen.txt', 'w').write(str(gdef))

    def extract(name):
        n = name
        # return n
        n = re.sub(r'^\^', '', n)
        n = re.sub(r':\d+$', '', n)
        return n

    nodedict = OrderedDict()
    for node in gdef.node:
        nd = MyGraph.MyNode()
        nd.name = node.name
        if node.op in opmap:
            nd.op = opmap[node.op]
        else:
            nd.op = node.op
        nd.input = node.input
        nd.attr = node.attr
        nd.input_norm = [extract(i) for i in node.input]
        nodedict[extract(node.name)] = nd
    mygraph = MyGraph(nodedict)
    mygraph.type = 'tf'
    return mygraph


if __name__ == '__main__':
    pbfile = sys.argv[1]

    opmap = {'Mean': 'GlobalAvgPool',
             'AvgPool': 'GlobalAvgPool',
             'Reshape' :'IdentityOp',
             'Squeeze' : 'IdentityOp',
    #         'Placeholder' : 'DarknetNet',
    }

    mygraph = buildGraph(pbfile, opmap)

    # 定义子图所需要的输出节点，输入节点，终止节点
    softmax_nodes = []
    for nodename in mygraph.nodedict:
        node = mygraph.nodedict[nodename]
        if node.op == 'Softmax':
            softmax_nodes.append(node.name)
    if len(softmax_nodes) != 1:
        raise ValueError("there should be 1 softmax node in the graph")

    outputNodes = softmax_nodes
    inputNodes = ['input']
    #inputNodes = ['Preprocessor/sub']
    #outputNodes = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6', 'concat_box_encodings', 'concat_cls_predictions_with_background']
    #outputNodes = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6']
    #outputNodes = ['BoxPredictor_%d/BoxEncodingPredictor/BiasAdd' %i for i in (0,1,2),
    # 'BoxPredictor_%d/ClassPredictor/BiasAdd' %i for i in (0,1,2)]
    stopNodes = ['Assert', 'Equal']
    node = mygraph.nodedict[inputNodes[0]]
    node.mean = 255/2.
    node.norm = 1./node.mean
    mygraph.extractSubGraph(inputNodes, outputNodes, stopNodes)
    
    name = pbfile
    mygraph.generateDot(name + '.dot')
    # 生成子图对应的代码
    mygraph.generateSource(name, name + '.cfg.ncnn', name + '.weights.ncnn')
