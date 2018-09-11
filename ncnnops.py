import os, sys, re, string
import numpy as np
import traceback

header = """\
#pragma once
#include "ncnn.hpp"
#include <stdio.h>
#include "darknet_head.h"
using namespace ncnn;

#ifdef __arm__
#define ARMWRAP(t) t##_arm
#else
#define ARMWRAP(t) t
#endif

class ${moduleName} {
public:
    void initialize(FILE* fp);
    void compute();

public:
${declaration}
}; // end of ${moduleName}

"""

source = '''\
#include "${moduleName}.hpp"


static void load_model(Layer& l, FILE* fp) {
    int pos1 = ftell(fp);
    l.load_model(fp);
    int pos2 = ftell(fp);
    printf("layer %s start %d, read %d\\n", l.name.c_str(), pos1, pos2-pos1);;  
}

static void forward_layer(Layer& l, const Tensor& in, Tensor& out) {
    Tensor tmp;
    int ret = l.forward(in, tmp);
    assert(ret == 0);
    tf::expand_dims(out, tmp, 0);
}

void print(const Mat& mat, const std::string& name);


void ${moduleName}::initialize(FILE* fp) {
${initializeBody}
}

void ${moduleName}::compute() {
    clock_t t1 = clock();
${computeBody}

    clock_t t2 = clock();
    printf("the cost time is %f ms\\n", float(t2-t1)*1e3/CLOCKS_PER_SEC);
}

'''



class MyTemplate(string.Template):
    delimiter = '$'

# how to process the const values!?
# for scalar const, use code
# otherwise, write to the file
class Op:
    clsName = "UnknownType"
    def __init__(self, graph, nodeName):
        self.clsName = self.clsName # weired!!!
        self.graph = graph
        self.nodeName = nodeName
        self.node = graph.nodedict[nodeName]
        self.opName = self.fancyName(graph.type, self.node)
        self.outVarName = self.opName + "_out"

        self.inNodes = []
        self.inOpNames = []
        self.inVarNames = []
        for i, name in enumerate(self.node.input_norm):
            inNode = graph.nodedict[name]
            inOpName = self.fancyName(graph.type, inNode)
            inVarName = inOpName + "_out"
            self.inNodes.append(inNode)
            self.inOpNames.append(inOpName)
            self.inVarNames.append(inVarName)
            setattr(self, 'inNode_%d' % i, inNode)
            setattr(self, 'inOpName_%d' % i, inOpName)
            setattr(self, 'inVarName_%d' % i, inVarName)

            if i == 0:
                self.inNode = inNode
                self.inOpName = inOpName
                self.inVarName = inVarName

        self.inVarNum = len(self.node.input_norm)

        pcls = self.clsName
        pcls = re.sub(r'.*\(', '', pcls)
        pcls = re.sub(r'\)', '', pcls)
        self.clsNameRoot = pcls

    def getInVarNames(self, sep = ', '):
        allstr = ""
        for i in range(0, self.inVarNum):
            allstr += getattr(self, 'inVarName_%d' % i) + sep

        allstr = allstr.strip(sep)
        return allstr


    def genDeclaration(self):
        return ""

    def genInitializeFun(self):
        return ""

    def genComputeFun(self):
        return ""

    def genWeightFun(self, weight_file):
        pass

    def genModelFun(self):
        return ""

    @staticmethod
    def fancyName(type, node):
        if type == 'darknet' and node.op != 'DarknetRegion' and node.op != 'DarknetNet' and node.op != 'Softmax':
            return node.name
        else:
            return node.op + "_" + str(node.lnum)

    @staticmethod
    def list2code(lst):
        l = len(lst)
        code = ''
        for i, v in enumerate(lst):
            if i < l-1:
                code += str(v) + ', '
            else:
                code += str(v)
        return code

    #@staticmethod
    def write2file(self, f, d):
        print('  %s write %d bytes' % (self.nodeName, len(d)))
        f.write(d)

    @staticmethod
    def parseConst(node):
        # node is a const node, return the corresponding numpy ndarray
        #print(vars(node))
        typemap = {
            1: (np.float32, 'float_val'),  # 'DT_FLOAT'
            3: (np.int32, 'int_val'), # 'DT_INT32'
        }
        dtype = node.attr['dtype'].type

        tensor = node.attr['value'].tensor
        shape = [i.size for i in tensor.tensor_shape.dim]
        #print(dtype, shape, type(dtype), type(shape))
        assert dtype in typemap
        item = typemap[dtype]
        if len(shape) == 0:
            # scalar
            val = getattr(tensor, item[1])
            #print('rank 0', type(val), val, len(val))
            array = np.array(val[0], dtype = item[0])
            # vector
        elif len(shape) == 1 and shape[0] == 1:
            val = getattr(tensor, item[1])
            #print('rank 1', type(val), val, len(val))
            array = np.array([val[0]], dtype=item[0])
        else:
            # tensor
            array = np.ndarray(shape=shape, dtype=item[0], buffer=tensor.tensor_content)
        #print(array)
        return array


class NCNNOp(Op):
    def genDeclaration(self):
        code = """\
    // $nodeName
    $clsName $opName;
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // ${nodeName}
    forward_layer($opName, $inVarName, $outVarName);
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    pass

    def genModelFun(self):
        code = '${clsNameRoot} ${opName} $inVarNum 1 ${allvars} ${outVarName}'
        code = MyTemplate(code).safe_substitute(
            allvars=self.getInVarNames(' '), **vars(self))
        return code


# only for scalar const
class Const(Op):
    clsName = 'Const'
    def __init__(self, graph, nodeName):
        super(Const, self).__init__(graph, nodeName)
        self.array = self.parseConst(self.node)
        self.data = self.array.sum()
        self.multi = self.array.ndim > 1 or (self.array.ndim == 1 and self.array.shape[0] > 1)
    def genDeclaration(self):

        #print('Block ', self.opName, self.outVarName, self.array.shape, self.multi)

        if self.multi:
            return ""

        code = """\
    // $nodeName
    Tensor $outVarName;
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genInitializeFun(self):

        if self.multi:
            return ""

        code = """\
    // $nodeName
    {
       $outVarName.create(1); $outVarName(0) = $data;
    }
    """
        return MyTemplate(code).safe_substitute(**vars(self))


class Placeholder(Op):
    clsName = 'Placeholder'
    def genDeclaration(self):
        code = """\
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = 'Input ${opName} 0 1 ${outVarName} 0=${width} 1=${height} 2=${channels}'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code


class Conv2D(NCNNOp):
    clsName = 'ARMWRAP(Convolution)'
    def __init__(self, graph, nodeName):
        super(Conv2D, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            format = self.node.attr['data_format'].s.decode()
            assert format == "NHWC"
            kernel = self.parseConst(self.inNodes[1])
            # transpose!
            # NCNN conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            kernel_T = np.transpose(kernel, [3, 2, 0, 1])
            self.kernel = np.reshape(kernel_T, kernel.shape)
            self.padding = self.node.attr['padding'].s.decode()
            self.strides = [i for i in self.node.attr['strides'].list.i]
            self.filters = self.kernel.shape[3]
        else:
            self.kernel = self.node.kernel
            self.padding = self.node.padding
            self.strides = self.node.strides
            self.filters = self.node.filters


        self.kernel_size = self.kernel.shape[-1]
        self.dilation = 1
        self.stride = self.strides[1]
        self.pad = -233 if self.padding == "SAME" else self.padding
        self.bias_term = 0
        self.weight_data_size = self.kernel.size
        self.groups = 1

    def genDeclaration(self):
        code = """\
    // $nodeName
    ${clsName} ${opName};
    Tensor ${opName}_weight;
    Tensor ${outVarName};"""
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genInitializeFun(self):
        code = """\
    // $nodeName
    ${opName}.num_output = $filters;
    ${opName}.kernel_size = $kernel_size;
    ${opName}.dilation = $dilation;
    ${opName}.stride = $stride;
    ${opName}.pad = $pad;
    ${opName}.bias_term = $bias_term;
    ${opName}.weight_data_size = $weight_data_size;
    ${opName}.load_model(fp);
    """
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genComputeFun(self):
        code = '''
    // $nodeName
    forward_layer(${opName}, ${inVarName}, ${outVarName});
    print(${opName}.weight_data, "${opName}-kernel");
    print(${inVarName}, "${opName}-bottom");
    print(${outVarName}, "${opName}-top");
    '''
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genModelFun(self):
        code = '${clsNameRoot} ${opName} 1 1 ${inVarName} ${outVarName} ' \
               '0=${filters} 1=${kernel_size} 2=${dilation} 3=${stride} ' \
               '4=${pad} 5=${bias_term} 6=${weight_data_size}'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genWeightFun(self, weight_file):
        quantization_tag = 0
        weight_file.write(quantization_tag.to_bytes(4, sys.byteorder))
        self.write2file(weight_file, self.kernel.data.tobytes())




class DepthwiseConv2dNative(Conv2D):
    clsName = "ARMWRAP(ConvolutionDepthWise)"

    def __init__(self, graph, nodeName):
        super(DepthwiseConv2dNative, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            self.groups = self.kernel.shape[2]
            self.filters = self.kernel.shape[2]
        else:
            self.groups = self.node.groups

    def genInitializeFun(self):
        code = """\
        // $nodeName
        ${opName}.num_output = $filters;
        ${opName}.kernel_size = $kernel_size;
        ${opName}.dilation = $dilation;
        ${opName}.stride = $stride;
        ${opName}.pad = $pad;
        ${opName}.bias_term = $bias_term;
        ${opName}.weight_data_size = $weight_data_size;
        ${opName}.group = $groups;
        load_model(${opName}, fp);
        """
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genModelFun(self):
        code = super(DepthwiseConv2dNative, self).genModelFun()
        code += """ 7=${groups}"""
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code



class Leaky(NCNNOp):
    clsName = "ARMWRAP(ReLU)"
    def __init__(self, graph, nodeName):
        super(Leaky, self).__init__(graph, nodeName)
        if hasattr(self.node, 'slope'):
            self.slope = str(self.node.slope)
            if len(self.slope) > 6:
                self.slope = '%.6e' % self.node.slope
        else:
            self.slope = '0.0'

    def genInitializeFun(self):
        code = """\
    // $nodeName
    {
        ${opName}.slope = ${slope};
    }"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = '${clsNameRoot} ${opName} 1 1 ${inVarName} ${outVarName} 0=${slope}'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

class Relu(Leaky):
    pass

class Relu6(NCNNOp):
    clsName = "ReLU6"


class Pooling(NCNNOp):
    clsName = "ARMWRAP(Pooling)"

    def genInitializeFun(self):
        code = '''
    // $nodeName
    {
        ${opName}.pooling_type = $pooling_type;
        ${opName}.kernel_size = $kernel_size;
        ${opName}.stride = $stride;
        ${opName}.pad = $pad;
        ${opName}.global_pooling = $global_pooling;
    }    
    '''
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

    def genModelFun(self):
        code = '${clsNameRoot} ${opName} 1 1 ${inVarName} ${outVarName} 0=${pooling_type}' \
               ' 1=${kernel_size} 2=${stride} 3=${pad} 4=${global_pooling}'
        code = MyTemplate(code).safe_substitute(
            pooling_type = 0 if self.node.mode == "max" else 1,
            kernel_size = self.node.size,
            stride = self.node.stride,
            pad = self.node.padding,
            global_pooling = self.node.global_pooling,
            **vars(self))
        return code


class FusedBatchNorm(NCNNOp):
    clsName = 'ARMWRAP(BatchNorm)'

    def __init__(self, graph, nodeName):
        super(FusedBatchNorm, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            assert "NHWC" == self.node.attr['data_format'].s.decode()
            self.gamma = self.parseConst(self.inNodes[1])
            self.beta = self.parseConst(self.inNodes[2])
            self.mean = self.parseConst(self.inNodes[3])
            self.variance = self.parseConst(self.inNodes[4])
        else:
            self.gamma = self.node.gamma
            self.beta  = self.node.beta
            self.mean  = self.node.mean
            self.variance = self.node.variance

    def genDeclaration(self):
        code = """\
    // $nodeName
    ${clsName} ${opName};
    Tensor ${opName}_gamma, ${opName}_beta, ${opName}_mean, ${opName}_variance;
    Tensor ${outVarName};"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genInitializeFun(self):
        code = """\
    // $nodeName
    {
        $clsName &l = $opName;
        l.channels = ${inOpName}.num_output;
        load_model(l, fp);
    }"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        supercode = super(FusedBatchNorm, self).genComputeFun()
        code = '''
    // $nodeName
    forward_layer(${opName}, ${inVarName}, ${outVarName});
    print(${opName}.slope_data, "bn-slope");
    print(${opName}.mean_data, "bn-mean");
    print(${opName}.var_data, "bn-var");
    print(${opName}.bias_data, "bn-bias");
    print(${opName}.a_data, "bn-a");
    print(${opName}.b_data, "bn-b");
    print(${inVarName}, "bn-bottom");
    print(${outVarName}, "bn-top");
    '''
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = 'BatchNorm ${opName} 1 1 ${inVarName} ${outVarName} 0=${channels}'
        code = MyTemplate(code).safe_substitute(channels=self.gamma.size, **vars(self))
        return code

    def genWeightFun(self, weight_file):
        self.write2file(weight_file, self.gamma.data.tobytes())
        self.write2file(weight_file, self.mean.data.tobytes())
        self.write2file(weight_file, self.variance.data.tobytes())
        self.write2file(weight_file, self.beta.data.tobytes())


class DarknetReorg(NCNNOp):
    clsName = 'Reorg'
    def genModelFun(self):
        code = super(DarknetReorg, self).genModelFun()
        code += ' 0=$stride'
        code = MyTemplate(code).safe_substitute(stride = self.node.stride)
        return code


class BiasAdd(NCNNOp):
    clsName = 'ARMWRAP(Bias)'

    def __init__(self, graph, nodeName):
        super(BiasAdd, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            format = self.node.attr['data_format'].s.decode()
            assert "NHWC" == self.node.attr['data_format'].s.decode()
            self.bias = self.parseConst(self.inNodes[1])
        else:
            self.bias = self.node.bias

    def genDeclaration(self):
        code = """\
    // $nodeName
    ${clsName} ${opName};
    Tensor ${opName}_bias;
    Tensor ${outVarName};"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genInitializeFun(self):
        code = """\
    // $nodeName
    {
        $clsName &l = $opName;
        l.bias_data_size = ${inOpName}.num_output;
        load_model(l, fp);
    }"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = 'Bias ${opName} 1 1 ${inVarName} ${outVarName} 0=${dataSize}'
        code = MyTemplate(code).safe_substitute(dataSize = self.bias.size, **vars(self))
        return code

    def genWeightFun(self, weight_file):
        self.write2file(weight_file, self.bias.data.tobytes())


class DarknetNet(Op):
    clsName = 'DarknetNet'

    def __init__(self, graph, nodeName):
        super(DarknetNet, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            shape = [i.size for i in self.node.attr['shape'].shape.dim]
            #traceback.print_stack()
            #print(shape) # why print twice??
            self.width, self.height, self.channels = shape[1:]
        else:
            self.width = self.node.width
            self.height = self.node.height
            self.channels = self.node.channels

    def genDeclaration(self):
        code = """\
    // $nodeName
    class $clsName {
    public:
        int width;
        int height;
        int channels;
    };
    $clsName $opName;
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genInitializeFun(self):
        code = """\
    // $nodeName
    {
        ${opName}.width = $width;
        ${opName}.height = $height;
        ${opName}.channels = $channels;
    }"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = 'Input ${opName} 0 1 ${outVarName} 0=${width} 1=${height} 2=${channels}'
        if hasattr(self.node, 'mean') and hasattr(self.node, 'norm'):
            self.mean = '%.6e' % self.node.mean
            self.norm = '%.6e' % self.node.norm
            code = 'InputSN ${opName} 0 1 ${outVarName} ' \
                   '0=${width} 1=${height} 2=${channels} 3=${mean} 4=${norm}'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code


class DarknetRegion(Op):
    clsName = 'DarknetRegion'

    def __init__(self, graph, nodeName):
        super(DarknetRegion, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            assert False, 'tf can not reach darknet region'
        else:
            self.classes = self.node.classes
            self.num = self.node.num
            self.anchors = self.list2code(self.node.anchors)
            self.softmax = self.node.softmax

    def genDeclaration(self):
        code = """\
    // $nodeName
    class $clsName {
    public:
        int classes;
        int num;
        int softmax;
        Tensor anchors;
    };
    $clsName $opName;
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genInitializeFun(self):
        code = """\
    // $nodeName
    {
        ${opName}.classes = $classes;
        ${opName}.num = $num;
        ${opName}.softmax = $softmax;
        float _anchors[] = { $anchors };
        ${opName}.anchors = Tensor(2, ${opName}.num ,_anchors).clone();
    }"""
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = '''
    // $nodeName
    region_layer(${outVarName}, ${inVarName}, ${opName}.classes, ${opName}.num, ${opName}.softmax);
    print(${inVarName}, "before_region");
    print(${outVarName}, "after_region");
    '''
        return MyTemplate(code).safe_substitute(**vars(self))

    def genModelFun(self):
        code = 'Region ${opName} 1 1 ${inVarName} ${outVarName} 0=$classes 1=$num 2=$softmax -23303=$length,$_anchors'
        code = MyTemplate(code).safe_substitute(
            length=len(self.node.anchors),
            _anchors = ','.join(['%f'%i for i in self.node.anchors]),
            **vars(self))
        return code



class Shape(Op):
    clsName = "Shape"


    def genDeclaration(self):
        code = '''
    // $nodeName
    Tensor ${outVarName};
    '''
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = '''
    // $nodeName
    tf::shape(${outVarName}, ${inVarName});
    '''
        return MyTemplate(code).safe_substitute(**vars(self))



class Reshape(NCNNOp):
    clsName = "Reshape"

    def genDeclaration(self):
        code = '''
    // $nodeName
    Tensor ${outVarName};
    '''
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = '''
    // $nodeName
    tf::reshape(${outVarName}, ${inVarName}, ${inVarName_1});
    '''
        return MyTemplate(code).safe_substitute(**vars(self))


class Shuffle(NCNNOp):
    clsName = "Shuffle"

    def genModelFun(self):
        code = '${clsName} ${opName} 1 1 ${inVarName} ${outVarName} 0=$groups'
        code = MyTemplate(code).safe_substitute(groups = self.node.groups, **vars(self))
        return code

class Softmax(NCNNOp):
    clsName = "Softmax"
    def __init__(self, graph, nodeName):
        super(Softmax, self).__init__(graph, nodeName)
        self.axis = 0

    def genModelFun(self):
        code = '${clsName} ${opName} 1 1 ${inVarName} ${outVarName} 0=$axis'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code

class BinaryOp(NCNNOp):
    clsName = "BinaryOp"

    def genModelFun(self):
        code = '${clsName} ${opName} 2 1 ${inVarName} ${inVarName_1} ${outVarName} 0=$op_type'
        code = MyTemplate(code).safe_substitute(op_type = self.node.op_type, **vars(self))
        return code

class NCNNConcat(NCNNOp):
    clsName = "Concat"

    def genModelFun(self):
        #print(vars(self.node))
        code = '${clsName} ${opName} $inVarNum 1 ${allvars} ${outVarName} 0=$axis'
        code = MyTemplate(code).safe_substitute(axis = self.node.axis,
                                                allvars = self.getInVarNames(' '),
                                                **vars(self))
        return code


class ConcatV2(Op):
    clsName = "ConcatV2"
    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    {
    Tensor _params[] = { ${allvars} };
    tf::concat_v2(${outVarName}, sizeof(_params)/sizeof(_params[0]), _params);
    }
    """
        return MyTemplate(code).safe_substitute(allvars = self.getInVarNames(), **vars(self))


class StridedSlice(Op):
    clsName = "StridedSlice"
    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    tf::strided_slice(${outVarName}, ${inVarName}, ${inVarName_1}, ${inVarName_2}, ${inVarName_3},
    $begein_mask, $end_mask, $ellipsis_mask, $new_axis_mask, $shrink_axis_mask
    );
    """
        self.begein_mask = self.node.attr['begein_mask'].i
        self.end_mask = self.node.attr['end_mask'].i
        self.ellipsis_mask = self.node.attr['ellipsis_mask'].i
        self.new_axis_mask = self.node.attr['new_axis_mask'].i
        self.shrink_axis_mask = self.node.attr['shrink_axis_mask'].i
        return MyTemplate(code).safe_substitute(**vars(self))

class Add(Op):
    clsName = "Add"

    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    tf::add(${outVarName}, ${inVarName}, ${inVarName_1});
    """
        return MyTemplate(code).safe_substitute(**vars(self))


class Pack(Op):
    clsName = "Pack"
    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    {
    Tensor _params[] = { ${allvars} };
    tf::pack(${outVarName}, sizeof(_params)/sizeof(_params[0]), _params, 0);
    }
    """
        return MyTemplate(code).safe_substitute(allvars=self.getInVarNames(), **vars(self))


class Fill(Op):
    clsName = "Fill"
    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    tf::fill(${outVarName}, ${inVarName}, ${inVarName_1});
    """
        return MyTemplate(code).safe_substitute(**vars(self))



class Squeeze(NCNNOp):
    clsName = "Squeeze"

    def __init__(self, graph, nodeName):
        super(Squeeze, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            self.squeeze_dims =[i for i in self.node.attr['squeeze_dims'].list.i]
            self.squeeze_dims = self.squeeze_dims[0]

    def genDeclaration(self):
        code = """
    // $nodeName
    Tensor ${outVarName};
    """
        return MyTemplate(code).safe_substitute(**vars(self))

    def genComputeFun(self):
        code = """
    // $nodeName
    tf::squeeze_dims(${outVarName}, ${inVarName}, ${squeeze_dims});
    """
        return MyTemplate(code).safe_substitute(**vars(self))


class GlobalAvgPool(Pooling):
    def __init__(self, graph, nodeName):
        super(GlobalAvgPool, self).__init__(graph, nodeName)
        self.node.mode = "avg"
        self.node.size = 1
        self.node.stride = 1
        self.node.padding = 0
        self.node.global_pooling = 1


class GlobalMaxPool(Pooling):
    def __init__(self, graph, nodeName):
        super(GlobalMaxPool, self).__init__(graph, nodeName)
        self.node.mode = "max"
        self.node.size = 1
        self.node.stride = 1
        self.node.padding = 0
        self.node.global_pooling = 1


class MaxPool(Pooling):
    def __init__(self, graph, nodeName):
        super(MaxPool, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            padding = self.node.attr['padding'].s.decode()
            strides = [i for i in self.node.attr['strides'].list.i]
            ksize = [i for i in self.node.attr['ksize'].list.i]
            self.node.mode = 'max'
            self.node.size = ksize[1]
            self.node.stride = strides[1]
            self.node.padding = -233 if padding == "SAME" else 0
            self.node.global_pooling = 0

class IdentityOp(NCNNOp):
    clsName = 'Identity'

    def genModelFun(self):
        code = '${clsNameRoot} ${opName} 1 1 ${inVarName} ${outVarName}'
        code = MyTemplate(code).safe_substitute(**vars(self))
        return code


class MatMul(NCNNOp):
    clsName = 'InnerProduct'
    def __init__(self, graph, nodeName):
        super(MatMul, self).__init__(graph, nodeName)
        if graph.type == 'tf':
            self.multiplier = self.parseConst(self.inNodes[1])
        else:
            self.multiplier = self.node.multiplier
    def genModelFun(self):
        code = '${clsNameRoot} ${opName} 1 1 ${inVarName} ${outVarName} 0=${num_output} ' \
               '1=${bias_term} 2=${weight_data_size}'
        code = MyTemplate(code).safe_substitute(
            num_output=self.multiplier.shape[1],
            bias_term = 0,
            weight_data_size = self.multiplier.size,
            **vars(self)
        )
        return code

    def genWeightFun(self, weight_file):
        self.write2file(weight_file, self.multiplier.data.tobytes())
