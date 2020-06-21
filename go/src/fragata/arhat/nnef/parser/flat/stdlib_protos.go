//
// Copyright (c) 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

//
// Based on the code of The Khronos Group Inc. NNEF Tools.
// Ported from C++ to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package flat

import "fragata/arhat/nnef/core"

var prototypes []*core.Prototype

func init() {
    primitiveType := core.GetPrimitiveType
    tensorType := core.GetTensorType
    arrayType := core.GetArrayType
    tupleType := core.GetTupleType

    scalarValue := core.NewScalarValue
    integerValue := core.NewIntegerValue
    logicalValue := core.NewLogicalValue
    stringValue := core.NewStringValue
    arrayValue := core.NewArrayValue

    scalar := primitiveType(core.TypenameScalar)
    integer := primitiveType(core.TypenameInteger)
    logical := primitiveType(core.TypenameLogical)
    strng := primitiveType(core.TypenameString)
    generic := primitiveType(core.TypenameGeneric)

    scalarTensor := tensorType(core.TypenameScalar)
    integerTensor := tensorType(core.TypenameInteger)
    logicalTensor := tensorType(core.TypenameLogical)
    genericTensor := tensorType(core.TypenameGeneric)

    integers := arrayType(integer)
    generics := arrayType(generic)
    tensors := arrayType(scalarTensor)
    genericTensors := arrayType(genericTensor)
    integerPair := tupleType([]core.Type{integer, integer})
    integerPairs := arrayType(integerPair)

    scalarZero := scalarValue(0.0)
    scalarOne := scalarValue(1.0)
    scalarHalf := scalarValue(0.5)

    integerMinusOne := integerValue(-1)
    integerZero := integerValue(0)
    integerOne := integerValue(1)

    logicalFalse := logicalValue(false)
//    logicalTrue := logicalValue(true)

    stringConstant := stringValue("constant")
    stringSymmetric := stringValue("symmetric")
    stringReplicate := stringValue("replicate")

    emptyArray := arrayValue(nil, false)
    integersOne := arrayValue([]core.Value{integerOne}, false)

    prototype := core.NewPrototype
    param := core.NewParam
    result := core.NewResult
    params := func(args ...*core.Param) []*core.Param { return args }
    results := func(args ...*core.Result) []*core.Result { return args }

    prototypes = []*core.Prototype{
        prototype(
            "external",
            params(
                param("shape", integers, nil)),
            results(
                result("output", genericTensor)), 
            scalar),
        prototype(
            "constant",
            params(
                param("shape", integers, nil),
                param("value", generics, nil)),
            results(
                result("output", genericTensor)), 
            scalar),
        prototype(
            "variable",
            params(
                param("shape", integers, nil),
                param("label", strng, nil)),
            results(
                result("output", genericTensor)), 
            scalar),
        prototype(
            "update",
            params(
                param("variable", genericTensor, nil),
                param("value", genericTensor, nil)),
            results(
                result("result", genericTensor)),
            nil),
        prototype(
            "reshape",
            params(
                param("input", genericTensor, nil),
                param("shape", integers, nil),
                param("axis_start", integer, integerZero),
                param("axis_count", integer, integerMinusOne)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "transpose",
            params(
                param("input", genericTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "concat",
            params(
                param("values", genericTensors, nil),
                param("axis", integer, nil)),
            results(
                result("value", genericTensor)),
            nil),
        prototype(
            "split", 
            params(
                param("value", genericTensor, nil),
                param("axis", integer, nil),
                param("ratios", integers, nil)),
            results(
                result("values", genericTensors)),
            nil),
        prototype(
            "slice",
            params(
                param("input", genericTensor, nil),
                param("axes", integers, nil),
                param("begin", integers, nil),
                param("end", integers, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "stack",
            params(
                param("values", genericTensors, nil),
                param("axis", integer, nil)),
            results(
                result("value", genericTensor)),
            nil),
        prototype(
            "unstack",
            params(
                param("value", genericTensor, nil),
                param("axis", integer, nil)),
            results(
                result("values", genericTensors)),
            nil),
        prototype(
            "squeeze",
            params(
                param("input", genericTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "unsqueeze",
            params(
                param("input", genericTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "pad",
            params(
                param("input", scalarTensor, nil),
                param("padding", integerPairs, nil),
                param("border", strng, stringConstant),
                param("value", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "tile",
            params(
                param("input", genericTensor, nil),
                param("repeats", integers, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "add",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "sub",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "mul",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "div",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "pow",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "min", 
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "max",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", scalarTensor)),
            nil),
        prototype(
            "lt",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),    
        prototype(
            "le",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype("gt",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype(
            "ge",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype(
            "eq", 
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),    
        prototype(
            "ne",
            params(
                param("x", scalarTensor, nil),
                param("y", scalarTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype(
            "and",
            params(
                param("x", logicalTensor, nil),
                param("y", logicalTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype(
            "or",
            params(
                param("x", logicalTensor, nil),
                param("y", logicalTensor, nil)),
            results(
                result("z", logicalTensor)),
            nil),
        prototype(
            "select",
            params(
                param("condition", logicalTensor, nil),
                param("true_value", genericTensor, nil),
                param("false_value", genericTensor, nil)),
            results(
                result("output", genericTensor)),
            nil),
        prototype(
            "clamp",
            params(
                param("x", scalarTensor, nil),
                param("a", scalarTensor, nil),
                param("b", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "copy",
            params(
                param("x", genericTensor, nil)),
            results(
                result("y", genericTensor)),
            nil),
        prototype(
            "neg",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "rcp",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "exp",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "log",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "sin",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "cos",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "abs",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "sign",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "floor",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "ceil",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "round",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "sqr",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "sqrt",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "rsqr",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "rsqrt",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "log2",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "not",
            params(
                param("x", logicalTensor, nil)),
            results(
                result("y", logicalTensor)),
            nil),
        prototype(
            "relu",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "sigmoid",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "tanh",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "elu",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "prelu",
            params(
                param("x", scalarTensor, nil),
                param("alpha", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "leaky_relu",
            params(
                param("x", scalarTensor, nil),
                param("alpha", scalar, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "softabs",
            params(
                param("x", scalarTensor, nil),
                param("epsilon", scalar, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "softplus",
            params(
                param("x", scalarTensor, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "softmax",
            params(
                param("x", scalarTensor, nil),
                param("axes", integers, integersOne)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "conv",
            params(
                param("input", scalarTensor, nil),
                param("filter", scalarTensor, nil),
                param("bias", scalarTensor, scalarZero),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("groups", integer, integerOne)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "deconv",
            params(
                param("input", scalarTensor, nil),
                param("filter", scalarTensor, nil),
                param("bias", scalarTensor, scalarZero),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("output_shape", integers, emptyArray),
                param("groups", integer, integerOne)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "box",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("normalize", logical, logicalFalse)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "debox",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("output_shape", integers, emptyArray),
                param("normalize", logical, logicalFalse)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "sample",
            params(
                param("input", scalarTensor, nil),
                param("index", integerTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "desample",
            params(
                param("input", scalarTensor, nil),
                param("index", integerTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("output_shape", integers, emptyArray)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "max_pool",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "argmax_pool",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("index", integerTensor)),
            nil),
        prototype(
            "max_pool_with_index",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("output", scalarTensor), 
                result("index", integerTensor)),
            nil),
        prototype(
            "avg_pool",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "rms_pool",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray)),
            results(
                result("output", scalarTensor)),
            nil),            
        prototype(
            "separable_conv",
            params(
                param("input", scalarTensor, nil),
                param("plane_filter", scalarTensor, nil),
                param("point_filter", scalarTensor, nil),
                param("bias", scalarTensor, scalarZero),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("groups", integer, integerOne)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "separable_deconv",
            params(
                param("input", scalarTensor, nil),
                param("plane_filter", scalarTensor, nil),
                param("point_filter", scalarTensor, nil),
                param("bias", scalarTensor, scalarZero),
                param("border", strng, stringConstant),
                param("padding", integerPairs, emptyArray),
                param("stride", integers, emptyArray),
                param("dilation", integers, emptyArray),
                param("output_shape", integers, emptyArray),
                param("groups", integer, integerOne)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "nearest_downsample",
            params(
                param("input", scalarTensor, nil),
                param("factor", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "nearest_upsample",
            params(
                param("input", scalarTensor, nil),
                param("factor", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "area_downsample",
            params(
                param("input", scalarTensor, nil),
                param("factor", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "multilinear_upsample",
            params(
                param("input", scalarTensor, nil),
                param("factor", integers, nil),
                param("method", strng, stringSymmetric),
                param("border", strng, stringReplicate)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "local_response_normalization",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("alpha", scalar, scalarOne),
                param("beta", scalar, scalarHalf),
                param("bias", scalar, scalarOne)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "local_mean_normalization",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "local_variance_normalization",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("bias", scalar, scalarZero),
                param("epsilon", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "local_contrast_normalization",
            params(
                param("input", scalarTensor, nil),
                param("size", integers, nil),
                param("bias", scalar, scalarZero),
                param("epsilon", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "l1_normalization",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil),
                param("bias", scalar, scalarZero),
                param("epsilon", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "l2_normalization",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil),
                param("bias", scalar, scalarZero),
                param("epsilon", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "batch_normalization",
            params(
                param("input", scalarTensor, nil),
                param("mean", scalarTensor, nil),
                param("variance", scalarTensor, nil),
                param("offset", scalarTensor, scalarZero),
                param("scale", scalarTensor, scalarOne),
                param("epsilon", scalar, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "sum_reduce",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil),
                param("normalize", logical, logicalFalse)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "min_reduce",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "max_reduce",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "mean_reduce",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
           "argmax_reduce",
           params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", integerTensor)),
            nil),
        prototype(
            "argmin_reduce",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", integerTensor)),
            nil),
        prototype(
            "any_reduce",
            params(
                param("input", logicalTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", logicalTensor)),
            nil),
        prototype(
            "all_reduce",
            params(
                param("input", logicalTensor, nil),
                param("axes", integers, nil)),
            results(
                result("output", logicalTensor)),
            nil),
        prototype(
            "moments",
            params(
                param("input", scalarTensor, nil),
                param("axes", integers, nil)),
            results(
                result("mean", scalarTensor), 
                result("variance", scalarTensor)),
            nil),
        prototype(
            "max_roi_pool",
            params(
                param("input", scalarTensor, nil),
                param("rois", scalarTensor, nil),
                param("batch_index", integerTensor, nil),
                param("output_size", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "avg_roi_pool",
            params(
                param("input", scalarTensor, nil),
                param("rois", scalarTensor, nil),
                param("batch_index", integerTensor, nil),
                param("output_size", integers, nil)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "roi_resample",
            params(
                param("input", scalarTensor, nil),
                param("rois", scalarTensor, nil),
                param("batch_index", integerTensor, nil),
                param("output_size", integers, nil),
                param("method", strng, stringSymmetric)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "max_roi_align",
            params(
                param("input", scalarTensor, nil),
                param("rois", scalarTensor, nil),
                param("batch_index", integerTensor, nil),
                param("output_size", integers, nil),
                param("sampling_rate", integers, nil),
                param("resize_method", strng, stringSymmetric)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "avg_roi_align",
            params(
                param("input", scalarTensor, nil),
                param("rois", scalarTensor, nil),
                param("batch_index", integerTensor, nil),
                param("output_size", integers, nil),
                param("sampling_rate", integers, nil),
                param("resize_method", strng, stringSymmetric)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "matmul",
            params(
                param("A", scalarTensor, nil),
                param("B", scalarTensor, nil),
                param("transposeA", logical, logicalFalse),
                param("transposeB", logical, logicalFalse)),
            results(
                result("C", scalarTensor)),
            nil),
        prototype(
            "linear",
            params(
                param("input", scalarTensor, nil),
                param("filter", scalarTensor, nil),
                param("bias", scalarTensor, scalarZero)),
            results(
                result("output", scalarTensor)),
            nil),
        prototype(
            "add_n",
            params(
                param("x", tensors, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "copy_n",
            params(
                param("x", genericTensor, nil),
                param("times", integer, nil)),
            results(
                result("y", genericTensors)),
            nil),
        prototype(
            "linear_quantize",
            params(
                param("x", scalarTensor, nil),
                param("min", scalarTensor, nil),
                param("max", scalarTensor, nil),
                param("bits", integer, nil)),
            results(
                result("y", scalarTensor)),
            nil),
        prototype(
            "logarithmic_quantize",
            params(
                param("x", scalarTensor, nil),
                param("max", scalarTensor, nil),
                param("bits", integer, nil)),
            results(
                result("y", scalarTensor)),
            nil),
    }
}

func StdlibPrototypes() []*core.Prototype {
    return prototypes
}

