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

package engine

import (
    "fmt"
    "io"
    "os"
    "strings"
    "fragata/arhat/nnef/core"
    dnn "fragata/arhat/nnef/dnn/api"
    "fragata/arhat/nnef/parser/comp"
    "fragata/arhat/nnef/runtime"
)

//
//    ShapeFunc
//
    
//
// Shape propagation function type
//
type ShapeFunc func(op *core.Operation, graph *core.Graph)

//
//    Engine
//

type Engine struct {
    dnn dnn.Engine
    contexts map[*core.Graph]*runtime.Context
}

func NewEngine(dnnEngine dnn.Engine) *Engine {
    e := new(Engine)
    e.dnn = dnnEngine
    e.contexts = make(map[*core.Graph]*runtime.Context)
    return e
}

//
// Parse the NNEF graph from file
//
// graphFn: name of the graph file
// quantFn: name of the quantization file
// graph: the graph data structure to fill in
// stdlib: the implementation of standard operations to use
// lowered: a list of operations to be lowered
//
// return error value or nil
//
func(e *Engine) ParseFile(
        graphFn string,
        quantFn string,
        graph *core.Graph,
        stdlib string,
        lowered map[string]bool) error {
    graphIs, err := os.Open(graphFn)
    if err != nil {
        return fmt.Errorf("Could not open graph file: %s", graphFn)
    }
    defer graphIs.Close()
    var quantIs *os.File
    if quantFn != "" {
        quantIs, err = os.Open(quantFn)
        if err != nil {
            return fmt.Errorf("Could not open quantization file: %s", quantFn)
        }
        defer quantIs.Close()
    }
    return parse(graphIs, graphFn, quantIs, quantFn, graph, stdlib, lowered)
}

//
// Parse the NNEF graph from string
//
// graphStr: the graph string
// quantStr: the quantization string
// graph: the graph data structure to fill in
// stdlib: the implementation of standard operations to use
// lowered: a list of operations to be lowered
//
// return error value or nil
//
func(e *Engine) ParseString(
        graphStr string, 
        quantStr string,
        graph *core.Graph, 
        stdlib string, 
        lowered map[string]bool) error {
    graphIs := strings.NewReader(graphStr)
    var quantIs *strings.Reader
    if quantStr != "" {
        quantIs = strings.NewReader(quantStr)
    }
    return parse(graphIs, "input", quantIs, "quantization", graph, stdlib, lowered)
}

func parse(
        graphIs io.Reader,
        graphFn string,
        quantIs io.Reader,
        quantFn string,
        graph *core.Graph,
        stdlib string, 
        lowered map[string]bool) (err error) {
    callback := NewParseCallback(graph, quantIs, quantFn)
    parser := comp.NewCompParser(stdlib, lowered)
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(*core.Error); ok {
                err = makeParseError(v)
            } else {
                panic(r)
            }
        }
    }()
    parser.Parse(graphIs, graphFn, callback)
    return
}

func makeParseError(e *core.Error) error {
    message := "Parse error in file " + formatErrorPosition(e.Position()) + " " + e.What()
    origin := e.Position().Origin
    for origin != nil {
        message += "\n... evaluated from file " + formatErrorPosition(e.Position())
        origin = origin.Origin
    }
    return fmt.Errorf("%s", message)
}

func formatErrorPosition(pos *core.Position) string {
    return "'" + pos.Filename + 
        "' [" + fmt.Sprintf("%d", pos.Line) + ":" + fmt.Sprintf("%d", pos.Column) + "]"
}

//
// Read a single tensor from binary stream
//
// is: the stream to read from
// tensor: the tensor object to fill into
//
// return error value or nil
//
func(e *Engine) ReadTensor(is io.Reader, tensor *core.Tensor) error {
    var err error
    var header core.TensorHeader
    err = readTensorHeader(is, &header)
    if err != nil {
        return err
    }
    rank := int(header.Rank)
    shape := make(core.Shape, rank)
    for i := 0; i < rank; i++ {
        shape[i] = int(header.Extents[i])
    }
    tensor.SetShape(shape)
    bitsPerItem := int(header.BitsPerItem)
    switch core.QuantCode(header.QuantCode) {
    case core.QuantCodeFloat:
        tensor.SetDtype("scalar")
        tensor.ResizeData()
        core.ReadScalarData(is, bitsPerItem, tensor.ScalarData())
    case core.QuantCodeInteger:
        if header.BitsPerItem == 1 {
            tensor.SetDtype("logical")
            tensor.ResizeData()
            core.ReadLogicalData(is, bitsPerItem, tensor.LogicalData())
        } else {
            core.Assert(header.QuantParams[0] == 1)
            tensor.SetDtype("integer")
            tensor.ResizeData()
            core.ReadIntegerData(is, bitsPerItem, tensor.IntegerData())
        }
    default:
        return fmt.Errorf(
            "Unsupported tensor item type code '%d' and bits per item '%d'", 
                header.QuantCode, header.BitsPerItem)
    }
    return err
}

func readTensorHeader(is io.Reader, header *core.TensorHeader) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(error); ok {
                err = v
            } else {
                panic(r)
            }
        }
    }()
    header.Read(is)
    header.Validate()
    return
}

//
// Write a single tensor to binary stream
//
// os: the stream to write to
// tensor: the tensor object to fill from
//
// return error value or nil
//
func(e *Engine) WriteTensor(os io.Writer, tensor *core.Tensor) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(error); ok {
                err = v
            } else {
                panic(r)
            }
        }
    }()
    dtype := tensor.Dtype()
    shape := tensor.Shape()
    rank := len(shape)
    if rank > core.MaxRank {
        return fmt.Errorf("Tensor rank %d exceeds maximum allowed rank (%d)", rank, core.MaxRank)
    }
    var quantCode core.QuantCode
    if dtype == "scalar" {
        quantCode = core.QuantCodeFloat
    } else {
        quantCode = core.QuantCodeInteger
    }   
    var header core.TensorHeader
    version := [2]int{1, 0}
    header.Fill(version, shape, itemBits(dtype), quantCode)
    header.Write(os)
    switch dtype {
    case "scalar":
        core.WriteScalarData(os, tensor.ScalarData())
    case "integer":
        core.WriteIntegerData(os, tensor.IntegerData())
    case "logical":
        core.WriteLogicalData(os, tensor.LogicalData())
    default:
        return fmt.Errorf("Invalid tensor data type: '%s'", dtype)
    }
    return
}

func itemBits(dtype string) int {
    switch dtype {
    case "scalar":
        return 32
    case "integer":
        return 32
    case "logical":
        return 1
    default:
        return 0
    }
}

func(e *Engine) ReadTensorFile(filename string, tensor *core.Tensor) error {
    fp, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer fp.Close()
    return e.ReadTensor(fp, tensor)
}

func(e *Engine) WriteTensorFile(filename string, tensor *core.Tensor) error {
    fp, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer fp.Close()
    return e.WriteTensor(fp, tensor)
}

//
// Load variables from set of files in a folder
//
// path: the path to the top level NNEF model folder
// graph: the graph object to load tensors into
//
// return error value or nil
//
func(e *Engine) LoadVariables(path string, graph *core.Graph) error {
    sep := ""
    if !strings.HasSuffix(path, "/") && !strings.HasSuffix(path, "\\") {
        sep = "/"
    }
    count := graph.OperationCount()
    for i := 0; i < count; i++ {
        op := graph.OperationAt(i)
        if op.Name() != "variable" {
            continue
        }
        label := op.GetAttrib("label").String()
        shape := valueToShape(op.GetAttrib("shape"))
        id := op.OutputAt(0).Identifier()
        tensor := graph.GetTensor(id)
        filename := path + sep + label + ".dat"
        err := e.ReadTensorFile(filename, tensor)
        if err != nil {
            return err
        }
        if tensor.Dtype() != op.Dtype() {
            return fmt.Errorf(
                "item type %s in variable file '%s' does not match "+
                "data type %s defined in network structure",
                    tensor.Dtype(), filename, op.Dtype())
        }
        tensorShape := core.Shape(tensor.Shape())
        if !tensorShape.Eq(shape) {
            return fmt.Errorf(
                "shape %s in variable file '%s' does not match shape %s "+
                "defined in network structure",
                    tensorShape.String(), filename, shape.String())
        }
    }
    return nil
}

func valueToShape(value core.Value) core.Shape {
    size := value.Size()
    shape := make(core.Shape, size)
    for i := 0; i < size; i++ {
        shape[i] = value.At(i).Integer()
    }
    return shape
}

//
// Load whole model from set of files in a folder
//
// path: the path to the top level NNEF model folder
// graph: the graph object to load tensors into
// stdlib: the implementation of standard operations to use
// lowered: a list of operations to be lowered
//
// return error value or nil
//
func(e *Engine) LoadGraph(
        path string, 
        graph *core.Graph, 
        stdlib string, 
        lowered map[string]bool) error {
    sep := ""
    if !strings.HasSuffix(path, "/") && !strings.HasSuffix(path, "\\") {
        sep = "/"
    }
    graphFn := path + sep + "graph.nnef"
    if !fileExists(graphFn) {
        return e.ParseFile(path, "", graph, stdlib, lowered)
    }
    quantFn := path + sep + "graph.quant"
    if !fileExists(quantFn) {
        quantFn = ""
    }
    err := e.ParseFile(graphFn, quantFn, graph, stdlib, lowered)
    if err != nil {
        return err
    }
    err = e.LoadVariables(path, graph)
    if err != nil {
        return err
    }
    return nil
}

func fileExists(path string) bool {
    info, err := os.Stat(path)
    return (err == nil && info.Mode().IsRegular())
}

var StandardShapeFuncs = map[string]ShapeFunc{
    "external": NullaryShapeFunc,
    "constant": ConstantShapeFunc,
    "variable": NullaryShapeFunc,
        
    "copy": UnaryShapeFunc,
    "neg": UnaryShapeFunc,
    "not": UnaryShapeFunc,
    "rcp": UnaryShapeFunc,
    "exp": UnaryShapeFunc,
    "log": UnaryShapeFunc,
    "sin": UnaryShapeFunc,
    "cos": UnaryShapeFunc,
    "abs": UnaryShapeFunc,
    "sign": UnaryShapeFunc,
    "floor": UnaryShapeFunc,
    "ceil": UnaryShapeFunc,
    "round": UnaryShapeFunc,
    "sqr": UnaryShapeFunc,
    "sqrt": UnaryShapeFunc,
    "rsqr": UnaryShapeFunc,
    "rsqrt": UnaryShapeFunc,
    "log2": UnaryShapeFunc,
        
    "relu": UnaryShapeFunc,
    "sigmoid": UnaryShapeFunc,
    "tanh": UnaryShapeFunc,
    "elu": UnaryShapeFunc,
    "softabs": UnaryShapeFunc,
    "softplus": UnaryShapeFunc,
    "leaky_relu": UnaryShapeFunc,
    "prelu": AsymmetricBinaryShapeFunc,
        
    "linear_quantize": LinearQuantizeShapeFunc,
    "logarithmic_quantize": LogarithmicQuantizeShapeFunc,
        
    "add": BinaryShapeFunc,
    "sub": BinaryShapeFunc,
    "mul": BinaryShapeFunc,
    "div": BinaryShapeFunc,
    "min": BinaryShapeFunc,
    "max": BinaryShapeFunc,
    "pow": BinaryShapeFunc,
    "lt": BinaryShapeFunc,
    "le": BinaryShapeFunc,
    "gt": BinaryShapeFunc,
    "ge": BinaryShapeFunc,
    "eq": BinaryShapeFunc,
    "ne": BinaryShapeFunc,
    "and": BinaryShapeFunc,
    "or": BinaryShapeFunc,
        
    "conv": ConvShapeFunc,
    "deconv": DeconvShapeFunc,
    "separable_conv": SeparableConvShapeFunc,
    "separable_deconv": SeparableDeconvShapeFunc,
        
    "box": PoolShapeFunc,
    "max_pool": PoolShapeFunc,
    "argmax_pool": PoolShapeFunc,
    "max_pool_with_index": PoolShapeFunc,
    "avg_pool": PoolShapeFunc,
    "rms_pool": PoolShapeFunc,
    "debox": UnpoolShapeFunc,
    "sample": SampleShapeFunc,
    "desample": DesampleShapeFunc,
        
    "sum_reduce": ReduceShapeFunc,
    "min_reduce": ReduceShapeFunc,
    "max_reduce": ReduceShapeFunc,
    "mean_reduce": ReduceShapeFunc,
    "argmax_reduce": ReduceShapeFunc,
    "argmin_reduce": ReduceShapeFunc,
    "any_reduce": ReduceShapeFunc,
    "all_reduce": ReduceShapeFunc,
    "moments": ReduceShapeFunc,
        
    "nearest_downsample": DownsampleShapeFunc,
    "area_downsample": DownsampleShapeFunc,
    "nearest_upsample": UpsampleShapeFunc,
    "multilinear_upsample": UpsampleShapeFunc,
        
    "local_response_normalization": NormalizeShapeSizeFunc,
    "local_mean_normalization": NormalizeShapeSizeFunc,
    "local_variance_normalization": NormalizeShapeSizeFunc,
    "local_contrast_normalization": NormalizeShapeSizeFunc,
    "l1_normalization": NormalizeShapeAxesFunc,
    "l2_normalization": NormalizeShapeAxesFunc,
    "batch_normalization": BatchnormShapeFunc,
        
    "avg_roi_pool": RoiShapeFunc,
    "max_roi_pool": RoiShapeFunc,
    "avg_roi_align": RoiShapeFunc,
    "max_roi_align": RoiShapeFunc,
    "roi_resample": RoiShapeResampleFunc,
        
    "reshape": ReshapeShapeFunc,
    "transpose": TransposeShapeFunc,
    "split": SplitShapeFunc,
    "concat": ConcatShapeFunc,
    "slice": SliceShapeFunc,
    "stack": StackShapeFunc,
    "unstack": UnstackShapeFunc,
    "squeeze": SqueezeShapeFunc,
    "unsqueeze": UnsqueezeShapeFunc,
    "tile": TileShapeFunc,
    "pad": PadShapeFunc,
    "matmul": MatmulShapeFunc,
    "linear": LinearShapeFunc,
    "update": UpdateShapeFunc,
    "softmax": SoftmaxShapeFunc,
    "copy_n": CopyNShapeFunc,
    "add_n": AddNShapeFunc,
    "select": TernaryShapeFunc,
    "clamp": TernaryShapeFunc,
}

//
// Perform shape inference on the graph
//
// graph: the graph object
// inputShapes: shapes of external tensors
// customShapes: shape inference functions for custom operations
//
// return error value or nil
//
func(e *Engine) InferShapes(
        graph *core.Graph, 
        inputShapes map[string]core.Shape, 
        customShapes map[string]ShapeFunc) error {
    count := graph.OperationCount()
    for i := 0; i < count; i++ {
        op := graph.OperationAt(i)
        name := op.Name()
        fn, ok := StandardShapeFuncs[name]
        if !ok {
            fn, ok = customShapes[name]
            if !ok {
                return fmt.Errorf("Shape function for operation '%s' is not provided", name)
            }
        }
        if name == "external" {
            id := op.GetOutput("output").Identifier()
            shape, ok := inputShapes[id]
            if ok {
                original := op.GetAttrib("shape")
                if len(shape) != original.Size() {
                    return fmt.Errorf(
                        "Overridden external shape rank (%d) does not match original rank (%d)",
                            len(shape), original.Size())
                }
            graph.GetTensor(id).SetShape(shape.Clone())
            }
        }
        err := callShapeFunc(fn, op, graph)
        if err != nil {
            return err
        }
    }
    return nil
}

func callShapeFunc(fn ShapeFunc, op *core.Operation, graph *core.Graph) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(error); ok {
                err = makeInferShapeError(op, v.Error())
            } else {
                panic(r)
            }
        }
    }()
    fn(op, graph)
    return
}

func makeInferShapeError(op *core.Operation, what string) error {
    output := op.OutputAt(0)
    var id string
    if output.Kind() == core.ValueKindIdentifier {
        id = output.Identifier()
    } else {
        id = output.At(0).Identifier()
    }
    return fmt.Errorf(
        "Shape error while inferring shape of tensor '%s' (operation '%s'): %s",
            id, op.Name(), what)
}

//
// Execute a graph
//
// graph: the graph object
//
// return error value or nil
//
func(e *Engine) Execute(graph *core.Graph) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(error); ok {
                err = v
            } else {
                panic(r)
            }
        }
    }()
    ctx, ok := e.contexts[graph]
    if !ok {
        ctx = runtime.NewContext(graph, e.dnn)
        e.contexts[graph] = ctx
        createTensors(graph, ctx)
        writeVariables(graph, ctx)
    }
    writeInputs(graph, ctx)
    opCount := graph.OperationCount()
    for i := 0; i < opCount; i++ {
        op := graph.OperationAt(i)
        name := op.Name()
        fn := runtime.FindExecutor(name)
        if fn == nil {
            return fmt.Errorf("operation not implemented: %s", name)
        }
        fn(ctx, op)
        // Enable for diagnostics
        // checkFinite(graph, ctx, i, op)
    }
    readOutputs(graph, ctx)
    return
}

func createTensors(graph *core.Graph, ctx *runtime.Context) {
    count := graph.TensorCount()
    for i := 0; i < count; i++ {
        ctx.CreateTensor(graph.TensorAt(i))
    }
}

func writeVariables(graph *core.Graph, ctx *runtime.Context) {
    count := graph.OperationCount()
    for i := 0; i < count; i++ {
        op :=  graph.OperationAt(i)
        if op.Name() == "variable" {
            core.Assert(op.OutputCount() == 1)
            name := op.OutputAt(0).Identifier()
            tensor := graph.GetTensor(name)
            checkData(tensor)
            ctx.WriteTensor(tensor)
        }
    }
}

func writeInputs(graph *core.Graph, ctx *runtime.Context) {
    count := graph.InputCount()
    for i := 0; i < count; i++ {
        name := graph.InputAt(i)
        tensor := graph.GetTensor(name)
        checkData(tensor)
        ctx.WriteTensor(tensor)
    }
}

func readOutputs(graph *core.Graph, ctx *runtime.Context) {
    count := graph.OutputCount()
    for i := 0; i < count; i++ {
        name := graph.OutputAt(i)
        tensor := graph.GetTensor(name)
        tensor.ResizeData()
        ctx.ReadTensor(tensor)
    }
}

func checkData(tensor *core.Tensor) {
    if tensor.Data() == nil {
        core.RuntimeError("missing data for tensor '%s'", tensor.Name())
    }
}

//
// Release runtime context of a graph
//
// graph: the graph object
//
// return error value or nil
//
func(e *Engine) Release(graph *core.Graph) error {
    delete(e.contexts, graph)
    return nil
}

