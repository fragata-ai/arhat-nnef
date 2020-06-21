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

package runtime

import (
    "fragata/arhat/nnef/core"
    dnn "fragata/arhat/nnef/dnn/api"
)

//
//    Context
//

type Context struct {
    graph *core.Graph
    dnn dnn.Engine
    tensorMap map[*core.Tensor]dnn.Tensor
}

// construction/destruction

func NewContext(graph *core.Graph, dnnEngine dnn.Engine) *Context {
    c := new(Context)
    c.graph = graph
    c.dnn = dnnEngine
    c.tensorMap = make(map[*core.Tensor]dnn.Tensor)
    return c
}

// interface

func(c *Context) CreateTensor(tensor *core.Tensor) {
    view, ok := c.tensorMap[tensor]
    if ok {
        core.RuntimeError("Tensor already exists: '%s'", tensor.Name())
    }
    var t dnn.Dtype
    dtype := tensor.Dtype()
    switch dtype {
    case "scalar":
        t = dnn.DtypeFloat
    case "integer":
        t = dnn.DtypeInt
    case "logical":
        t = dnn.DtypeBool
    default:
         core.RuntimeError("data type not supported: %s", dtype)
    }
    view, err := c.dnn.NewTensor(t, tensor.Shape())
    if err != nil {
        signalError(err)
    }
    c.tensorMap[tensor] = view    
}

func(c *Context) WriteTensor(tensor *core.Tensor) {
    view := c.MapTensor(tensor)
    err := c.dnn.Fill(view, tensor.Data())
    if err != nil {
        signalError(err)
    }
}

func(c *Context) ReadTensor(tensor *core.Tensor) {
    view := c.MapTensor(tensor)
    err := c.dnn.Read(view, tensor.Data())
    if err != nil {
        signalError(err)
    }
}

func(c *Context) MapTensor(tensor *core.Tensor) dnn.Tensor {
    view, ok := c.tensorMap[tensor]
    if !ok {
        core.RuntimeError("Invalid tensor '%s'", tensor.Name())
    }
    return view
}

// implementation

func(c *Context) getTensor(name string) dnn.Tensor {
    tensor := c.graph.GetTensor(name)
    core.Assert(tensor != nil)
    return c.MapTensor(tensor)
}

//
//    Executor
//

type Executor func(ctx *Context, op *core.Operation)

func FindExecutor(name string) Executor {
    fn, ok := executorMap[name]
    if !ok {
        return nil
    }
    return fn
}

//
//    Executor map
//

var executorMap = map[string]Executor {
    "external": executeExternal,
    "constant": executeConstant,
    "variable": executeVariable,
        
    "neg": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpNeg),
    "not": makeUnaryExecutor(dnn.DtypeBool, dnn.OpNot),
    "abs": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpAbs),
    "sign": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSign),
    "exp": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpExp),
    "log": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpLog),
    "log2": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpLog2),
    "sin": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSin),
    "cos": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpCos),
    "round": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpRound),
    "floor": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpFloor),
    "ceil": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpCeil),
    "sqrt": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSqrt),
    "sqr": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSqr),
    "rsqrt": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpRsqrt),
    "rsqr": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpRsqr),
    "rcp": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpRcp),
    "copy": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpCopy),

    "sigmoid": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSigmoid),
    "tanh": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpTanh),
    "relu": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpRelu),
    "elu": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpElu),
    "softplus": makeUnaryExecutor(dnn.DtypeFloat, dnn.OpSoftplus),

    "add": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpAdd),
    "sub": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpSub),
    "mul": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpMul),
    "div": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpDiv),
    "pow": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpPow),
    "min": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpMin),
    "max": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeFloat, dnn.OpMax),
    "and": makeBinaryExecutor(dnn.DtypeBool, dnn.DtypeBool, dnn.OpAnd),
    "or": makeBinaryExecutor(dnn.DtypeBool, dnn.DtypeBool, dnn.OpOr),
    "lt": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpLt),
    "gt": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpGt),
    "le": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpLe),
    "ge": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpGe),
    "eq": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpEq),
    "ne": makeBinaryExecutor(dnn.DtypeFloat, dnn.DtypeBool, dnn.OpNe),

    "select": executeSelect,

    "sum_reduce": makeReduceExecutor(dnn.DtypeFloat, dnn.OpSumReduce),
    "mean_reduce": makeReduceExecutor(dnn.DtypeFloat, dnn.OpMeanReduce),
    "min_reduce": makeReduceExecutor(dnn.DtypeFloat, dnn.OpMinReduce),
    "max_reduce": makeReduceExecutor(dnn.DtypeFloat, dnn.OpMaxReduce),
    "any_reduce": makeReduceExecutor(dnn.DtypeBool, dnn.OpAnyReduce),
    "all_reduce": makeReduceExecutor(dnn.DtypeBool, dnn.OpAllReduce),

    "conv": makeConvExecutor(false, dnn.DtypeFloat),
    "deconv": makeConvExecutor(true, dnn.DtypeFloat),

    "box": makePoolExecutor(false, dnn.DtypeFloat),
    "debox": makePoolExecutor(true, dnn.DtypeFloat),
    "avg_pool": makePoolExecutor(false, dnn.DtypeFloat),
    "max_pool": makePoolExecutor(false, dnn.DtypeFloat),

    "reshape": executeReshape,
    "squeeze": executeReshape,
    "unsqueeze": executeReshape,
    "transpose": executeTranspose,

    "concat": executeConcat,
    "split": executeSplit,
    "stack": executeConcat,
    "unstack": executeSplit,
    "pad": makePadExecutor(dnn.DtypeFloat),
    "tile": executeTile,
    "slice": executeSlice,

    "matmul": makeMatmulExecutor(dnn.DtypeFloat),
    "linear": makeLinearExecutor(dnn.DtypeFloat),
        
    "softmax": makeSoftmaxExecutor(dnn.DtypeFloat),
    "argmin_reduce": makeArgReduceExecutor(dnn.DtypeFloat, dnn.DtypeInt, dnn.OpArgminReduce),
    "argmax_reduce": makeArgReduceExecutor(dnn.DtypeFloat, dnn.DtypeInt, dnn.OpArgmaxReduce),

    "multilinear_upsample": makeMultilinearUpsampleExecutor(dnn.DtypeFloat),
        
    "update": executeUpdate,
}

//
//    Standard executors
//

func executeExternal(ctx *Context, op *core.Operation) {
    // nothing to do
}

func executeVariable(ctx *Context, op *core.Operation) {
    // nothing to do
}

func executeConstant(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    output := op.GetOutput("output")
    value := op.GetAttrib("value")
    view := mapTensor(ctx, t, output)
    n := view.Volume()
    var data interface{}
    switch t {
    case dnn.DtypeFloat:
        if value.Kind() == core.ValueKindArray {
            size := value.Size()
            if size == n {
                v := make([]float32, n)
                for i := 0; i < n; i++ {
                    v[i] = value.At(i).Scalar()
                }
                data = v
            } else {
                data = value.At(0).Scalar()
            }
        } else {
            data = value.Scalar()
        }
    case dnn.DtypeInt:
        if value.Kind() == core.ValueKindArray {
            size := value.Size()
            if size == n {
                v := make([]int, n)
                for i := 0; i < n; i++ {
                    v[i] = value.At(i).Integer()
                }
                data = v
            } else {
                data = value.At(0).Integer()
            }
        } else {
            data = value.Integer()
        }
    case dnn.DtypeBool:
        if value.Kind() == core.ValueKindArray {
            size := value.Size()
            if size == n {
                v := make([]bool, n)
                for i := 0; i < n; i++ {
                    v[i] = value.At(i).Logical()
                }
                data = v
            } else {
                data = value.At(0).Logical()
            }
        } else {
            data = value.Logical()
        }
    default:
        core.Assert(false)
    }
    err := ctx.dnn.Fill(view, data)
    if err != nil {
        signalError(err)
    }
} 

func makeUnaryExecutor(t dnn.Dtype, f dnn.UnaryOp) Executor {
    return func(ctx *Context, op *core.Operation) {
        x := op.GetInput("x")
        y := op.GetOutput("y")
        xView := mapTensor(ctx, t, x)
        yView := mapTensor(ctx, t, y)
        err := ctx.dnn.Unary(f, xView, yView)
        if err != nil {
            signalError(err)
        }
    }
}

func makeBinaryExecutor(t dnn.Dtype, r dnn.Dtype, f dnn.BinaryOp) Executor {
    return func(ctx *Context, op *core.Operation) {
        x := op.GetInput("x")
        y := op.GetInput("y")
        z := op.GetOutput("z")
        xView := mapTensor(ctx, t, x)
        yView := mapTensor(ctx, t, y)
        zView := mapTensor(ctx, t, z)
        err := ctx.dnn.Binary(f, xView, yView, zView)
        if err != nil {
            signalError(err)
        }
    }
}

func makeReduceExecutor(t dnn.Dtype, f dnn.ReduceOp) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("input")
        output := op.GetOutput("output")
        inputView := mapTensor(ctx, t, input)
        outputView := mapTensor(ctx, t, output)
        err := ctx.dnn.Reduce(f, inputView, outputView)
        if err != nil {
            signalError(err)
        }
    }
}

func executeSelect(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    c := op.GetInput("condition")
    x := op.GetInput("true_value")
    y := op.GetInput("false_value")
    z := op.GetOutput("output")
    cView := mapTensor(ctx, t, c)
    xView := mapTensor(ctx, t, x)
    yView := mapTensor(ctx, t, y)
    zView := mapTensor(ctx, t, z)
    err := ctx.dnn.Select(cView, xView, yView, zView)
    if err != nil {
        signalError(err)
    }
}

func makeConvExecutor(transposed bool, t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("input")
        filter := op.GetInput("filter")
        bias := op.GetInput("bias")
        output := op.GetOutput("output")
        padding := op.GetAttrib("padding")
        stride := op.GetAttrib("stride")
        dilation := op.GetAttrib("dilation")
        groups := op.GetAttrib("groups").Integer()
        border := op.GetAttrib("border").String()
        if border != "constant" {
            core.RuntimeError(
                "operation not implemented: %s with border = '%s'", 
                    op.Name(), border)
        }
        var inputView, outputView dnn.Tensor
        if transposed {
            inputView = mapTensor(ctx, t, output)
            outputView = mapTensor(ctx, t, input)
        } else {
            inputView = mapTensor(ctx, t, input)
            outputView = mapTensor(ctx, t, output)
        }
        filterView := mapTensor(ctx, t, filter)
        biasView := mapTensor(ctx, t, bias)
        d := inputView.Rank() - 2
        checkSupportedRank(op.Name(), d, 3)
        var strideShape core.Shape
        if stride.Size() != 0 {
            strideShape = extractItems(stride)
        } else {
            strideShape = makeSingletonShape(d)
        }
        var dilationShape core.Shape
        if dilation.Size() != 0 {
            dilationShape = extractItems(dilation)
        } else {
            dilationShape = makeSingletonShape(d)
        }
        var paddingShape core.Shape
        if padding.Size() != 0 {
            paddingShape = extractItems(padding)
        } else {
            paddingShape = 
                makePadding(
                    d,
                    inputView.Shape()[2:],
                    outputView.Shape()[2:],
                    filterView.Shape()[2:],
                    strideShape,
                    dilationShape)
        }
        if groups == 1 {
            err := 
                ctx.dnn.Conv(
                    transposed, 
                    inputView, 
                    filterView, 
                    biasView, 
                    outputView,
                    paddingShape, 
                    strideShape, 
                    dilationShape)
            if err != nil {
                signalError(err)
            }
        } else if groups == 0 || groups == inputView.Shape()[1] {
            err :=
                ctx.dnn.DepthwiseConv(
                    transposed,
                    inputView, 
                    filterView, 
                    biasView, 
                    outputView,
                    paddingShape, 
                    strideShape, 
                    dilationShape)
            if err != nil {
                signalError(err)
            }
        } else {
            err :=
                ctx.dnn.GroupedConv(
                    transposed,
                    inputView, 
                    filterView, 
                    biasView, 
                    outputView,
                    paddingShape, 
                    strideShape, 
                    dilationShape, 
                    groups)
            if err != nil {
                signalError(err)
            }
        }
    }
}

func makePoolExecutor(transposed bool, t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        var f dnn.PoolOp
        switch op.Name() {
        case "box", "debox":
            core.Assert(transposed == (op.Name() == "debox"))
            normalize := op.GetAttrib("normalize").Logical()
            if normalize {
                f = dnn.OpAvgPool
            } else {
                f = dnn.OpSumPool
            }
        case "avg_pool":
            f = dnn.OpAvgPool
        case "max_pool":
            f = dnn.OpMaxPool
        default:
            core.Assert(false)
        }
        input := op.GetInput("input")
        output := op.GetOutput("output")
        size := op.GetAttrib("size")
        padding := op.GetAttrib("padding")
        stride := op.GetAttrib("stride")
        dilation := op.GetAttrib("dilation")
        border := op.GetAttrib("border").String()
        if border != "constant" && border != "ignore" {
            core.RuntimeError(
                "operation not implemented: %s with border = '%s'", 
                    op.Name(), border)
        }
        includeBorder := (border != "ignore")
        var inputView, outputView dnn.Tensor
        if transposed {
            inputView = mapTensor(ctx, t, output)
            outputView = mapTensor(ctx, t, input)
        } else {
            inputView = mapTensor(ctx, t, input)
            outputView = mapTensor(ctx, t, output)
        }
        d := inputView.Rank()
        checkSupportedRank(op.Name(), d, 5)
        sizeShape := extractItems(size)
        var strideShape core.Shape
        if stride.Size() != 0 {
            strideShape = extractItems(stride)
        } else {
            strideShape = makeSingletonShape(d)
        }
        var dilationShape core.Shape
        if dilation.Size() != 0 {
            dilationShape = extractItems(dilation)
        } else {
            dilationShape = makeSingletonShape(d)
        }
        var paddingShape core.Shape
        if padding.Size() != 0 {
            paddingShape = extractItems(padding)
        } else {
            paddingShape =
                makePadding(
                    d, 
                    inputView.Shape(), 
                    outputView.Shape(),
                    sizeShape, 
                    strideShape,
                    dilationShape)
        }
        err :=
            ctx.dnn.Pool(
                f, 
                transposed,
                inputView, 
                outputView, 
                sizeShape, 
                paddingShape, 
                strideShape, 
                dilationShape,
                includeBorder)
        if err != nil {
            signalError(err)
        }
    }
}

func executeReshape(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    input := op.GetInput("input")
    output := op.GetOutput("output")
    inputView := mapTensor(ctx, t, input)
    outputView := mapTensor(ctx, t, output)
    err := ctx.dnn.Copy(inputView, outputView)
    if err != nil {
        signalError(err)
    }
}

func executeTranspose(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    input := op.GetInput("input")
    output := op.GetOutput("output")
    axes := op.GetAttrib("axes")
    inputView := mapTensor(ctx, t, input)
    outputView := mapTensor(ctx, t, output)
    rank := inputView.Rank()
    checkSupportedRank(op.Name(), rank, 5)
    perm := make([]int, rank)
    size := axes.Size()
    for i := 0; i < size; i++ {
        perm[i] = axes.At(i).Integer()
    }
    for i := size; i < rank; i++ {
        perm[i] = i
    }
    err := ctx.dnn.Transpose(inputView, outputView, perm)      
    if err != nil {
        signalError(err)
    }
}

func executeConcat(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    values := op.GetInput("values")
    value := op.GetOutput("value")
    axis := op.GetAttrib("axis").Integer()
    size := values.Size()
    v := make([]dnn.Tensor, size)
    for i := 0; i < size; i++ {
        v[i] = mapTensor(ctx, t, values.At(i))
    }
    valueView := mapTensor(ctx, t, value)
    singular := (op.Name() == "stack")
    err := ctx.dnn.Concat(singular, v, valueView, axis)
    if err != nil {
        signalError(err)
    }
}

func executeSplit(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    value := op.GetInput("value")
    values := op.GetOutput("values")
    axis := op.GetAttrib("axis").Integer()
    size := values.Size()
    v := make([]dnn.Tensor, size)
    for i := 0; i < size; i++ {
        v[i] = mapTensor(ctx, t, values.At(i))
    }
    valueView := mapTensor(ctx, t, value)
    singular := (op.Name() == "unstack")
    err := ctx.dnn.Split(singular, valueView, v, axis)
    if err != nil {
        signalError(err)
    }
}

func makePadExecutor(t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("input")
        output := op.GetOutput("output")
        padding := op.GetAttrib("padding")
        border := op.GetAttrib("border").String()
        value := op.GetAttrib("value")
        inputView := mapTensor(ctx, t, input)
        outputView := mapTensor(ctx, t, output)
        paddingShape := extractItems(padding)
        d := inputView.Rank()
        checkSupportedRank(op.Name(), d, 5)
        switch border {
        case "constant":
            err := ctx.dnn.PadConstant(inputView, outputView, paddingShape, mapValue(value, t))
            if err != nil {
                signalError(err)
            }
        case "replicate":
            err := ctx.dnn.PadReplicate(inputView, outputView, paddingShape)
            if err != nil {
                signalError(err)
            }
        default:
            core.RuntimeError("operation not implemented: pad with border == '%s'", border)
        }
    }
}

func executeTile(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    input := op.GetInput("input")
    output := op.GetOutput("output")
    inputView := mapTensor(ctx, t, input)
    outputView := mapTensor(ctx, t, output)
    d := inputView.Rank()
    checkSupportedRank(op.Name(), d, 5)
    err := ctx.dnn.Tile(inputView, outputView)
    if err != nil {
        signalError(err)
    }
}

func executeSlice(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    input := op.GetInput("input")
    output := op.GetOutput("output")
    axes := op.GetAttrib("axes")
    begin := op.GetAttrib("begin")
    inputView := mapTensor(ctx, t, input)
    outputView := mapTensor(ctx, t, output)
    d := inputView.Rank()
    checkSupportedRank(op.Name(), d, 5)
    size := axes.Size()
    offset := make([]int, d)
    for i := 0; i < size; i++ {
        offset[axes.At(i).Integer()] = begin.At(i).Integer()
    }
    // have offset[i] = 0 for i in [size, d)
    err := ctx.dnn.Slice(inputView, outputView, offset)
    if err != nil {
        signalError(err)
    }
}

func makeMatmulExecutor(t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        a := op.GetInput("A")
        b := op.GetInput("B")
        // ACHTUNG: Apparent bug in original code: "C" is input instead of output
        c := op.GetOutput("C")
        trA := op.GetAttrib("transposeA").Logical()
        // ACHTUNG: Apparent bug in original code: "transposeA" instead of "transposeB"
        trB := op.GetAttrib("transposeB").Logical()
        aView := mapTensor(ctx, t, a)
        bView := mapTensor(ctx, t, b)
        cView := mapTensor(ctx, t, c)
        err := ctx.dnn.Matmul(trA, trB, aView, bView, cView)
        if err != nil {
            signalError(err)
        }
    }
}

func makeLinearExecutor(t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("input")
        filter := op.GetInput("filter")
        bias := op.GetInput("bias")
        output := op.GetOutput("output")
        inputView := mapTensor(ctx, t, input)
        filterView := mapTensor(ctx, t, filter)
        biasView := mapTensor(ctx, t, bias)
        outputView := mapTensor(ctx, t, output)
        err := ctx.dnn.Linear(inputView, filterView, biasView, outputView)
        if err != nil {
            signalError(err)
        }
    }
}

func makeSoftmaxExecutor(t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("x")
        output := op.GetOutput("y")
        axes := op.GetAttrib("axes")
        inputView := mapTensor(ctx, t, input)
        outputView := mapTensor(ctx, t, output)
        if axes.Size() != 1 {
            core.RuntimeError("operation not implemented: softmax with multiple axes")
        }
        axis := axes.At(0).Integer()
        err := ctx.dnn.Softmax(inputView, outputView, axis)
        if err != nil {
            signalError(err)
        }
    }
}

func makeArgReduceExecutor(t dnn.Dtype, i dnn.Dtype, f dnn.ArgReduceOp) Executor {
    return func(ctx *Context, op *core.Operation) {
        input := op.GetInput("input")
        output := op.GetOutput("output")
        axes := op.GetAttrib("axes")
        inputView := mapTensor(ctx, t, input)
        outputView := mapTensor(ctx, i, output)
        if axes.Size() != 1 {
            core.RuntimeError("operation not implemented: argmax_reduce with multiple axes")
        }
        axis := axes.At(0).Integer()
        err := ctx.dnn.ArgReduce(f, inputView, outputView, axis)
        if err != nil {
            signalError(err)
        }
    }
}

func makeMultilinearUpsampleExecutor(t dnn.Dtype) Executor {
    return func(ctx *Context, op *core.Operation) {
        // TODO
        core.RuntimeError("operation not implemented: %s", op.Name())
    }
}

func executeUpdate(ctx *Context, op *core.Operation) {
    t := mapOpDtype(op)
    value := op.GetInput("value")
    result := op.GetOutput("result")
    inputView := mapTensor(ctx, t, value)
    outputView := mapTensor(ctx, t, result)
    err := ctx.dnn.Copy(inputView, outputView)
    if err != nil {
        signalError(err)
    }
}

//
//    Utility functions
//

func checkSupportedRank(op string, rank int, max int) {
    if rank > max {
        core.RuntimeError("operation not implemented: %s with rank = %d", op, rank)
    }
}

func extractItems(value core.Value) core.Shape {
    size := value.Size()
    items := make(core.Shape, size)
    for i := 0; i < size; i++ {
        v := value.At(i)
        if v.Kind() == core.ValueKindTuple {
            items[i] = v.At(0).Integer()
        } else {
            items[i] = v.Integer()
        }
    }
    return items
}

func makePadding(
        rank int, 
        input []int, 
        output []int, 
        filter []int, 
        stride []int, 
        dilation []int) core.Shape {
    padding := make(core.Shape, rank)
    for i := 0; i < rank; i++ {
        v := (output[i] - 1) * stride[i] + (filter[i] - 1) * dilation[i] + 1 - input[i]
        if v < 0 {
            v = 0
        }
        padding[i] = v / 2
    }
    return padding
}

func makeSingletonShape(rank int) core.Shape {
    shape := make(core.Shape, rank)
    for i := 0; i < rank; i++ {
        shape[i] = 1
    }
    return shape
}

func mapOpDtype(op *core.Operation) dnn.Dtype {
    dtype := op.Dtype()
    switch dtype {
    case "scalar":
        return dnn.DtypeFloat
    case "integer":
        return dnn.DtypeInt
    case "logical":
        return dnn.DtypeBool
    default:
         core.RuntimeError("operation not implemented: %s<%s>", op.Name(), dtype)
         return 0
    }
}

func mapTensor(ctx *Context, t dnn.Dtype, value core.Value) dnn.Tensor {
    switch value.Kind() {
    case core.ValueKindIdentifier:
        view := ctx.getTensor(value.Identifier())
        core.Assert(view.Dtype() == t)
        return view
    case core.ValueKindScalar:
        v := value.Scalar()
        return makeSingleton(ctx, t, v)
    case core.ValueKindInteger:
        v := value.Integer()
        return makeSingleton(ctx, t, v)
    case core.ValueKindLogical:
        v := value.Logical()
        return makeSingleton(ctx, t, v)
    default:
        core.Assert(false)
        return nil
    }
}

func makeSingleton(ctx *Context, t dnn.Dtype, value interface{}) dnn.Tensor {
    // TODO: Cache singleton tensors?
    view, err := ctx.dnn.NewTensor(t, nil)
    if err != nil {
        signalError(err)
    }
    err = ctx.dnn.Fill(view, value)
    if err != nil {
        signalError(err)
    }
    return view
}

func mapValue(value core.Value, t dnn.Dtype) interface{} {
    switch t {
    case dnn.DtypeBool:
        return value.Logical()
    case dnn.DtypeInt:
        return value.Integer()
    case dnn.DtypeFloat:
        return value.Scalar()
    default:
        core.Assert(false)
        return nil
    }
}

func signalError(err error) {
    panic(err)
}

