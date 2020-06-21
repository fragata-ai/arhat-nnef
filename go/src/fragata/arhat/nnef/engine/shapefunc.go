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

import "fragata/arhat/nnef/core"

func ShapeOf(graph *core.Graph, value core.Value) core.Shape {
    if value.Kind() == core.ValueKindIdentifier {
        return graph.GetTensor(value.Identifier()).Shape()
    } else {
        return nil
    }
}

func SetShape(graph *core.Graph, value core.Value, shape core.Shape) {
    // no shape cloning required: shape functions always create a fresh copy
    graph.GetTensor(value.Identifier()).SetShape([]int(shape))
}

func InputShape(graph *core.Graph, op *core.Operation, idx int) core.Shape {
    return ShapeOf(graph, op.InputAt(idx))
}

func InputShapes(graph *core.Graph, op *core.Operation) []core.Shape {
    inputs := op.InputAt(0)
    size := inputs.Size()
    shapes := make([]core.Shape, size)
    for idx := 0; idx < size; idx++ {
        shapes[idx] = ShapeOf(graph, inputs.At(idx))
    }
    return shapes
}

func SetOutputShape(graph *core.Graph, op *core.Operation, shape core.Shape) {
    count := op.OutputCount()
    for idx := 0; idx < count; idx++ {
        SetShape(graph, op.OutputAt(idx), shape)
    }
}

func SetOutputShapes(graph *core.Graph, op *core.Operation, shapes []core.Shape) {
    outputs := op.OutputAt(0)
    size := outputs.Size()
    core.Check(len(shapes) == size, 
        "number of shapes (%d) does not match number of outputs (%d)", 
            len(shapes), size)
    for idx := 0; idx < size; idx++ {
        SetShape(graph, outputs.At(idx), shapes[idx])
    }
}

func NullaryShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.NullaryShape(op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func ConstantShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.ConstantShape(op.AttribAt(0), op.AttribAt(1))
    SetOutputShape(graph, op, shape)
}

func UnaryShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.UnaryShape(InputShape(graph, op, 0))
    SetOutputShape(graph, op, shape)
}

func BinaryShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.BinaryShape(InputShape(graph, op, 0), InputShape(graph, op, 1))
    SetOutputShape(graph, op, shape)
}

func AsymmetricBinaryShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.AsymmetricBinaryShape(InputShape(graph, op, 0), InputShape(graph, op, 1))
    SetOutputShape(graph, op, shape)
}

func TernaryShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.TernaryShape(
            InputShape(graph, op, 0), 
            InputShape(graph, op, 1), 
            InputShape(graph, op, 2))
    SetOutputShape(graph, op, shape)
}

func ReduceShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.ReduceShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func DownsampleShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.DownsampleShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func UpsampleShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.UpsampleShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func ConvShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.ConvShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4))
    SetOutputShape(graph, op, shape)
}

func DeconvShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.DeconvShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4),
            op.AttribAt(5))
    SetOutputShape(graph, op, shape)
}

func SeparableConvShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.SeparableConvShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            InputShape(graph, op, 3),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4))
    SetOutputShape(graph, op, shape)
}

func SeparableDeconvShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.SeparableDeconvShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            InputShape(graph, op, 3),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4),
            op.AttribAt(5))
    SetOutputShape(graph, op, shape)
}

func PoolShapeFunc(op *core.Operation, graph *core.Graph) {
    // [5] "box.normalize" not used
    shape :=
        core.PoolShape(
            InputShape(graph, op, 0),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4))
    SetOutputShape(graph, op, shape)
}

func UnpoolShapeFunc(op *core.Operation, graph *core.Graph) {
    // [6] "debox.normalize" not used
    shape :=
        core.UnpoolShape(
            InputShape(graph, op, 0),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4),
            op.AttribAt(5))
    SetOutputShape(graph, op, shape)
}

func SampleShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.SampleShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4))
    SetOutputShape(graph, op, shape)
}

func DesampleShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.DesampleShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2),
            op.AttribAt(3),
            op.AttribAt(4),
            op.AttribAt(5))
    SetOutputShape(graph, op, shape)
}

func NormalizeShapeAxesFunc(op *core.Operation, graph *core.Graph) {
    // [1] "bias" [2] "epsilon" not used
    shape := core.NormalizeShapeAxes(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func NormalizeShapeSizeFunc(op *core.Operation, graph *core.Graph) {
    // [1] "bias" [2] "epsilon" not used
    shape := core.NormalizeShapeSize(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func BatchnormShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.BatchnormShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            InputShape(graph, op, 3),
            InputShape(graph, op, 4),
            op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func RoiShapeFunc(op *core.Operation, graph *core.Graph) {
    // [3] "sampling_rate" [4] "resize_method" not used
    shape :=
        core.RoiShape(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func RoiShapeResampleFunc(op *core.Operation, graph *core.Graph) {
    // [2] "method" not used
    shape :=
        core.RoiShapeResample(
            InputShape(graph, op, 0),
            InputShape(graph, op, 1),
            InputShape(graph, op, 2),
            op.AttribAt(0),
            op.AttribAt(1))
    SetOutputShape(graph, op, shape)
}

func ReshapeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape :=
        core.ReshapeShape(
            InputShape(graph, op, 0),
            op.AttribAt(0),
            op.AttribAt(1),
            op.AttribAt(2))
    SetOutputShape(graph, op, shape)
}

func TransposeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.TransposeShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func SplitShapeFunc(op *core.Operation, graph *core.Graph) {
    shapes := core.SplitShape(InputShape(graph, op, 0), op.AttribAt(0), op.AttribAt(1))
    SetOutputShapes(graph, op, shapes)
}

func ConcatShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.ConcatShape(InputShapes(graph, op), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func SliceShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.SliceShape(
            InputShape(graph, op, 0), 
            op.AttribAt(0), 
            op.AttribAt(1), 
            op.AttribAt(2))
    SetOutputShape(graph, op, shape)
}

func StackShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.StackShape(InputShapes(graph, op), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func UnstackShapeFunc(op *core.Operation, graph *core.Graph) {
    shapes := core.UnstackShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShapes(graph, op, shapes)
}

func SqueezeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.SqueezeShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func UnsqueezeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.UnsqueezeShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func TileShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.TileShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func PadShapeFunc(op *core.Operation, graph *core.Graph) {
    // [1] "border" [2] "value" not used
    shape := core.PadShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func MatmulShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.MatmulShape(
            InputShape(graph, op, 0), 
            InputShape(graph, op, 1), 
            op.AttribAt(0), 
            op.AttribAt(1))
    SetOutputShape(graph, op, shape)
}

func LinearShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.LinearShape(
            InputShape(graph, op, 0), 
            InputShape(graph, op, 1), 
            InputShape(graph, op, 2))
    SetOutputShape(graph, op, shape)
}

func UpdateShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.UpdateShape(InputShape(graph, op, 0), InputShape(graph, op, 1))
    SetOutputShape(graph, op, shape)
}

func SoftmaxShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.SoftmaxShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func CopyNShapeFunc(op *core.Operation, graph *core.Graph) {
    shapes := core.CopyNShape(InputShape(graph, op, 0), op.AttribAt(0))
    SetOutputShapes(graph, op, shapes)
}

func AddNShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := core.AddNShape(InputShapes(graph, op))
    SetOutputShape(graph, op, shape)
}

func LinearQuantizeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.LinearQuantizeShape(
            InputShape(graph, op, 0), 
            InputShape(graph, op, 1), 
            InputShape(graph, op, 2),
            op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

func LogarithmicQuantizeShapeFunc(op *core.Operation, graph *core.Graph) {
    shape := 
        core.LogarithmicQuantizeShape(
            InputShape(graph, op, 0), 
            InputShape(graph, op, 1), 
            op.AttribAt(0))
    SetOutputShape(graph, op, shape)
}

