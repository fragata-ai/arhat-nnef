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
// Ported from C++ to Go and modified by FRAGATA COMPUTER SYSTEMS AG.
//

package reference

import "fragata/arhat/nnef/dnn/api"

//
//    Engine
//

type Engine struct {}

func NewEngine() *Engine {
    return new(Engine)
}

// interface

func(e *Engine) NewTensor(dtype api.Dtype, shape []int) (api.Tensor, error) {
    tensor, err := NewTensor(dtype, shape)
    if err != nil {
        return nil, err
    }
    return tensor, nil
}

func(e *Engine) Fill(tensor api.Tensor, data interface{}) error {
    return Fill(tensor.(*Tensor), data)
}

func(e *Engine) Read(tensor api.Tensor, data interface{}) error {
    return Read(tensor.(*Tensor), data)
}

func(e *Engine) Copy(input api.Tensor, output api.Tensor) error {
    return Copy(input.(*Tensor), output.(*Tensor))
}

func(e *Engine) Unary(op api.UnaryOp, x api.Tensor, y api.Tensor) error {
    return Unary(op, x.(*Tensor), y.(*Tensor))
}

func(e *Engine) Binary(op api.BinaryOp, x api.Tensor, y api.Tensor, z api.Tensor) error {
    return Binary(op, x.(*Tensor), y.(*Tensor), z.(*Tensor))
}

func(e *Engine) Reduce(op api.ReduceOp, input api.Tensor, output api.Tensor) error {
    return Reduce(op, input.(*Tensor), output.(*Tensor))
}

func(e *Engine) Select(c api.Tensor, x api.Tensor, y api.Tensor, z api.Tensor) error {
    return Select(c.(*Tensor), x.(*Tensor), y.(*Tensor), z.(*Tensor))
}

func(e *Engine) Conv(
        transposed bool,
        input api.Tensor,
        filter api.Tensor,
        bias api.Tensor,
        output api.Tensor,
        padding []int,
        stride []int,
        dilation []int) error {
    return Conv(
        transposed,
        input.(*Tensor),
        filter.(*Tensor),
        bias.(*Tensor),
        output.(*Tensor),
        padding,
        stride,
        dilation)
}

func(e *Engine) DepthwiseConv(
        transposed bool,
        input api.Tensor,
        filter api.Tensor,
        bias api.Tensor,
        output api.Tensor,
        padding []int,
        stride []int,
        dilation []int) error {
    return DepthwiseConv(
        transposed,
        input.(*Tensor),
        filter.(*Tensor),
        bias.(*Tensor),
        output.(*Tensor),
        padding,
        stride,
        dilation)
}

func(e *Engine) GroupedConv(
        transposed bool,
        input api.Tensor,
        filter api.Tensor,
        bias api.Tensor,
        output api.Tensor,
        padding []int,
        stride []int,
        dilation []int,
        groups int) error {
    return GroupedConv(
        transposed,
        input.(*Tensor),
        filter.(*Tensor),
        bias.(*Tensor),
        output.(*Tensor),
        padding,
        stride,
        dilation,
        groups)
}

func(e *Engine) Pool(
        op api.PoolOp,
        transposed bool,
        input api.Tensor,
        output api.Tensor,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) error {
    return Pool(
        op,
        transposed,
        input.(*Tensor),
        output.(*Tensor),
        size,
        padding,
        stride,
        dilation,
        includeBorder)
}

func(e *Engine) Matmul(trA bool, trB bool, a api.Tensor, b api.Tensor, c api.Tensor) error {
    return Matmul(trA, trB, a.(*Tensor), b.(*Tensor), c.(*Tensor))
}

func(e *Engine) Linear(
        input api.Tensor, 
        filter api.Tensor, 
        bias api.Tensor, 
        output api.Tensor) error {
    return Linear(input.(*Tensor), filter.(*Tensor), bias.(*Tensor), output.(*Tensor))
}

func(e *Engine) Softmax(input api.Tensor, output api.Tensor, axis int) error {
    return Softmax(input.(*Tensor), output.(*Tensor), axis)
}

func(e *Engine) ArgReduce(op api.ArgReduceOp, input api.Tensor, output api.Tensor, axis int) error {
    return ArgReduce(op, input.(*Tensor), output.(*Tensor), axis)
}

func(e *Engine) Transpose(input api.Tensor, output api.Tensor, perm []int) error {
    return Transpose(input.(*Tensor), output.(*Tensor), perm)
}

func(e *Engine) Concat(singular bool, x []api.Tensor, y api.Tensor, axis int) error {
    return Concat(singular, castTensors(x), y.(*Tensor), axis)
}

func(e *Engine) Split(singular bool, x api.Tensor, y []api.Tensor, axis int) error {
    return Split(singular, x.(*Tensor), castTensors(y), axis)
}

func(e *Engine) PadConstant(
        input api.Tensor, 
        output api.Tensor, 
        padding []int, 
        value interface{}) error {
    return PadConstant(input.(*Tensor), output.(*Tensor), padding, value)
}

func(e *Engine) PadReplicate(input api.Tensor, output api.Tensor, padding []int) error {
    return PadReplicate(input.(*Tensor), output.(*Tensor), padding)
}

func(e *Engine) Tile(input api.Tensor, output api.Tensor) error {
    return Tile(input.(*Tensor), output.(*Tensor))
}

func(e *Engine) Slice(input api.Tensor, output api.Tensor, offset []int) error {
    return Slice(input.(*Tensor), output.(*Tensor), offset)
}

// implementation

func castTensors(x []api.Tensor) []*Tensor {
    n := len(x)
    y := make([]*Tensor, n)
    for i := 0; i < n; i++ {
        y[i] = x[i].(*Tensor)
    }
    return y
}

