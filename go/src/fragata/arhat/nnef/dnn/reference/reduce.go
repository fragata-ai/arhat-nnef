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

// interface

func Reduce(op api.ReduceOp, input *Tensor, output *Tensor) error {
    reduceInit(op, output)
    var xv, yv TensorView
    rank := output.rank
    xv.Init(input, rank)
    yv.Init(output, rank)
    kernel := getReduceKernel(op)
    reduceLoop(0, &xv, &yv, kernel)
    if op == api.OpMeanReduce {
        scale := float32(output.volume) / float32(input.volume)
        scaleFloat(output.FloatData(), scale)
    }
    return nil
}

// implementation

func reduceInit(op api.ReduceOp, output *Tensor) {
    switch op {
    case api.OpSumReduce, api.OpMeanReduce:
        fillFloat(output.FloatData(), 0.0)
    case api.OpMinReduce:
        fillFloat(output.FloatData(), posInf)
    case api.OpMaxReduce:
        fillFloat(output.FloatData(), negInf)
    case api.OpAnyReduce:
        fillBool(output.BoolData(), false)
    case api.OpAllReduce:
        fillBool(output.BoolData(), true)
    default:
        assert(false)
    }
}

type reduceKernel func(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int)

var reduceKernels = [...]reduceKernel{
    api.OpSumReduce: reduceSumFloat,
    api.OpMeanReduce: reduceSumFloat,
    api.OpMinReduce: reduceMinFloat,
    api.OpMaxReduce: reduceMaxFloat,
    api.OpAnyReduce: reduceAnyBool,
    api.OpAllReduce: reduceAllBool,
}

func getReduceKernel(op api.ReduceOp) reduceKernel {
    return reduceKernels[op]
}

func reduceLoop(level int, x *TensorView, y *TensorView, kernel reduceKernel) {
    xVolume := x.volume[level]
    yVolume := y.volume[level]
    if yVolume == xVolume || yVolume == 1 {
        dy := btoi(yVolume == xVolume)
        kernel(
            xVolume, 
            x.data, 
            x.offset[level], 
            1,
            y.data, 
            y.offset[level], 
            dy)
    } else {
        assert(level + 1 < y.rank)
        x.Start(level)
        y.Start(level)
        n := x.shape[level]
        for i := 0; i < n; i++ {
            reduceLoop(level+1, x, y, kernel) 
            x.Next(level)
            y.Next(level)
        }
    }
}

// kernels

func reduceSumFloat(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i*dy+py] = x[i*dx+px] + y[i*dy+py]
    }
}

func reduceMinFloat(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i*dy+py] = min(x[i*dx+px], y[i*dy+py])
    }
}

func reduceMaxFloat(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i*dy+py] = max(x[i*dx+px], y[i*dy+py])
    }
}

func reduceAnyBool(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int) {
    x := ax.([]bool)
    y := ay.([]bool)
    for i := 0; i < n; i++ {
        y[i*dy+py] = (x[i*dx+px] || y[i*dy+py])
    }
}

func reduceAllBool(
        n int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int) {
    x := ax.([]bool)
    y := ay.([]bool)
    for i := 0; i < n; i++ {
        y[i*dy+py] = (x[i*dx+px] && y[i*dy+py])
    }
}

