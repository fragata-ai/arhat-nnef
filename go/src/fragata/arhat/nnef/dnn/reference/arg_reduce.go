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

func ArgReduce(op api.ArgReduceOp, input *Tensor, output *Tensor, axis int) error {
    kernel := getArgReduceKernel(op)
    argReduceLoopFloat(input, output, axis, kernel)
    return nil
}

// implementation

type argReduceKernel func(n int, ax interface{}, px int, dx int) int

var argReduceKernels = [...]argReduceKernel{
    api.OpArgminReduce: argReduceArgminFloat,
    api.OpArgmaxReduce: argReduceArgmaxFloat,
}

func getArgReduceKernel(op api.ArgReduceOp) argReduceKernel {
    return argReduceKernels[op]
}

func argReduceLoopFloat(input *Tensor, output *Tensor, axis int, kernel argReduceKernel) {
    batch := volumeOf(input.shape[:axis])
    channels := input.shape[axis]
    size := volumeOf(input.shape[axis+1:])
    volume := channels * size
    x := input.FloatData()
    y := output.IntData()
    for i := 0; i < batch; i++ {
        for j := 0; j < size; j++ {
            y[i*size+j] = kernel(channels, x, i*volume+j, size)
        }
    }
}

// kernels

func argReduceArgminFloat(n int, ax interface{}, px int, dx int) int {
    x := ax.([]float32)
    idx := 0
    val := x[0]
    for i := 1; i < n; i++ {
        xi := x[i*dx+px]
        if xi < val {
            val = xi
            idx = i
        }
    }
    return idx
}

func argReduceArgmaxFloat(n int, ax interface{}, px int, dx int) int {
    x := ax.([]float32)
    idx := 0
    val := x[0]
    for i := 1; i < n; i++ {
        xi := x[i*dx+px]
        if xi > val {
            val = xi
            idx = i
        }
    }
    return idx
}

