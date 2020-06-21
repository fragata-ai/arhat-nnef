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

func Transpose(input *Tensor, output *Tensor, perm []int) error {
    kernel := getTransposeKernel(output.dtype, output.rank)
    kernel(input, output, perm)
    return nil
}

// implementation

type transposeKernel func(input *Tensor, output *Tensor, perm []int)

var transposeKernels = [...][5]transposeKernel{
    api.DtypeBool: [5]transposeKernel{
        transpose1Bool,
        transpose2Bool,
        transpose3Bool,
        transpose4Bool,
        transpose5Bool,
    },
    api.DtypeInt: [5]transposeKernel{
        transpose1Int,
        transpose2Int,
        transpose3Int,
        transpose4Int,
        transpose5Int,
    },
    api.DtypeFloat: [5]transposeKernel{
        transpose1Float,
        transpose2Float,
        transpose3Float,
        transpose4Float,
        transpose5Float,
    },
}

func getTransposeKernel(dtype api.Dtype, rank int) transposeKernel {
    assert(rank >= 1 && rank <= 5)
    return transposeKernels[dtype][rank-1]
}

// kernels (bool)

func transpose1Bool(input *Tensor, output *Tensor, perm []int) {
    copy(output.BoolData(), input.BoolData())
}

func transpose2Bool(input *Tensor, output *Tensor, perm []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [2]int
    var loop NdLoop2
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        inputOffset := NdOffset2(inputShape, inputIndex)
        outputOffset := NdOffset2(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose3Bool(input *Tensor, output *Tensor, perm []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [3]int
    var loop NdLoop3
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        inputOffset := NdOffset3(inputShape, inputIndex)
        outputOffset := NdOffset3(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose4Bool(input *Tensor, output *Tensor, perm []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [4]int
    var loop NdLoop4
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        inputOffset := NdOffset4(inputShape, inputIndex)
        outputOffset := NdOffset4(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose5Bool(input *Tensor, output *Tensor, perm []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [5]int
    var loop NdLoop5
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        outputIndex[4] = inputIndex[perm[4]]
        inputOffset := NdOffset5(inputShape, inputIndex)
        outputOffset := NdOffset5(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (int)

func transpose1Int(input *Tensor, output *Tensor, perm []int) {
    copy(output.IntData(), input.IntData())
}

func transpose2Int(input *Tensor, output *Tensor, perm []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [2]int
    var loop NdLoop2
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        inputOffset := NdOffset2(inputShape, inputIndex)
        outputOffset := NdOffset2(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose3Int(input *Tensor, output *Tensor, perm []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [3]int
    var loop NdLoop3
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        inputOffset := NdOffset3(inputShape, inputIndex)
        outputOffset := NdOffset3(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose4Int(input *Tensor, output *Tensor, perm []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [4]int
    var loop NdLoop4
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        inputOffset := NdOffset4(inputShape, inputIndex)
        outputOffset := NdOffset4(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose5Int(input *Tensor, output *Tensor, perm []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [5]int
    var loop NdLoop5
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        outputIndex[4] = inputIndex[perm[4]]
        inputOffset := NdOffset5(inputShape, inputIndex)
        outputOffset := NdOffset5(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (float)

func transpose1Float(input *Tensor, output *Tensor, perm []int) {
    copy(output.FloatData(), input.FloatData())
}

func transpose2Float(input *Tensor, output *Tensor, perm []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [2]int
    var loop NdLoop2
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        inputOffset := NdOffset2(inputShape, inputIndex)
        outputOffset := NdOffset2(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose3Float(input *Tensor, output *Tensor, perm []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [3]int
    var loop NdLoop3
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        inputOffset := NdOffset3(inputShape, inputIndex)
        outputOffset := NdOffset3(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose4Float(input *Tensor, output *Tensor, perm []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [4]int
    var loop NdLoop4
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        inputOffset := NdOffset4(inputShape, inputIndex)
        outputOffset := NdOffset4(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func transpose5Float(input *Tensor, output *Tensor, perm []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var outputIndex [5]int
    var loop NdLoop5
    for loop.Start(inputShape); loop.Test(); loop.Next() {
        inputIndex := loop.Index()
        outputIndex[0] = inputIndex[perm[0]]
        outputIndex[1] = inputIndex[perm[1]]
        outputIndex[2] = inputIndex[perm[2]]
        outputIndex[3] = inputIndex[perm[3]]
        outputIndex[4] = inputIndex[perm[4]]
        inputOffset := NdOffset5(inputShape, inputIndex)
        outputOffset := NdOffset5(outputShape, outputIndex[:])
        outputData[outputOffset] = inputData[inputOffset]
    }
}

