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

func Slice(input *Tensor, output *Tensor, offset []int) error {
    kernel := getSliceKernel(output.dtype, output.rank)
    kernel(input, output, offset)
    return nil
}

// implementation

type sliceKernel func(input *Tensor, output *Tensor, offset []int)

var sliceKernels = [...][5]sliceKernel{
    api.DtypeBool: [5]sliceKernel{
        slice1Bool,
        slice2Bool,
        slice3Bool,
        slice4Bool,
        slice5Bool,
    },
    api.DtypeInt: [5]sliceKernel{
        slice1Int,
        slice2Int,
        slice3Int,
        slice4Int,
        slice5Int,
    },
    api.DtypeFloat: [5]sliceKernel{
        slice1Float,
        slice2Float,
        slice3Float,
        slice4Float,
        slice5Float,
    },
}

func getSliceKernel(dtype api.Dtype, rank int) sliceKernel {
    assert(rank >= 1 && rank <= 5)
    return sliceKernels[dtype][rank-1]
}

// kernels (bool)

func slice1Bool(input *Tensor, output *Tensor, offset []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    shape0 := output.shape[0]
    offset0 := offset[0]
    for outputIndex := 0; outputIndex < shape0; outputIndex++ {
        inputIndex := outputIndex + offset0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func slice2Bool(input *Tensor, output *Tensor, offset []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice3Bool(input *Tensor, output *Tensor, offset []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice4Bool(input *Tensor, output *Tensor, offset []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice5Bool(input *Tensor, output *Tensor, offset []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputIndex[4] = outputIndex[4] + offset[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (int)

func slice1Int(input *Tensor, output *Tensor, offset []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    shape0 := output.shape[0]
    offset0 := offset[0]
    for outputIndex := 0; outputIndex < shape0; outputIndex++ {
        inputIndex := outputIndex + offset0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func slice2Int(input *Tensor, output *Tensor, offset []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice3Int(input *Tensor, output *Tensor, offset []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice4Int(input *Tensor, output *Tensor, offset []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice5Int(input *Tensor, output *Tensor, offset []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputIndex[4] = outputIndex[4] + offset[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (float)

func slice1Float(input *Tensor, output *Tensor, offset []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    shape0 := output.shape[0]
    offset0 := offset[0]
    for outputIndex := 0; outputIndex < shape0; outputIndex++ {
        inputIndex := outputIndex + offset0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func slice2Float(input *Tensor, output *Tensor, offset []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice3Float(input *Tensor, output *Tensor, offset []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice4Float(input *Tensor, output *Tensor, offset []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func slice5Float(input *Tensor, output *Tensor, offset []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] + offset[0]
        inputIndex[1] = outputIndex[1] + offset[1]
        inputIndex[2] = outputIndex[2] + offset[2]
        inputIndex[3] = outputIndex[3] + offset[3]
        inputIndex[4] = outputIndex[4] + offset[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

