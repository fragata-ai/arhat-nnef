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

func Tile(input *Tensor, output *Tensor) error {
    kernel := getTileKernel(output.dtype, output.rank)
    kernel(input, output)
    return nil
}

// implementation

type tileKernel func(input *Tensor, output *Tensor)

var tileKernels = [...][5]tileKernel{
    api.DtypeBool: [5]tileKernel{
        tile1Bool,
        tile2Bool,
        tile3Bool,
        tile4Bool,
        tile5Bool,
    },
    api.DtypeInt: [5]tileKernel{
        tile1Int,
        tile2Int,
        tile3Int,
        tile4Int,
        tile5Int,
    },
    api.DtypeFloat: [5]tileKernel{
        tile1Float,
        tile2Float,
        tile3Float,
        tile4Float,
        tile5Float,
    },
}

func getTileKernel(dtype api.Dtype, rank int) tileKernel {
    assert(rank >= 1 && rank <= 5)
    return tileKernels[dtype][rank-1]
}

// kernels (bool)

func tile1Bool(input *Tensor, output *Tensor) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex % inputShape0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func tile2Bool(input *Tensor, output *Tensor) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile3Bool(input *Tensor, output *Tensor) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile4Bool(input *Tensor, output *Tensor) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile5Bool(input *Tensor, output *Tensor) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputIndex[4] = outputIndex[4] % inputShape[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (int)

func tile1Int(input *Tensor, output *Tensor) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex % inputShape0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func tile2Int(input *Tensor, output *Tensor) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile3Int(input *Tensor, output *Tensor) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile4Int(input *Tensor, output *Tensor) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile5Int(input *Tensor, output *Tensor) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputIndex[4] = outputIndex[4] % inputShape[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (float)

func tile1Float(input *Tensor, output *Tensor) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex % inputShape0
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func tile2Float(input *Tensor, output *Tensor) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile3Float(input *Tensor, output *Tensor) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile4Float(input *Tensor, output *Tensor) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func tile5Float(input *Tensor, output *Tensor) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] % inputShape[0]
        inputIndex[1] = outputIndex[1] % inputShape[1]
        inputIndex[2] = outputIndex[2] % inputShape[2]
        inputIndex[3] = outputIndex[3] % inputShape[3]
        inputIndex[4] = outputIndex[4] % inputShape[4]
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

