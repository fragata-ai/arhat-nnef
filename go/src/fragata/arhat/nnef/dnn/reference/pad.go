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

func PadConstant(input *Tensor, output *Tensor, padding []int, value interface{}) error {
    kernel := getPadConstantKernel(output.dtype, output.rank)
    kernel(input, output, padding, value)
    return nil
}

func PadReplicate(input *Tensor, output *Tensor, padding []int) error {
    kernel := getPadReplicateKernel(output.dtype, output.rank)
    kernel(input, output, padding)
    return nil
}

// implementation

type padConstantKernel func(input *Tensor, output *Tensor, padding []int, value interface{})

var padConstantKernels = [...][5]padConstantKernel{
    api.DtypeBool: [5]padConstantKernel{
        padConstant1Bool,
        padConstant2Bool,
        padConstant3Bool,
        padConstant4Bool,
        padConstant5Bool,
    },
    api.DtypeInt: [5]padConstantKernel{
        padConstant1Int,
        padConstant2Int,
        padConstant3Int,
        padConstant4Int,
        padConstant5Int,
    },
    api.DtypeFloat: [5]padConstantKernel{
        padConstant1Float,
        padConstant2Float,
        padConstant3Float,
        padConstant4Float,
        padConstant5Float,
    },
}

func getPadConstantKernel(dtype api.Dtype, rank int) padConstantKernel {
    assert(rank >= 1 && rank <= 5)
    return padConstantKernels[dtype][rank-1]
}

type padReplicateKernel func(input *Tensor, output *Tensor, padding []int)

var padReplicateKernels = [...][5]padReplicateKernel{
    api.DtypeBool: [5]padReplicateKernel{
        padReplicate1Bool,
        padReplicate2Bool,
        padReplicate3Bool,
        padReplicate4Bool,
        padReplicate5Bool,
    },
    api.DtypeInt: [5]padReplicateKernel{
        padReplicate1Int,
        padReplicate2Int,
        padReplicate3Int,
        padReplicate4Int,
        padReplicate5Int,
    },
    api.DtypeFloat: [5]padReplicateKernel{
        padReplicate1Float,
        padReplicate2Float,
        padReplicate3Float,
        padReplicate4Float,
        padReplicate5Float,
    },
}

func getPadReplicateKernel(dtype api.Dtype, rank int) padReplicateKernel {
    assert(rank >= 1 && rank <= 5)
    return padReplicateKernels[dtype][rank-1]
}

// kernels (constant, bool)

func padConstant1Bool(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    c := value.(bool)
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex - padding0
        v := c
        if inputIndex >= 0 && inputIndex < inputShape0 {
            v = inputData[inputIndex]
        }
        outputData[outputIndex] = v
    }
}

func padConstant2Bool(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(bool)
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        outputOffset := NdOffset2(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) {
            inputOffset := NdOffset2(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant3Bool(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(bool)
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        outputOffset := NdOffset3(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) {
            inputOffset := NdOffset3(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant4Bool(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(bool)
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        outputOffset := NdOffset4(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) {
            inputOffset := NdOffset4(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant5Bool(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(bool)
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        inputIndex[4] = outputIndex[4] - padding[4]
        outputOffset := NdOffset5(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) &&
                (inputIndex[4] >= 0 && inputIndex[4] < inputShape[4]) {
            inputOffset := NdOffset5(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

// kernels (constant, int)

func padConstant1Int(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    c := value.(int)
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex - padding0
        v := c
        if inputIndex >= 0 && inputIndex < inputShape0 {
            v = inputData[inputIndex]
        }
        outputData[outputIndex] = v
    }
}

func padConstant2Int(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(int)
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        outputOffset := NdOffset2(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) {
            inputOffset := NdOffset2(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant3Int(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(int)
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        outputOffset := NdOffset3(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) {
            inputOffset := NdOffset3(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant4Int(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(int)
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        outputOffset := NdOffset4(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) {
            inputOffset := NdOffset4(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant5Int(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(int)
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        inputIndex[4] = outputIndex[4] - padding[4]
        outputOffset := NdOffset5(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) &&
                (inputIndex[4] >= 0 && inputIndex[4] < inputShape[4]) {
            inputOffset := NdOffset5(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

// kernels (constant, float)

func padConstant1Float(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    c := value.(float32)
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := outputIndex - padding0
        v := c
        if inputIndex >= 0 && inputIndex < inputShape0 {
            v = inputData[inputIndex]
        }
        outputData[outputIndex] = v
    }
}

func padConstant2Float(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(float32)
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        outputOffset := NdOffset2(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) {
            inputOffset := NdOffset2(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant3Float(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(float32)
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        outputOffset := NdOffset3(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) {
            inputOffset := NdOffset3(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant4Float(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(float32)
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        outputOffset := NdOffset4(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) {
            inputOffset := NdOffset4(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

func padConstant5Float(input *Tensor, output *Tensor, padding []int, value interface{}) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    c := value.(float32)
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = outputIndex[0] - padding[0]
        inputIndex[1] = outputIndex[1] - padding[1]
        inputIndex[2] = outputIndex[2] - padding[2]
        inputIndex[3] = outputIndex[3] - padding[3]
        inputIndex[4] = outputIndex[4] - padding[4]
        outputOffset := NdOffset5(outputShape, outputIndex)
        v := c
        if (inputIndex[0] >= 0 && inputIndex[0] < inputShape[0]) &&
                (inputIndex[1] >= 0 && inputIndex[1] < inputShape[1]) &&
                (inputIndex[2] >= 0 && inputIndex[2] < inputShape[2]) &&
                (inputIndex[3] >= 0 && inputIndex[3] < inputShape[3]) &&
                (inputIndex[4] >= 0 && inputIndex[4] < inputShape[4]) {
            inputOffset := NdOffset5(inputShape, inputIndex[:])
            v = inputData[inputOffset]
        }
        outputData[outputOffset] = v
    }
}

// kernels (replicate, bool)

func padReplicate1Bool(input *Tensor, output *Tensor, padding []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := clipInt(outputIndex-padding0, 0, inputShape0-1)
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func padReplicate2Bool(input *Tensor, output *Tensor, padding []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate3Bool(input *Tensor, output *Tensor, padding []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate4Bool(input *Tensor, output *Tensor, padding []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate5Bool(input *Tensor, output *Tensor, padding []int) {
    inputData := input.BoolData()
    outputData := output.BoolData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputIndex[4] = clipInt(outputIndex[4]-padding[4], 0, inputShape[4]-1)
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (replicate, int)

func padReplicate1Int(input *Tensor, output *Tensor, padding []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := clipInt(outputIndex-padding0, 0, inputShape0-1)
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func padReplicate2Int(input *Tensor, output *Tensor, padding []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate3Int(input *Tensor, output *Tensor, padding []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate4Int(input *Tensor, output *Tensor, padding []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate5Int(input *Tensor, output *Tensor, padding []int) {
    inputData := input.IntData()
    outputData := output.IntData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputIndex[4] = clipInt(outputIndex[4]-padding[4], 0, inputShape[4]-1)
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

// kernels (replicate, float)

func padReplicate1Float(input *Tensor, output *Tensor, padding []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape0 := input.shape[0]
    outputShape0 := output.shape[0]
    padding0 := padding[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        inputIndex := clipInt(outputIndex-padding0, 0, inputShape0-1)
        outputData[outputIndex] = inputData[inputIndex]
    }
}

func padReplicate2Float(input *Tensor, output *Tensor, padding []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [2]int
    var loop NdLoop2
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputOffset := NdOffset2(inputShape, inputIndex[:])
        outputOffset := NdOffset2(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate3Float(input *Tensor, output *Tensor, padding []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [3]int
    var loop NdLoop3
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputOffset := NdOffset3(inputShape, inputIndex[:])
        outputOffset := NdOffset3(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate4Float(input *Tensor, output *Tensor, padding []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [4]int
    var loop NdLoop4
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputOffset := NdOffset4(inputShape, inputIndex[:])
        outputOffset := NdOffset4(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

func padReplicate5Float(input *Tensor, output *Tensor, padding []int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    outputShape := output.shape
    var inputIndex [5]int
    var loop NdLoop5
    for loop.Start(outputShape); loop.Test(); loop.Next() {
        outputIndex := loop.Index()
        inputIndex[0] = clipInt(outputIndex[0]-padding[0], 0, inputShape[0]-1)
        inputIndex[1] = clipInt(outputIndex[1]-padding[1], 0, inputShape[1]-1)
        inputIndex[2] = clipInt(outputIndex[2]-padding[2], 0, inputShape[2]-1)
        inputIndex[3] = clipInt(outputIndex[3]-padding[3], 0, inputShape[3]-1)
        inputIndex[4] = clipInt(outputIndex[4]-padding[4], 0, inputShape[4]-1)
        inputOffset := NdOffset5(inputShape, inputIndex[:])
        outputOffset := NdOffset5(outputShape, outputIndex)
        outputData[outputOffset] = inputData[inputOffset]
    }
}

