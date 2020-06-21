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

func Pool(
        op api.PoolOp,
        transposed bool,
        input *Tensor,
        output *Tensor,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) error {
    if transposed {
        poolInit(op, input)
    } else {
        poolInit(op, output)
    }
    kernel := getPoolKernelFloat(transposed, input.rank, op)
    kernel(
        input.FloatData(),
        output.FloatData(),
        input.shape,
        output.shape,
        size,
        padding,
        stride,
        dilation,
        includeBorder)
    if op == api.OpAvgPool {
        poolAverage(
            transposed, 
            input, 
            output, 
            size, 
            padding,
            stride,
            dilation,
            includeBorder)
    } 
    return nil
}

// implementation

func poolInit(op api.PoolOp, tensor *Tensor) {
    switch op {
    case api.OpSumPool, api.OpAvgPool:
        fillFloat(tensor.FloatData(), 0.0)
    case api.OpMaxPool:
        fillFloat(tensor.FloatData(), negInf)
    default:
        assert(false)
    }
}

func poolAverage(
        transposed bool,
        input *Tensor,
        output *Tensor,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    if includeBorder {
        if transposed {
            poolAverageVolume(input, size)
        } else {
            poolAverageVolume(output, size)
        }
    } else {
        poolAverageArea(
            transposed,
            input,
            output,
            size,
            padding,
            stride,
            dilation)
    }
}

func poolAverageVolume(tensor *Tensor, size []int) {
    scale := 1.0 / float32(volumeOf(size))
    scaleFloat(tensor.FloatData(), scale)
}

func poolAverageArea(
        transposed bool,
        input *Tensor,
        output *Tensor,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var resultData []float32
    var resultVolume int
    if transposed {
        resultData = input.FloatData()
        resultVolume = input.volume
    } else {
        resultData = output.FloatData()
        resultVolume = output.volume
    }
    counterData := make([]float32, resultVolume)
    kernel := getPoolAreaKernel(transposed, input.rank)
    kernel(
        counterData,
        input.shape,
        output.shape,
        size,
        padding,
        stride,
        dilation)
    for i := 0; i < resultVolume; i++ {
        resultData[i] /= counterData[i]
    }
}

type poolKernelFloat func(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool)

var poolKernelsNSumFloat = [...]poolKernelFloat {
    poolCoreN1SumFloat,
    poolCoreN2SumFloat,
    poolCoreN3SumFloat,
    poolCoreN4SumFloat,
    poolCoreN5SumFloat,
}

var poolKernelsNMaxFloat = [...]poolKernelFloat {
    poolCoreN1MaxFloat,
    poolCoreN2MaxFloat,
    poolCoreN3MaxFloat,
    poolCoreN4MaxFloat,
    poolCoreN5MaxFloat,
}

var poolKernelsTSumFloat = [...]poolKernelFloat {
    poolCoreT1SumFloat,
    poolCoreT2SumFloat,
    poolCoreT3SumFloat,
    poolCoreT4SumFloat,
    poolCoreT5SumFloat,
}

var poolKernelsTMaxFloat = [...]poolKernelFloat {
    poolCoreT1MaxFloat,
    poolCoreT2MaxFloat,
    poolCoreT3MaxFloat,
    poolCoreT4MaxFloat,
    poolCoreT5MaxFloat,
}

func getPoolKernelFloat(transposed bool, rank int, op api.PoolOp) poolKernelFloat {
    assert(rank >= 1 && rank <= 5)
    if transposed {
        switch op {
        case api.OpSumPool, api.OpAvgPool:
            return poolKernelsTSumFloat[rank-1]
        case api.OpMaxPool:
            return poolKernelsTMaxFloat[rank-1]
        }
    } else {
        switch op {
        case api.OpSumPool, api.OpAvgPool:
            return poolKernelsNSumFloat[rank-1]
        case api.OpMaxPool:
            return poolKernelsNMaxFloat[rank-1]
        }
    }
    assert(false)
    return nil
}

type poolAreaKernel func(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int)

var poolAreaKernelsN = [...]poolAreaKernel{
    poolAreaN1,
    poolAreaN2,
    poolAreaN3,
    poolAreaN4,
    poolAreaN5,
}

var poolAreaKernelsT = [...]poolAreaKernel{
    poolAreaT1,
    poolAreaT2,
    poolAreaT3,
    poolAreaT4,
    poolAreaT5,
}

func getPoolAreaKernel(transposed bool, rank int) poolAreaKernel {
    assert(rank >= 1 && rank <= 5)
    if transposed {
        return poolAreaKernelsT[rank-1]
    } else {
        return poolAreaKernelsN[rank-1]    
    }    
}

// kernels (normal, sum)

func poolCoreN1SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                outputData[outputIndex] += inputData[inputIndex]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreN2SumFloat(
        inputData []float32,
        outputData []float32, 
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                outputData[outputOffset] += inputData[inputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreN3SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                outputData[outputOffset] += inputData[inputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreN4SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset4(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                inputOffset := NdOffset4(inputShape, inputIndex[:])
                outputData[outputOffset] += inputData[inputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreN5SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset5(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                inputOffset := NdOffset5(inputShape, inputIndex[:])
                outputData[outputOffset] += inputData[inputOffset]
            }
            // ignore includeBorder
        }
    }
}

// kernels (normal, max)

func poolCoreN1MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                outputData[outputIndex] = max(outputData[outputIndex], inputData[inputIndex])
            } else if includeBorder {
                outputData[outputIndex] = max(outputData[outputIndex], float32(0.0))
            }
        }
    }
}

func poolCoreN2MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                outputData[outputOffset] = max(outputData[outputOffset], inputData[inputOffset])
            } else if includeBorder {
                outputData[outputOffset] = max(outputData[outputOffset], float32(0.0))
            }
        }
    }
}

func poolCoreN3MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                outputData[outputOffset] = max(outputData[outputOffset], inputData[inputOffset])
            } else if includeBorder {
                outputData[outputOffset] = max(outputData[outputOffset], float32(0.0))
            }
        }
    }
}

func poolCoreN4MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset4(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                inputOffset := NdOffset4(inputShape, inputIndex[:])
                outputData[outputOffset] = max(outputData[outputOffset], inputData[inputOffset])
            } else if includeBorder {
                outputData[outputOffset] = max(outputData[outputOffset], float32(0.0))
            }
        }
    }
}

func poolCoreN5MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset5(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                inputOffset := NdOffset5(inputShape, inputIndex[:])
                outputData[outputOffset] = max(outputData[outputOffset], inputData[inputOffset])
            } else if includeBorder {
                outputData[outputOffset] = max(outputData[outputOffset], float32(0.0))
            }
        }
    }
}

// kernels (transposed, sum)

func poolCoreT1SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                inputData[inputIndex] += outputData[outputIndex]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreT2SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                inputData[inputOffset] += outputData[outputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreT3SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                inputData[inputOffset] += outputData[outputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreT4SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset4(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                inputOffset := NdOffset4(inputShape, inputIndex[:])
                inputData[inputOffset] += outputData[outputOffset]
            }
            // ignore includeBorder
        }
    }
}

func poolCoreT5SumFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset5(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                inputOffset := NdOffset5(inputShape, inputIndex[:])
                inputData[inputOffset] += outputData[outputOffset]
            }
            // ignore includeBorder
        }
    }
}

// kernels (transposed, max)

func poolCoreT1MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                inputData[inputIndex] = max(inputData[inputIndex], outputData[outputIndex])
            }
            // includeBorder unused
        }
    }
}

func poolCoreT2MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                inputData[inputOffset] = max(inputData[inputOffset], outputData[outputOffset])
            }
            // includeBorder unused
        }
    }
}

func poolCoreT3MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                inputData[inputOffset] = max(inputData[inputOffset], outputData[outputOffset])
            }
            // includeBorder unused
        }
    }
}

func poolCoreT4MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset4(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                inputOffset := NdOffset4(inputShape, inputIndex[:])
                inputData[inputOffset] = max(inputData[inputOffset], outputData[outputOffset])
            }
            // includeBorder unused
        }
    }
}

func poolCoreT5MaxFloat(
        inputData []float32,
        outputData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset5(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                inputOffset := NdOffset5(inputShape, inputIndex[:])
                inputData[inputOffset] = max(inputData[inputOffset], outputData[outputOffset])
            }
            // includeBorder unused
        }
    }
}

// kernels: area (normal)

func poolAreaN1(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                counterData[outputIndex] += float32(1.0)
            }
        }
    }
}

func poolAreaN2(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                counterData[outputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaN3(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                counterData[outputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaN4(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset4(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                counterData[outputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaN5(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset5(outputShape, outputIndex)
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                counterData[outputOffset] += float32(1.0)
            }
        }
    }
}

// kernels: area (transposed)

func poolAreaT1(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    inputShape0 := inputShape[0]
    outputShape0 := outputShape[0]
    size0 := size[0]
    padding0 := padding[0]
    stride0 := stride[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for kernelIndex := 0; kernelIndex < size0; kernelIndex++ {
            inputIndex := outputIndex * stride0 + kernelIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                counterData[inputIndex] += float32(1.0)
            }
        }
    }
}

func poolAreaT2(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [2]int
    var outputLoop, kernelLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                counterData[inputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaT3(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [3]int
    var outputLoop, kernelLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                counterData[inputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaT4(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [4]int
    var outputLoop, kernelLoop NdLoop4
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] {
                inputOffset := NdOffset4(inputShape, inputIndex[:])
                counterData[inputOffset] += float32(1.0)
            }
        }
    }
}

func poolAreaT5(
        counterData []float32,
        inputShape []int,
        outputShape []int,
        size []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [5]int
    var outputLoop, kernelLoop NdLoop5
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        for kernelLoop.Start(size); kernelLoop.Test(); kernelLoop.Next() {
            kernelIndex := kernelLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + kernelIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + kernelIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + kernelIndex[2] * dilation[2] - padding[2]
            inputIndex[3] = outputIndex[3] * stride[3] + kernelIndex[3] * dilation[3] - padding[3]
            inputIndex[4] = outputIndex[4] * stride[4] + kernelIndex[4] * dilation[4] - padding[4]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] &&
                    inputIndex[3] >= 0 && inputIndex[3] < inputShape[3] &&
                    inputIndex[4] >= 0 && inputIndex[4] < inputShape[4] {
                inputOffset := NdOffset5(inputShape, inputIndex[:])
                counterData[inputOffset] += float32(1.0)
            }
        }
    }
}

