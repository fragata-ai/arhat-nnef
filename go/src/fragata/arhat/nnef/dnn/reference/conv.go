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

// interface

func Conv(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int) error {
    kernel := getConvKernelFloat(transposed, input.rank)
    convLoopFloat(
        transposed,
        input,
        filter,
        bias,
        output,
        padding,
        stride,
        dilation,
        kernel)
    return nil
}

func DepthwiseConv(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int) error {
    kernel := getConvKernelFloat(transposed, input.rank)
    depthwiseConvLoopFloat(
        transposed,
        input,
        filter,
        bias,
        output,
        padding,
        stride,
        dilation,
        kernel)
    return nil
}

func GroupedConv(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int,
        groups int) error {
    kernel := getConvKernelFloat(transposed, input.rank)
    groupedConvLoopFloat(
        transposed,
        input,
        filter,
        bias,
        output,
        padding,
        stride,
        dilation,
        groups,
        kernel)
    return nil
}

// implementation

type convKernelFloat func(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int)

var convKernelsNFloat = [...]convKernelFloat{
    convCoreN1Float,
    convCoreN2Float,
    convCoreN3Float,
}

var convKernelsTFloat = [...]convKernelFloat{
    convCoreT1Float,
    convCoreT2Float,
    convCoreT3Float,
}

func getConvKernelFloat(transposed bool, rank int) convKernelFloat {
    assert(rank >= 3 && rank <= 5)
    if transposed {
        return convKernelsTFloat[rank-3]
    } else {
        return convKernelsNFloat[rank-3]
    }
}

func convLoopFloat(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int,
        kernel convKernelFloat) {
    inputData := input.FloatData()
    filterData := filter.FloatData()
    biasData := bias.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    filterShape := filter.shape
    biasShape := bias.shape
    outputShape := output.shape
    if transposed {
        convBiasFloat(biasData, inputData, biasShape, inputShape)
    } else {
        convBiasFloat(biasData, outputData, biasShape, outputShape)
    }
    outputShape0 := outputShape[0]
    outputShape1 := outputShape[1]
    inputShape1 := inputShape[1]
    for b := 0; b < outputShape0; b++ {
        for z := 0; z < outputShape1; z++ {
            for c := 0; c < inputShape1; c++ {
                inputOffset := getConvOffset2(inputShape, b, c)
                filterOffset := getConvOffset2(filterShape, z, c)
                outputOffset := getConvOffset2(outputShape, b, z)
                kernel(
                    inputData[inputOffset:], 
                    filterData[filterOffset:],
                    outputData[outputOffset:],
                    inputShape[2:],
                    filterShape[2:],
                    outputShape[2:],
                    padding,
                    stride,
                    dilation)
            }
        }
    }
}

func depthwiseConvLoopFloat(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int,
        kernel convKernelFloat) {
    inputData := input.FloatData()
    filterData := filter.FloatData()
    biasData := bias.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    filterShape := filter.shape
    biasShape := bias.shape
    outputShape := output.shape
    if transposed {
        convBiasFloat(biasData, inputData, biasShape, inputShape)
    } else {
        convBiasFloat(biasData, outputData, biasShape, outputShape)
    }
    inputShape0 := inputShape[0]
    inputShape1 := inputShape[1]
    multiplier := outputShape[1] / inputShape1
    broadcast := (filterShape[0] == 1)
    for b := 0; b < inputShape0; b++ {
        for c := 0; c < inputShape1; c++ {
            for m := 0; m < multiplier; m++ {
                z := multiplier * c + m
                inputOffset := getConvOffset2(inputShape, b, c)
                filterOffset := 0
                if !broadcast {
                    filterOffset = getConvOffset2(filterShape, z, 0)
                }
                outputOffset := getConvOffset2(outputShape, b, z)
                kernel(
                    inputData[inputOffset:], 
                    filterData[filterOffset:],
                    outputData[outputOffset:],
                    inputShape[2:],
                    filterShape[2:],
                    outputShape[2:],
                    padding,
                    stride,
                    dilation)
            }
        }
    }
}

func groupedConvLoopFloat(
        transposed bool,
        input *Tensor,
        filter *Tensor,
        bias *Tensor,
        output *Tensor,
        padding []int,
        stride []int,
        dilation []int,
        groups int,
        kernel convKernelFloat) {
    inputData := input.FloatData()
    filterData := filter.FloatData()
    biasData := bias.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    filterShape := filter.shape
    biasShape := bias.shape
    outputShape := output.shape
    if transposed {
        convBiasFloat(biasData, inputData, biasShape, inputShape)
    } else {
        convBiasFloat(biasData, outputData, biasShape, outputShape)
    }
    inputShape0 := input.shape[0]
    inputBlock := input.shape[1] / groups
    outputBlock := output.shape[1] / groups
    for b := 0; b < inputShape0; b++ {
        for g := 0; g < groups; g++ {
            for z := 0; z < outputBlock; z++ {
                for c := 0; c < inputBlock; c++ {
                    inputOffset := getConvOffset2(inputShape, b, g*inputBlock+c)
                    filterOffset := getConvOffset2(filterShape, g*outputBlock+z, c)
                    outputOffset := getConvOffset2(outputShape, b, g*outputBlock+z)
                    kernel(
                        inputData[inputOffset:], 
                        filterData[filterOffset:],
                        outputData[outputOffset:],
                        inputShape[2:],
                        filterShape[2:],
                        outputShape[2:],
                        padding,
                        stride,
                        dilation)
                }
            }
        }
    }
}

func convBiasFloat(biasData []float32, tensorData []float32, biasShape []int, tensorShape []int) {
    biasVolume := volumeOf(biasShape)
    if biasVolume == 1 {
        tensorVolume := volumeOf(tensorShape)
        fillFloat(tensorData[:tensorVolume], biasData[0])
    } else {
        size := volumeOf(tensorShape[2:])
        shape0 := tensorShape[0]
        shape1 := tensorShape[1]
        offset := 0
        for b := 0; b < shape0; b++ {
            for c := 0; c < shape1; c++ {
                fillFloat(tensorData[offset:offset+size], biasData[c])
                offset += size
            }
        }
    }
}

func getConvOffset2(shape []int, index0 int, index1 int) int {
    return (index0 * shape[1] + index1) * volumeOf(shape[2:])
}

// kernels (normal)

func convCoreN1Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    inputShape0 := inputShape[0]
    filterShape0 := filterShape[0]
    outputShape0 := outputShape[0]
    stride0 := stride[0]
    padding0 := padding[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        sum := outputData[outputIndex]
        for filterIndex := 0; filterIndex < filterShape0; filterIndex++ {
            inputIndex := outputIndex * stride0 + filterIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                sum += inputData[inputIndex] * filterData[filterIndex]
            }
        }
        outputData[outputIndex] = sum
    }
}

func convCoreN2Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [2]int
    var outputLoop, filterLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        sum := outputData[outputOffset]
        for filterLoop.Start(filterShape); filterLoop.Test(); filterLoop.Next() {
            filterIndex := filterLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + filterIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + filterIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                filterOffset := NdOffset2(filterShape, filterIndex)
                sum += inputData[inputOffset] * filterData[filterOffset]
            }
        }
        outputData[outputOffset] = sum
    }
}

func convCoreN3Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [3]int
    var outputLoop, filterLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        sum := outputData[outputOffset]
        for filterLoop.Start(filterShape); filterLoop.Test(); filterLoop.Next() {
            filterIndex := filterLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + filterIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + filterIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + filterIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                filterOffset := NdOffset3(filterShape, filterIndex)
                sum += inputData[inputOffset] * filterData[filterOffset]
            }
        }
        outputData[outputOffset] = sum
    }
}

// kernels (transposed)

func convCoreT1Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    inputShape0 := inputShape[0]
    filterShape0 := filterShape[0]
    outputShape0 := outputShape[0]
    stride0 := stride[0]
    padding0 := padding[0]
    dilation0 := dilation[0]
    for outputIndex := 0; outputIndex < outputShape0; outputIndex++ {
        for filterIndex := 0; filterIndex < filterShape0; filterIndex++ {
            inputIndex := outputIndex * stride0 + filterIndex * dilation0 - padding0
            if inputIndex >= 0 && inputIndex < inputShape0 {
                inputData[inputIndex] += outputData[outputIndex] * filterData[filterIndex]
            }
        }
    }
}

func convCoreT2Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [2]int
    var outputLoop, filterLoop NdLoop2
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset2(outputShape, outputIndex)
        for filterLoop.Start(filterShape); filterLoop.Test(); filterLoop.Next() {
            filterIndex := filterLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + filterIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + filterIndex[1] * dilation[1] - padding[1]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] {
                inputOffset := NdOffset2(inputShape, inputIndex[:])
                filterOffset := NdOffset2(filterShape, filterIndex)
                inputData[inputOffset] += outputData[outputOffset] * filterData[filterOffset]
            }
        }
    }
}

func convCoreT3Float(
        inputData []float32,
        filterData []float32,
        outputData []float32,
        inputShape []int,
        filterShape[]int,
        outputShape []int,
        padding []int,
        stride []int,
        dilation []int) {
    var inputIndex [3]int
    var outputLoop, filterLoop NdLoop3
    for outputLoop.Start(outputShape); outputLoop.Test(); outputLoop.Next() {
        outputIndex := outputLoop.Index()
        outputOffset := NdOffset3(outputShape, outputIndex)
        for filterLoop.Start(filterShape); filterLoop.Test(); filterLoop.Next() {
            filterIndex := filterLoop.Index()
            inputIndex[0] = outputIndex[0] * stride[0] + filterIndex[0] * dilation[0] - padding[0]
            inputIndex[1] = outputIndex[1] * stride[1] + filterIndex[1] * dilation[1] - padding[1]
            inputIndex[2] = outputIndex[2] * stride[2] + filterIndex[2] * dilation[2] - padding[2]
            if inputIndex[0] >= 0 && inputIndex[0] < inputShape[0] &&
                    inputIndex[1] >= 0 && inputIndex[1] < inputShape[1] &&
                    inputIndex[2] >= 0 && inputIndex[2] < inputShape[2] {
                inputOffset := NdOffset3(inputShape, inputIndex[:])
                filterOffset := NdOffset3(filterShape, filterIndex)
                inputData[inputOffset] += outputData[outputOffset] * filterData[filterOffset]
            }
        }
    }
}

