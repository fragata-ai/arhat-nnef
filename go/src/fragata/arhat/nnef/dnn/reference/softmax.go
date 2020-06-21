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

func Softmax(input *Tensor, output *Tensor, axis int) error {
    softmaxLoopFloat(input, output, axis)
    return nil
}

// implementation

func softmaxLoopFloat(input *Tensor, output *Tensor, axis int) {
    inputData := input.FloatData()
    outputData := output.FloatData()
    inputShape := input.shape
    batch := volumeOf(inputShape[:axis])
    channels := inputShape[axis]
    size := volumeOf(inputShape[axis+1:])
    volume := channels * size
    for i := 0; i < batch; i++ {
        for j := 0; j < size; j++ {
            offset := volume * i + j
            softmaxFloat(channels, size, inputData[offset:], outputData[offset:])
        }
    }
}

func softmaxFloat(n int, m int, x []float32, y []float32) {
    xmax := x[0]
    for i := 1; i < n; i++ {
        xmax = max(xmax, x[i*m])
    }
    ysum := float32(0)
    for i := 0; i < n; i++ {
        yval := exp(x[i*m]-xmax)
        y[i*m] = yval
        ysum += yval
    }
    for i := 0; i < n; i++ {
        y[i*m] /= ysum
    }
}

