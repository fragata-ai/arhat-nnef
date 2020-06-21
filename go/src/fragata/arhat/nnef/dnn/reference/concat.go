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

func Concat(singular bool, x []*Tensor, y *Tensor, axis int) error {
    concatLoop(singular, x, y, axis)
    return nil
}

// implementation

type concatKernel func(n int, ax interface{}, px int, ay interface{}, py int)

func concatLoop(singular bool, x []*Tensor, y *Tensor, axis int) {
    yShape := y.shape
    b := volumeOf(yShape[:axis])
    m := volumeOf(yShape[axis+1:])
    kernel := getConcatKernel(y.dtype)
    n := len(x)
    size := m // singular
    py := 0
    for i := 0; i < b; i++ {
        for j := 0; j < n; j++ {
            if !singular {
                size = x[j].shape[axis] * m 
            }
            kernel(size, x[j].data, i*size, y.data, py)
            py += size
        }
    }
}

func getConcatKernel(dtype api.Dtype) concatKernel {
    switch dtype {
    case api.DtypeBool:
        return concatBool
    case api.DtypeInt:
        return concatInt
    case api.DtypeFloat:
        return concatFloat
    default:
        assert(false)
        return nil
    }
}

// kernels

func concatBool(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]bool)
    y := ay.([]bool)
    copy(y[py:py+n], x[px:])
}

func concatInt(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]int)
    y := ay.([]int)
    copy(y[py:py+n], x[px:])
}

func concatFloat(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]float32)
    y := ay.([]float32)
    copy(y[py:py+n], x[px:])
}

