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

func Split(singular bool, x *Tensor, y []*Tensor, axis int) error {
    splitLoop(singular, x, y, axis)
    return nil
}

// implementation

type splitKernel func(n int, ax interface{}, px int, ay interface{}, py int)

func splitLoop(singular bool, x *Tensor, y []*Tensor, axis int) {
    xShape := x.shape
    b := volumeOf(xShape[:axis])
    m := volumeOf(xShape[axis+1:])
    kernel := getSplitKernel(x.dtype)
    n := len(y)
    size := m // singular
    px := 0
    for i := 0; i < b; i++ {
        for j := 0; j < n; j++ {
            if !singular {
                size = y[j].shape[axis]
            }
            kernel(size, x.data, px, y[j].data, i*size)
            px += size
        }
    }
}

func getSplitKernel(dtype api.Dtype) splitKernel {
    switch dtype {
    case api.DtypeBool:
        return splitBool
    case api.DtypeInt:
        return splitInt
    case api.DtypeFloat:
        return splitFloat
    default:
        assert(false)
        return nil
    }
}

// kernels

func splitBool(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]bool)
    y := ay.([]bool)
    copy(y[py:py+n], x[px:])
}

func splitInt(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]int)
    y := ay.([]int)
    copy(y[py:py+n], x[px:])
}

func splitFloat(n int, ax interface{}, px int, ay interface{}, py int) {
    x := ax.([]float32)
    y := ay.([]float32)
    copy(y[py:py+n], x[px:])
}

