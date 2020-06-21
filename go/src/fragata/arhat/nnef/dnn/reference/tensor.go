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

//
//    Tensor
//

type Tensor struct {
    dtype api.Dtype
    rank int
    volume int
    shape []int
    data interface{}
}

func NewTensor(dtype api.Dtype, shape []int) (*Tensor, error) {
    t := new(Tensor)
    t.Init(dtype, shape)
    return t, nil
}

func(t *Tensor) Init(dtype api.Dtype, shape []int) {
    t.dtype = dtype
    t.rank = len(shape)
    t.volume = volumeOf(shape)
    t.shape = cloneShape(shape)
    t.data = makeData(t.dtype, t.volume)
}

func(t *Tensor) Dtype() api.Dtype {
    return t.dtype
}

func(t *Tensor) Rank() int {
    return t.rank
}

func(t *Tensor) Volume() int {
    return t.volume
}

func(t *Tensor) Shape() []int {
    return t.shape
}

func(t *Tensor) BoolData() []bool {
    assert(t.dtype == api.DtypeBool)
    return t.data.([]bool)
}

func(t *Tensor) IntData() []int {
    assert(t.dtype == api.DtypeInt)
    return t.data.([]int)
}

func(t *Tensor) FloatData() []float32 {
    assert(t.dtype == api.DtypeFloat)
    return t.data.([]float32)
}

//
//    TensorView
//

type TensorView struct {
    rank int
    shape []int
    volume []int
    step []int
    data interface{}
    offset []int
}

func(t *TensorView) Init(tensor *Tensor, rank int) {
    var shape []int
    n := len(tensor.shape)
    if n == rank {
        shape = tensor.shape
    } else {
        shape = make([]int, rank)
        copy(shape, tensor.shape)
        for i := n; i < rank; i++ {
            shape[i] = 1
        }
    }
    volume := make([]int, rank)
    step := make([]int, rank)
    offset := make([]int, rank)
    prod := 1
    for i := rank - 1; i >= 0; i-- {
        dim := shape[i]
        prod *= dim
        volume[i] = prod
        if dim == 1 {
            step[i] = 0
        } else {
            step[i] = 1
        }
    }
    t.rank = rank
    t.shape = shape
    t.volume = volume
    t.step = step
    t.data = tensor.data    
    t.offset = offset
}

func(t *TensorView) Start(level int) {
    t.offset[level+1] = t.offset[level]
}

func(t *TensorView) Next(level int) {
    t.offset[level+1] += t.step[level] * t.volume[level+1]
}

