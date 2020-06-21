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

import (
    "fmt"
    "fragata/arhat/nnef/dnn/api"
)

//
//    Utility functions
//

func cloneShape(shape []int) []int {
    size := len(shape)
    if size == 0 {
        return nil
    }
    result := make([]int, size)
    copy(result, shape)
    return result
}

func volumeOf(shape []int) int {
    result := 1
    for _, dim := range shape {
        result *= dim
    }
    return result
}

func makeData(dtype api.Dtype, volume int) interface{} {
    switch dtype {
    case api.DtypeBool:
        return make([]bool, volume)
    case api.DtypeInt:
        return make([]int, volume)
    case api.DtypeFloat:
        return make([]float32, volume)
    default:
        assert(false)
        return nil
    }
}

func fillBool(data []bool, value bool) {
    n := len(data)
    for i := 0; i < n; i++ {
        data[i] = value
    }
}

func fillInt(data []int, value int) {
    n := len(data)
    for i := 0; i < n; i++ {
        data[i] = value
    }
}

func fillFloat(data []float32, value float32) {
    n := len(data)
    for i := 0; i < n; i++ {
        data[i] = value
    }
}

func scaleFloat(data []float32, scale float32) {
    n := len(data)
    for i := 0; i < n; i++ {
        data[i] *= scale
    }
}

func clipInt(value int, low int, high int) int {
    if value < low {
        return low
    }
    if value > high {
        return high
    }
    return value
}

func btoi(b bool) int {
    if b {
        return 1
    } else {
        return 0
    }
}

func assert(cond bool) {
    if !cond {
        panic(fmt.Errorf("Assertion failed"))
    }
}

