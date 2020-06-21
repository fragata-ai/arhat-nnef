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

func Fill(tensor *Tensor, data interface{}) error {
    switch tensor.dtype {
    case api.DtypeBool:
        x := tensor.BoolData()
        switch v := data.(type) {
        case bool:
            fillBool(x, v)
        case []bool:
            copy(x, v)
        default:
            assert(false)
        }
    case api.DtypeInt:
        x := tensor.IntData()
        switch v := data.(type) {
        case int:
            fillInt(x, v)
        case []int:
            copy(x, v)
        default:
            assert(false)
        }
    case api.DtypeFloat:
        x := tensor.FloatData()
        switch v := data.(type) {
        case float32:
            fillFloat(x, v)
        case []float32:
            copy(x, v)
        default:
            assert(false)
        }
    default:
        assert(false)
    }
    return nil
}

func Read(tensor *Tensor, data interface{}) error {
    switch tensor.dtype {
    case api.DtypeBool:
        x := tensor.BoolData()
        v := data.([]bool)
        copy(v, x)
    case api.DtypeInt:
        x := tensor.IntData()
        v := data.([]int)
        copy(v, x)
    case api.DtypeFloat:
        x := tensor.FloatData()
        v := data.([]float32)
        copy(v, x)
    default:
        assert(false)
    }
    return nil
}

func Copy(input *Tensor, output *Tensor) error {
    switch output.dtype {
    case api.DtypeBool:
        copy(output.BoolData(), input.BoolData())
    case api.DtypeInt:
        copy(output.IntData(), input.IntData())
    case api.DtypeFloat:
        copy(output.FloatData(), input.FloatData())
    default:
        assert(false)
    }
    return nil
}

