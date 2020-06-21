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

func Unary(op api.UnaryOp, x *Tensor, y *Tensor) error {
    kernel := getUnaryKernel(op)
    kernel(x.volume, x.data, y.data)
    return nil
}

// implementation

type unaryKernel func(n int, ax interface{}, ay interface{})

var unaryKernels = [...]unaryKernel{
    api.OpNeg: unaryNegFloat,
    api.OpNot: unaryNotBool,
    api.OpAbs: unaryAbsFloat,
    api.OpSign: unarySignFloat,
    api.OpExp: unaryExpFloat,
    api.OpLog: unaryLogFloat,
    api.OpLog2: unaryLog2Float,
    api.OpSin: unarySinFloat,
    api.OpCos: unaryCosFloat,
    api.OpRound: unaryRoundFloat,
    api.OpFloor: unaryFloorFloat,
    api.OpCeil: unaryCeilFloat,
    api.OpSqrt: unarySqrtFloat,
    api.OpSqr: unarySqrFloat,
    api.OpRsqrt: unaryRsqrtFloat,
    api.OpRsqr: unaryRsqrFloat,
    api.OpRcp: unaryRcpFloat,
    api.OpCopy: unaryCopyFloat,
    api.OpSigmoid: unarySigmoidFloat,
    api.OpTanh: unaryTanhFloat,
    api.OpRelu: unaryReluFloat,
    api.OpElu: unaryEluFloat,
    api.OpSoftplus: unarySoftplusFloat,
}

func getUnaryKernel(op api.UnaryOp) unaryKernel {
    return unaryKernels[op]
}

// kernels

func unaryNegFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = -x[i]
    }    
}

func unaryNotBool(n int, ax interface{}, ay interface{}) {
    x := ax.([]bool)
    y := ay.([]bool)
    for i := 0; i < n; i++ {
        y[i] = !x[i]
    }    
}

func unaryAbsFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = abs(x[i])
    }    
}

func unarySignFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = sign(x[i])
    }    
}

func unaryExpFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = exp(x[i])
    }    
}

func unaryLogFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = log(x[i])
    }    
}

func unaryLog2Float(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = log2(x[i])
    }    
}

func unarySinFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = sin(x[i])
    }    
}

func unaryCosFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = cos(x[i])
    }    
}

func unaryRoundFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = round(x[i])
    }    
}

func unaryFloorFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = floor(x[i])
    }    
}

func unaryCeilFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = ceil(x[i])
    }    
}

func unarySqrtFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = sqrt(x[i])
    }    
}

func unarySqrFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        t := x[i]
        y[i] = t * t
    }    
}

func unaryRsqrtFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = rsqrt(x[i])
    }    
}

func unaryRsqrFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        t := x[i]
        y[i] = float32(1.0) / (t * t)
    }    
}

func unaryRcpFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = float32(1.0) / x[i]
    }    
}

func unaryCopyFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = x[i]
    }    
}

func unarySigmoidFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = sigmoid(x[i])
    }    
}

func unaryTanhFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = tanh(x[i])
    }    
}

func unaryReluFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = relu(x[i])
    }    
}

func unaryEluFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = elu(x[i])
    }    
}

func unarySoftplusFloat(n int, ax interface{}, ay interface{}) {
    x := ax.([]float32)
    y := ay.([]float32)
    for i := 0; i < n; i++ {
        y[i] = softplus(x[i])
    }    
}

