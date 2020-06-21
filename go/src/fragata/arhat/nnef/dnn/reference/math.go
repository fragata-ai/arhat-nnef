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
    "math"
)

//
//    Constants
//

var (
    posInf = float32(math.Inf(1))
    negInf = float32(math.Inf(-1))
)

//
//    Unary math functions
//

func abs(x float32) float32 {
    return float32(math.Abs(float64(x)))
}

func sign(x float32) float32 {
    if x > float32(0.0) {
        return float32(1.0)
    }
    if x < float32(0.0) {
        return float32(-1.0)
    }
    return float32(0.0)
}

func exp(x float32) float32 {
    return float32(math.Exp(float64(x)))
}

func log(x float32) float32 {
    return float32(math.Log(float64(x)))
}

func log2(x float32) float32 {
    return float32(math.Log2(float64(x)))
}

func sin(x float32) float32 {
    return float32(math.Sin(float64(x)))
}

func cos(x float32) float32 {
    return float32(math.Cos(float64(x)))
}

func round(x float32) float32 {
    return float32(math.Round(float64(x)))
}

func floor(x float32) float32 {
    return float32(math.Floor(float64(x)))
}

func ceil(x float32) float32 {
    return float32(math.Ceil(float64(x)))
}

func sqrt(x float32) float32 {
    return float32(math.Sqrt(float64(x)))
}

func rsqrt(x float32) float32 {
    return float32(1.0) / float32(math.Sqrt(float64(x)))
}

func sigmoid(x float32) float32 {
    return float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-x))))
}

func tanh(x float32) float32 {
    return float32(math.Tanh(float64(x)))
}

func relu(x float32) float32 {
    return float32(math.Max(float64(x), 0.0))
}

func elu(x float32) float32 {
    if x < float32(0.0) {
        return float32(math.Exp(float64(x))) - float32(1.0)
    }
    return x
}

func softplus(x float32) float32 {
    return float32(math.Log(math.Exp(float64(x))+1.0))
}

//
//    Binary math functions
//

func pow(x float32, y float32) float32 {
    return float32(math.Pow(float64(x), float64(y)))
}

func min(x float32, y float32) float32 {
    if x <= y {
        return x
    } else {
        return y
    }
}

func max(x float32, y float32) float32 {
    if x >= y {
        return x
    } else {
        return y
    }
}

//
//    Diagnostic functions
//

func checkFinite(tag string, data []float32) {
    posInfCount := 0
    negInfCount := 0
    nanCount := 0
    for _, value := range data {
        switch {
        case math.IsInf(float64(value), 1):
            posInfCount++
        case math.IsInf(float64(value), -1):
            negInfCount++
        case math.IsNaN(float64(value)):
            nanCount++
        }
    }
    fmt.Printf("%s: +Inf %d, -Inf %d, NaN %d out of %d\n", 
        tag, posInfCount, negInfCount, nanCount, len(data))
}

