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

func Binary(op api.BinaryOp, x *Tensor, y *Tensor, z *Tensor) error {
    var xv, yv, zv TensorView
    rank := z.rank
    xv.Init(x, rank)
    yv.Init(y, rank)
    zv.Init(z, rank)
    kernel := getBinaryKernel(op)
    binaryLoop(0, &xv, &yv, &zv, kernel)
    return nil
}

// binary loop

type binaryKernel func(
    n int, 
    ax interface{}, 
    px int, 
    dx int, 
    ay interface{}, 
    py int, 
    dy int, 
    az interface{},
    pz int,
    dz int)

var binaryKernels = [...]binaryKernel{
    api.OpAdd: binaryAddFloat,
    api.OpSub: binarySubFloat,
    api.OpMul: binaryMulFloat,
    api.OpDiv: binaryDivFloat,
    api.OpPow: binaryPowFloat,
    api.OpMin: binaryMinFloat,
    api.OpMax: binaryMaxFloat,
    api.OpAnd: binaryAndBool,
    api.OpOr: binaryOrBool,
    api.OpLt: binaryLtFloat,
    api.OpGt: binaryGtFloat,
    api.OpLe: binaryLeFloat,
    api.OpGe: binaryGeFloat,
    api.OpEq: binaryEqFloat,
    api.OpNe: binaryNeFloat,
}

func getBinaryKernel(op api.BinaryOp) binaryKernel {
    return binaryKernels[op]
}

func binaryLoop(level int, x *TensorView, y *TensorView, z *TensorView, kernel binaryKernel) {
    xVolume := x.volume[level]
    yVolume := y.volume[level]
    zVolume := z.volume[level]
    if (xVolume == zVolume || xVolume == 1) && (yVolume == zVolume || yVolume == 1) {
        dx := btoi(xVolume == zVolume)
        dy := btoi(yVolume == zVolume)
        kernel(
            zVolume, 
            x.data, 
            x.offset[level], 
            dx,
            y.data, 
            y.offset[level], 
            dy,
            z.data, 
            z.offset[level], 
            1)
    } else {
        assert(level + 1 < z.rank)
        x.Start(level)
        y.Start(level)
        z.Start(level)
        n := z.shape[level]
        for i := 0; i < n; i++ {
            binaryLoop(level+1, x, y, z, kernel) 
            x.Next(level)
            y.Next(level)
            z.Next(level)
        }
    }
}

// kernels

func binaryAddFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = x[i*dx+px] + y[i*dy+py]
    }
}

func binarySubFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = x[i*dx+px] - y[i*dy+py]
    }
}

func binaryMulFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = x[i*dx+px] * y[i*dy+py]
    }
}

func binaryDivFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = x[i*dx+px] / y[i*dy+py]
    }
}

func binaryPowFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = pow(x[i*dx+px], y[i*dy+py])
    }
}

func binaryMinFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = min(x[i*dx+px], y[i*dy+py])
    }
}

func binaryMaxFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = max(x[i*dx+px], y[i*dy+py])
    }
}

func binaryAndBool(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]bool)
    y := ay.([]bool)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] && y[i*dy+py])
    }
}

func binaryOrBool(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]bool)
    y := ay.([]bool)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] || y[i*dy+py])
    }
}

func binaryLtFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] < y[i*dy+py])
    }
}

func binaryGtFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] > y[i*dy+py])
    }
}

func binaryLeFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] <= y[i*dy+py])
    }
}

func binaryGeFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] >= y[i*dy+py])
    }
}

func binaryEqFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] == y[i*dy+py])
    }
}

func binaryNeFloat(
        n int, 
        ax interface{}, 
        px int, 
        dx int, 
        ay interface{}, 
        py int, 
        dy int, 
        az interface{}, 
        pz int, 
        dz int) {
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        z[i*dz+pz] = (x[i*dx+px] != y[i*dy+py])
    }
}

