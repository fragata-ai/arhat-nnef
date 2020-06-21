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

type selectKernel func(
        n int,
        ac interface{},
        pc int,
        dc int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int,
        az interface{},
        pz int,
        dz int)

func Select(c *Tensor, x *Tensor, y *Tensor, z *Tensor) error {
    var cv, xv, yv, zv TensorView
    rank := z.rank
    cv.Init(c, rank)
    xv.Init(x, rank)
    yv.Init(y, rank)
    zv.Init(z, rank)
    var kernel selectKernel
    switch z.dtype {
    case api.DtypeBool:
        kernel = selectBool
    case api.DtypeInt:
        kernel = selectInt
    case api.DtypeFloat:
        kernel = selectFloat
    default:
        assert(false)
    }
    selectLoop(0, &cv, &xv, &yv, &zv, kernel)
    return nil
}

func selectLoop(
        level int,
        c *TensorView, 
        x *TensorView, 
        y *TensorView, 
        z *TensorView, 
        kernel selectKernel) {
    cVolume := c.volume[level]
    xVolume := x.volume[level]
    yVolume := y.volume[level]
    zVolume := z.volume[level]
    if (cVolume == zVolume || cVolume == 1) && 
            (xVolume == zVolume || xVolume == 1) && 
            (yVolume == zVolume || yVolume == 1) {
        dc := btoi(cVolume == zVolume)
        dx := btoi(xVolume == zVolume)
        dy := btoi(yVolume == zVolume)
        kernel(
            zVolume, 
            c.data, 
            c.offset[level], 
            dc, 
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
        c.Start(level)
        x.Start(level)
        y.Start(level)
        z.Start(level)
        n := z.shape[level]
        for i := 0; i < n; i++ {
            selectLoop(level+1, c, x, y, z, kernel) 
            c.Next(level)
            x.Next(level)
            y.Next(level)
            z.Next(level)
        }
    }
}

func selectBool(
        n int,
        ac interface{},
        pc int,
        dc int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int,
        az interface{},
        pz int,
        dz int) {
    c := ac.([]bool)
    x := ax.([]bool)
    y := ay.([]bool)
    z := az.([]bool)
    for i := 0; i < n; i++ {
        if c[i*dc+pc] {
            z[i*dz+pz] = x[i*dx+pz]
        } else {
            z[i*dz+pz] = y[i*dy+py]
        }
    }
}

func selectInt(
        n int,
        ac interface{},
        pc int,
        dc int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int,
        az interface{},
        pz int,
        dz int) {
    c := ac.([]bool)
    x := ax.([]int)
    y := ay.([]int)
    z := az.([]int)
    for i := 0; i < n; i++ {
        if c[i*dc+pc] {
            z[i*dz+pz] = x[i*dx+pz]
        } else {
            z[i*dz+pz] = y[i*dy+py]
        }
    }
}

func selectFloat(
        n int,
        ac interface{},
        pc int,
        dc int,
        ax interface{},
        px int,
        dx int,
        ay interface{},
        py int,
        dy int,
        az interface{},
        pz int,
        dz int) {
    c := ac.([]bool)
    x := ax.([]float32)
    y := ay.([]float32)
    z := az.([]float32)
    for i := 0; i < n; i++ {
        if c[i*dc+pc] {
            z[i*dz+pz] = x[i*dx+pz]
        } else {
            z[i*dz+pz] = y[i*dy+py]
        }
    }
}

