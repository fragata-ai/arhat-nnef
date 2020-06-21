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

func Matmul(trA bool, trB bool, a *Tensor, b *Tensor, c *Tensor) error {
    matmulLoopFloat(trA, trB, a, b, c)
    return nil
}

func Linear(input *Tensor, filter *Tensor, bias *Tensor, output *Tensor) error {
    linearFloat(input, filter, bias, output)
    return nil
}

// implementation

type matmulKernelFloat func(m int, n int, k int, a []float32, b []float32, c []float32)

func matmulLoopFloat(trA bool, trB bool, a *Tensor, b *Tensor, c *Tensor) {
    aData := a.FloatData()
    bData := b.FloatData()
    cData := c.FloatData()
    aShape := a.shape
    bShape := b.shape
    cShape := c.shape
    fillFloat(cData, float32(0.0))
    offset := c.rank - 2
    dA := aShape[offset] * aShape[offset+1]
    dB := bShape[offset] * bShape[offset+1]
    dC := cShape[offset] * cShape[offset+1]
    m := cShape[offset]
    n := cShape[offset+1]
    var k int
    if trA {
        k = aShape[offset]
    } else {
        k = aShape[offset+1]
    }
    kernel := getMatmulKernelFloat(trA, trB)
    v := volumeOf(cShape[:offset])
    for i := 0; i < v; i++ {
        kernel(m, n, k, aData[dA*i:], bData[dB*i:], cData[dC*i:])
    }
}

func getMatmulKernelFloat(trA bool, trB bool) matmulKernelFloat {
    if trA {
        if trB {
            return matmulTTFloat
        } else {
            return matmulTNFloat
        }
    } else {
        if trB {
            return matmulNTFloat
        } else {
            return matmulNNFloat
        }
    }
}

func linearFloat(input *Tensor, filter *Tensor, bias *Tensor, output *Tensor) {
    inputData := input.FloatData()
    filterData := filter.FloatData()
    biasData := bias.FloatData()
    outputData := output.FloatData()
    m := output.shape[0]
    n := output.shape[1]
    k := input.shape[1]
    if bias.volume == 1 {
        fillFloat(outputData, biasData[0])
    } else {
        for i := 0; i < m; i++ {
            copy(outputData[i*n:(i+1)*n], biasData)
        }
    }
    matmulNTFloat(m, n, k, inputData, filterData, outputData)
}

// kernels

func matmulNNFloat(m int, n int, k int, a []float32, b []float32, c []float32) {
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            ij := i * n + j
            x := c[ij]
            for l := 0; l < k; l++ {
                x += a[i*k+l] * b[l*n+j]
            }
            c[ij] = x
        }
    }
}

func matmulNTFloat(m int, n int, k int, a []float32, b []float32, c []float32) {
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            ij := i * n + j
            x := c[ij]
            for l := 0; l < k; l++ {
                x += a[i*k+l] * b[j*k+l]
            }
            c[ij] = x
        }
    }
}

func matmulTNFloat(m int, n int, k int, a []float32, b []float32, c []float32) {
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            ij := i * n + j
            x := c[ij]
            for l := 0; l < k; l++ {
                x += a[l*m+i] * b[l*n+j]
            }
            c[ij] = x
        }
    }
}

func matmulTTFloat(m int, n int, k int, a []float32, b []float32, c []float32) {
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            ij := i * n + j
            x := c[ij]
            for l := 0; l < k; l++ {
                x += a[l*m+i] * b[j*k+l]
            }
            c[ij] = x
        }
    }
}

