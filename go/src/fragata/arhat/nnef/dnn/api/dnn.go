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
// Partly based on the code of The Khronos Group Inc. NNEF Tools.
// Ported from C++ to Go, modified and extended by FRAGATA COMPUTER SYSTEMS AG.
//

package api

//
//    Dtype
//

type Dtype int

const (
    DtypeBool Dtype = iota
    DtypeInt
    DtypeFloat
)

//
//    UnaryOp
//

type UnaryOp int

const (
    OpNeg UnaryOp = iota
    OpNot
    OpAbs
    OpSign
    OpExp
    OpLog
    OpLog2
    OpSin
    OpCos
    OpRound
    OpFloor
    OpCeil
    OpSqrt
    OpSqr
    OpRsqrt
    OpRsqr
    OpRcp
    OpCopy
    OpSigmoid
    OpTanh
    OpRelu
    OpElu
    OpSoftplus
)

//
//    BinaryOp
//

type BinaryOp int

const (
    OpAdd BinaryOp = iota
    OpSub
    OpMul
    OpDiv
    OpPow
    OpMin
    OpMax
    OpAnd
    OpOr
    OpLt
    OpGt
    OpLe
    OpGe
    OpEq
    OpNe
)

//
//    ReduceOp
//

type ReduceOp int

const (
    OpSumReduce ReduceOp = iota
    OpMeanReduce
    OpMinReduce
    OpMaxReduce
    OpAnyReduce
    OpAllReduce
)

//
//    ArgReduceOp
//

type ArgReduceOp int

const (
    OpArgminReduce ArgReduceOp = iota
    OpArgmaxReduce
)

//
//    PoolOp
//

type PoolOp int

const (
    OpSumPool PoolOp = iota
    OpAvgPool
    OpMaxPool
)

//
//    Tensor
//

type Tensor interface {
    Dtype() Dtype
    Rank() int
    Volume() int
    Shape() []int
}

//
//    Engine
//

type Engine interface {
    NewTensor(dtype Dtype, shape []int) (Tensor, error)
    Fill(tensor Tensor, data interface{}) error
    Read(tensor Tensor, data interface{}) error
    Copy(input Tensor, output Tensor) error
    Unary(op UnaryOp, x Tensor, y Tensor) error
    Binary(op BinaryOp, x Tensor, y Tensor, z Tensor) error
    Reduce(op ReduceOp, input Tensor, output Tensor) error
    Select(c Tensor, x Tensor, y Tensor, z Tensor) error
    Conv(
        transposed bool,
        input Tensor,
        filter Tensor,
        bias Tensor,
        output Tensor,
        padding []int,
        stride []int,
        dilation []int) error
    DepthwiseConv(
        transposed bool,
        input Tensor,
        filter Tensor,
        bias Tensor,
        output Tensor,
        padding []int,
        stride []int,
        dilation []int) error
    GroupedConv(
        transposed bool,
        input Tensor,
        filter Tensor,
        bias Tensor,
        output Tensor,
        padding []int,
        stride []int,
        dilation []int,
        groups int) error
    Pool(
        op PoolOp,
        transposed bool,
        input Tensor,
        output Tensor,
        size []int,
        padding []int,
        stride []int,
        dilation []int,
        includeBorder bool) error
    Matmul(trA bool, trB bool, a Tensor, b Tensor, c Tensor) error
    Linear(input Tensor, filter Tensor, bias Tensor, output Tensor) error
    Softmax(input Tensor, output Tensor, axis int) error
    ArgReduce(op ArgReduceOp, input Tensor, output Tensor, axis int) error
    Transpose(input Tensor, output Tensor, perm []int) error
    Concat(singular bool, x []Tensor, y Tensor, axis int) error
    Split(singular bool, x Tensor, y []Tensor, axis int) error
    PadConstant(input Tensor, output Tensor, padding []int, value interface{}) error
    PadReplicate(input Tensor, output Tensor, padding []int) error
    Tile(input Tensor, output Tensor) error
    Slice(input Tensor, output Tensor, offset []int) error
}

