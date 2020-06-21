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
// Ported from C++ to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package comp

import (
    "fmt"
    "fragata/arhat/nnef/core"
)

//
//    ExprKind
//

type ExprKind int

const (
    ExprKindLiteral ExprKind = iota
    ExprKindIdentifier
    ExprKindArray
    ExprKindTuple
    ExprKindSubscript
    ExprKindComprehension
    ExprKindUnary
    ExprKindBinary
    ExprKindSelect
    ExprKindInvocation
    ExprKindBuiltin
)

//
//    Expr
//

type Expr interface {
    Kind() ExprKind
    Type() core.Type
    String() string
    Position() *core.Position
}

//
//    ExprBase
//

type ExprBase struct {
    position *core.Position
}

func(e *ExprBase) Init(position *core.Position) {
    e.position = position.Clone()
}

func(e *ExprBase) Position() *core.Position {
    return e.position
}

//
//    LiteralExpr
//

type LiteralExpr struct {
    ExprBase
    typ core.Type
}

func(e *LiteralExpr) Init(position *core.Position, typ core.Type) {
    e.ExprBase.Init(position)
    e.typ = typ
}

func(e *LiteralExpr) Kind() ExprKind {
    return ExprKindLiteral
}

func(e *LiteralExpr) Type() core.Type {
    return e.typ
}

//
//    ScalarExpr
//

type ScalarExpr struct {
    LiteralExpr
    value float32
}

func NewScalarExpr(position *core.Position, value float32, typ core.Type) *ScalarExpr {
    e := new(ScalarExpr)
    e.Init(position, value, typ)
    return e
}

func(e *ScalarExpr) Init(position *core.Position, value float32, typ core.Type) {
    e.LiteralExpr.Init(position, typ)
    e.value = value
}

func(e *ScalarExpr) Value() float32 {
    return e.value
}

func(e *ScalarExpr) String() string {
    return fmt.Sprintf("%g", e.value)
}

//
//    IntegerExpr
//

type IntegerExpr struct {
    LiteralExpr
    value int
}

func NewIntegerExpr(position *core.Position, value int, typ core.Type) *IntegerExpr {
    e := new(IntegerExpr)
    e.Init(position, value, typ)
    return e
}

func(e *IntegerExpr) Init(position *core.Position, value int, typ core.Type) {
    e.LiteralExpr.Init(position, typ)
    e.value = value
}

func(e *IntegerExpr) Value() int {
    return e.value
}

func(e *IntegerExpr) String() string {
    return fmt.Sprintf("%d", e.value)
}

//
//    LogicalExpr
//

type LogicalExpr struct {
    LiteralExpr
    value bool
}

func NewLogicalExpr(position *core.Position, value bool, typ core.Type) *LogicalExpr {
    e := new(LogicalExpr)
    e.Init(position, value, typ)
    return e
}

func(e *LogicalExpr) Init(position *core.Position, value bool, typ core.Type) {
    e.LiteralExpr.Init(position, typ)
    e.value = value
}

func(e *LogicalExpr) Value() bool {
    return e.value
}

func(e *LogicalExpr) String() string {
    return fmt.Sprintf("%t", e.value)
}

//
//    StringExpr
//

type StringExpr struct {
    LiteralExpr
    value string
}

func NewStringExpr(position *core.Position, value string, typ core.Type) *StringExpr {
    e := new(StringExpr)
    e.Init(position, value, typ)
    return e
}

func(e *StringExpr) Init(position *core.Position, value string, typ core.Type) {
    e.LiteralExpr.Init(position, typ)
    e.value = value
}

func(e *StringExpr) Value() string {
    return e.value
}

func(e *StringExpr) String() string {
    return fmt.Sprintf("'%s'", e.value)
}

//
//    IdentifierExpr
//

type IdentifierExpr struct {
    ExprBase
    name string
    typ core.Type
}

func NewIdentifierExpr(position *core.Position, name string, typ core.Type) *IdentifierExpr {
    e := new(IdentifierExpr)
    e.Init(position, name, typ)
    return e
}

func(e *IdentifierExpr) Init(position *core.Position, name string, typ core.Type) {
    e.ExprBase.Init(position)
    e.name = name
    e.typ = typ
}

func(e *IdentifierExpr) Kind() ExprKind {
    return ExprKindIdentifier
}

func(e *IdentifierExpr) Name() string {
    return e.name
}

func(e *IdentifierExpr) Type() core.Type {
    return e.typ
}

func(e *IdentifierExpr) String() string {
    return fmt.Sprintf("%s", e.name)
}

//
//    SubscriptExpr
//

type SubscriptExpr struct {
    ExprBase
    sequence Expr
    begin Expr
    end Expr
    typ core.Type
}

func NewSubscriptExpr(
        position *core.Position,
        sequence Expr,
        begin Expr,
        end Expr,
        typ core.Type) *SubscriptExpr {
    e := new(SubscriptExpr)
    e.Init(position, sequence, begin, end, typ)
    return e
}

func(e *SubscriptExpr) Init(
        position *core.Position,
        sequence Expr,
        begin Expr,
        end Expr,
        typ core.Type) {
    e.ExprBase.Init(position)
    e.sequence = sequence
    e.begin = begin
    e.end = end
    e.typ = typ
}

func(e *SubscriptExpr) Kind() ExprKind {
    return ExprKindSubscript
}

func(e *SubscriptExpr) IsRange() bool {
    return (e.begin != e.end && e.begin != nil)
}

func(e *SubscriptExpr) Sequence() Expr {
    return e.sequence
}

func(e *SubscriptExpr) Begin() Expr {
    return e.begin
}

func(e *SubscriptExpr) End() Expr {
    return e.end
}

func(e *SubscriptExpr) Type() core.Type {
    return e.typ
}

func(e *SubscriptExpr) String() string {
    s := e.sequence.String()
    s += "["
    if e.begin != nil {
        s += e.begin.String()
    }
    if  e.IsRange() {
        s += ":"
    }
    if e.end != nil {
        s += e.end.String()
    }
    s += "]"
    return s
}

//
//    ItemExpr
//

type ItemExpr struct {
    ExprBase
    items []Expr
    typ core.Type
}

func(e *ItemExpr) Init(position *core.Position, items []Expr, typ core.Type) {
    e.ExprBase.Init(position)
    e.items = items // no cloning, unique instance expected
    e.typ = typ
}

func(e *ItemExpr) Size() int {
    return len(e.items)
}

func(e *ItemExpr) At(idx int) Expr {
    return e.items[idx]
}

func(e *ItemExpr) Type() core.Type {
    return e.typ
}

//
//    ArrayExpr
//

type ArrayExpr struct {
    ItemExpr
}

func NewArrayExpr(position *core.Position, items []Expr, typ core.Type) *ArrayExpr {
    e := new(ArrayExpr)
    e.Init(position, items, typ)
    return e
}

func(e *ArrayExpr) Init(position *core.Position, items []Expr, typ core.Type) {
    e.ItemExpr.Init(position, items, typ)
}

func(e *ArrayExpr) Kind() ExprKind {
    return ExprKindArray
}

func(e *ArrayExpr) String() string {
    s := "["
    for i, item := range e.items {
        if i != 0 {
            s += ","
        }
        s += item.String()
    }
    s += "]"
    return s
}

//
//    TupleExpr
//

type TupleExpr struct {
    ItemExpr
}

func NewTupleExpr(position *core.Position, items []Expr, typ core.Type) *TupleExpr {
    e := new(TupleExpr)
    e.Init(position, items, typ)
    return e
}

func(e *TupleExpr) Init(position *core.Position, items []Expr, typ core.Type) {
    e.ItemExpr.Init(position, items, typ)
}

func(e *TupleExpr) Kind() ExprKind {
    return ExprKindTuple
}

func(e *TupleExpr) String() string {
    s := "("
    for i, item := range e.items {
        if i != 0 {
            s += ","
        }
        s += item.String()
    }
    s += ")"
    return s
}

//
//    ComprehensionExpr
//

type ComprehensionExpr struct {
    ExprBase
    iterators []Expr
    iterables []Expr
    condition Expr
    item Expr
    typ core.Type
}

func NewComprehensionExpr(
        position *core.Position,
        iterators []Expr,
        iterables []Expr,
        condition Expr,
        item Expr,
        typ core.Type) *ComprehensionExpr {
    e := new(ComprehensionExpr)
    e.Init(position, iterators, iterables, condition, item, typ)
    return e
}

func(e *ComprehensionExpr) Init(
        position *core.Position,
        iterators []Expr,
        iterables []Expr,
        condition Expr,
        item Expr,
        typ core.Type) {
    e.ExprBase.Init(position)
    e.iterators = iterators // no cloning, unique instance expected
    e.iterables = iterables // --"--
    e.condition = condition
    e.item = item
    e.typ = typ
}

func(e *ComprehensionExpr) Kind() ExprKind {
    return ExprKindComprehension
}

func(e *ComprehensionExpr) IteratorCount() int {
    return len(e.iterators)
}

func(e *ComprehensionExpr) Iterator(idx int) Expr {
    return e.iterators[idx]
}

func(e *ComprehensionExpr) Iterable(idx int) Expr {
    return e.iterables[idx]
}

func(e *ComprehensionExpr) Condition() Expr {
    return e.condition
}

func(e *ComprehensionExpr) Item() Expr {
    return e.item
}

func(e *ComprehensionExpr) Type() core.Type {
    return e.typ
}

func(e *ComprehensionExpr) String() string {
    s := "[for "
    size := len(e.iterators)
    for i := 0; i < size; i++ {
        if i != 0 {
            s += ", "
        }
        s += e.iterators[i].String()
        s += " in "
        s += e.iterables[i].String()
    }
    if e.condition != nil {
        s += " if "
        s += e.condition.String()
    }
    s += " yield "
    s += e.item.String()
    s += "]"
    return s
}

//
//    UnaryExpr
//

type UnaryExpr struct {
    ExprBase
    right Expr
    op core.Token
    typ core.Type
}

func NewUnaryExpr(
        position *core.Position, 
        right Expr, 
        op core.Token, 
        typ core.Type) *UnaryExpr {
    e := new(UnaryExpr)
    e.Init(position, right, op, typ)
    return e
}

func(e *UnaryExpr) Init(
        position *core.Position, 
        right Expr, 
        op core.Token, 
        typ core.Type) {
    e.ExprBase.Init(position)
    e.right = right
    e.op = op
    e.typ = typ
}

func(e *UnaryExpr) Kind() ExprKind {
    return ExprKindUnary
}

func(e *UnaryExpr) Right() Expr {
    return e.right
}

func(e *UnaryExpr) Op() core.Token {
    return e.op
}

func(e *UnaryExpr) Type() core.Type {
    return e.typ
}

func(e *UnaryExpr) String() string {
    s := e.op.String()
    par := (len(s) > 1)
    if par {
        s += "("
    }
    s += e.right.String()
    if par {
        s += ")"
    }
    return s
}

//
//    BinaryExpr
//

type BinaryExpr struct {
    ExprBase
    left Expr
    right Expr
    op core.Token
    typ core.Type
}

func NewBinaryExpr(
        position *core.Position, 
        left Expr,
        right Expr, 
        op core.Token, 
        typ core.Type) *BinaryExpr {
    e := new(BinaryExpr)
    e.Init(position, left, right, op, typ)
    return e
}

func(e *BinaryExpr) Init(
        position *core.Position, 
        left Expr,
        right Expr, 
        op core.Token, 
        typ core.Type) {
    e.ExprBase.Init(position)
    e.left = left
    e.right = right
    e.op = op
    e.typ = typ
}

func(e *BinaryExpr) Kind() ExprKind {
    return ExprKindBinary
}

func(e *BinaryExpr) Left() Expr {
    return e.left
}

func(e *BinaryExpr) Right() Expr {
    return e.right
}

func(e *BinaryExpr) Op() core.Token {
    return e.op
}

func(e *BinaryExpr) Type() core.Type {
    return e.typ
}

func(e *BinaryExpr) String() string {
    s := ""
    lpar := (e.left.Kind() == ExprKindBinary)
    if lpar {
        s += "("
    }
    s += e.left.String()
    if lpar {
        s += ")"
    }
    s += " "
    s += e.op.String()
    s += " "
    rpar := (e.right.Kind() == ExprKindBinary)
    if rpar {
        s += "("
    }
    s += e.right.String()
    if rpar {
        s += ")"
    }
    return s
}

//
//    BuiltinExpr
//

type BuiltinExpr struct {
    ExprBase
    arg Expr
    op core.Token
    typ core.Type
}

func NewBuiltinExpr(
        position *core.Position,
        arg Expr,
        op core.Token,
        typ core.Type) *BuiltinExpr {
    e := new(BuiltinExpr)
    e.Init(position, arg, op, typ)
    return e
}

func(e *BuiltinExpr) Init(
        position *core.Position,
        arg Expr,
        op core.Token,
        typ core.Type) {
    e.ExprBase.Init(position)
    e.arg = arg
    e.op = op
    e.typ = typ
}

func(e *BuiltinExpr) Kind() ExprKind {
    return ExprKindBuiltin
}

func(e *BuiltinExpr) Arg() Expr {
    return e.arg
}

func(e *BuiltinExpr) Op() core.Token {
    return e.op
}

func(e *BuiltinExpr) Type() core.Type {
    return e.typ
}

func(e *BuiltinExpr) String() string {
    s := e.op.String()
    s += "("
    s += e.arg.String()
    s += ")"
    return s
}

//
//    SelectExpr
//

type SelectExpr struct {
    ExprBase
    condition Expr
    trueValue Expr
    falseValue Expr
    typ core.Type
}

func NewSelectExpr(
        position *core.Position,
        condition Expr,
        trueValue Expr,
        falseValue Expr,
        typ core.Type) *SelectExpr {
    e := new(SelectExpr)
    e.Init(position, condition, trueValue, falseValue, typ)
    return e
}

func(e *SelectExpr) Init(
        position *core.Position,
        condition Expr,
        trueValue Expr,
        falseValue Expr,
        typ core.Type) {
    e.ExprBase.Init(position)
    e.condition = condition
    e.trueValue = trueValue
    e.falseValue = falseValue
    e.typ = typ
}

func(e *SelectExpr) Kind() ExprKind {
    return ExprKindSelect
}

func(e *SelectExpr) Condition() Expr {
    return e.condition
}

func(e *SelectExpr) TrueValue() Expr {
    return e.trueValue
}

func(e *SelectExpr) FalseValue() Expr {
    return e.falseValue
}

func(e *SelectExpr) Type() core.Type {
    return e.typ
}

func(e *SelectExpr) String() string {
    s := e.trueValue.String()
    s += " if "
    s += e.condition.String()
    s += " else "
    s += e.falseValue.String()
    return s
}

//
//    InvocationExpr
//

type InvocationExpr struct {
    ExprBase
    target string
    argNames []string
    argValues []Expr
    typ core.Type 
    dataType *core.PrimitiveType
}

func NewInvocationExpr(
        position *core.Position,
        target string,
        argNames []string,
        argValues []Expr,
        typ core.Type,
        dataType *core.PrimitiveType) *InvocationExpr {
    e := new(InvocationExpr)
    e.Init(position, target, argNames, argValues, typ, dataType)
    return e
}

func(e *InvocationExpr) Init(
        position *core.Position,
        target string,
        argNames []string,
        argValues []Expr,
        typ core.Type,
        dataType *core.PrimitiveType) {
    e.ExprBase.Init(position)
    e.target = target
    e.argNames = argNames   // no cloning, unique instance expected
    e.argValues = argValues // --"--
    e.typ = typ
    e.dataType = dataType
}

func(e *InvocationExpr) Kind() ExprKind {
    return ExprKindInvocation
}

func(e *InvocationExpr) Target() string {
    return e.target
}

func(e *InvocationExpr) ArgCount() int {
    return len(e.argNames)
}

func(e *InvocationExpr) ArgName(idx int) string {
    return e.argNames[idx]
}

func(e *InvocationExpr) ArgValue(idx int) Expr {
    return e.argValues[idx]
}

func(e *InvocationExpr) Arg(name string) Expr {
    n := len(e.argNames)
    for i := 0; i < n; i++ {
        if e.argNames[i] == name {
            return e.argValues[i]
        }
    }
    return nil
}

func(e *InvocationExpr) Type() core.Type {
    return e.typ
}

func(e *InvocationExpr) DataType() *core.PrimitiveType {
    return e.dataType
}

func(e *InvocationExpr) String() string {
    s := e.target
    if e.dataType != nil {
        s += "<"
        s += e.dataType.String()
        s += ">"
    }
    s += "("
    n := len(e.argNames)
    for i := 0; i < n; i++ {
        if i != 0 {
            s += ", "
        }
        s += e.argNames[i]
        s += " = "
        s += e.argValues[i].String()
    }
    s += ")"
    return s
}

