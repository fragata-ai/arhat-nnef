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
    "math"
    "strconv"
    "strings"
    "fragata/arhat/nnef/core"
)

//
//    binary operations
//

type binaryArithmeticOp struct {
    integer func(x int, y int) int
    scalar func(x float32, y float32) float32
}

var (
    plus = &binaryArithmeticOp{
        func(x int, y int) int { return x + y },
        func(x float32, y float32) float32 { return x + y },
    }
    minus = &binaryArithmeticOp{
        func(x int, y int) int { return x - y },
        func(x float32, y float32) float32 { return x - y },
    }
    multiplies = &binaryArithmeticOp{
        func(x int, y int) int { return x * y },
        func(x float32, y float32) float32 { return x * y },
    }
    divides = &binaryArithmeticOp{
        func(x int, y int) int { return x / y },
        func(x float32, y float32) float32 { return x / y },
    }
    power = &binaryArithmeticOp{
        func(x int, y int) int { return int(math.Pow(float64(x), float64(y))) },
        func(x float32, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) },
    }
)

type binaryComparisonOp struct {
    integer func(x int, y int) bool
    scalar func(x float32, y float32) bool
}

var (
    less = &binaryComparisonOp{
        func(x int, y int) bool { return (x < y) },
        func(x float32, y float32) bool { return (x < y) },
    }
    greater = &binaryComparisonOp{
        func(x int, y int) bool { return (x > y) },
        func(x float32, y float32) bool { return (x > y) },
    }
    lessEqual = &binaryComparisonOp{
        func(x int, y int) bool { return (x <= y) },
        func(x float32, y float32) bool { return (x <= y) },
    }
    greaterEqual = &binaryComparisonOp{
        func(x int, y int) bool { return (x >= y) },
        func(x float32, y float32) bool { return (x >= y) },
    }
    equalTo = &binaryComparisonOp{
        func(x int, y int) bool { return (x == y) },
        func(x float32, y float32) bool { return (x == y) },
    }
    notEqualTo = &binaryComparisonOp{
        func(x int, y int) bool { return (x != y) },
        func(x float32, y float32) bool { return (x != y) },
    }
)

//
//    Evaluation
//

type Evaluation struct {
    fragments map[string]*Fragment
    lowered map[string]bool
    tensorCounts map[string]int
    reservedIds map[string]bool
}

func NewEvaluation(
        assignments []*Assignment, 
        fragments map[string]*Fragment, 
        lowered map[string]bool) *Evaluation {
    e := new(Evaluation)
    e.Init(assignments, fragments, lowered)
    return e
}

func(e *Evaluation) Init(
        assignments []*Assignment, 
        fragments map[string]*Fragment, 
        lowered map[string]bool) {
    e.fragments = fragments // immutable: store references
    e.lowered = lowered     // --"--
    e.tensorCounts = make(map[string]int)
    e.reservedIds = make(map[string]bool)
    for _, assignment := range assignments {
        e.addReservedIdentifiers(assignment.Lhs())
    }
}

func EvaluateLvalue(expr Expr, values map[string]core.Value, fallbackToIds bool) core.Value {
    switch expr.Kind() {
    case ExprKindIdentifier:
        identifier := expr.(*IdentifierExpr)
        name := identifier.Name()
        if value, ok := values[name]; ok {
            return value
        }
        if fallbackToIds {
            return core.NewIdentifierValue(name)
        }
        return core.None()
    case ExprKindArray:
        array := expr.(*ArrayExpr)
        size := array.Size()
        items := make(core.Items, size)
        for i := 0; i < size; i++ {
            items[i] = EvaluateLvalue(array.At(i), values, fallbackToIds)
        }
        return core.NewArrayValue(items, false)
    case ExprKindTuple:
        tuple := expr.(*TupleExpr)
        size := tuple.Size()
        items := make(core.Items, size)
        for i := 0; i < size; i++ {
            items[i] = EvaluateLvalue(tuple.At(i), values, fallbackToIds)
        }
        return core.NewTupleValue(items, false)
    default:
        core.Assert(false)
        return core.None()
    }
}

func EvaluateRvalue(expr Expr) core.Value {
    switch expr.Kind() {
    case ExprKindLiteral:
        return evaluateLiteral(expr)
    case ExprKindArray:
        array := expr.(*ArrayExpr)
        size := array.Size()
        items := make(core.Items, size)
        for i := 0; i < size; i++ {
            items[i] = EvaluateRvalue(array.At(i))
        }
        return core.NewArrayValue(items, false)
    case ExprKindTuple:
        tuple := expr.(*TupleExpr)
        size := tuple.Size()
        items := make(core.Items, size)
        for i := 0; i < size; i++ {
            items[i] = EvaluateRvalue(tuple.At(i))
        }
        return core.NewTupleValue(items, false)
    case ExprKindUnary:
        unary := expr.(*UnaryExpr)
        if unary.Op() == '-' {
            arg := EvaluateRvalue(unary.Right())
            switch arg.Kind() {
            case core.ValueKindInteger:
                return core.NewIntegerValue(-arg.Integer())
            case core.ValueKindScalar:
                return core.NewScalarValue(-arg.Scalar())
            }
        }
        fallthrough
    default:
        core.Assert(false)
        return core.None()
    }
}

func(e *Evaluation) EvaluateAssign(
        lhs Expr,
        rhs Expr, 
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) {
    value := e.evaluate(rhs, values, dtypes, callback, dtype, context)
    e.assign(lhs, value, values, dtypes, callback)
}

func(e *Evaluation) evaluate(
        expr Expr, 
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    switch expr.Kind() {
    case ExprKindLiteral:
        return evaluateLiteral(expr)
    case ExprKindIdentifier:
        return evaluateIdentifier(expr.(*IdentifierExpr), values)
    case ExprKindArray:
        return e.evaluateArray(expr.(*ArrayExpr), values, dtypes, callback, dtype, context)
    case ExprKindTuple:
        return e.evaluateTuple(expr.(*TupleExpr), values, dtypes, callback, dtype, context)
    case ExprKindSubscript:
        return e.evaluateSubscript(expr.(*SubscriptExpr), values, dtypes, callback, dtype)
    case ExprKindUnary:
        return e.evaluateUnary(expr.(*UnaryExpr), values, dtypes, callback, dtype)
    case ExprKindBinary:
        return e.evaluateBinary(expr.(*BinaryExpr), values, dtypes, callback, dtype)
    case ExprKindSelect:
        return e.evaluateSelect(expr.(*SelectExpr), values, dtypes, callback, dtype, context)
    case ExprKindComprehension:
        return e.evaluateComprehension(
            expr.(*ComprehensionExpr), values, dtypes, callback, dtype, context)
    case ExprKindBuiltin:
        return e.evaluateBuiltin(expr.(*BuiltinExpr), values, dtypes, callback, dtype)
    case ExprKindInvocation:
        return e.evaluateInvocation(
            expr.(*InvocationExpr), values, dtypes, callback, dtype, context)
    default:
        core.Assert(false)
        return core.None()
    }
}

func evaluateLiteral(expr Expr) core.Value {
    typ := expr.Type().(*core.PrimitiveType)
    switch typ.Name() {
    case core.TypenameInteger:
        return evaluateInteger(expr.(*IntegerExpr))
    case core.TypenameScalar:
        return evaluateScalar(expr.(*ScalarExpr))
    case core.TypenameLogical:
        return evaluateLogical(expr.(*LogicalExpr))
    case core.TypenameString:
        return evaluateString(expr.(*StringExpr))
    default:
        core.Assert(false)
        return core.None()
    }
}

func evaluateScalar(scalar *ScalarExpr) core.Value {
    return core.NewScalarValue(scalar.Value())
}

func evaluateInteger(integer *IntegerExpr) core.Value {
    return core.NewIntegerValue(integer.Value())
}

func evaluateLogical(logical *LogicalExpr) core.Value {
    return core.NewLogicalValue(logical.Value())
}

func evaluateString(str *StringExpr) core.Value {
    return core.NewStringValue(str.Value())
}

func evaluateIdentifier(
        identifier *IdentifierExpr, values map[string]core.Value) core.Value {
    name := identifier.Name()
    value, ok := values[name]
    if !ok {
        core.RaiseError(identifier.Position(), "undefined identifier '%s'", name)
    }
    return value
}

func(e *Evaluation) evaluateArray(
        array *ArrayExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    size := array.Size()
    items := make(core.Items, size)
    for i := 0; i < size; i++ {
        var ctx core.Value
        if context.Kind() == core.ValueKindArray {
            ctx = context.At(i)
        } else {
            ctx = core.None()
        }
        items[i] = e.evaluate(array.At(i), values, dtypes, callback, dtype, ctx)
    }
    return core.NewArrayValue(items, false)
}
        
func(e *Evaluation) evaluateTuple(
        tuple *TupleExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    size := tuple.Size()
    items := make(core.Items, size)
    for i := 0; i < size; i++ {
        var ctx core.Value
        if context.Kind() == core.ValueKindTuple {
            ctx = context.At(i)
        } else {
            ctx = core.None()
        }
        items[i] = e.evaluate(tuple.At(i), values, dtypes, callback, dtype, ctx)
    }
    return core.NewTupleValue(items, false)
}

func(e *Evaluation) evaluateSubscript(
        subscript *SubscriptExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType) core.Value {
    sequence := e.evaluate(subscript.Sequence(), values, dtypes, callback, dtype, core.None())
    if subscript.IsRange() {
        size := sequence.Size()
        var i int
        begin := subscript.Begin()
        if begin != nil {
            i = e.evaluate(begin, values, dtypes, callback, dtype, core.None()).Integer()
        } else {
            i = 0
        }
        if i < 0 {
            i += size
        }
        if i < 0 || i > size {
            core.RaiseError(
                subscript.Position(), 
                "range begin (%d) out of bounds (size = %d)", 
                i, 
                size)
        }
        var j int
        end := subscript.End()
        if end != nil {
            j = e.evaluate(end, values, dtypes, callback, dtype, core.None()).Integer()
        } else {
            j = size
        }
        if j < 0 {
            j += size
        }
        if j < 0 || j > size {
            core.RaiseError(
                subscript.Position(), 
                "range end (%d) out of bounds (size = %d)", 
                j, 
                size)
        }
        if j < i {
            core.RaiseError(subscript.Position(), "invalid range: %d:%d", i, j)
        }
        if sequence.Kind() == core.ValueKindString {
            return core.NewStringValue(sequence.String()[i:j])
        } else {
            items := sequence.Items()[i:j]
            return core.NewArrayValue(items, true)
        }
    } else {
        size := sequence.Size()
        begin := subscript.Begin()
        index := e.evaluate(begin, values, dtypes, callback, dtype, core.None()).Integer()
        if index < 0 {
            index += size
        }
        if index < 0 || index >= size {
            core.RaiseError(
                subscript.Position(), 
                "index (%d) out of bounds (size = %d)", 
                index, 
                size)
        }
        if sequence.Kind() == core.ValueKindString {
            return core.NewStringValue(sequence.String()[index:index+1])
        } else {
            return sequence.At(index)
        }
    }
}

func(e *Evaluation) evaluateUnary(
        unary *UnaryExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType) core.Value {
    right := e.evaluate(unary.Right(), values, dtypes, callback, dtype, core.None())
    switch unary.Op() {
    case '!':
        if right.Kind() == core.ValueKindLogical {
            return core.NewLogicalValue(!right.Logical())
        }
    case '-':
         switch right.Kind() {
         case core.ValueKindInteger:
             return core.NewIntegerValue(-right.Integer())
         case core.ValueKindScalar:
             return core.NewScalarValue(-right.Scalar())
        }
    case '+':
        return right
    }
    core.Assert(false)
    return core.None()
}

func(e *Evaluation) evaluateBinary(
        binary *BinaryExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType) core.Value {
    op := binary.Op()
    lazy := (op == core.TokenAnd || op == core.TokenOr)
    left := e.evaluate(binary.Left(), values, dtypes, callback, dtype, core.None())
    var right core.Value
    if lazy {
        right = core.None()
    } else {
        right = e.evaluate(binary.Right(), values, dtypes, callback, dtype, core.None())
    }
    switch op {
    case '+':
        if left.Kind() == core.ValueKindString && right.Kind() == core.ValueKindString {
            return core.NewStringValue(left.String()+right.String())
        }
        if left.Kind() == core.ValueKindArray && right.Kind() == core.ValueKindArray {
            items := concatItems(left.Array(), right.Array())
            return core.NewArrayValue(items, false)
        }
        return evaluateBinaryArithmeticOp(plus, left, right)
    case '*':
        if left.Kind() == core.ValueKindString && right.Kind() == core.ValueKindInteger {
             str := strings.Repeat(left.String(), right.Integer())
             return core.NewStringValue(str)
        }
        if left.Kind() == core.ValueKindArray && right.Kind() == core.ValueKindInteger {
             items := repeatItems(left.Array(), right.Integer())
             return core.NewArrayValue(items, false)
        }
        return evaluateBinaryArithmeticOp(multiplies, left, right)
    case '-':
        return evaluateBinaryArithmeticOp(minus, left, right)
    case '/':
        return evaluateBinaryArithmeticOp(divides, left, right)
    case '^':
        return evaluateBinaryArithmeticOp(power, left, right)
    case '<':
        return evaluateBinaryComparisonOp(less, left, right)
    case '>':
        return evaluateBinaryComparisonOp(greater, left, right)
    case core.TokenLe:
        return evaluateBinaryComparisonOp(lessEqual, left, right)
    case core.TokenGe:
        return evaluateBinaryComparisonOp(greaterEqual, left, right)
    case core.TokenEq:
        return evaluateBinaryComparisonOp(equalTo, left, right)
    case core.TokenNe:
        return evaluateBinaryComparisonOp(notEqualTo, left, right)
    case core.TokenAnd:
        if !left.Logical() {
            return left
        }
        return e.evaluate(binary.Right(), values, dtypes, callback, dtype, core.None())
    case core.TokenOr:
        if left.Logical() {
            return left
        }
        return e.evaluate(binary.Right(), values, dtypes, callback, dtype, core.None())
    case core.TokenIn:
        items := right.Array()
        contains := false
        for _, item := range items {
            if item == left {
                contains = true
                break
            }
        }
        return core.NewLogicalValue(contains)
    }
    core.Assert(false)
    return core.None()
}

func concatItems(left core.Items, right core.Items) core.Items {
    leftSize := len(left)
    if leftSize == 0 {
        return left.Clone()
    }
    rightSize := len(right)
    if  rightSize == 0 {
        return right.Clone()
    }
    result := make(core.Items, leftSize+rightSize)
    copy(result, left)
    copy(result[leftSize:], right)
    return result
}

func repeatItems(items core.Items, count int) core.Items {
    size := len(items)
    if size == 0 {
        return nil
    }
    result := make(core.Items, size*count)
    for i := 0; i < count; i++ {
        copy(result[size*i:], items)
    }
    return result
}

func(e *Evaluation) evaluateSelect(
        sel *SelectExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    condition := e.evaluate(sel.Condition(), values, dtypes, callback, dtype, core.None())
    if condition.Logical() {
        return e.evaluate(sel.TrueValue(), values, dtypes, callback, dtype, context)
    } else {
        return e.evaluate(sel.FalseValue(), values, dtypes, callback, dtype, context)
    }
}
        
func(e *Evaluation) evaluateComprehension(
        comprehension *ComprehensionExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    size := comprehension.IteratorCount()
    iterables := make([]core.Value, size)
    for i := 0; i < size; i++ {
        iterables[i] = 
            e.evaluate(
                comprehension.Iterable(i), values, dtypes, callback, dtype, core.None())
    }
    length := iterables[0].Size()
    for i := 1; i < size; i++ {
        if iterables[i].Size() != length {
            core.RaiseError(
                comprehension.Position(), 
                "iterables must have the same length in array comprehension")
        }
    }
    var items core.Items
    ids := make(map[string]core.Value)
    for key, value := range values {
        ids[key] = value
    }
    for i := 0; i < length; i++ {
        for k := 0; k < size; k++ {
            e.assign(comprehension.Iterator(k), iterables[k].At(i), ids, dtypes, callback)
        }
        accept := true
        if condition := comprehension.Condition(); condition != nil {
            accept = e.evaluate(condition, ids, dtypes, callback, dtype, core.None()).Logical()
        }
        if accept {
            var ctx core.Value
            if context.Kind() == core.ValueKindArray && len(items) < context.Size() {
                ctx = context.At(len(items))
            } else {
                ctx = core.None()
            }
            item := e.evaluate(comprehension.Item(), ids, dtypes, callback, dtype, ctx)
            items = append(items, item)
        }
        for k := 0; k < size; k++ {
            e.unassign(comprehension.Iterator(k), ids)
        }
    }
    return core.NewArrayValue(items, false)
}

func(e *Evaluation) evaluateInvocation(
        invocation *InvocationExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType, 
        context core.Value) core.Value {
    fragment := e.fragments[invocation.Target()]
    proto := fragment.Prototype()
    ids := make(map[string]core.Value)
    paramCount := proto.ParamCount()
    for i := 0; i < paramCount; i++ {
        param := proto.ParamAt(i)
        name := param.Name()
        arg := invocation.Arg(name)
        if arg != nil {
            ids[name] = e.evaluate(arg, values, dtypes, callback, dtype, core.None())
        } else {
            ids[name] = param.DefaultValue()
        }
    }
    dataType := invocation.DataType()
    if dataType == core.GetPrimitiveType(core.TypenameGeneric) {
        dataType = dtype
    }
    if dataType != nil {
        ids["?"] = core.NewStringValue(dataType.String())
    }    
    resultCount := proto.ResultCount()
    if !invocation.Type().IsAttribute() {
        value := context
        if value.IsNone() {
            value = core.NewIdentifierValue("")
        }
        checkStructure(value, invocation.Type(), invocation.Position())
        if resultCount == 1 {
            name := proto.ResultAt(0).Name()
            if !context.IsNone() {
                ids[name] = context
            } else {
                ids[name] = e.makeResultValue(proto.Name())
            }
        } else {
            for i := 0; i < resultCount; i++  {
                name := proto.ResultAt(i).Name()
                if !context.IsNone() {
                    ids[name] = context.At(i)
                } else {
                    ids[name] = e.makeResultValue(proto.Name())
                }
            }
        }
    }
    assignmentCount := fragment.AssignmentCount()
    lower := (assignmentCount != 0 && e.lowered[proto.Name()])
    if lower {
        for i := 0; i < assignmentCount; i++ {
            assignment := fragment.AssignmentAt(i)
            lhs := assignment.Lhs()
            rhs := assignment.Rhs()
            ctx := EvaluateLvalue(lhs, ids, false)
            func() {
                defer func() {
                    if r := recover(); r != nil {
                        if e, ok := r.(*core.Error); ok {
                            core.RaiseError(chain(e.Position(), invocation.Position()), e.What())
                        } else {
                            panic(r)
                        }
                    }
                }()
                e.EvaluateAssign(lhs, rhs, ids, dtypes, callback, dataType, ctx)
            }()
        }
    }
    var value core.Value
    if resultCount == 1 {
        value = ids[proto.ResultAt(0).Name()]
    } else {
        items := make(core.Items, resultCount)
        for i := 0; i < resultCount; i++ {
            items[i] = ids[proto.ResultAt(i).Name()]
        }
        value = core.NewTupleValue(items, false)
    }        
    if !lower {
        declareValue(value, invocation.Type(), dtypes, dtype)
        callback.Operation(proto, ids, dtypes)
    }        
    return value
}
        
func(e *Evaluation) evaluateBuiltin(
        builtin *BuiltinExpr,
        values map[string]core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback, 
        dtype *core.PrimitiveType) core.Value {
    arg := e.evaluate(builtin.Arg(), values, dtypes, callback, dtype, core.None())
    switch builtin.Op() {
    case core.TokenLengthOf:
        var length int
        if arg.Kind() == core.ValueKindString {
            length = len(arg.String())
        } else {
            length = len(arg.Array())
        }
        return core.NewIntegerValue(length)
    case core.TokenRangeOf:
        var length int
        if arg.Kind() == core.ValueKindString {
            length = len(arg.String())
        } else {
            length = len(arg.Array())
        }
        items := make(core.Items, length)
        for i := 0; i < length; i++ {
            items[i] = core.NewIntegerValue(i)
        }
        return core.NewArrayValue(items, false)
    case core.TokenShapeOf:
        core.RaiseError(
            builtin.Position(), 
            "the use of operator 'shape_of' is deprecated and is not supported")
    case core.TokenInteger:
        switch arg.Kind() {
        case core.ValueKindInteger:
            return arg
        case core.ValueKindScalar:
            return core.NewIntegerValue(int(arg.Scalar()))
        case core.ValueKindLogical:
            value := 0
            if arg.Logical() {
                value = 1
            }
            return core.NewIntegerValue(value)
        case core.ValueKindString:
            str := arg.String()
            value, err := strconv.Atoi(str)
            if err != nil {
                core.RaiseError(
                    builtin.Position(), 
                    "cannot convert string '%s' to integer", 
                    str)
            }
            return core.NewIntegerValue(value)
        }
    case core.TokenScalar:
        switch arg.Kind() {
        case core.ValueKindScalar:
            return arg
        case core.ValueKindInteger:
            return core.NewScalarValue(float32(arg.Integer()))
        case core.ValueKindLogical:
            value := float32(0.0)
            if arg.Logical() {
                value = float32(1.0)
            }
            return core.NewScalarValue(value)
        case core.ValueKindString:
            str := arg.String()
            value, err := strconv.ParseFloat(str, 32)
            if err != nil {
                core.RaiseError(
                    builtin.Position(), 
                    "cannot convert string '%s' to scalar", 
                    str)
            }
            return core.NewScalarValue(float32(value))
        }
    case core.TokenLogical:
        switch arg.Kind() {
        case core.ValueKindLogical:
            return arg
        case core.ValueKindInteger:
            return core.NewLogicalValue(arg.Integer() != 0)
        case core.ValueKindScalar:
            return core.NewLogicalValue(arg.Scalar() != 0.0)
        case core.ValueKindString:
            return core.NewLogicalValue(len(arg.String()) != 0)
        }
    case core.TokenString:
        switch arg.Kind() {
        case core.ValueKindLogical:
            return core.NewStringValue(fmt.Sprintf("%t", arg.Logical()))
        case core.ValueKindInteger:
            return core.NewStringValue(fmt.Sprintf("%d", arg.Integer()))
        case core.ValueKindScalar:
            // TODO: Enforce decimal point (need common function, e.g. formatScalar)
            return core.NewStringValue(fmt.Sprintf("%g", arg.Scalar()))
        case core.ValueKindString:
            return arg
        }
    }
    core.Assert(false)
    return core.None()
}

func evaluateBinaryArithmeticOp(
        op *binaryArithmeticOp, left core.Value, right core.Value) core.Value {
    if left.Kind() == core.ValueKindInteger && right.Kind() == core.ValueKindInteger {
        return core.NewIntegerValue(op.integer(left.Integer(), right.Integer()))
    }
    if left.Kind() == core.ValueKindScalar && right.Kind() == core.ValueKindScalar {
        return core.NewScalarValue(op.scalar(left.Scalar(), right.Scalar()))
    }
    core.Assert(false)
    return core.None()
}

func evaluateBinaryComparisonOp(
        op *binaryComparisonOp, left core.Value, right core.Value) core.Value {
    if left.Kind() == core.ValueKindInteger && right.Kind() == core.ValueKindInteger {
        return core.NewLogicalValue(op.integer(left.Integer(), right.Integer()))
    }
    if left.Kind() == core.ValueKindScalar && right.Kind() == core.ValueKindScalar {
        return core.NewLogicalValue(op.scalar(left.Scalar(), right.Scalar()))
    }
    core.Assert(false)
    return core.None()
}

func dtypeOf(value core.Value, dtypes map[string]core.Typename) core.Typename {
    switch value.Kind() {
    case core.ValueKindScalar:
        return core.TypenameScalar
    case core.ValueKindInteger:
        return core.TypenameInteger
    case core.ValueKindLogical:
        return core.TypenameLogical
    case core.ValueKindString:
        return core.TypenameString
    case core.ValueKindIdentifier:
        return dtypes[value.Identifier()]
    default:
        core.Assert(false)
        return core.TypenameGeneric
    }
}

func(e *Evaluation) insertCopy(
        lvalue core.Value, 
        rvalue core.Value, 
        dtypes map[string]core.Typename,
        callback core.ParserCallback) {
    dtype := dtypeOf(rvalue, dtypes)
    dvalue := core.NewStringValue(dtype.String())
    proto := e.fragments["copy"].Prototype()
    args := map[string]core.Value{
        "x": rvalue,
        "y": lvalue,
        "?": dvalue,
    }       
    dtypes[lvalue.Identifier()] = dtype
    callback.Operation(proto, args, dtypes)
}

func(e *Evaluation) assign(
        lhs Expr, 
        rvalue core.Value, 
        ids map[string]core.Value,
        dtypes map[string]core.Typename,
        callback core.ParserCallback) {
    switch lhs.Kind() {
    case ExprKindArray:
        array := lhs.(*ArrayExpr)
        size := array.Size()
        if size != rvalue.Size() {
            core.RaiseError(
                lhs.Position(), 
                "cannot assign array of length %d to array of length %d",
                rvalue.Size(), 
                size)
        }
        for i := 0; i < size; i++ {
            e.assign(array.At(i), rvalue.At(i), ids, dtypes, callback)
        }
    case ExprKindTuple:
        tuple := lhs.(*TupleExpr)
        size := tuple.Size()
        core.Assert(size == rvalue.Size())
        for i := 0; i < size; i++ {
            e.assign(tuple.At(i), rvalue.At(i), ids, dtypes, callback)
        }
    case ExprKindIdentifier:
        identifier := lhs.(*IdentifierExpr)
        name := identifier.Name()
        lvalue, ok := ids[name]
        if ok {
            if !lvalue.Eq(rvalue) {
                kind := lvalue.Kind()
                if kind == core.ValueKindArray || kind == core.ValueKindTuple {
                    size := lvalue.Size()
                    if kind == core.ValueKindArray && size != rvalue.Size() {
                        core.RaiseError(
                            lhs.Position(), 
                            "cannot assign array of length %d to array of length %d",
                             rvalue.Size(), 
                             size)
                    }
                    for i := 0; i < size; i++ {
                        e.insertCopy(lvalue.At(i), rvalue.At(i), dtypes, callback)
                    }
                } else {
                    core.Assert(kind == core.ValueKindIdentifier)
                    e.insertCopy(lvalue, rvalue, dtypes, callback)
                }
            }
        } else {
            ids[name] = rvalue
        }
    default:
        core.Assert(false)
    }
}      

func(e *Evaluation) unassign(lhs Expr, ids map[string]core.Value) {
    switch lhs.Kind() {
    case ExprKindArray:
        array := lhs.(*ArrayExpr)
        size := array.Size()
        for i := 0; i < size; i++ {
            e.unassign(array.At(i), ids)
        }
    case ExprKindTuple:
        tuple := lhs.(*TupleExpr)
        size := tuple.Size()
        for i := 0; i < size; i++ {
            e.unassign(tuple.At(i), ids)
        }
    case ExprKindIdentifier:
        identifier := lhs.(*IdentifierExpr)
        delete(ids, identifier.Name())
    default:
        core.Assert(false)
    }
}      

func declareValue(
        arg core.Value, 
        typ core.Type, 
        dtypes map[string]core.Typename, 
        dtype *core.PrimitiveType) {
    switch arg.Kind() {
    case core.ValueKindIdentifier:
        core.Assert(typ.Kind() == core.TypeKindTensor)
        id := arg.Identifier()
        tensorType := typ.(*core.TensorType)
        core.Assert(tensorType.DataType().Kind() == core.TypeKindPrimitive)
        dataType := tensorType.DataType().(*core.PrimitiveType)
        name := dataType.Name()
        if name == core.TypenameGeneric {
            name = dtype.Name()
        }
        oldName, ok := dtypes[id]
        if ok {
            core.Assert(oldName == name)
        } else {
            dtypes[id] = name
        }
    case core.ValueKindArray:
        core.Assert(typ.Kind() == core.TypeKindArray)
        arrayType := typ.(*core.ArrayType)
        itemType := arrayType.ItemType()
        size := arg.Size()
        for i := 0; i < size; i++ {
            declareValue(arg.At(i), itemType, dtypes, dtype)
        }
    case core.ValueKindTuple:
        core.Assert(typ.Kind() == core.TypeKindTuple)
        tupleType := typ.(*core.TupleType)
        size := arg.Size()
        for i := 0; i < size; i++ {
            declareValue(arg.At(i), tupleType.ItemType(i), dtypes, dtype)
        }
    }
}

func checkStructure(value core.Value, typ core.Type, position *core.Position) {
    switch typ.Kind() {
    case core.TypeKindPrimitive, core.TypeKindTensor:
        if value.Kind() != core.ValueKindIdentifier {
            core.RaiseError(
                position, 
                "invocation context mismatch: "+
                    "expected identifier on left hand side to match type '%s'",
                typ.String())
        }
    case core.TypeKindArray:
        if value.Kind() != core.ValueKindArray {
            core.RaiseError(
                position, 
                "invocation context mismatch: "+
                    "expected array on left hand side to match type '%s'",
                typ.String())
        }
        size := value.Size()
        array := typ.(*core.ArrayType)
        itemType := array.ItemType()
        for i := 0; i < size; i++ {
            checkStructure(value.At(i), itemType, position)
        }
    case core.TypeKindTuple:
        if value.Kind() != core.ValueKindTuple {
            core.RaiseError(
                position, 
                "invocation context mismatch: "+
                    "expected tuple on left hand side to match type '%s'",
                typ.String())
        }
        size := value.Size()
        tuple := typ.(*core.TupleType)
        for i := 0; i < size; i++ {
            checkStructure(value.At(i), tuple.ItemType(i), position)
        }
    }
}

func chain(position *core.Position, origin *core.Position) *core.Position {
    return &core.Position{position.Line, position.Column, position.Filename, origin}
}

func(e *Evaluation) nextTensorId(op string) string {
    count := e.tensorCounts[op] + 1
    e.tensorCounts[op] = count
    return fmt.Sprintf("%s%d", op, count)
}

func(e *Evaluation) makeTensorId(op string) string {
    var id string
    for {
        id = e.nextTensorId(op)
        if !e.isReservedId(id) {
            break
        }
    }
    e.reservedIds[id] = true
    return id
}

func(e *Evaluation) makeResultValue(op string) core.Value {
    return core.NewIdentifierValue(e.makeTensorId(op))
}

func(e *Evaluation) addReservedIdentifiers(expr Expr) {
    switch expr.Kind() {
    case ExprKindIdentifier:
        identifier := expr.(*IdentifierExpr)
        e.reservedIds[identifier.Name()] = true
    case ExprKindArray:
        array := expr.(*ArrayExpr)
        size := array.Size()
        for i := 0; i < size; i++ {
            e.addReservedIdentifiers(array.At(i))
        }
    case ExprKindTuple:
        tuple := expr.(*TupleExpr)
        size := tuple.Size()
        for i := 0; i < size; i++ {
            e.addReservedIdentifiers(tuple.At(i))
        }
    default:
        core.Assert(false)
    }
}

func(e *Evaluation) isReservedId(id string) bool {
    _, ok := e.reservedIds[id]
    return ok
}

