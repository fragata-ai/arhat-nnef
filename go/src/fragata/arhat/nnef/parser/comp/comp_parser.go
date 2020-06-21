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
    "io"
    "strings"
    "fragata/arhat/nnef/core"
)

//
//    Dclarations
//

// Need this warpper around a map for correct representation of nil declaration
// objects for lvalues; comparison to nil in Go does not distinguish between
// nil and empty maps.

type Declarations struct {
    decls map[string]core.Type
}

func NewDeclarations() *Declarations {
    return &Declarations{make(map[string]core.Type)}
}

func(d *Declarations) Find(name string) core.Type {
    typ, ok := d.decls[name]
    if !ok {
        return nil
    }
    return typ
}

func(d *Declarations) Enter(name string, typ core.Type) {
    d.decls[name] = typ
}

func(d *Declarations) Delete(name string) {
    delete(d.decls, name)
}

//
//    CompParser
//

type CompParser struct {
    stdlibSource string
    lowered map[string] bool
    flags core.Flags
}

func NewCompParser(stdlib string, lowered map[string] bool) *CompParser {
    p := new(CompParser)
    p.Init(stdlib, lowered)
    return p
}

func(p *CompParser) Init(stdlib string, lowered map[string] bool) {
    if stdlib != "" {
        p.stdlibSource = stdlib
    } else {
        p.stdlibSource = StdlibSource()
    }
    p.lowered = lowered
    p.flags = 0
}

func(p *CompParser) Parse(is io.Reader, filename string, callback core.ParserCallback) {
    lexer := core.NewLexer(is, filename)
    lexer.Next()
    version := core.ReadVersion(lexer)
    callback.BeginDocument(filename, version)
    p.flags = 0
    handler := func(ext string) bool {
        return callback.HandleExtension(ext) || p.handleExtension(ext)
    }
    core.ReadExtensions(lexer, handler)
    prototypes := make(map[string]*core.Prototype)
    fragments := make(map[string]*Fragment)
    parseFragments(p.stdlibSource, "stdlib", prototypes, fragments)
    allowOperator := ((p.flags & core.KhrEnableOperatorExpressions) != 0)
    if (p.flags & core.KhrEnableFragmentDefinitions) != 0 {
        for lexer.Token() == core.TokenFragment {
            fragment := parseFragment(lexer, prototypes, allowOperator)
            fragments[fragment.Prototype().Name()] = fragment
        }
    }
    lexer.ReadToken(core.TokenGraph)
    graph := parsePrototype(lexer, prototypes, false, true)
    assignments := parseAssignments(lexer, graph, prototypes, allowOperator, true)
    callback.BeginGraph(graph, prototypes)
    values := make(map[string]core.Value)
    dtypes := make(map[string]core.Typename)
    vars := make(map[string]bool)
    evaluation := NewEvaluation(assignments, fragments, p.lowered)
    for _, assignment := range assignments {
        lhs := assignment.Lhs()
        rhs := assignment.Rhs()
        checkExternalsAndVariables(lhs, rhs, graph, vars)
        context := EvaluateLvalue(lhs, nil, true)
        evaluation.EvaluateAssign(lhs, rhs, values, dtypes, callback, nil, context)
    }
     callback.EndGraph(graph, dtypes)
     callback.EndDocument(filename)
     lexer.ReadToken(core.TokenEof)
}

func(p *CompParser) handleExtension(ext string) bool {
    switch ext {
    case "KHR_enable_fragment_definitions":
        p.flags |= core.KhrEnableFragmentDefinitions
        return true
    case "KHR_enable_operator_expressions":
        p.flags |= core.KhrEnableOperatorExpressions
        return true
    }
    return false
}

func parseFragments(
        text string, 
        filename string, 
        prototypes map[string]*core.Prototype, 
        fragments map[string]*Fragment) {
    ss := strings.NewReader(text)
    lexer := core.NewLexer(ss, filename)
    lexer.Next()
    for lexer.Token() != core.TokenEof {
        fragment := parseFragment(lexer, prototypes, true)
        fragments[fragment.Prototype().Name()] = fragment
    }
}

func parsePrototype(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        allowTypespec bool, 
        graph bool) *core.Prototype {
    var position = *lexer.Position()
    name := lexer.Str()
    lexer.ReadToken(core.TokenIdentifier)
    if _, ok := prototypes[name]; ok {
        core.RaiseError(&position, "operation '%s' already defined", name)
    }
    isGenericDecl := false
    var genericParamDefault *core.PrimitiveType
    if !graph && lexer.ReadIfToken('<') {
        isGenericDecl = true
        lexer.ReadToken('?')
        if lexer.ReadIfToken('=') {
            genericParamDefault = core.GetPrimitiveType(core.GetTypename(lexer))
            lexer.Next()
        }        
        lexer.ReadToken('>')
    }
    params := parseParams(lexer, name, allowTypespec, graph)
    lexer.ReadToken(core.TokenArrow)
    results := parseResults(lexer, name, allowTypespec, !graph)
    for _, result := range results {
        for _, param := range params {
            if param.Name() == result.Name() {
                core.RaiseError(
                    &position, 
                    "invalid definition of operation '%s'; "+
                        "'%s' is defined both as parameter and as result",
                    name, 
                    result.Name())
            }
        }
    }
    attribute := results[0].Type().IsAttribute()
    for _, result := range results[1:] {
        if result.Type().IsAttribute() != attribute {
            core.RaiseError(
                &position, 
                "result types of fragment must be all tensor types or all attribute types")
        }
    }
    hasGenericParams := false
    for _, param := range params {
        if param.Type().IsGeneric() {
            hasGenericParams = true
            break
        }
    }
    hasGenericResults := false
    for _, result := range results {
        if result.Type().IsGeneric() {
            hasGenericResults = true
            break
        }
    }
    if (hasGenericParams || hasGenericResults) && !isGenericDecl {
        core.RaiseError(
            nil, 
            "fragment with generic parameter or result types must be declared generic using <?>")
    }
    if isGenericDecl && !hasGenericParams && !hasGenericResults {
        core.RaiseError(
            nil,
            "fragment declared as generic must have at least one generic parameter or result type")
    }
    return core.NewPrototype(name, params, results, genericParamDefault)
}

func parseParams(
        lexer *core.Lexer, 
        op string, 
        allowTypespec bool, 
        forceDefaults bool) []*core.Param {
    var params []*core.Param
    lexer.ReadToken('(')
    expectAttribute := false
    var position core.Position
    for {
        position = *lexer.Position()
        name := lexer.Str()
        lexer.ReadToken(core.TokenIdentifier)
        var typ core.Type = core.UnboundTensorType()
        if allowTypespec {
            lexer.ReadToken(':')
            typ = parseTypespec(lexer, true)
        }
        if expectAttribute && !typ.IsAttribute() {
            core.RaiseError(
                &position, 
                "expected attribute, found parameter of type '%s'", 
                typ.String())
        }
        expectAttribute = (expectAttribute && typ.IsAttribute())
        var defaultValue core.Value = core.None()
        if lexer.Token() == '=' {
            lexer.Next()
            expr := parseExpression(lexer, nil, nil, true, false, false, false)
            if !core.IsCastable(expr.Type(), typ, true) {
                core.RaiseError(
                    expr.Position(), 
                    "default value type '%s' cannot be cast to parameter type '%s'",
                    expr.Type().String(), 
                    typ.String())
            }
            defaultValue = EvaluateRvalue(expr)
        } else if forceDefaults && typ.IsAttribute() {
            core.RaiseError(&position, "expected default value for parameter '%s'", name)
        }
        for _, param := range params {
            if param.Name() == name {
                core.RaiseError(
                    &position, 
                    "duplicate parameter definition for fragment '%s'; "+
                        "parameter '%s' is already defined",
                op, 
                name)
            }
        }
        params = append(params, core.NewParam(name, typ, defaultValue))
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    lexer.ReadToken(')')
    return params
}

func parseResults(
        lexer *core.Lexer, 
        op string, 
        allowTypespec bool, 
        allowAttribute bool) []*core.Result {
    var results []*core.Result
    lexer.ReadToken('(')
    var position core.Position
    for {
        position = *lexer.Position()
        name := lexer.Str()
        lexer.ReadToken(core.TokenIdentifier)
        var typ core.Type = core.UnboundTensorType()
        if allowTypespec {
            lexer.ReadToken(':')
            typ = parseTypespec(lexer, false)
            if !allowAttribute && typ.IsAttribute() {
                core.RaiseError(&position, "non-tensor type not allowed in this context")
            }
        }
        for _, result := range results {
            if result.Name() == name {
                core.RaiseError(
                    &position, 
                    "duplicate result definition for operation '%s'; "+
                        "result '%s' is already defined",
                    op, 
                    name)
            }
        }
        results = append(results, core.NewResult(name, typ))
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    lexer.ReadToken(')')
    return results
}

func parseFragment(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        allowOperator bool) *Fragment {
    lexer.ReadToken(core.TokenFragment)
    proto := parsePrototype(lexer, prototypes, true, false)
    prototypes[proto.Name()] = proto
    var assignments []*Assignment
    if !lexer.ReadIfToken(';') {
        assignments = parseAssignments(lexer, proto, prototypes, allowOperator, false)
    }
    return NewFragment(proto, assignments)
}

func parseAssignments(
        lexer *core.Lexer, 
        proto *core.Prototype, 
        prototypes map[string]*core.Prototype, 
        allowOperator bool, 
        graph bool) []*Assignment {
    decls := NewDeclarations()
    paramCount := proto.ParamCount()
    for i := 0; i < paramCount; i++ {
        param := proto.ParamAt(i)
        if !graph || param.Type().IsAttribute() {
            decls.Enter(param.Name(), param.Type())
        }
    }
    var assignments []*Assignment
    lexer.ReadToken('{')
    for {
        lhs := parseTuple(lexer, nil, nil, false, true, false)
        lexer.ReadToken('=')
        var rhs Expr
        if allowOperator {
            rhs = parseExpression(lexer, prototypes, decls, true, true, true, true)
        } else {
            rhs = parseInvocation(lexer, prototypes, decls)
        }
        lexer.ReadToken(';')
        declare(lhs, rhs.Type(), decls)
        if !graph {
            checkOperationsAllowed(rhs)
        }
        assignments = append(assignments, NewAssignment(lhs, rhs))
        if lexer.Token() == '}' {
            break
        }
    }
    if graph {
        for i := 0; i < paramCount; i++ {
            param := proto.ParamAt(i)
            if decls.Find(param.Name()) == nil {
                core.RaiseError(
                    lexer.Position(), 
                    "graph parameter '%s' is not assigned", 
                    param.Name())
            }
        }
    }
    resultCount := proto.ResultCount()
    for i := 0; i < resultCount; i++ {
        result := proto.ResultAt(i)
        decl := decls.Find(result.Name())
        if decl == nil {
            core.RaiseError(
                lexer.Position(), 
                "result '%s' of operation '%s' is not assigned",
                result.Name(), 
                proto.Name())
        }
        if !core.IsCastable(decl, result.Type(), true) {
            core.RaiseError(
                lexer.Position(), 
                "result '%s' of operation '%s' is declared as '%s' "+
                    "but assignment has incompatible type '%s'",
                result.Name(), 
                proto.Name(), 
                result.Type().String(), 
                decl.String())
        }
    }
    lexer.ReadToken('}')
    return assignments
}

func checkOperationsAllowed(rhs Expr) {
    traverse(rhs, func(expr Expr) {
        if expr.Kind() == ExprKindInvocation {
            invocation := expr.(*InvocationExpr)
            target := invocation.Target()
            if target == "external" || target == "variable" || target == "update" {
                core.RaiseError(
                    invocation.Position(), 
                    "operation '%s' not allowed inside fragments", 
                    target)
            }
        }
    })
}

func checkExternalsAndVariables(
        lhs Expr, 
        rhs Expr, 
        graph *core.Prototype, 
        vars map[string]bool) {
    lhsKind := lhs.Kind()
    rhsKind := rhs.Kind()
    switch {
    case lhsKind == ExprKindArray && rhsKind == ExprKindArray:
        left := lhs.(*ArrayExpr)
        right := rhs.(*ArrayExpr)
        size := left.Size()
        for i := 0; i < size; i++ {
            checkExternalsAndVariables(left.At(i), right.At(i), graph, vars)
        }
    case lhsKind == ExprKindTuple && rhsKind == ExprKindTuple:
        left := lhs.(*TupleExpr)
        right := rhs.(*TupleExpr)
        size := left.Size()
        for i := 0; i < size; i++ {
            checkExternalsAndVariables(left.At(i), right.At(i), graph, vars)
        }
    case lhsKind == ExprKindIdentifier && rhsKind == ExprKindInvocation:
        identifier := lhs.(*IdentifierExpr)
        invocation := rhs.(*InvocationExpr)
        name := identifier.Name()
        target := invocation.Target()
        if target == "external" {
            if graph.GetParam(name) == nil {
                core.RaiseError(
                    identifier.Position(), 
                    "identifiers assigned by operation 'external' must be graph parameters")
            }
        } else {
            if graph.GetParam(name) != nil {
                core.RaiseError(
                    identifier.Position(), 
                    "graph parameter '%s' can only be assigned by operation 'external'",
                    name)
            }
        }
        switch target {    
        case "variable":
            vars[name] = true
        case "update":
            arg := invocation.Arg("variable")
            name := arg.(*IdentifierExpr).Name()
            if arg.Kind() != ExprKindIdentifier || !vars[name] {
                core.RaiseError(
                    arg.Position(), 
                    "first argument to operation 'update' must be a variable")
            }
        }
    }
}

func traverse(expr Expr, fn func(Expr)) {
    fn(expr)
    switch expr.Kind() {
    case ExprKindLiteral, ExprKindIdentifier:
        // ok
    case ExprKindBuiltin:
        builtin := expr.(*BuiltinExpr)
        traverse(builtin.Arg(), fn)
    case ExprKindArray:
        array := expr.(*ArrayExpr)
        size := array.Size()
        for i := 0; i < size; i++ {
            traverse(array.At(i), fn)
        }
    case ExprKindTuple:
        tuple := expr.(*TupleExpr)
        size := tuple.Size()
        for i := 0; i < size; i++ {
            traverse(tuple.At(i), fn)
        }
    case ExprKindSubscript:
        subscript := expr.(*SubscriptExpr)
        traverse(subscript.Sequence(), fn)
        if begin := subscript.Begin(); begin != nil {
            traverse(begin, fn)
        }
        if end := subscript.End(); end != nil {
            traverse(end, fn)
        }
    case ExprKindComprehension:
        comprehension := expr.(*ComprehensionExpr)
        count := comprehension.IteratorCount()
        for i := 0; i < count; i++ {
            traverse(comprehension.Iterator(i), fn)
            traverse(comprehension.Iterable(i), fn)
        }
        if condition := comprehension.Condition(); condition != nil {
            traverse(condition, fn)
        }
        traverse(comprehension.Item(), fn)
    case ExprKindUnary:
        unary := expr.(*UnaryExpr)
        traverse(unary.Right(), fn)
    case ExprKindBinary:
        binary := expr.(*BinaryExpr)
        traverse(binary.Left(), fn)
        traverse(binary.Right(), fn)
    case ExprKindSelect:
        sel := expr.(*SelectExpr)
        traverse(sel.Condition(), fn)
        traverse(sel.TrueValue(), fn)
        traverse(sel.FalseValue(), fn)
    case ExprKindInvocation:
        invocation := expr.(*InvocationExpr)
        count := invocation.ArgCount()
        for i := 0; i < count; i++ {
            traverse(invocation.ArgValue(i), fn)
        }
    }
}

func parseArrayTypespec(lexer *core.Lexer, typ core.Type) core.Type {
    for lexer.ReadIfToken('[') {
        lexer.ReadToken(']')
        typ = core.GetArrayType(typ)
    }
    return typ
}

func parseTupleTypespec(lexer *core.Lexer, allowUnboundTensor bool) core.Type {
    position := *lexer.Position()
    lexer.Next()
    var items []core.Type
    for {
        items = append(items, parseTypespec(lexer, allowUnboundTensor))
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    lexer.ReadToken(')')
    attribute := items[0].IsAttribute()
    for _, item := range items[1:] {
        if item.IsAttribute() != attribute {
            core.RaiseError(
                &position, 
                "item types in tuple type must be all attribute types or all tensor types")
        }
    }
    return parseArrayTypespec(lexer, core.GetTupleType(items))
}

func parseTypespec(lexer *core.Lexer, allowUnboundTensor bool) core.Type {
    if lexer.Token() == '(' {
        return parseTupleTypespec(lexer, allowUnboundTensor)
    }
    var typ core.Type
    if lexer.ReadIfToken(core.TokenTensor) {
        lexer.ReadToken('<')
        typ = core.UnboundTensorType()
        if lexer.Token() != '>' {
            typ = core.GetTensorType(core.GetTypename(lexer))
            lexer.Next()
        } else if !allowUnboundTensor {
            core.RaiseError(lexer.Position(), "unbound tensor not allowed in this context")
        }
        lexer.ReadToken('>')
    } else {
        name := core.GetTypename(lexer)
        lexer.Next()
        typ = core.GetPrimitiveType(name)
    }
    return parseArrayTypespec(lexer, typ)
}

func parseExpression(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype,
        decls *Declarations,
        allowLiteral bool,
        allowIdentifier bool, 
        allowOperator bool,
        allowSelect bool) Expr {
    expr := parsePrimary(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator)
    if expr.Kind() != ExprKindLiteral && allowOperator {
        expr = parseSubscripts(lexer, prototypes, decls, expr)
    }
    if allowOperator {
        expr = parseBinary(lexer, prototypes, decls, expr, 0)
        if lexer.Token() == core.TokenIf && allowSelect {
             expr = parseSelect(lexer, prototypes, decls, expr)
        }
    }
    return expr
}

func parsePrimary(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations,
        allowLiteral bool, 
        allowIdentifier bool, 
        allowOperator bool) Expr {
    switch lexer.Token() {
    case core.TokenTrue, core.TokenFalse:
        if allowLiteral {
            return parseLogical(lexer)
        }
    case core.TokenFractional:
        if allowLiteral {
            return parseScalar(lexer)
        }
    case core.TokenDecimal:
        if allowLiteral {
            return parseInteger(lexer)
        }
    case core.TokenCharacters:
        if allowLiteral {
            return parseString(lexer)
        }
    case core.TokenIdentifier:
        if allowIdentifier {
            return parseIdentifier(
                lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator)
        }
    case '[':
        return parseArray(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator)
    case '(':
        return parseTuple(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator)
    case '-':
        return parseUnary(lexer, prototypes, decls)
    case '!':
        if allowOperator {
            return parseUnary(lexer, prototypes, decls)
        }
    case core.TokenShapeOf:
        core.RaiseError(
            lexer.Position(), 
            "the use of operator 'shape_of' is deprecated and is not supported")
    case core.TokenLengthOf, 
            core.TokenRangeOf, 
            core.TokenInteger, 
            core.TokenScalar, 
            core.TokenLogical, 
            core.TokenString:
        if allowOperator {
            return parseBuiltin(lexer, prototypes, decls)
        }
    default:
        core.RaiseError(
            lexer.Position(), 
            "unexpected token '%s'", 
            lexer.Token().String())
    }
    core.RaiseError(
        lexer.Position(), 
        "token '%s' not allowed in this context", 
        lexer.Token().String())
    return nil
}

func parseInteger(lexer *core.Lexer) Expr {
    position := *lexer.Position()
    value := core.GetIntegerValue(lexer)
    lexer.Next()
    return NewIntegerExpr(&position, value, core.GetPrimitiveType(core.TypenameInteger))
}

func parseScalar(lexer *core.Lexer) Expr {
    position := *lexer.Position()
    value := core.GetScalarValue(lexer)
    lexer.Next() 
    return NewScalarExpr(&position, value, core.GetPrimitiveType(core.TypenameScalar))
}

func parseLogical(lexer *core.Lexer) Expr {
    position := *lexer.Position()
    value := (lexer.Token() == core.TokenTrue)
    lexer.Next()
    return NewLogicalExpr(&position, value, core.GetPrimitiveType(core.TypenameLogical))
}

func parseString(lexer *core.Lexer) Expr {
    position := *lexer.Position()
    value := lexer.Str()
    lexer.Next()
    return NewStringExpr(&position, value, core.GetPrimitiveType(core.TypenameString))
}

func parseIdentifier(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations,
        allowLiteral bool,
        allowIdentifier bool, 
        allowOperator bool) Expr  {
    position := *lexer.Position()
    str := lexer.Str()
    lexer.ReadToken(core.TokenIdentifier)
    if lexer.Token() == '(' || (lexer.Token() == '<' && prototypes[str] != nil) {
        return parseInvocationArguments(
            lexer, prototypes, decls, &position, str, allowLiteral, allowIdentifier, allowOperator)
    } else {
        return makeIdentifier(&position, str, decls)
    }
}

func makeIdentifier(position *core.Position, name string, decls *Declarations) Expr {
    var typ core.Type
    if decls != nil {
        typ = decls.Find(name)
        if typ == nil {
            core.RaiseError(position, "undeclared identifier '%s'", name)
        }
    }
    return NewIdentifierExpr(position, name, typ)
}

func parseArray(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations,
        allowLiteral bool, 
        allowIdentifier bool, 
        allowOperator bool) Expr {
    position := *lexer.Position()
    lexer.Next() 
    var items []Expr
    var typ core.Type
    if lexer.Token() != ']' {
        if lexer.Token() == core.TokenFor {
            return parseComprehension(lexer, prototypes, decls, &position)
        }
        first := 
            parseExpression(
                lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator, true)
        items = append(items, first)
        typ = first.Type()
        for lexer.ReadIfToken(',') {
            item := 
                parseExpression(
                    lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator, true)
            items = append(items, item)
            if decls != nil {
                typ = core.CommonType(typ, item.Type(), true)
                if typ == nil {
                    core.RaiseError(
                        &position, 
                        "incompatible item types (%s vs %s) in array",
                        first.Type().String(), 
                        item.Type().String())
                }
            }
        }
    }        
    lexer.ReadToken(']')
    return NewArrayExpr(&position, items, core.GetArrayType(typ))
}

func parseTuple(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations,
        allowLiteral bool, 
        allowIdentifier bool, 
        allowOperator bool) Expr {
    position := *lexer.Position()
    parenthesized := (lexer.Token() == '(')
    if parenthesized {
        lexer.Next()
    }
    var items []Expr
    var types []core.Type
    first := 
        parseExpression(
            lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator, true)
    if lexer.Token() == ',' {
        items = append(items, first)
        types = append(types, first.Type())
        for lexer.ReadIfToken(',') {
            item := 
                parseExpression(
                    lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator, true)
            items = append(items, item)
            types = append(types, item.Type())
        }
    }        
    if parenthesized {
        lexer.ReadToken(')')
    }
    if len(items) == 0 {
        return first
    }
    return NewTupleExpr(&position, items, core.GetTupleType(types))
}

func parseInvocation(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations) Expr {
    position := *lexer.Position()
    str := lexer.Str()
    lexer.ReadToken(core.TokenIdentifier)
    if lexer.Token() != '(' && lexer.Token() != '<' {
        core.RaiseError(&position, "expected operation invocation")
    }
    return parseInvocationArguments(lexer, prototypes, decls, &position, str, true, true, false)
}

func parseInvocationArguments(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations,
        position *core.Position, 
        target string,
        allowLiteral bool, 
        allowIdentifier bool, 
        allowOperator bool) Expr {
    proto, ok := prototypes[target]
    if !ok {
        core.RaiseError(position, "undefined operation '%s'", target)
    }        
    dataType := proto.GenericParamDefault()
    if lexer.ReadIfToken('<') {
        dataType = core.GetPrimitiveType(core.GetTypename(lexer))
        lexer.Next() 
        lexer.ReadToken('>')
    }
    lexer.ReadToken('(')
    args := make(map[string]Expr)
    var argNames []string
    var argValues []Expr
    expectNamed := false
    for {
        pos := *lexer.Position()
        if len(args) >= proto.ParamCount() {
            core.RaiseError(
                &pos, 
                "too many positional arguments; definition of '%s' has only %d parameters",
                proto.Name(), 
                proto.ParamCount())
        }        
        var param *core.Param
        var arg Expr
        named := false
        if lexer.Token() == core.TokenIdentifier {
            str := lexer.Str()
            lexer.Next()
            if lexer.ReadIfToken('=') {
                param = proto.GetParam(str)
                if param == nil {
                    core.RaiseError(
                        &pos, 
                        "operation '%s' has no parameter called '%s'",
                        proto.Name(), 
                        str)
                }        
                arg = 
                    parseExpression(
                        lexer, 
                        prototypes, 
                        decls, 
                        allowLiteral, 
                        allowIdentifier, 
                        allowOperator, 
                        true)
                named = true
            } else {
                param = proto.ParamAt(len(args))
                if lexer.Token() == '(' {
                    arg = 
                        parseInvocationArguments(
                             lexer, 
                             prototypes, 
                             decls, 
                             &pos, 
                             str, 
                             allowLiteral, 
                             allowIdentifier, 
                             allowOperator)
                } else {
                    arg = makeIdentifier(&pos, str, decls)
                }
                arg = parseSubscripts(lexer, prototypes, decls, arg)
                arg = parseBinary(lexer, prototypes, decls, arg, 0)
                if lexer.Token() == core.TokenIf {
                    arg = parseSelect(lexer, prototypes, decls, arg)
                }
            }
        } else {
            param = proto.ParamAt(len(args))
            arg = 
                parseExpression(
                    lexer, 
                    prototypes, 
                    decls, 
                    allowLiteral, 
                    allowIdentifier, 
                    allowOperator, 
                    true)
        }
        paramName := param.Name()
        paramType := param.Type()
        if dataType != nil {
            paramType = core.BindDataType(paramType, dataType)
        }
        if !core.IsCastable(arg.Type(), paramType, true) {
            core.RaiseError(
                &pos, 
                "argument of type '%s' cannot be cast to type '%s' for parameter '%s'",
                arg.Type().String(), 
                paramType.String(), 
                paramName)
        }        
        expectNamed = (expectNamed || named || paramType.IsAttribute())
        if expectNamed && !named {
            core.RaiseError(&pos, "expected named argument")
        }
        if contained, ok := args[paramName]; ok {
            cpos := contained.Position()
            core.RaiseError(
                &pos, 
                "duplicate arguments: parameter '%s' already assigned (%u, %u)",
                paramName, 
                cpos.Line, 
                cpos.Column)
        }
        args[paramName] = arg
        argNames = append(argNames, paramName)
        argValues = append(argValues, arg)
        if !lexer.ReadIfToken(',') {
            break
        }
    }       
    paramCount := proto.ParamCount()
    for i := 0; i < paramCount; i++ {
        param := proto.ParamAt(i)
        paramName := param.Name()
        if _, ok := args[paramName]; !ok {
            defaultValue := param.DefaultValue()
            if defaultValue.IsNone() {
                core.RaiseError(
                    lexer.Position(), 
                    "missing argument for fragment '%s'; parameter '%s' not assigned",
                    proto.Name(), 
                    paramName)
            }
            paramType := param.Type()
            if paramType.IsGeneric() {
                valueType := typeOf(defaultValue)
                if dataType != nil {
                    paramType = core.BindDataType(paramType, dataType)
                }
                if !core.IsCastable(valueType, paramType, true) {
                    core.RaiseError(
                        lexer.Position(), 
                        "default value type '%s' cannot be cast to type '%s' for parameter '%s'",
                        valueType.String(), 
                        paramType.String(), 
                        paramName)
                }
            }
        }
    }
    lexer.ReadToken(')')
    if proto.IsGeneric() && dataType == nil {
        dataType = deduceDataType(proto, args, dataType, position)
        if dataType == nil {
            core.RaiseError(position, "could not deduce generic data type")
        }
    }        
    typ := resultType(proto, dataType)
    return NewInvocationExpr(position, target, argNames, argValues, typ, dataType)
}

func parseUnary(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations) Expr {
    position := *lexer.Position()
    op := lexer.Token()
    lexer.Next()        
    rhs := parseExpression(lexer, prototypes, decls, true, true, true, true)
    typ := unaryResultType(rhs.Type(), op)
    if typ == nil {
        core.RaiseError(
            &position, 
            "invalid operand type '%s' for operation '%s'",
            rhs.Type().String(), 
            op.String())
    }        
    if typ.Kind() == core.TypeKindTensor {
        target := unaryOpName(op)
        argNames, argValues := makeUnaryOpArgs(rhs)
        return NewInvocationExpr(&position, target, argNames, argValues, typ, nil)
    } else {
        return NewUnaryExpr(&position, rhs, op, typ)
    }
}

func parseBinary(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations, 
        lhs Expr,
        exprPrec int) Expr {
    position := *lhs.Position()
    for {
        tokPrec := tokenPrecedence(lexer.Token())
        if tokPrec < exprPrec {
            break
        }        
        op := lexer.Token()
        lexer.Next()
        rhs := parsePrimary(lexer, prototypes, decls, true, true, true)
        rhs = parseSubscripts(lexer, prototypes, decls, rhs)
        nextPrec := tokenPrecedence(lexer.Token())
        if tokPrec < nextPrec {
            rhs = parseBinary(lexer, prototypes, decls, rhs, tokPrec+1)
        }        
        typ := binaryResultType(lhs.Type(), rhs.Type(), op)
        if typ == nil {
            core.RaiseError(
                &position, 
                "invalid operand types '%s' and '%s' for operation '%s'",
                lhs.Type().String(),
                rhs.Type().String(),
                op.String())
        }        
        if typ.Kind() == core.TypeKindTensor {
            target := binaryOpName(op)
            argNames,  argValues := makeBinaryOpArgs(lhs, rhs)
            lhs = NewInvocationExpr(&position, target, argNames, argValues, typ, nil)
        } else {
            lhs = NewBinaryExpr(&position, lhs, rhs, op, typ)
        }
    }
    return lhs
}

func parseBuiltin(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        decls *Declarations) Expr {
    position := *lexer.Position()
    op := lexer.Token()
    lexer.Next()
    lexer.ReadToken('(') 
    arg := parseExpression(lexer, prototypes, decls, true, true, true, true)
    typ := builtinResultType(op)
    if typ == nil {
        core.RaiseError(
            &position, 
            "invalid operand type '%s' for operation '%s'",
            arg.Type().String(),
            op.String())
    }        
    lexer.ReadToken(')')
    switch op {
    case core.TokenLengthOf:
        argType := arg.Type()
        if argType.Kind() != core.TypeKindArray && 
                argType != core.GetPrimitiveType(core.TypenameString) {
            core.RaiseError(
                &position, 
                "argument of length_of() must be an array or string (found %s)", 
                argType.String())
        }
    case core.TokenShapeOf:
        argType := arg.Type()
        if argType.Kind() != core.TypeKindTensor && argType.Kind() != core.TypeKindPrimitive {
            core.RaiseError(
                &position, 
                "argument of shape_of() must be of tensor or primitive type (found %s)",
                argType.String())
        }
    case core.TokenRangeOf:
        argType := arg.Type()
        if argType.Kind() != core.TypeKindArray && 
                argType != core.GetPrimitiveType(core.TypenameString) {
            core.RaiseError(
                &position, 
                "argument of range_of() must be an array or string (found %s)",
                argType.String())
        }
    case core.TokenInteger, core.TokenScalar, core.TokenLogical, core.TokenString:
        argType := arg.Type()
        if argType.Kind() != core.TypeKindPrimitive {
            core.RaiseError(
                &position, 
                "argument of %s() must be of non-tensor primitive type (found %s)",
                op.String(), 
                argType.String())
        }
    }        
    return NewBuiltinExpr(&position, arg, op, typ)
}

func parseSubscript(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations, 
        sequence Expr) Expr {
    lexer.Next()
    var beg, end Expr
    var typ core.Type
    sequenceType := sequence.Type()
    switch {
    case sequenceType.Kind() == core.TypeKindTuple:
        beg = parseExpression(lexer, prototypes, decls, true, true, true, true)
        if beg.Kind() != ExprKindLiteral || 
                beg.Type() != core.GetPrimitiveType(core.TypenameInteger) {
            core.RaiseError(beg.Position(), "tuple index must be an integer literal")
        }
        idx := beg.(*IntegerExpr).Value()
        lexer.ReadToken(']')
        typ = sequenceType.(*core.TupleType).ItemType(idx)
    case sequenceType.Kind() == core.TypeKindArray || 
            sequenceType == core.GetPrimitiveType(core.TypenameString):
        if lexer.Token() != ':' {
            beg = parseExpression(lexer, prototypes, decls, true, true, true, true)
            if beg.Type() != core.GetPrimitiveType(core.TypenameInteger) {
                core.RaiseError(
                    beg.Position(), 
                    "array index must be of type integer, found '%s'", 
                    beg.Type().String())
            }
        }
        rng := false
        if lexer.ReadIfToken(':') {
            rng = true
            if lexer.Token() != ']' {
                end = parseExpression(lexer, prototypes, decls, true, true, true, true)
                if end.Type() != core.GetPrimitiveType(core.TypenameInteger) {
                    core.RaiseError(
                        end.Position(), 
                        "array index must be of type integer, found '%s'", 
                        end.Type().String())
                }
            }
        } else {
            end = beg
        }
        lexer.ReadToken(']')
        if sequenceType.Kind() == core.TypeKindArray {
            arrayType := sequenceType.(*core.ArrayType)
            if rng {
                typ = arrayType
            } else {
                typ = arrayType.ItemType()
            }
        } else {
            typ = core.GetPrimitiveType(core.TypenameString)
        }
    default:
        core.RaiseError(
            sequence.Position(), 
            "subscripted expression must be of type array, tuple, or string; found '%s'",
            sequenceType.String())
    }
    return NewSubscriptExpr(sequence.Position(), sequence, beg, end, typ)
}

func parseSubscripts(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations, 
        sequence Expr) Expr {
    for lexer.Token() == '[' {
        sequence = parseSubscript(lexer, prototypes, decls, sequence)
    }
    return sequence
}

func parseSelect(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations, 
        trueValue Expr) Expr {
    lexer.ReadToken(core.TokenIf)
    condition := parseExpression(lexer, prototypes, decls, true, true, true, true)
    if condition.Type() != core.GetPrimitiveType(core.TypenameLogical) {
        core.RaiseError(condition.Position(), "condition must be a logical value")
    }
    lexer.ReadToken(core.TokenElse)
    falseValue := parseExpression(lexer, prototypes, decls, true, true, true, true)
    typ := core.CommonType(trueValue.Type(), falseValue.Type(), true)
    if typ == nil {
        core.RaiseError(
            trueValue.Position(), 
            "incompatible types in if-else expression (%s vs %s)",
            trueValue.Type().String(), 
            falseValue.Type().String())
    }
    return NewSelectExpr(trueValue.Position(), condition, trueValue, falseValue, typ)
}

func parseComprehension(
        lexer *core.Lexer,
        prototypes map[string]*core.Prototype, 
        decls *Declarations, 
        position *core.Position) Expr {
    lexer.ReadToken(core.TokenFor)
    var iterators, iterables []Expr
    for {
        iterator := parseIterator(lexer, decls)
        lexer.ReadToken(core.TokenIn)
        iterable := parseExpression(lexer, prototypes, decls, true, true, true, false)
        if iterable.Type().Kind() != core.TypeKindArray {
            core.RaiseError(iterable.Position(), "expression not iterable")
        }        
        iterators = append(iterators, iterator)
        iterables = append(iterables, iterable)
        itemType := iterable.Type().(*core.ArrayType).ItemType()
        declare(iterator, itemType, decls)
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    var condition Expr
    if lexer.ReadIfToken(core.TokenIf) {
        condition = parseExpression(lexer, prototypes, decls, true, true, true, true)
        if condition.Type() != core.GetPrimitiveType(core.TypenameLogical) {
            core.RaiseError(
                condition.Position(), 
                "condition in comprehension expression must be a logical expression")
        }
    }
    lexer.ReadToken(core.TokenYield)
    item := parseExpression(lexer, prototypes, decls, true, true, true, true)
    typ := core.GetArrayType(item.Type())
    for _, iterator := range iterators {
        undeclare(iterator, decls)
    }
    lexer.ReadToken(']')
    return NewComprehensionExpr(position, iterators, iterables, condition, item, typ)
}

func parseIterator(lexer *core.Lexer, decls *Declarations) Expr {
    if lexer.Token() == core.TokenIdentifier {
        iterator := NewIdentifierExpr(lexer.Position(), lexer.Str(), nil)
        lexer.ReadToken(core.TokenIdentifier)
        return iterator
    }        
    if lexer.Token() != '(' {
        core.RaiseError(lexer.Position(), "expected tuple or identifier")
    }
    lexer.Next()
    position := *lexer.Position()
    var items []Expr
    var types []core.Type
    first := parseIterator(lexer, decls)
    if lexer.Token() == ',' {
        items = append(items, first)
        types = append(types, first.Type())
        for lexer.ReadIfToken(',') {
            item := parseIterator(lexer, decls)
            items = append(items, item)
            types = append(types, item.Type())
        }
    }        
    lexer.ReadToken(')')
    if len(items) == 0 {
        return first
    } 
    return NewTupleExpr(&position, items, core.GetTupleType(types))
}

func declare(expr Expr, typ core.Type, declared *Declarations) {
    switch expr.Kind() {
    case ExprKindIdentifier:
        identifier := expr.(*IdentifierExpr)
        name := identifier.Name()
        if declared.Find(name) != nil {
            core.RaiseError(expr.Position(), "identifier '%s' is already declared", name)
        }
        declared.Enter(name, typ)
    case ExprKindArray:
        if typ.Kind() != core.TypeKindArray {
            core.RaiseError(
                expr.Position(), 
                "cannot assign result of type '%s' to array", 
                typ.String())
        }
        array := expr.(*ArrayExpr)
        arrayType := typ.(*core.ArrayType)
        size := array.Size()
        for i := 0; i < size; i++ {
            declare(array.At(i), arrayType.ItemType(), declared)
        }
    case ExprKindTuple:
        if typ.Kind() != core.TypeKindTuple {
            core.RaiseError(
                expr.Position(), 
                "cannot assign result of type '%s' to tuple", 
                typ.String())
        }
        tuple := expr.(*TupleExpr)
        tupleType := typ.(*core.TupleType)
        size := tuple.Size()
        if tupleType.Size() != size {
            core.RaiseError(
                expr.Position(), 
                "cannot assign result of type '%s' to a tuple of size %d", 
                typ.String(), 
                size)
        }
        for i := 0; i < size; i++ {
            declare(tuple.At(i), tupleType.ItemType(i), declared)
        }
    default:
       core.RaiseError(expr.Position(), "expression not allowed in this context")
    }
}

func undeclare(expr Expr, declared *Declarations) {
    switch expr.Kind() {
    case ExprKindIdentifier:
        identifier := expr.(*IdentifierExpr)
        declared.Delete(identifier.Name())
    case ExprKindArray:
        array := expr.(*ArrayExpr)
        size := array.Size()
        for i := 0; i < size; i++ {
            undeclare(array.At(i), declared)
        }
    case ExprKindTuple:
        tuple := expr.(*TupleExpr)
        size := tuple.Size()
        for i := 0; i < size; i++ {
            undeclare(tuple.At(i), declared)
        }
    default:
        core.RaiseError(expr.Position(), "expression not allowed in this context")
    }
}

func deduceDataType(
        proto *core.Prototype, 
        args map[string]Expr,
        dataType *core.PrimitiveType,
        position *core.Position) *core.PrimitiveType {
    types := make(map[string]core.Type)
    for name, arg := range args {
        types[name] = arg.Type()
    }
    count := proto.ParamCount()
    for i := 0; i < count; i++ {
        param := proto.ParamAt(i)
        name := param.Name()
        if _, ok := types[name]; !ok {
            defaultValue := param.DefaultValue()
            core.Assert(!defaultValue.IsNone())
            types[name] = typeOf(defaultValue)
        }
    }
    dataType, mismatch := core.DeduceDataType(proto, types, dataType)
    if mismatch != nil {
        core.RaiseError(
            position, 
            "could not deduce data type: ambiguous candidates '%s' vs '%s'",
            mismatch[0].String(),
            mismatch[1].String())
    }
    return dataType
}

func resultType(proto *core.Prototype, dataType *core.PrimitiveType) core.Type {
    count := proto.ResultCount()
    if count == 1 {
        typ := proto.ResultAt(0).Type()
        if dataType != nil {
            return core.BindDataType(typ, dataType)
        } else {
            return typ
        }
    }
    types := make([]core.Type, count)
    for i := 0; i < count; i++ {
        typ := proto.ResultAt(i).Type()
        if dataType != nil {
            types[i] = core.BindDataType(typ, dataType)
        } else {
            types[i] = typ
        }
    }
    return core.GetTupleType(types)
}

func unaryResultType(argType core.Type, op core.Token) core.Type {
    switch op {
    case '-', '+':
        if argType == core.GetPrimitiveType(core.TypenameInteger) ||
                argType == core.GetPrimitiveType(core.TypenameScalar) ||
                argType == core.GetTensorType(core.TypenameScalar) {
            return argType
        }
    case '!':
        if argType == core.GetPrimitiveType(core.TypenameLogical) ||
                argType == core.GetTensorType(core.TypenameScalar) {
            return argType
        }
    }
    return nil
}

func binaryResultType(lhsType core.Type, rhsType core.Type, op core.Token) core.Type {
    if op == core.TokenIn && rhsType.Kind() == core.TypeKindArray {
        return core.GetPrimitiveType(core.TypenameLogical)
    }
    if op == '+' && lhsType.Kind() == core.TypeKindArray && rhsType == lhsType {
        return lhsType
    }
    if op == '*' {
        if lhsType.Kind() == core.TypeKindArray && 
                rhsType == core.GetPrimitiveType(core.TypenameInteger) {
            return lhsType
        }
        if rhsType.Kind() == core.TypeKindArray && 
                lhsType == core.GetPrimitiveType(core.TypenameInteger) {
            return rhsType
        }
    }        
    argType := core.CommonType(lhsType, rhsType, true)
    switch op {
    case '<', '>', core.TokenLe, core.TokenGe, core.TokenEq, core.TokenNe:
        if argType == core.GetTensorType(core.TypenameScalar) {
            return core.GetTensorType(core.TypenameLogical)
        } else {
            return core.GetPrimitiveType(core.TypenameLogical)
        }
    case '+', '*':
        if argType == core.GetPrimitiveType(core.TypenameString) {
            return argType
        }
        fallthrough
    case '-', '/', '^':
        if argType == core.GetPrimitiveType(core.TypenameInteger) ||
                argType == core.GetPrimitiveType(core.TypenameScalar) ||
                argType == core.GetTensorType(core.TypenameScalar) {
            return argType
        }
    case core.TokenAnd, core.TokenOr:
        if argType == core.GetPrimitiveType(core.TypenameLogical) ||
                argType == core.GetTensorType(core.TypenameScalar) {
            return argType
        }
    }
    return nil
}

func builtinResultType(op core.Token) core.Type {
    switch op {
    case core.TokenLengthOf:
        return core.GetPrimitiveType(core.TypenameInteger)
    case core.TokenShapeOf:
        return core.GetArrayType(core.GetPrimitiveType(core.TypenameInteger))
    case core.TokenRangeOf:
        return core.GetArrayType(core.GetPrimitiveType(core.TypenameInteger))
    case core.TokenInteger:
        return core.GetPrimitiveType(core.TypenameInteger)
    case core.TokenScalar:
        return core.GetPrimitiveType(core.TypenameScalar)
    case core.TokenString:
        return core.GetPrimitiveType(core.TypenameString)
    case core.TokenLogical:
        return core.GetPrimitiveType(core.TypenameLogical)
    }
    return nil
}

func typeOf(value core.Value) core.Type {
    switch value.Kind() {
    case core.ValueKindInteger:
        return core.GetPrimitiveType(core.TypenameInteger)
    case core.ValueKindScalar:
        return core.GetPrimitiveType(core.TypenameScalar)
    case core.ValueKindLogical:
        return core.GetPrimitiveType(core.TypenameLogical)
    case core.ValueKindString:
        return core.GetPrimitiveType(core.TypenameString)
    case core.ValueKindArray:
        var itemType core.Type
        if value.Size() != 0 {
            itemType = typeOf(value.At(0))
        }
        return core.GetArrayType(itemType)
    case core.ValueKindTuple:
        size := value.Size()
        itemTypes := make([]core.Type, size)
        for i := 0; i < size; i++ {
            itemTypes[i] = typeOf(value.At(i))
        }
        return core.GetTupleType(itemTypes)
    case core.ValueKindIdentifier, core.ValueKindNone:
        return nil
    }
    core.Assert(false)
    return nil
}

var precedenceMap = map[core.Token]int{
    core.TokenIn: 10,
    core.TokenAnd: 20,
    core.TokenOr: 20,
    core.TokenLe: 30,
    core.TokenGe: 30,
    core.TokenEq: 30,
    core.TokenNe: 30,
    '<': 30,
    '>': 30,
    '+': 40,
    '-': 40,
    '*': 50,
    '/': 50,
    '^': 60,
}

func tokenPrecedence(token core.Token) int {
    prec, ok := precedenceMap[token]
    if !ok {
        return -1
    }
    return prec
}

func unaryOpName(op core.Token) string {
    switch op {
    case '+':
        return "copy"
    case '-':
        return "neg"
    case '!':
        return "not"
    default:
        return ""
    }
}

func binaryOpName(op core.Token) string {
    switch op {
    case '+':
        return "add"
    case '-':
        return "sub"
    case '*':
        return "mul"
    case '/':
        return "div"
    case '^':
        return "pow"
    case '<':
        return "lt"
    case '>':
        return "gt"
    case core.TokenLe:
        return "le"
    case core.TokenGe:
        return "ge"
    case core.TokenEq:
        return "eq"
    case core.TokenNe:
        return "ne"
    case core.TokenAnd:
        return "and"
    case core.TokenOr:
        return "or"
    default:
        return ""
    }
}

func makeUnaryOpArgs(right Expr) ([]string, []Expr) {
    return []string{"x"}, []Expr{right}
}

func makeBinaryOpArgs(left Expr, right Expr) ([]string, []Expr) {
    return []string{"x", "y"}, []Expr{left, right}
}

func checkGraphParamType(value core.Value, typ core.Type) bool {
    switch value.Kind() {
    case core.ValueKindInteger:
        return (typ == core.GetPrimitiveType(core.TypenameInteger))
    case core.ValueKindScalar:
        return (typ == core.GetPrimitiveType(core.TypenameScalar))
    case core.ValueKindLogical:
        return (typ == core.GetPrimitiveType(core.TypenameLogical))
    case core.ValueKindString:
        return (typ == core.GetPrimitiveType(core.TypenameString))
    case core.ValueKindIdentifier:
        return (typ == core.UnboundTensorType())
    case core.ValueKindArray:
        if typ.Kind() != core.TypeKindArray {
            return false
        }
        arrayType := typ.(*core.ArrayType)
        itemType := arrayType.ItemType()
        size := value.Size()
        for i := 0; i < size; i++ {
            if !checkGraphParamType(value.At(i), itemType) {
                return false
            }
        }
        return true
    case core.ValueKindTuple:
        if typ.Kind() != core.TypeKindTuple {
            return false
        }
        tupleType := typ.(*core.TupleType)
        size := value.Size()
        for i := 0; i < size; i++ {
            if !checkGraphParamType(value.At(i), tupleType.ItemType(i)) {
                return false
            }
        }
        return true
    case core.ValueKindNone:
        return false
    }
    return false
}

