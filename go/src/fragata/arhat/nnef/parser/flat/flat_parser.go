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

package flat

import (
    "io"
    "fragata/arhat/nnef/core"
)

//
//    FlatParser
//

type FlatParser struct {}

func(p *FlatParser) Parse(is io.Reader, filename string, callback core.ParserCallback) {
    lexer := core.NewLexer(is, filename)
    lexer.Next()
    version := core.ReadVersion(lexer)
    callback.BeginDocument(filename, version)
    handle := func(ext string) bool { 
        return callback.HandleExtension(ext)
    }
    core.ReadExtensions(lexer, handle)
    prototypes := buildPrototypes()
    parseGraph(lexer, prototypes, callback)
    callback.EndDocument(filename)
}

func parseGraph(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype, 
        callback core.ParserCallback) {
    lexer.ReadToken(core.TokenGraph)
    name := lexer.Str()  
    lexer.ReadToken(core.TokenIdentifier)
    params := parseParamIdentifiers(lexer)
    lexer.ReadToken(core.TokenArrow)
    results := parseResultIdentifiers(lexer)
    graph := core.NewPrototype(name, params, results, nil)
    callback.BeginGraph(graph, prototypes)
    lexer.ReadToken('{')
    dtypes := make(map[string]core.Typename)
    for lexer.Token() != '}' {
        parseAssignment(lexer, graph, prototypes, dtypes, callback)
    }        
    checkGraphParamsAssigned(graph, dtypes, lexer.Position())
    lexer.ReadToken('}')
    callback.EndGraph(graph, dtypes)
    lexer.ReadToken(core.TokenEof)
}

func parseParamIdentifiers(lexer *core.Lexer) []*core.Param {
    var identifiers []*core.Param
    lexer.ReadToken('(')
    for {
        id := lexer.Str()
        lexer.ReadToken(core.TokenIdentifier)
        param := core.NewParam(id, core.GetTensorType(core.TypenameScalar), core.None())
        identifiers = append(identifiers, param)
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    lexer.ReadToken(')') 
    return identifiers
}

func parseResultIdentifiers(lexer *core.Lexer) []*core.Result {
    var identifiers []*core.Result
    lexer.ReadToken('(')
    for {
        id := lexer.Str()
        lexer.ReadToken(core.TokenIdentifier)
        result := core.NewResult(id, core.GetTensorType(core.TypenameScalar))
        identifiers = append(identifiers, result)
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    lexer.ReadToken(')') 
    return identifiers
}

func checkGraphParam(
        arg core.Value, graph *core.Prototype, target string, position *core.Position) {
    switch arg.Kind() {
    case core.ValueKindIdentifier:
        name := arg.Identifier()
        if target == "external" {
            if graph.GetParam(name) == nil {
                core.RaiseError(
                    position, 
                    "identifier '%s' assigned by operation 'external' must be a graph parameter",
                    name)
            }
        } else {
            if graph.GetParam(name) != nil {
                core.RaiseError(
                    position, 
                    "graph parameter '%s' can only be assigned by operation 'external'",
                    name)
            }
        }
    case core.ValueKindArray, core.ValueKindTuple:
        size := arg.Size()
        for i := 0; i < size; i++ {
            checkGraphParam(arg.At(i), graph, target, position)
        }
    default:
        core.Assert(false)
    }
}

func checkGraphParamsAssigned(
        graph *core.Prototype, declared map[string]core.Typename, position *core.Position) {
    for i := 0; i < graph.ParamCount(); i++ {
        param := graph.ParamAt(i)
        if _, ok := declared[param.Name()]; !ok {
            core.RaiseError(position, "graph parameter '%s' not assigned", param.Name())
        }
    }        
    for i := 0; i < graph.ResultCount(); i++ {
        result := graph.ResultAt(i)
        if _, ok := declared[result.Name()]; !ok {
            core.RaiseError(position, "graph result '%s' not assigned", result.Name())
        }
    }
}

func parseAssignment(
        lexer *core.Lexer, 
        graph *core.Prototype, 
        prototypes map[string]*core.Prototype,
        dtypes map[string]core.Typename, 
        callback core.ParserCallback) {
    position := lexer.Position()
    results := parseTuple(lexer, nil, false, true)
    lexer.ReadToken('=')
    target := lexer.Str()
    lexer.ReadToken(core.TokenIdentifier)
    proto, ok := prototypes[target]
    if !ok {
        core.RaiseError(lexer.Position(), "undefined operation '%s'", target)
    }        
    checkGraphParam(results, graph, proto.Name(), position)
    dataType := proto.GenericParamDefault()
    if lexer.ReadIfToken('<') {
        if (lexer.Token() == '?') {
            core.RaiseError(lexer.Position(), "expected type name")
        }        
        dataType = core.GetPrimitiveType(core.GetTypename(lexer))
        lexer.Next()
        lexer.ReadToken('>')
    }
    lexer.ReadToken('(')
    args := parseArguments(proto, lexer, dtypes, dataType, true, false, nil)
    lexer.ReadToken(')')
    lexer.ReadToken(';')
    if results.Size() != proto.ResultCount() {
        // ACHTUNG: Apparent bug in original code: "%s" instead of "%d"
        core.RaiseError(
            position, 
            "left-hand-side item count must match result count of operation (%d)",
            proto.ResultCount())
    }
    if proto.IsGeneric() && dataType == nil {
        dataType = deduceDataType(proto, args, dtypes, position)
        if dataType == nil {
            core.RaiseError(position, "could not deduce generic data type")
        }
    }
    if dataType != nil {
        args["?"] = core.NewStringValue(dataType.String())
    }
    for i := 0; i < proto.ResultCount(); i++ {
        result := proto.ResultAt(i)
        typ := result.Type()
        if dataType != nil {
            typ = core.BindDataType(typ, dataType)
        }
        declare(results.At(i), typ, dtypes, position)
        args[result.Name()] = results.At(i)
    }        
    callback.Operation(proto, args, dtypes)
}

func parseArguments(
        proto *core.Prototype, 
        lexer *core.Lexer,
        decls map[string]core.Typename,
        dataType *core.PrimitiveType,
        allowIdentifier bool, 
        expectNamed bool,
        exclusion *core.Param) map[string]core.Value {
    args := make(map[string]core.Value)
    var position core.Position
    for {
        position = *lexer.Position()
        if len(args) >= proto.ParamCount() {
            core.RaiseError(
                &position, 
                "too many arguments; definition of '%s' has only %d parameters",
                proto.Name(), 
                proto.ParamCount())
        }
        var param *core.Param
        var arg core.Value = core.None()
        named := false
        if lexer.Token() == core.TokenIdentifier {
            str := lexer.Str()
            lexer.Next()
            if lexer.Token() == '=' {
                lexer.Next()
                param = proto.GetParam(str)
                if param == nil {
                    core.RaiseError(
                        &position, 
                        "operation '%s' has no parameter called '%s'",
                        proto.Name(), 
                        str)
                }
                arg = parseValue(lexer, decls, true, allowIdentifier)
                named = true
            } else if allowIdentifier {
                param = proto.ParamAt(len(args))
                arg = makeIdentifier(str, &position, decls)
            } else {
                core.RaiseError(&position, "token 'identifier' not allowed in this context")
            }
        } else {
            param = proto.ParamAt(len(args))
            arg = parseValue(lexer, decls, true, allowIdentifier)
        }
        paramType := param.Type()
        if dataType != nil {
            paramType = core.BindDataType(paramType, dataType)
        }
        argType := typeOf(arg, decls)
        if !core.IsCastable(argType, paramType, true) {
            core.RaiseError(
                &position,
                "argument of type '%s' cannot be cast to type '%s' for parameter '%s'",
                argType.String(), 
                paramType.String(), 
                param.Name())
        }
        expectNamed = (expectNamed || named || paramType.IsAttribute())
        if expectNamed && !named {
            core.RaiseError(&position, "expected named argument")
        }
        if _, ok := args[param.Name()]; ok {
            // ACHTUNG: Apparent bug in origina code (extra " (%u,%u)")
            core.RaiseError(
                &position, 
                "duplicate arguments: parameter '%s' already assigned",
                param.Name())
        }
        if param == exclusion {
            core.RaiseError(
                lexer.Position(), 
                "argument '%s' of operation '%s' must not be provided in this context",
                param.Name(), 
                proto.Name())
        }
        args[param.Name()] = arg
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    for i := 0; i < proto.ParamCount(); i++ {
        param := proto.ParamAt(i)
        if param == exclusion {
            continue
        }
        if _, ok := args[param.Name()]; !ok {
            defaultValue := param.DefaultValue()
            if defaultValue != nil {
                if param.Type().IsGeneric() {
                    valueType := typeOf(defaultValue, decls)
                    paramType := param.Type()
                    if dataType != nil {
                        paramType = core.BindDataType(paramType, dataType)
                    }
                    if !core.IsCastable(valueType, paramType, true) {
                        core.RaiseError(
                            lexer.Position(), 
                            "default value type '%s' cannot be cast to "+
                                "type '%s' for parameter '%s'",
                            valueType.String(), 
                            paramType.String(), 
                            param.Name())
                    }
                }
                args[param.Name()] = defaultValue
            } else {
                core.RaiseError(
                    lexer.Position(), 
                    "missing argument for operation '%s'; parameter '%s' not assigned",
                    proto.Name(), 
                    param.Name())
            }
        }
    }
    return args
}

func declare(
        arg core.Value, 
        typ core.Type, 
        dtypes map[string]core.Typename, 
        position *core.Position) {
    switch arg.Kind() {
    case core.ValueKindIdentifier:
        if typ.Kind() != core.TypeKindTensor {
            core.RaiseError(
                position, 
                "cannot assign result of type '%s' to tensor identifier", 
                typ.String())
        }
        id := arg.Identifier()
        if _, ok := dtypes[id]; ok {
            core.RaiseError(position, "identifier '%s' already declared", id)
        }
        dataType := typ.(*core.TensorType).DataType()
        core.Assert(dataType.Kind() == core.TypeKindPrimitive)
        dtypes[id] = dataType.(*core.PrimitiveType).Name()
    case core.ValueKindArray:
        if typ.Kind() != core.TypeKindArray {
            core.RaiseError(position, "cannot assign result of type '%s' to array", typ.String())
        }
        arrayType := typ.(*core.ArrayType)
        for i := 0; i < arg.Size(); i++ {
            declare(arg.At(i), arrayType.ItemType(), dtypes, position)
        }
    case core.ValueKindTuple:
        if typ.Kind() != core.TypeKindTuple {
            core.RaiseError(position, "cannot assign result of type '%s' to tuple", typ.String())
        }
        tupleType := typ.(*core.TupleType)
        for i := 0; i < arg.Size(); i++ {
            declare(arg.At(i), tupleType.ItemType(i), dtypes, position)
        }
    default:
        core.RaiseError(position, "literal expression not allowed in this context")
    }
}

func parseValue(
        lexer *core.Lexer, 
        decls map[string]core.Typename, 
        allowLiteral bool, 
        allowIdentifier bool) core.Value {
    switch lexer.Token() {
    case core.TokenTrue, core.TokenFalse:
        if allowLiteral {
            return parseLogical(lexer)
        }
    case '-', core.TokenDecimal, core.TokenFractional:
        if allowLiteral {
            return parseNumber(lexer)
        }
    case core.TokenCharacters:
        if allowLiteral {
            return parseString(lexer)
            }
    case '[':
        return parseArray(lexer, decls, allowLiteral, allowIdentifier)
    case '(':
        return parseTuple(lexer, decls, allowLiteral, allowIdentifier)
    case core.TokenIdentifier:
        if allowIdentifier {
            return parseIdentifier(lexer, decls)
        }
    default:
        core.RaiseError(lexer.Position(), "unexpected token '%s'", lexer.Token().String())
    }
    core.RaiseError(
        lexer.Position(), 
        "token '%s' not allowed in this context", 
        lexer.Token().String())
    return nil
}

func parseNumber(lexer *core.Lexer) core.Value {
    negative := (lexer.Token() == '-')
    if negative {
        lexer.Next()
    }
    switch lexer.Token() {
    case core.TokenDecimal:
        return parseInteger(lexer, negative)
    case core.TokenFractional:
        return parseScalar(lexer, negative)
    default:
        core.RaiseError(lexer.Position(), "expected number")
    }
    return nil
}

func parseInteger(lexer *core.Lexer, negative bool) core.Value {
    value := core.GetIntegerValue(lexer)
    lexer.Next()
    if negative {
        value = -value
    }
    return core.NewIntegerValue(value)
}

func parseScalar(lexer *core.Lexer, negative bool) core.Value {
    value := core.GetScalarValue(lexer)
    lexer.Next()
    if negative {
        value = -value
    }
    return core.NewScalarValue(value)
}

func parseLogical(lexer *core.Lexer) core.Value {
    value := (lexer.Token() == core.TokenTrue)
    lexer.Next()
    return core.NewLogicalValue(value)
}

func parseString(lexer *core.Lexer) core.Value {
    value := lexer.Str()
    lexer.Next()
    return core.NewStringValue(value)
}

func parseIdentifier(lexer *core.Lexer, decls map[string]core.Typename) core.Value {
    value := makeIdentifier(lexer.Str(), lexer.Position(), decls)
    lexer.Next()
    return value
}

func makeIdentifier(
        name string, 
        position *core.Position, 
        decls map[string]core.Typename) core.Value {
    if _, ok := decls[name]; !ok {
        core.RaiseError(position, "undeclared identifier '%s'", name)
    }
    return core.NewIdentifierValue(name)
}

func parseArray(
       lexer *core.Lexer, 
       decls map[string]core.Typename, 
       allowLiteral bool, 
       allowIdentifier bool) core.Value {
    lexer.ReadToken('[')
    var items []core.Value
    if lexer.Token() != ']' {
        for {
            item := parseValue(lexer, decls, allowLiteral, allowIdentifier)
            items = append(items, item)
            if !lexer.ReadIfToken(',') {
                break
            }
        }
    }
    lexer.ReadToken(']')
    return core.NewArrayValue(items, false)
}

func parseTuple(
        lexer *core.Lexer, 
        decls map[string]core.Typename, 
        allowLiteral bool, 
        allowIdentifier bool) core.Value {
    var items []core.Value
    parenthesized := (lexer.Token() == '(')
    if parenthesized {
        lexer.Next()
        first := parseValue(lexer, decls, allowLiteral, allowIdentifier)
        lexer.ReadToken(',')
        items = append(items, first)
    }
    for {
        item := parseValue(lexer, decls, allowLiteral, allowIdentifier)
        items = append(items, item)
        if !lexer.ReadIfToken(',') {
            break
        }
    }
    if parenthesized {
        lexer.ReadToken(')')
    }
    return core.NewTupleValue(items, false)
}

func typeOf(value core.Value, declared map[string]core.Typename) core.Type {
    switch value.Kind() {
    case core.ValueKindInteger:
        return core.GetPrimitiveType(core.TypenameInteger)
    case core.ValueKindScalar:
        return core.GetPrimitiveType(core.TypenameScalar)
    case core.ValueKindLogical:
        return core.GetPrimitiveType(core.TypenameLogical)
    case core.ValueKindString:
        return core.GetPrimitiveType(core.TypenameString)
    case core.ValueKindIdentifier:
        name := value.Identifier()
        return core.GetTensorType(declared[name])
    case core.ValueKindArray:
        var itemType core.Type
        if value.Size() != 0 {
            itemType = typeOf(value.At(0), declared)
        }
        return core.GetArrayType(itemType)
    case core.ValueKindTuple:
        itemTypes := make([]core.Type, value.Size())
        for i := 0; i < value.Size(); i++ {
            itemTypes[i] = typeOf(value.At(i), declared)
        }
        return core.GetTupleType(itemTypes)
    case core.ValueKindNone:
        return nil
    }
    core.Assert(false)
    return nil
}

func deduceDataType(
        proto *core.Prototype, 
        args map[string]core.Value, 
        declared map[string]core.Typename,
        position *core.Position) *core.PrimitiveType {
    types := make(map[string]core.Type)
    for name, arg := range args {
        types[name] = typeOf(arg, declared)
    }
    for i := 0; i < proto.ParamCount(); i++ {
        param := proto.ParamAt(i)
        if _, ok := types[param.Name()]; !ok {
            core.Assert(param.DefaultValue() != nil)
            types[param.Name()] = typeOf(param.DefaultValue(), declared)
        }
    }
    dataType, mismatch := core.DeduceDataType(proto, types, nil)
    if mismatch != nil {
        core.RaiseError(
            position, 
            "could not deduce data-type: ambiguous candidates '%s' vs '%s'",
            mismatch[0].String(),
            mismatch[1].String())
    }
    return dataType
}

var stdlibPrototypes = StdlibPrototypes()

func buildPrototypes() map[string]*core.Prototype {
    stdlibPrototypes := stdlibPrototypes
    prototypes := make(map[string]*core.Prototype)
    for _, proto := range stdlibPrototypes {
        prototypes[proto.Name()] = proto
    }
    return prototypes
}

