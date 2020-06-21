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

func ParseQuant(
        is io.Reader, 
        filename string, 
        prototypes map[string]*core.Prototype) map[string]map[string]core.Value {
    lexer := core.NewLexer(is, filename)
    lexer.Next()
    quantization := make(map[string]map[string]core.Value)
    for line := uint(0); lexer.Token() != core.TokenEof; line++ {
        tensor := lexer.Str()
        if _, ok := quantization[tensor]; ok {
            core.RaiseError(
                lexer.Position(), "duplicate quantization entries for tensor '%s'", tensor)
        }
        lexer.ReadToken(core.TokenCharacters)
        lexer.ReadToken(':')
        args := parseInvocation(lexer, prototypes)
        quantization[tensor] = args
    }
    return quantization
}

func parseInvocation(
        lexer *core.Lexer, 
        prototypes map[string]*core.Prototype) map[string]core.Value {
    position := *lexer.Position()
    op := lexer.Str()
    lexer.ReadToken(core.TokenIdentifier)
    proto, ok := prototypes[op]
    if !ok {
        core.RaiseError(&position, "undefined quantization operation '%s'", op)
    }
    if proto.ParamCount() == 0 {
        core.RaiseError(&position, "quantization operation must have at least one parameter")
    }
    if proto.ParamAt(0).Type().Kind() != core.TypeKindTensor {
        core.RaiseError(
            &position, "first parameter of quantization operation must be of type tensor")
    }
    lexer.ReadToken('(')
    args := parseArguments(proto, lexer, nil, nil, false, false, proto.ParamAt(0))
    lexer.ReadToken(')')
    lexer.ReadToken(';')
    args["op-name"] = core.NewStringValue(op)
    return args
}

