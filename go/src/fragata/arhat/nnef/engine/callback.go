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

package engine

import (
    "io"
    "sort"
    "fragata/arhat/nnef/core"
    "fragata/arhat/nnef/parser/flat"
)

//
//    ParseCallback
//

type ParseCallback struct {
    core.ParserCallbackBase
    graph *core.Graph
    qis io.Reader
    qfn string
    quantizations map[string]map[string]core.Value
}

func NewParseCallback(graph *core.Graph, qis io.Reader, qfn string) *ParseCallback {
    c := new(ParseCallback)
    c.Init(graph, qis, qfn)
    return c
}

func(c *ParseCallback) Init(graph *core.Graph, qis io.Reader, qfn string) {
    c.graph = graph
    c.qis = qis
    c.qfn = qfn
}

func(c *ParseCallback) BeginGraph(
        proto *core.Prototype, fragments map[string]*core.Prototype) {
    c.graph.SetName(proto.Name())
    c.graph.ClearOperations()
    c.graph.ClearTensors()
    c.graph.ResizeInputs(proto.ParamCount())
    for i := 0; i < proto.ParamCount(); i++ {
        c.graph.SetInput(i, proto.ParamAt(i).Name())
    }
    c.graph.ResizeOutputs(proto.ResultCount())
    for i := 0; i < proto.ResultCount(); i++ {
        c.graph.SetOutput(i, proto.ResultAt(i).Name())
    }        
    if c.qis != nil {
        c.quantizations = flat.ParseQuant(c.qis, c.qfn, fragments)
    }
}

func(c *ParseCallback) EndGraph(proto *core.Prototype, dtypes map[string]core.Typename) {
    names := sortDtypeNames(dtypes)
    for _, name := range names {
        dtype := dtypes[name]
        tensor := new(core.Tensor)
        tensor.SetName(name)
        tensor.SetDtype(dtype.String())
        if quant, ok := c.quantizations[name]; ok {
            for key, value := range quant {
                tensor.AddQuantization(key, value)
            }
        }
        c.graph.AddTensor(name, tensor)
    }
}

func(c *ParseCallback) Operation(
        proto *core.Prototype, 
        args map[string]core.Value, 
        dtypes map[string]core.Typename) {
    operation := new(core.Operation)
    operation.SetName(proto.Name())
    if genericDtype, ok := args["?"]; ok {
        operation.SetDtype(genericDtype.String())
    } else {
        operation.SetDtype("")
    }
    for i := 0; i < proto.ParamCount(); i++ {
        param := proto.ParamAt(i)
        name := param.Name()
        value := args[name]
        if param.Type().IsAttribute() {
            operation.AddAttrib(name, value)
        } else {
            operation.AddInput(name, value)
        }
    }
    for i := 0; i < proto.ResultCount(); i++ {
        result := proto.ResultAt(i)
        name := result.Name()
        value := args[name]
        operation.AddOutput(name, value)
    }        
    c.graph.AddOperation(operation)
}

func sortDtypeNames(dtypes map[string]core.Typename) []string {
    // establish deterministic ordering: required by code emitters
    var result []string
    for name, _ := range dtypes {
        result = append(result, name)
    }
    sort.Strings(result)
    return result
}

