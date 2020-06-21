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

package main

import (
    "fmt"
    "os"
    "fragata/arhat/nnef/core"
    "fragata/arhat/nnef/engine"
)

func main() {
    if len(os.Args) < 2 {
        fmt.Fprintf(os.Stderr,  "Input file name must be provided\n")
        os.Exit(1)
    }
    path := os.Args[1]
    nnef := engine.NewEngine(nil)
    graph := new(core.Graph)
    err := nnef.LoadGraph(path, graph, "", nil)
    if  err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
    err = nnef.InferShapes(graph, nil, nil)
    if  err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
    fmt.Fprintf(os.Stderr, "Successfully parsed file: %s\n", path)
    fmt.Fprintf(os.Stderr, "Graph '%s': operations %d, tensors %d\n",
        graph.Name(), graph.OperationCount(), graph.TensorCount())
}

