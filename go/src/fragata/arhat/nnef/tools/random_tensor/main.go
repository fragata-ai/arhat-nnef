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
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package main

import (
    "fmt"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "fragata/arhat/nnef/tools/io/input"
    "fragata/arhat/nnef/tools/util"
)

//
//    Main program
//

func main() {
    err := run()
    if err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
}

func run() error {
    var err error
    args, err := getArgs()
    if err != nil {
        return err
    }
    if args.seed >= 0 {
        rand.Seed(int64(args.seed))
    }
    source, err := createInput(args.params)
    if err != nil {
        return err
    }
    data, shape, err := source.CreateInput(args.dtype, args.shape, false)
    if err != nil {
        return err
    }
    err = util.WriteTensorFile(args.output, args.dtype, shape, data)
    if err != nil {
        return err
    }
    return nil
}

//
//    Local functions
//

func createInput(params []string) (input.InputSource, error) {
    switch params[0] {
    case "uniform":
        p, err := getFloats(params, 2)
        if err != nil {
            return nil, err
        }
        source := input.NewUniformInput(p[0], p[1])
        return source, nil
    case "normal":
        p, err := getFloats(params, 2)
        if err != nil {
            return nil, err
        }
        source := input.NewNormalInput(p[0], p[1])
        return source, nil
    case "binomial":
        p, err := getFloats(params, 2)
        if err != nil {
            return nil, err
        }
        source := input.NewBinomialInput(int(p[0]), p[1])
        return source, nil
    case "bernoulli":
        p, err := getFloats(params, 1)
        if err != nil {
            return nil, err
        }
        source := input.NewBernoulliInput(p[0])
        return source, nil
    default:
        err := fmt.Errorf("Invalid random input algorithm: %s", params[0])
        return nil, err
    }
}

func getFloats(params []string, count int) ([]float32, error) {
    if len(params) - 1 != count {
        err := 
            fmt.Errorf(
                "%s: invalid number of parameters: need %d, got %d", 
                    params[0], count, len(params)-1)
        return nil, err
    }
    result := make([]float32, count)
    for i := 0; i < count; i++ {
        value, err := strconv.ParseFloat(params[i+1], 32)
        if err != nil {
            err = 
                fmt.Errorf(
                    "%s: parameter %d: invalid value: %s", 
                        params[0], i+1, params[i+1])
            return nil, err
        }
        result[i] = float32(value)
    }
    return result, nil
}

//
//    Argument parser
//

type Args struct {
    params []string
    output string
    shape []int
    seed int
    dtype string
}

func getArgs() (*Args, error) {
    var err error
    var params []string
    var output string
    var shape []int
    seed := -1
    var dtype string
    parser := util.NewArgParser(os.Args)
    params, err = parser.GetStrings(0, -1)
    if err != nil {
        return nil, err
    }
    for !parser.Done() {
        switch option := parser.Option(); option {
        case "--output":
            output, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        case "--shape":
            shape, err = parser.GetInts(1, -1)
            if err != nil {
                return nil, err
            }
        case "--seed":
            seed, err = parser.GetInt()
            if err != nil {
                return nil, err
            }
        case "--dtype":
            dtype, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        default:
            err = fmt.Errorf("Invalid option: %s", option)
            return nil, err
        }
    }
    if len(params) == 0 {
        params = []string{"uniform", "0", "1"}
    }
    params[0] = strings.ToLower(params[0])
    if len(output) == 0 {
        err = fmt.Errorf("Missing --output option")
        return nil, err
    }
    if len(shape) == 0 {
        err = fmt.Errorf("Missing --shape option")
        return nil, err
    }
    if dtype == "" {
        switch params[0] {
        case "binomial":
            dtype = "integer"
        case "bernoulli":
            dtype = "logical"
        default:
            dtype = "scalar"
        }
    }
    args := &Args{
        params: params,
        output: output,
        shape: shape,
        seed: seed,
        dtype: dtype,
    }
    return args, nil
}

