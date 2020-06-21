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
    "io/ioutil"
    "math"
    "math/rand"
    "os"
    "strings"
    "fragata/arhat/nnef/core"
    "fragata/arhat/nnef/dnn/reference"
    "fragata/arhat/nnef/engine"
)

var lowered = map[string]bool {
    "separable_conv": true,
    "separable_deconv": true,
    "rms_pool": true,
    "local_response_normalization": true,
    "local_mean_normalization": true,
    "local_variance_normalization": true,
    "local_contrast_normalization": true,
    "l1_normalization": true,
    "l2_normalization": true,
    "batch_normalization": true,
    "area_downsample": true,
    "nearest_downsample": true,
    "nearest_upsample": true,
    "linear_quantize": true,
    "logarithmic_quantize": true,
    "leaky_relu": true,
    "prelu": true,
    "clamp": true,
}

func main() {
    argv := os.Args
    argc := len(argv) 
    if argc < 2 {
        fmt.Fprintf(os.Stderr, "Input file name must be provided\n")
        os.Exit(1)
    }
    var err error
    path := argv[1]
    var stdlib string
    var inputs []string
    var outputs []string
    compare := false
    for i := 2; i < argc; i++ {
        arg := argv[i]
        switch arg {
        case "--stdlib":
            i++
            if i == argc {
                fmt.Fprintf(os.Stderr, 
                    "Stdlib file name must be provided after --stdlib; ignoring option\n")
            }
            stdlib, err = readFile(argv[i])
            if err != nil {
                fmt.Fprintf(os.Stderr, "%s\n", err.Error())
            }
        case "--input":
            if i + 1 == argc {
                fmt.Fprintf(os.Stderr,
                    "Input file name(s) must be provided after --input; ignoring option\n")
            }
            for i + 1 < argc && !strings.HasPrefix(argv[i+1], "-") {
                i++
                inputs = append(inputs, argv[i])
            }
        case "--output":
            if i + 1 == argc {
                fmt.Fprintf(os.Stderr,
                    "Output file name(s) must be provided after --output; ignoring option\n")
            }
            for i + 1 < argc && !strings.HasPrefix(argv[i+1], "-") {
                i++
                outputs = append(outputs, argv[i])
            }
        case "--compare":
            compare = true
        default:
            fmt.Fprintf(os.Stderr, "Unrecognized option: '%s'; ignoring\n", argv[i])
        }
    }
    dnn := reference.NewEngine()
    nnef := engine.NewEngine(dnn)
    graph := new(core.Graph)
    err = nnef.LoadGraph(path, graph, stdlib, lowered)
    if  err != nil {
        signalError(err)
    }
    inputShapes := make(map[string]core.Shape)
    if len(inputs) != 0 {
        err = readInputsFromFile(nnef, graph, inputs)
        if err != nil {
            signalError(err)
        }
        count := graph.InputCount()
        for i := 0; i < count; i++ {
            input := graph.InputAt(i)
            inputShapes[input] = graph.GetTensor(input).Shape()
        }
    }
    err = nnef.InferShapes(graph, inputShapes, nil)
    if err != nil {
        signalError(err)
    }
    if len(inputs) == 0 {
        err = generateRandomInputs(nnef, graph)
        if err != nil {
            signalError(err)
        }
    }
    fmt.Fprintf(os.Stderr, "Executing model: %s\n", path)
    err = nnef.Execute(graph)
    if err != nil {
        signalError(err)
    }
    if len(outputs) != 0 {
        if compare {
            count := graph.OutputCount()
            for i := 0; i < count; i++ {
                name := graph.OutputAt(i)
                output := graph.GetTensor(name)
                tensor := new(core.Tensor)
                err = nnef.ReadTensorFile(outputs[i], tensor)
                if err != nil {
                    signalError(err)
                }
                outputShape := core.Shape(output.Shape())
                tensorShape := core.Shape(tensor.Shape())
                switch {
                // ACHTUNG: Apparent bug in original code: 'tensor.dtype != tensor.dtype'
                case output.Dtype() != tensor.Dtype():
                    fmt.Fprintf(os.Stderr, 
                        "data type %s of '%s' does not match reference data type %s\n",
                            output.Dtype(), name, tensor.Dtype())
                case !outputShape.Eq(tensorShape):
                    fmt.Fprintf(os.Stderr, 
                        "shape %s of '%s' does not match reference shape %s\n",
                            outputShape.String(), name, tensorShape.String())
                default:
                    tensorData := tensor.ScalarData()
                    outputData := output.ScalarData()
                    if len(outputData) != len(tensorData) {
                        fmt.Fprintf(os.Stderr, 
                            "volume %d does not match reference volume %d\n",
                                len(outputData), len(tensorData))

                    } else {
                        diff := relativeDifference(tensorData, outputData)
                        fmt.Printf("'%s' diff = %g\n", name, diff)
                    }
                }
            }
        } else {
            err = writeOutputToFile(nnef, graph, outputs)
            if err != nil {
                signalError(err)
            }
        }
    }
    // ACHTUNG: For diagnostic of Imagent models only; disable otherwise
    printClasses(graph, 3)
}

func readFile(fn string) (string, error) {
    buf, err := ioutil.ReadFile(fn)
    if err != nil {
        return "", err
    }
    return string(buf), nil
}

func readInputsFromFile(nnef *engine.Engine, graph *core.Graph, inputs []string) error {
    count := graph.InputCount()
    for i := 0; i < count; i++ {
        input := graph.InputAt(i)
        tensor := graph.GetTensor(input)
        err := nnef.ReadTensorFile(inputs[i], tensor)
        if err != nil {
            return err
        }
    }
    return nil
}

func generateRandomInputs(nnef *engine.Engine, graph *core.Graph) error {
    count := graph.InputCount()
    for i := 0; i < count; i++ {
        input := graph.InputAt(i)
        tensor := graph.GetTensor(input)
        tensor.ResizeData()
        volume := core.Shape(tensor.Shape()).VolumeOf()
        switch tensor.Dtype() {
        case "scalar":
            data := make([]float32, volume)
            for i := 0; i < volume; i++ {
                data[i] = rand.Float32()
            }
            copy(tensor.ScalarData(), data)
        case "integer":
            data := make([]int, volume)
            for i := 0; i < volume; i++ {
                data[i] = rand.Int()
            }
            copy(tensor.IntegerData(), data)
        case "logical":
            data := make([]bool, volume)
            for i := 0; i < volume; i++ {
                data[i] = (rand.Int() >= 0)
            }
            copy(tensor.LogicalData(), data)
        }
    }
    return nil
}

func writeOutputToFile(nnef *engine.Engine, graph *core.Graph, outputs []string) error {
    count := graph.OutputCount()
    for i := 0; i < count; i++ {
        output := graph.OutputAt(i)
        tensor := graph.GetTensor(output)
        err := nnef.WriteTensorFile(outputs[i], tensor)
        if err != nil {
            return err
        }
    }
    return nil
}

// ACHTUNG: Temporary code for printing Imagenet classes
func printClasses(graph *core.Graph, topk int) {
    if (topk < 1) {
        topk = 1
    } else if (topk > 5) {
        topk = 5
    }
    output := graph.OutputAt(0)
    tensor := graph.GetTensor(output)
    data := tensor.ScalarData();
    for b := 0; b < 10; b++ {
        pos, val := TopK(data[1000*b:1000*(b+1)], topk)
        for i := 0; i < topk; i++ {
            className := ClassName(pos[i])
            if i == 0 {
                fmt.Printf("[%d]", b)
            } else {
                fmt.Printf("   ")
            }
            fmt.Printf(" class %d prob %5.2f%% %s\n", pos[i], 100.0*val[i], className)
        }
    }
}

func relativeDifference(ref []float32, dat []float32) float32 {
    diff := float32(0.0)
    rng := float32(0.0)
    n := len(ref)
    for i := 0; i < n; i++ {
        diff += sqr(ref[i]-dat[i])
        rng += sqr(ref[i])
    }
    return float32(math.Sqrt(float64(diff/rng)))
}

func sqr(x float32) float32 {
    return x * x
}

func signalError(err error) {
    fmt.Fprintf(os.Stderr, "%s\n", err.Error())
    os.Exit(1)
}

