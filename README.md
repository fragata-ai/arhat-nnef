# Arhat NNEF Tools

Arhat NNEF Tools is a software package implemented in pure [Go](https://golang.org/)
containing libraries and command line tools supporting 
[Khronos Neural Network Exchange Format](https://www.khronos.org/nnef)
(NNEF™).

NNEF represents a powerful, flexible, and elegant domain-specific language for 
description of neural networks. Its primary goal is enabling easy transfer of 
trained models between various platforms, machine learning frameworks, and inference engines. 
The qualities that set NNEF apart are concise human-readable model descriptions as well as 
a self-contained design with no external dependencies.

Arhat NNEF Tools implementation is based on Khronos 
[NNEF Tools](https://github.com/KhronosGroup/NNEF-Tools).
We have ported to Go and partly redesigned sections of the original C++ code.

The package currently includes:

* NNEF parser
* NNEF reference intepreter backend
* Tools for generation of NNEF tensor data

Arhat NNEF Tools are fully compatible with the original Khronos
NNEF Tools, in particular, the original format converters can be used
with Arhat NNEF Tools.

## Coming soon

In the near future we plan to publish a turorial on using Arhat NNEF Toools
together with deep learning models from
[NNEF Model Zoo](https://github.com/KhronosGroup/NNEF-Tools/tree/master/models#nnef-model-zoo)
for image classification.

## Trademarks

NNEF and the NNEF logo are trademarks of the Khronos Group Inc.

## License

We are releasing Arhat NNEF Tools under an open source 
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License.

 
