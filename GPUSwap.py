#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, warnings, sys


# https://www.sharcnet.ca/help/index.php/Porting_CUDA_to_OpenCL
class gpuswap():
    """
        Basic qualifiers
    """
    cuda_qualifiers = [
        "__global__",
        "__constant",
        "__device__",
        "__shared__"
    ]

    opencl_qualifiers = [
        "__kernel",
        "__constant",
        "__global",
        "__local"
    ]

    """
        Synchronization references
    """

    cuda_syncs = [
        "__syncthreads()",
        "__threadfence()",
        "__threadfence_block()",
        "cudaDeviceSynchronize()",
        "cudaStreamSynchronize()"
    ]

    opencl_syncs = [
        "barrier()",
        None,
        "mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)",
        "clFinish()",
        "clFinish()"
    ]

    """
        Indexing references
    """

    indexing_dims = [
        "x",
        "y",
        "z"
    ]
    cuda_indexing = [
        "gridDim.$dims$",
        "blockDim.$dims$",
        "blockIdx.$dims$",
        "threadIdx.$dims$"
    ]

    opencl_indexing = [
        "get_num_groups($dims_num$)",
        "get_local_size($dims_num$)",
        "get_group_id($dims_num$)",
        "get_local_id($dims_num$)"
    ]
    """
        Full patch
    """
    def cuda2opencl_full(self, src):
        src = """
/*
 d888b  d8888b. db    db .d8888. db   d8b   db  .d8b.  d8888b.
88' Y8b 88  `8D 88    88 88'  YP 88   I8I   88 d8' `8b 88  `8D
88      88oodD' 88    88 `8bo.   88   I8I   88 88ooo88 88oodD'
88  ooo 88~~~   88    88   `Y8b. Y8   I8I   88 88~~~88 88~~~
88. ~8~ 88      88b  d88 db   8D `8b d8'8b d8' 88   88 88
 Y888P  88      ~Y8888P' `8888Y'  `8b8' `8d8'  YP   YP 88

                            --- by github.com/bananaproject ---
*/\n""" + src

        src = self.cuda2opencl_kernel(src)
        src = self.cuda2opencl_qualifiers(src)
        src = self.cuda2opencl_syncitems(src)
        src = self.cuda2opencl_indexing(src)

        return src

    """
        Patch indexing
    """

    def cuda2opencl_indexing(self, src):
        for i in range(0, len(self.cuda_indexing)):
            for j in range(0, len(self.indexing_dims)):
                cuda_item = self.cuda_indexing[i].replace("$dims$", self.indexing_dims[j])
                opencl_item = self.opencl_indexing[i].replace("$dims_num$", str(j))

                src = src.replace(cuda_item, opencl_item)
        return src
    """
        Patch kernels
    """

    @staticmethod
    def cuda2opencl_kernel(src):
        cuda_kernel_regex_vs = r"[a-zA-Z0-9_.+-]+ << <[a-zA-Z0-9\d ,.\(\)]*>> >\([a-zA-Z0-9\d ,._\(\)]*\)"
        comp0 = re.compile(cuda_kernel_regex_vs)
        for element in comp0.findall(src):
            function_name = element.split("<")[0]
            args_aux_0 = element.split(">(")[1]
            args_aux_pos = args_aux_0.rfind(")")
            args = args_aux_0[0:args_aux_pos].split(", ")

            inject = ""
            for arg in args:
                if inject != "":
                    inject += "\t"
                inject += 'clSetKernelArg(' + function_name + ', ' + arg + ');\n'

            inject += "\tclEnqueueNDRangeKernel(" + function_name + ");\n"

            src = src.replace(element + ";", inject)
            
        return src

    """
        Patch sync functions
    """

    def cuda2opencl_syncitems(self, src):
        for i in range(0, len(self.cuda_syncs)):
            if self.opencl_syncs[i] is not None:
                src = src.replace(self.cuda_syncs[i], self.opencl_syncs[i])

        return src

    """
        Patch qualifiers
    """

    def cuda2opencl_qualifiers(self, src):
        for i in range(0, len(self.cuda_qualifiers)):
            comp0 = re.compile("__device__ \w* [a-zA-Z0-9_.+-]+\([a-zA-Z0-9\d ,.\(\)]*\)")

            for match in comp0.findall(src):
                src = src.replace(match, match.replace("__device__ ", ""))

            comp1 = re.compile("__device__ \w* [a-zA-Z0-9_.+-]+ \([a-zA-Z0-9\d ,.\(\)]*\)")

            for match in comp1.findall(src):
                src = src.replace(match, match.replace("__device__ ", ""))
            for i in range(0, len(self.cuda_qualifiers)):
                src = src.replace(self.cuda_qualifiers[i], self.opencl_qualifiers[i])
        return src

if len(sys.argv) != 4:
    print "Invalid syntax! gpuswap.py -type -src -output (Where -type can be -f (file) or -d (directory), -src is the file that will be converted, and -otuput the output path.)"
    exit(-1)

type = sys.argv[1]
src = sys.argv[2]
output = sys.argv[3]

gps = gpuswap()

if type == "-f":
    src_file = open(src, "r")
    out_file = open(output, "w")

    final = gps.cuda2opencl_full(src_file.read())

    out_file.write(final)

    src_file.close()
    out_file.close()
