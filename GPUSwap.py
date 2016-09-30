import re


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
        Patch kernels
    """

    def cuda2opencl_kernel(self, src):
        cuda_kernel_regex_vs = r"[a-zA-Z0-9_.+-]+ << <[a-zA-Z0-9\d ,.\(\)]*>> >\([a-zA-Z0-9\d ,.\(\)]*\)"
        comp0 = re.compile(cuda_kernel_regex_vs)
        for element in comp0.findall(src):
            function_name = element.split("<")[0]
            args_aux_0 = element.split(">(")[1]
            args_aux_pos = args_aux_0.rfind(")")
            args = args_aux_0[0:args_aux_pos].split(", ")

            inject_point = src.find(element + ";")

            src = src.replace(element + ";", "")

            for arg in args:
                inject = 'clSetKernelArg(' + function_name + ', ' + arg + ');\n'
                if src[inject_point - 1] != '\t':
                    inject = '\t' + inject
                src = src[:inject_point] + inject + src[inject_point:]
                inject_point += len(inject)

            src = src[:inject_point] + "\tclEnqueueNDRangeKernel(" + function_name + ");\n" + src[inject_point:]
        return src

    """
        Patch sync functions
    """

    def cuda2opencl_syncitems(self, src):
        for i in range(0, len(self.cuda_syncs)):
            src = src.replace(self.cuda_syncs[i], self.opencl_qualifiers[i])

    """
        Patch qualifiers
    """

    def cuda2opencl_qualifiers(self, src):
        for i in range(0, len(self.cuda_qualifiers)):

            # Invalid group in regex Mmm
            comp0 = re.compile("__device__ \w* [a-zA-Z0-9_.+-]+\([a-zA-Z0-9\d ,.\(\)]*\)")

            for match in comp0.findall(src):
                src = src.replace(match, match.replace("__device__ ", ""))

            comp1 = re.compile("__device__ \w* [a-zA-Z0-9_.+-]+ \([a-zA-Z0-9\d ,.\(\)]*\)")

            for match in comp1.findall(src):
                src = src.replace(match, match.replace("__device__ ", ""))

            # Simply replace
            for i in range(0, len(self.cuda_qualifiers)):
                src = src.replace(self.cuda_qualifiers[i], self.opencl_qualifiers[i])
        return src