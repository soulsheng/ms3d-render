// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
// 
// WARRANTY DISCLAIMER
// 
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

__kernel void
SimpleKernel( const __global float *input, __global float *output)
{
	size_t index = get_global_id(0);
    output[index] = rsqrt( input[index] );
}

__kernel /*__attribute__((vec_type_hint(float4))) */ void
SimpleKernel4( const __global float4 *input, __global float4 *output)
{
	size_t index = get_global_id(0);
    output[index] = rsqrt( input[index] );
}

__kernel void
updateVectorByMatrix( const __global float *pInput, const __global int *pIndex,const __global float *pMatrix,__global float *pOutput)
{
	size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	
	int offset =  pIndex[index]*3*4;

	pOutput[4*index+0] = pInput[4*index] * pMatrix[offset+0] + pInput[4*index+1] * pMatrix[offset+1] + pInput[4*index+2] * pMatrix[offset+2]  + pMatrix[offset+3];
	pOutput[4*index+1] = pInput[4*index] * pMatrix[offset+4] + pInput[4*index+1] * pMatrix[offset+5] + pInput[4*index+2] * pMatrix[offset+6]  + pMatrix[offset+7];
	pOutput[4*index+2] = pInput[4*index] * pMatrix[offset+8] + pInput[4*index+1] * pMatrix[offset+9] + pInput[4*index+2] * pMatrix[offset+10]  + pMatrix[offset+11];
}


__kernel void
updateVectorByMatrix4( const __global float4 *pInput, const __global int *pIndex,__constant float4 *pMatrix,__global float4 *pOutput)
{
	size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	
	int offset = pIndex[index]*3;

	float4 vIn = pInput[index]; 
	
	pOutput[index] = (float4)( 
		vIn.x * pMatrix[offset+0].x + vIn.y * pMatrix[offset+0].y + vIn.z * pMatrix[offset+0].z  + pMatrix[offset+0].w ,
		vIn.x * pMatrix[offset+1].x + vIn.y * pMatrix[offset+1].y + vIn.z * pMatrix[offset+1].z  + pMatrix[offset+1].w ,
		vIn.x * pMatrix[offset+2].x + vIn.y * pMatrix[offset+2].y + vIn.z * pMatrix[offset+2].z  + pMatrix[offset+2].w ,
		1.0f);

}