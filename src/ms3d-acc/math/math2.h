/*************************************************************/
/*                           MATH.H                          */
/*                                                           */
/* Purpose: Header file to include in order to get full math */
/*          support for 2 and 3 element vectors, 3x3 and 4x4 */
/*          matrices and quaternions                         */
/*      Evan Pipho (May 27, 2002)                            */
/*                                                           */
/*************************************************************/
#ifndef MS3D_MATH_H
#define MS3D_MATH_H


namespace vgMs3d {

	const float Pi = 3.141592f;

	#undef SQU
	#define SQU(x) (x) * (x)

#include "vector.inl"

#include "matrix.inl"


#include "quaternion.inl"
#include "vectorImplementation.inl"
#include "matrixImplementation.inl"
#include "quaternionImplementation.inl"

}


#endif //MS3D_MATH_H