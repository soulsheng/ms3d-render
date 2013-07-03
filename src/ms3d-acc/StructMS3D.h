
#pragma  once

#include "math/math2.h"


/* 
	MS3D STRUCTURES 
*/

// byte-align structures
#ifdef _MSC_VER
#	pragma pack( push, packing )
#	pragma pack( 1 )
#	define PACK_STRUCT
#elif defined( __GNUC__ )
#	define PACK_STRUCT	__attribute__((packed))
#else
#	error you must byte-align these structures with the appropriate compiler directives
#endif

typedef unsigned char byte;
typedef unsigned short word;

// File header
struct MS3DHeader
{
	char m_ID[10];
	int m_version;
} PACK_STRUCT;

// Vertex information
struct MS3DVertex
{
	byte m_flags;
	float m_vertex[3];
	char m_cBone;
	byte m_refCount;
} PACK_STRUCT;

// Triangle information
struct MS3DTriangle
{
	word m_flags;
	word m_usVertIndices[3];
	vgMs3d::CVector3 m_vNormals[3];
	vgMs3d::CVector3 m_s, m_t;
	byte m_smoothingGroup;
	byte m_groupIndex;
} PACK_STRUCT;

// Material information
struct MS3DMaterial
{
    char m_name[32];
    float m_ambient[4];
    float m_diffuse[4];
    float m_specular[4];
    float m_emissive[4];
    float m_shininess;	// 0.0f - 128.0f
    float m_transparency;	// 0.0f - 1.0f
    byte m_mode;	// 0, 1, 2 is unused now
    char m_texture[128];
    char m_alphamap[128];
} PACK_STRUCT;

// Keyframe data
struct MS3DKeyframe
{
	float m_fTime;
	float m_fParam[3];
} PACK_STRUCT;

//	Joint information
struct MS3DJoint
{
	//Data from file
	unsigned char m_ucpFlags;
	char m_cName[32];
	char m_cParent[32];
	float m_fRotation[3];
	float m_fPosition[3];
	unsigned short m_usNumRotFrames;
	unsigned short m_usNumTransFrames;

	MS3DKeyframe * m_RotKeyFrames;       //Rotation keyframes
	MS3DKeyframe * m_TransKeyFrames;     //Translation keyframes
	
	//Data not loaded from file
	short m_sParent;                     //Parent joint index


	vgMs3d::CMatrix4X4 m_matLocal;       
	vgMs3d::CMatrix4X4 m_matAbs;			
	vgMs3d::CMatrix4X4 m_matFinal;

	unsigned short m_usCurRotFrame;
	unsigned short m_usCurTransFrame;

	//Clean up after itself like usual
	MS3DJoint()
	{
		m_RotKeyFrames = 0;
		m_TransKeyFrames = 0;
		m_usCurRotFrame = 0;
		m_usCurTransFrame = 0;
	}
	~MS3DJoint()
	{
		if(m_RotKeyFrames)
		{
			delete [] m_RotKeyFrames;
			m_RotKeyFrames = 0;
		}
		if(m_TransKeyFrames)
		{
			delete [] m_TransKeyFrames;
			m_TransKeyFrames = 0;
		}
	}
} PACK_STRUCT;

//	Mesh
struct Mesh
{
	int m_materialIndex;
	int m_usNumTris;
	int *m_uspIndices;
};

//	Material properties
struct Material
{
	float m_ambient[4], m_diffuse[4], m_specular[4], m_emissive[4];
	float m_shininess;
	GLuint m_texture;
	char *m_pTextureFilename;
};

//	Triangle structure
struct Triangle
{
	vgMs3d::CVector3 m_vNormals[3];
	float m_s[3], m_t[3];
	int m_usVertIndices[3];
};

//	Vertex structure
struct Vertex
{
	char m_cBone;	// for skeletal animation
	vgMs3d::CVector3 m_vVert;
	float m_texcoord[2];
};


struct VboFaceIndex
{
	union{
		struct{
			GLuint _point0;
			GLuint _point1;
			GLuint _point2;
		};

		GLuint _point[3];
	};

};
class  VboMetaFaceStruct
{
public:
	VboMetaFaceStruct();
	~VboMetaFaceStruct();

	string _texFileName;

	GLuint _elementBufferObjectID;

	// 注意,这里是int类型的原子数目.即3的倍数
	long _numOfElements;

	VboFaceIndex* pFaceIndex;
};


class Ms3dVertexArrayMesh 
{
public:
	Ms3dVertexArrayMesh()
	{
		materialID = 0;
		pVertexArray = NULL;
		numOfVertex = 0;
	}

	~Ms3dVertexArrayMesh()
	{
		if (pVertexArray != NULL)
		{
			delete pVertexArray;
			pVertexArray = NULL;
		}

		numOfVertex = 0;
	}

	int materialID;

	float * pVertexArray;

	int numOfVertex;
};

class Ms3dIntervelData 
{
public:
	Ms3dIntervelData()
	{
		m_pMesh = NULL;
		m_numberOfMesh = 0;
	}

	~Ms3dIntervelData()
	{
		if (m_pMesh != NULL)
		{
			delete[] m_pMesh;
			m_pMesh = NULL;
		}
	}

public:
	Ms3dVertexArrayMesh* m_pMesh;
	int m_numberOfMesh;
};

// Default alignment
#ifdef _MSC_VER
#	pragma pack( pop, packing )
#endif

#undef PACK_STRUCT
