// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;
  float sx, sy;
  sx = (float)x/((float)resolution.x-1);
  sy = (float)y/((float)resolution.y-1);
  glm::vec3 A = glm::cross(view,up);
  glm::vec3 B = glm::cross(A,view);
  double radian = fov.y/180.0*PI;
  float tmp = tan(radian) * glm::length(view)/glm::length(B);
  glm::vec3 V = B;
  V*= tmp;
  tmp = (float)resolution.x/(float)resolution.y*glm::length(view)/glm::length(A);
  glm::vec3 H = A;
  H*=tmp;
  glm::vec3 p = eye + view + (2*sx-1)*H + (1-2*sy)*V;
  r.direction = p-eye;
  r.direction = glm::normalize(r.direction);
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,material* mats,int* lightFlag,int lightNum){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  //////////construct light index array based on lightNum////
  int* lightIndex = new int[lightNum];
  int index = 0;
  for(int i = 0;i<numberOfGeoms;++i)
  {
	  if(lightFlag[i] == 1)
	  {
		  lightIndex[index] = i;
		  index++;
	  }
  }
  ////////////////////////////////////

  if((x<=resolution.x && y<=resolution.y)){
    ray Ri = raycastFromCameraKernel(resolution,time, x, y, cam.position, cam.view, cam.up, cam.fov);
	TraceRay(Ri,eye,depth,color[index],geoms,umberOfGeoms,mats,lightIndex,lightNum,time);

   }
}
void TraceRay(ray r,glm::vec eye, int depth, glm::vec3& color, staticGeom* geoms, 
	int numberOfGeoms,material* mats,int* lightIndex,int lightNum, float time)
{

  /////////////variables//////////////
  glm::vec3 intersectionPoint(0,0,0);
  glm::vec3 normal(0,0,0);
  glm::vec3 tmpnormal(0,0,0);
  float interPointDist = -1;
  float tmpDist = 0;
  glm::vec3 diffuseColor;
  glm::vec3 specularColor;  
  glm::vec3 reflectedColor;
  glm::vec3 refractColor(0,0,0);
  int nearestObjIndex = -1; // use the index to get material color
  ///////////////////////////////////
	if(rayDepth>MaxDepth)
	{
		colors[index].x = bgColorR;
		colors[index].y = bgColorG;
		colors[index].z = bgColorB;
		return;
	}
	//intersect with objects
	for(int i = 0;i<numberOfGeoms;++i)
	{
		if(lightFlag[i]==1)
			continue;
		if(geoms[i].type == SPHERE )
		{
			tmpDist = sphereIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);
			if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
			{
				interPointDist = tmpDist;
				normal = tmpnormal;
				nearestObjIndex = i;
			}
		}
		else if(geoms[i].type == CUBE)
		{
			tmpDist = boxIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);
			if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
			{
				interPointDist = tmpDist;
				normal = tmpnormal;
				nearestObjIndex = i;
			}
		}
	/*	else if(geoms[i].type == MESH)
		{
			;
		}*/

	}
	//after getting the intersect point, tell if it is -1
	//if it is -1, means didn't hit any object
	if(interPointDist == -1)
	{
		//check if the ray is parallel to any of the light source)
		for(int l = 0; l<lightNum;++l)
		{
			//if parallel
			color[index] = mats[lightIndex[l]].color;
			//else
			color[index] = glm::vec3(bgColorR,bgColorG,bgColorB);
		}
	}
	else // if the ray hits any of the object in the scene
	{
		//calculate local illumination

		if(mats[nearestObjIndex].specularExponent>0)
		{
			//non-zero specular reflectivity
			//compute direction of reflected ray
			//computer direction of reflected ray Rr = Ri-2N*dot(Ri*N)
			glm::vec3 Rr = glm::dot(Ri,normal);
			Rr *= -2.0; Rr*=normal;
			Rr += Ri;
			//raytraceRay again
			raytraceRay(Rr,depth+1,reflectedColor,geoms,numberOfGeoms, mats,lightIndex,lightNum,time)
			specularColor = reflectedColor;
			specularColor *= Kspecular;
		}
		else
		{
			specularColor = glm::vec3(0,0,0);
		}
		if(mats[nearestObjIndex].hasReflective>0)
		{
			//non-zero refraction 
			refractColor = glm::vec3(0,0,0);
		}
		//amibient term
		glm::vec3 ambient(ambientColorR,ambientColorG,ambientColorB);
		ambient *= Kambient;
		color = ambient + refractColor;
		//for all light check if the object is in shadow		
		for(int j = 0;j<lightNum;++j)
		{

			if(shadowRayUnblocked(intersectionPoint,geoms[j].translation,numberOfGeoms,mats) == true)
			{
				//not in shadow
				diffuseColor = mats[nearestObjIndex].color;
				diffuseColor *= Kdiffuse;
				diffuseColor *= glm::dot(normal,geoms[j].translation);
				diffuseColor *= mats[j].color;
				color += diffuseColor;
				specularColor *= pow(glm::dot(Rr,eye-intersectionPoint),mats[nearestObjIndex].specularExponent);
				specularColor *= mats[j].color;
				color += specularColor;
			}			
		}
	}
}
bool ShadowRayUnblocked(glm::vec3 point,glm::vec3 lightPos,staticGeom* geoms, int numberOfGeoms,material* mats)
{
	float tmpDist = -1;
	for(int i = 0;i<numberOfGeoms;++i)
	{
		if(mats[i].emittance>0) continue;
		if(geoms[i].type == SPHERE )
		{
			tmpDist = sphereIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);
			if(tmpDist != -1)
				return false;
		}
		else if(geoms[i].type == CUBE)
		{
			tmpDist = boxIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);
			if(tmpDist != -1)
				return false;
		}
	/*	else if(geoms[i].type == MESH)
		{
			;
		}*/
	}
	return true;
}
//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  material* matList = new material[numberOfGeoms];
  //use this flag to check if the object is light source
  //use 0 to represent normal object and 1 to represent light;
  int* lightFlag = new int[numberOfGeoms];
  int lightNum = 0;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;	
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;

	//material	
	matList[i] = materials[newStaticGeom.materialid];
	if(matList[i].emittance>0)
	{
		lightFlag[i] = 1;
		lightNum++;
	}
	else
	{
		lightFlag[i] = 0;
	}
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;


  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material* cudamat = NULL;
  cudaMalloc((void**)&cudamat, numberOfGeoms*sizeof(material));
  cudaMemcpy(cudamat,matList,numberOfGeoms*sizeof(staticGeom),cudaMemcpyHostToDevice);



  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam,
	  traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamat,lightFlag,lightNum);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
