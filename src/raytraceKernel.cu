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

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define  printf(f, ...) ((void)(f, __VA_ARGS__),0)  
//#endif

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
  //printf("%d,%d",x,y);
  sx = (float)x/((float)resolution.x-1);
  sy = (float)y/((float)resolution.y-1);
  glm::vec3 A = glm::normalize(glm::cross(view,up));
  glm::vec3 B = glm::normalize(glm::cross(A,view));
  double radian = (float)fov.y/180.0*PI;
  float tmp = tan(radian) * glm::length(view)/glm::length(B);
  glm::vec3 V = B;
  V*= tmp;
  tmp = tan(radian) * (float)resolution.x/(float)resolution.y*glm::length(view)/glm::length(A);
  glm::vec3 H = A;
  H*=tmp;
  H *= (2.0*sx-1);
  V *= (1-2.0*sy);
  glm::vec3 p = eye + view + H + V;
  r.direction = p-eye;
  r.direction = glm::normalize(r.direction);
  //r.direction = glm::normalize(r.direction);
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
/////shadow check
__host__ __device__ bool ShadowRayUnblocked(glm::vec3 point,glm::vec3 lightPos,staticGeom* geoms, int numberOfGeoms,material* mats)
{
	//return true;
	float tmpDist = -1;
	glm::vec3 tmpnormal;
	glm::vec3 intersectionPoint;
	ray r; r.origin = point;
	r.direction = lightPos-point; r.direction = glm::normalize(r.direction);	
	float objDist = -1;
	for(int i = 0;i<numberOfGeoms;++i)
	{
		//if light source no need to check
		if(mats[i].emittance >0)
			continue;
		if(geoms[i].type == SPHERE )
		{			
			//if same location as light, no need to check
			if(glm::length(geoms[i].translation - lightPos)<0.001)
				continue;
			tmpDist = sphereIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);							
			if(abs(tmpDist+1)<EPSILON || abs(tmpDist)<=0.03)
				continue;// obj not in shadow
			else
			{
				objDist = tmpDist;
			}
			
		}
		else if(geoms[i].type == CUBE)
		{		
			if(glm::length(geoms[i].translation - lightPos)<0.0001)
				continue;
			tmpDist = boxIntersectionTest(geoms[i],r,intersectionPoint,tmpnormal);		
			if(abs(tmpDist+1)< EPSILON || abs(tmpDist)<=0.03)	
				continue;
			else
			{
				objDist = tmpDist;
			}				
		}
	}
	if(objDist == -1)
		return true;
	else
		return false;	
}

//recursive raytrace
__host__ __device__ void raytrace(ray Ri,glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3& color,
                            staticGeom* geoms, int numberOfGeoms,material* mats,int* lightIndex,int lightNum)
{
	/////////////variables//////////////	
	ray Rr; //reflect ray
	glm::vec3 intersectionPoint(0,0,0);
	glm::vec3 normal(0,0,0);
	glm::vec3 tmpnormal(0,0,0);
	float interPointDist = -1;
	float tmpDist = -1;
	glm::vec3 diffuseColor(0,0,0);
	glm::vec3 specularColor(0,0,0);  
	glm::vec3 reflectedColor(0,0,0);
	glm::vec3 refractColor(0,0,0);
	glm::vec3 localColor(0,0,0);
	int nearestObjIndex = -1; // nearest intersect object index
	glm::vec3 ambient(ambientColorR,ambientColorG,ambientColorB);
	ambient *= Kambient;
	int nearestLight = -1;
	float lightDist = -1;
	float tmplightDist = -1;
	////////////////////////////////////////////
	color = glm::vec3(0,0,0);

	for(int i = 0;i<numberOfGeoms;++i)
	{
		//if the object is light, continue to another
		if(geoms[i].type == SPHERE )
		{

			tmpDist =  sphereIntersectionTest(geoms[i],Ri,intersectionPoint,tmpnormal);
			if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
			{
					
				if(mats[i].emittance>0)
				{
					nearestLight = i;
				}
				else
				{
					nearestLight = -1;
					interPointDist = tmpDist;
					normal = tmpnormal;
					nearestObjIndex = i;
				}
			}
				
		}
		else if(geoms[i].type == CUBE)
		{
			
			tmpDist = boxIntersectionTest(geoms[i],Ri,intersectionPoint,tmpnormal);
			if(tmpDist!=-1 &&(interPointDist==-1 ||(interPointDist!=-1 && tmpDist<interPointDist)))
			{
				if(mats[i].emittance>0)
					nearestLight = i;
				else
				{
					nearestLight = -1;
					interPointDist = tmpDist;
					normal = tmpnormal;
					nearestObjIndex = i;
				}
			}
				
		}
	}

	//if first ray didn't hit any object,color set to light / bg color
	if(interPointDist == -1 ||(interPointDist!=-1 && nearestLight!=-1))
	{
		if(nearestLight != -1)
		{
			color = mats[nearestLight].color;
		}
		else
			color = glm::vec3(bgColorR,bgColorG,bgColorB);
		return;
	}
	//did hit object
	else
	{			
		//printf("%d ",nearestObjIndex);
		/*colors[index] = glm::vec3(abs(normal.x),abs(normal.y),abs(normal.z));
		return;*/
		Rr.direction = normal; 
		Rr.direction *= glm::dot(Ri.direction,normal);
		Rr.direction *= -2.0;
		Rr.direction += Ri.direction;
		Rr.origin = intersectionPoint;	
		localColor += ambient* mats[nearestObjIndex].color;
			
		//shadow check
		for(int j = 0;j<lightNum;++j)
		{
			
			if(ShadowRayUnblocked(intersectionPoint,geoms[lightIndex[j]].translation,geoms,numberOfGeoms,mats) == true)
			{
				//not in shadow
				//add diffuse 			
				diffuseColor = mats[nearestObjIndex].color;								
				glm::vec3 L = glm::normalize(geoms[lightIndex[j]].translation - intersectionPoint);
				float diffuseCon = glm::dot(normal,L);
				if(diffuseCon<0)
					diffuseColor = glm::vec3(0,0,0);
				else
				{
					diffuseColor *= diffuseCon;
					diffuseColor *= Kdiffuse;
					diffuseColor *= mats[lightIndex[j]].color;
				}			
				localColor += diffuseColor;
			/*	specularColor *= pow(glm::dot(Rr.direction,(cam.position-intersectionPoint)),mats[nearestObjIndex].specularExponent);
				specularColor *= mats[lightIndex[j]].color;
				localColor += specularColor;*/
			}
		}
	}
	color += localColor;	
}



//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,material* mats,int* lightIndex,int lightNum){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if((x<=resolution.x && y<=resolution.y)){
		ray Ri; //indice ray
		Ri = raycastFromCameraKernel(resolution, time, x,y,cam.position, cam.view, cam.up, cam.fov);
		raytrace(Ri,resolution,time,cam,rayDepth,colors[index],geoms,numberOfGeoms,mats,lightIndex,lightNum);
   }
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
  //material
  material* matList = new material[numberOfGeoms];
  int lightNum = 0;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
	matList[i] = materials[newStaticGeom.materialid];
	if(matList[i].emittance >0)
		lightNum++;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  int* lightIndex = new int[lightNum];
  int lin = 0;
  for(int i = 0;i<numberOfGeoms;i++)
  {
	  if(matList[i].emittance ==0) continue;
	  lightIndex[lin] = i;
	  lin++;
  }
  int* cudalightindex = NULL;
  cudaMalloc((void**)&cudalightindex,lightNum*sizeof(int));
  cudaMemcpy(cudalightindex,lightIndex,lightNum*sizeof(int),cudaMemcpyHostToDevice);
  material* cudamat = NULL;
  cudaMalloc((void**)&cudamat,numberOfGeoms*sizeof(material));
  cudaMemcpy(cudamat,matList,numberOfGeoms*sizeof(material),cudaMemcpyHostToDevice);
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamat,cudalightindex,lightNum);

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