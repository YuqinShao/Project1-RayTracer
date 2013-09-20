-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Yuqin Shao
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
VEDIO DEMO
-------------------------------------------------------------------------------
[![ScreenShot](youtube.PNG)](http://youtu.be/D12Ol3EwDHI)

-------------------------------------------------------------------------------
FEATURES
-------------------------------------------------------------------------------
* Raycasting from a camera into a scene through a pixel grid
* Phong lighting for one point light source
* Diffuse lambertian surfaces
* Raytraced shadows
* Cube intersection testing
* Sphere surface point sampling
* Specular reflection 
* Soft shadows and area lights 
* Interactive camera 
	Mouse interaction with right click and drag to zoom in/out, with left click and drag to rotate

-------------------------------------------------------------------------------
SCREEN SHOTS
-------------------------------------------------------------------------------
Diffuse Only
![Alt test](/renders/diffuse.bmp " ")

Specular Highlights
![Alt test](/renders/specular.bmp "")

Reflection
![Alt test](/renders/reflect.bmp "")

Soft Shadow
![Alt test](/renders/softshadow.bmp "")

![Alt test](/renders/img3.bmp "")

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
perform at least one experiment on your code to investigate the positive or
negative effects on performance. 

One such experiment would be to investigate the performance increase involved 
with adding a spatial data-structure to your scene data.

Another idea could be looking at the change in timing between various block
sizes.

A good metric to track would be number of rays per second, or frames per 
second, or number of objects displayable at 60fps.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain and
performance differences.


-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam,
  liamboone+cis565@gmail.com, with a one paragraph explanation.  Be concise and
  realistic.  Recall that we reserve 30 points as a sanity check to adjust your
  grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We
  hope to only use this in extreme cases when your grade does not realistically
  reflect your work - it is either too high or too low.  In most cases, we plan
  to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as
  the path tracer.  We will determine the weighting at the end of the semester
  based on the size of each project.

-------------------------------------------------------------------------------
SUBMISSION
-------------------------------------------------------------------------------
As with the previous project, you should fork this project and work inside of
your fork. Upon completion, commit your finished project back to your fork, and
make a pull request to the master repository.  You should include a README.md
file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running, and at least one screenshot
  of the final rendered output of your raytracer
* A link to a video of your raytracer running.
* Instructions for building and running your project if they differ from the
  base code
* A performance writeup as detailed above.
* A list of all third-party code used.
* This Readme file, augmented or replaced as described above in the README section.
