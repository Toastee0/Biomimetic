Design a chain of tech that starts with a empty room in a quake 3 .map file, adds object to it as primatives then compiles the map file and sends the BSP to the engine to render a frame, that frame is then used as the virtual twin image to give persistence to what it currently sees with its camera. The engine should be able to sit loaded in the latest state so the camera can be moved by the AI 
You said:
The modelling engine will assume that all objects are convex polygons.  But I do like the idea is being able to insert and remove objects live. I want to be able to have an object contain it's own estimated trajectory so the engine helps estimate motion
You said:
Need to be able to detect walls/floors/stairs and make them permanent until actively decided to be otherwise
You said:
Ok, I'll build the 2 camera array, with the 60 deg field 8*8 tof. On 360° mast