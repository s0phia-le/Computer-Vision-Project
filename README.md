## 1. Pre-Processing 
- Voxel downsampling

## 2. Ground-Removal
- Use RANSAC to calculate road plan, then compare remaining points to road height. Only points that are a certain distance above the road are kept for clustering

## 3. Background Subtraction
- Code compares the current frame to a base cloud. Any points that haven't moved or changed are discarded, leaving only potential moving objects

## 4. Detecting Position (Clustering)
- DBSCAN
- Centroid calculation: position of car is defined as the average/mean of the points in that cluster

## 5. Motion Estimation
- Association: for every car in the current frame, it looks for the closest car in the previous frame using Euclidean distance
- - Match
  - Velocity Calculation
 
  translate backsub from 2d to 3d
ideas
- larger voxel grid and look for pixels that move
- 

- naive way: subtract from curr frames to first frame
- voxel downsample most naive thing we do
- calculate distance btwn height of each obj (if obj height much higher than ground, remove)
- background: 
  - first downsample, 
 - keep remaining points of building
 - use the frames 10 seconds ago rather than 1st frame 
 - use frames that are slightly further away so they aren't captured by distance threshold easily


- eps not good enough
- back sub: doesn't work for first 2-3 bc using 1st frame as reference
- perhaps throw away the first frame
- back sub might remove points from cars
- distance threshold needs tweaked: threshold above, if dist < threshold, static. if dist >, keep bc nonstatic
- RANSAC detects ground sometimes
- clustering is categorizing ground as noise
- if points far enough away from each other, should be fine
- consider what is considered a cluster
- voxel downsampling tweaked?
