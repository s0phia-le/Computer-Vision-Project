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
