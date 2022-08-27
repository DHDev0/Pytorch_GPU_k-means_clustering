# Pytorch_GPU_k-means_clustering
A pytorch implementation of k-means_clustering

The algorithm is an adaptation of MiniBatchKMeans sklearn with an autoscaling of the batch base on your VRAM memory. 
The algorithm is als N dimensional (will do tranform any input to 2D).
( you still have to insert the amount of VRAM because only Pytorch 1.11+ support command call to retrieve VRAM information)

The code is using very simple torch operation so it should be compatible with a lot of legacy pytorch version.

You will find a tutorial base on jupyter lab block that describe the different function.

Benchmark: 21.5x speed gain compare to sklearn.kmeans (even if the comparaison isn't fair)
( a bit faster than kmeans on RAPIDS[ having 20.5x speed gain against sklearn.kmeans] )

|X| Multi-GPU (coming soon...)
|X| Multi-CPU (coming soon...)
|X| Will also be using: https://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html for a better cluster accuracy. (coming soon...)
