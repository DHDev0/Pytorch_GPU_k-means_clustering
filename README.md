# Pytorch_GPU_k-means_clustering<br />
A pytorch implementation of k-means_clustering<br />
<br />
The algorithm is an adaptation of MiniBatchKMeans sklearn with an autoscaling of the batch base on your VRAM memory.<br /> 
The algorithm is N dimensional, it will transform any input to 2D.<br />
( you still have to insert the amount of VRAM because only Pytorch 1.11+ support command call to retrieve VRAM information)<br />
<br />
The code is using very simple torch operation so it should be compatible with a lot of legacy pytorch version.<br />
<br />
You will find a tutorial base on jupyter lab block that describe the different function.<br />
<br />
Benchmark: 21.5x speed gain compare to sklearn.kmeans (even if the comparaison isn't fair)<br />
( a bit faster than kmeans on RAPIDS[ having 20.5x speed gain against sklearn.kmeans] )<br />
<br />
|X| Multi-GPU (coming soon...)<br />
|X| Multi-CPU (coming soon...)<br />
|X| Will also be using: https://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html for a better cluster accuracy.<br />
|X| Going to try to limit the amount of data per cluster. Auto-clustering using sample of random sample to determine an optimal cluster size for the amount of data desire<br />
|X| Join, implement or make a generalize dataloader for anytype of structured or unstructured data<br />
|X| Perhaps a tensorflow implementation or more direct python cuda<br />
<br />
Let me know on "issues" if you have any idea or problem.
