# Pytorch_GPU_k-means_clustering<br />
Pytorch GPU friendly implementation of k means clustering (and k-nearest neighbors algorithm) <br />
<br />
The algorithm is an adaptation of MiniBatchKMeans sklearn with an autoscaling of the batch base on your VRAM memory. The algorithm is N dimensional, it will transform any input to 2D. You can compute arbitrary big dataset by splitting them to 10 millions datapoints per sample(file) using only 2.5GB of VRAM. <br />
( you still have to insert the amount of VRAM because only Pytorch 1.11+ support command call to retrieve VRAM information )<br />
<br />
The code is using very simple torch operation so it should be compatible with a lot of legacy pytorch version.<br />
<br />
Tutorial: https://github.com/DHDev0/Pytorch_GPU_k-means_clustering/blob/main/tutorial_kmean_gpu.ipynb <br />
<br />
Benchmark (cpu mode vs cpu sklearn vs gpu mode): https://github.com/DHDev0/Pytorch_GPU_k-means_clustering/blob/main/tutorial_kmean_gpu.ipynb <br />
<br />
Incoming/idea:<br />
|X| Multi-GPU (coming soon...)<br />
|X| Multi-CPU (coming soon...)<br />
|X| Add Float 8/16/32/64 mode <br />
|X| Add other metric(euclidean distance only) such as Taxicab Geometry, Minkowski distance, Jaccard index, Hamming distance, Fractal dimension(with some scaling constant), CrossEntropy(with some scaling constant)<br />
|X| Will also be using: https://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html for a better cluster accuracy.<br />
|X| Going to try to limit the amount of data per cluster. Auto-clustering using sample of random sample to determine an optimal cluster size for the amount of data desire per cluster<br />
|X| Join, implement or make a generalize dataloader for anytype of structured or unstructured data<br />
|X| Perhaps a tensorflow implementation or more direct python cuda<br />
<br />
Let me know on "issues" if you have any idea or problem.

