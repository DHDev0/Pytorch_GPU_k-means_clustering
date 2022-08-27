import numpy as np
import math
import torch
#TODO a generalize dataloader for unstructured and structured with scaler and category

class k_means_gpu:
    def __init__(self, k=2, convergence = 0.01, num_iteration = 100, compute_device = "default", vram = "2.5GB"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if compute_device == "default" else torch.device(compute_device) #option: "cuda:0", "cuda:1", etc.. and , "cpu"
        print(self.device)
        self.clusters = torch.tensor(k).to(self.device)
        self.convergence = torch.tensor(convergence).to(self.device)
        self.iteration = torch.tensor(num_iteration).to(self.device)
        self.has_converged = False
        self.performance = []
        self.gpu_split = self.gpu_allocation(vram)
        self.verbose = True
        
    def fit(self, data, retrain = False, verbose = True):
        
        self.verbose = verbose
        
        #splitting is done on cpu then batching and computation on gpu if possible
        data = self.d2_flatten(torch.tensor(data).to("cpu"))
        shape= data.size()
        
        #build centroids
        initial_random_value = torch.randint(0, len(data)-1,(self.clusters,))
        
        if retrain:
            self.centroids = self.centroids.to(self.device)
        else:
            self.centroids=data[initial_random_value].to(self.device)
        
        #hyperparameter
        centroid_k_category=torch.arange(self.clusters).reshape(self.clusters,1).to(self.device)
        excluder = torch.tensor([0]).to(device= self.device,dtype= data.dtype).float()
        self.dim_metric_for_compute = len(shape) if len(shape) >= 2 else 2      #TODO can reduce amount of hyperparameter since code run in 2D always mode              
        self.dim_argmin_reshape =  len(shape)-1 if len(shape) >= 2 else 2
        gpu_allocation = int(1/((self.gpu_split/(self.clusters/10))/shape[0]))
        data_split = torch.tensor_split(data[:,None,:],gpu_allocation if gpu_allocation > 1 else 2) 

        #start training
        for epoch in range(self.iteration):  
            #building block to rebuild gradient from batch
            reminder_val = torch.ones(self.clusters,shape[-1]).to(self.device)
            reminder_op = torch.ones(self.clusters,shape[-1]).to(self.device)
            
            #Training on random batch block and compute average gradient at the end of eapoch
            for batchs in torch.randint(0, len(data_split)-1,(len(data_split),)):
                batch = data_split[batchs].to(self.device)
                centroid_reshape_for_compute =  self.centroids[None,:,:] 
                metric_between_input_and_centroid = torch.sqrt(torch.sum(torch.pow((centroid_reshape_for_compute -batch),2),dim = self.dim_metric_for_compute ))# custom distance metric calculation can be  implement here
                centroid_category = torch.argmin(metric_between_input_and_centroid,dim = self.dim_argmin_reshape)
                boolean_mask_with_reshape = (centroid_k_category == centroid_category).T[:,:,None] 
                extract_close_metric_using_batch_on_mask = (boolean_mask_with_reshape * batch ).type(torch.float)
                latest_centroid = extract_close_metric_using_batch_on_mask.sum(dim=0) / (~torch.isclose(torch.nan_to_num(extract_close_metric_using_batch_on_mask),excluder)).long().sum(dim=0)
                reminder_val += (torch.nan_to_num(latest_centroid,nan=0.0) - (self.centroids * torch.nan_to_num(latest_centroid,nan=0.0).bool().float()) )
                reminder_op +=torch.nan_to_num(latest_centroid,nan=0.0).bool().float()
                
            #update centroids from batch gradient
            latest_centroid = self.centroids + torch.nan_to_num(reminder_val/reminder_op)

            #Estimator to determine continuity of training
            centroid_convergence =torch.sqrt(torch.sum(((self.centroids - latest_centroid)**2),dim=1))
            mean_convergence = torch.mean(centroid_convergence)
            self.performance.append(mean_convergence)
            self.centroids = latest_centroid
                
            #Use estimator to end or continue iteration
            if epoch > 1 and self.convergence_eval():
                print(self.convergence_eval())
                print('End compute')
                break
        self.centroids = self.centroids.to("cpu")
        torch.cuda.empty_cache()
        
    def convergence_eval(self):
        try:
            evolution = self.performance[-2]-self.performance[-1]
            if self.verbose:
                print(f"|actual convergence: {self.performance[-1]} | rate of convergence: {evolution} |converge min / rate: {self.convergence}")
                
            if self.performance[-1] <= self.convergence and self.performance[-1] < min(self.performance[:-1]) and evolution >= self.convergence:
                self.has_converged = False 
                
            if self.performance[-1] >= min(self.performance[:-1]) or evolution <= self.convergence: 
                self.has_converged=True
            
        except:
            pass
        return self.has_converged
    
    def gpu_allocation(self,vram):
        list_of_numbers = [float(i) for i in ''.join((ch if ch in '0123456789.-e' else ' ') for ch in vram).split()]
        indice_of_number = [str(i) for i in ''.join((ch if ch in "BKMGTPEZY" else ' ') for ch in vram).split()]

        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        size_byte = list_of_numbers[0] * math.pow(1024, int(size_name.index(indice_of_number[0])))
        min_size_byte = 858993459

        res = (int(np.ceil(size_byte/1342.17728)))   #trial and error constant find for scaling gpu
        res = res if res % 2 == 0 else res + 1

        if (size_byte < min_size_byte):
            raise Exception("You need at least 0.8GB of VRAM")
        return res
    
    def d2_flatten(self,x):
        if len(x.size()) <= 1:
            return x[:,None]
        return x.reshape(torch.prod(torch.tensor(x.size()[:-1])),x.size()[-1])

    def compute_opt(self,compute_device):
        if compute_device != "default": #option: "cuda:0", "cuda:1", etc.. and , "cpu" 
            self.device = torch.device(compute_device)
            
    def encode(self,pointer_to_encode, compute_device = "default",save_array = False, data_path = "encoded_array"):
        self.compute_opt(compute_device)
        pointer_to_encode_tenso = torch.tensor(compress_input).to(self.device)
        centroid_to_device = self.centroids.to(self.device)
        tensor_2d = self.d2_flatten(pointer_to_encode_tenso)
        distance_merge = torch.sum((centroid_to_device[None,:,:]  - tensor_2d[:,None,:] )**2,dim=self.dim_metric_for_compute)
        result  = torch.argmin(distance_merge,dim=self.dim_argmin_reshape).reshape(tuple(pointer_to_encode_tensor.size())[:-1]+(1,)).type(torch.uint8).cpu().numpy()
        if save_array:
            np.save(f'{data_path}.npy', result)
            return f'Save encode array file to {data_path}.npy'
        return result
    
    
    def decode(self,pointer_to_decode, compute_device = "default",save_array = False, data_path = "decoded_array"):
        self.compute_opt(compute_device)
        pointer_to_decode = torch.tensor(pointer_to_encode).to(self.device)
        tensor_flatten = pointer_to_encode_tensor.flatten()
        result = self.centroids.to(self.device)[tensor_flatten].reshape(tuple(pointer_to_decode.size())[:-1]+tuple(self.centroids.size()[-1])).type(torch.float64).cpu().numpy()
        if save_array:
            np.save(f'{data_path}.npy', result)
            return f'Save decode array file to {data_path}.npy'
        return result
   
    def compress_input(self,compress_input, compute_device = "default",save_array = False, data_path = "compress_array"):
        self.compute_opt(compute_device)
        compress_input_to_torch = torch.tensor(compress_input).to(self.device)
        centroid_to_device = self.centroids.to(self.device)
        tensor_2d = self.d2_flatten(compress_input_to_torch)
        distance_merge = torch.sum((centroid_to_device[None,:,:]  - tensor_2d[:,None,:] )**2,dim=self.dim_metric_for_compute)
        metric_between_input_and_centroid  = torch.argmin(distance_merge,dim=self.dim_argmin_reshape)
        result = centroid_to_device[metric_between_input_and_centroid].reshape(compress_input_to_torch.size()).type(compress_input_to_torch.dtype).cpu().numpy()
        if save_array:
            save(f'{data_path}.npy', result)
            return f'Save compressed array file to {data_path}.npy'
        return result
    
    def save_centroid(self,model_path = "",model_name = "centroids"):
        torch.save(self.centroids, f'{model_path}{model_name}.pt')
        print(f"Model svae at : {model_path}{model_name}.pt")
        
    def load_centroid(self,path_to_model_to_load=""):
        self.centroids = torch.load(path_to_model_to_load).to(self.device)
        print("The model | {path_to_model_to_load} | is load into self.centroids")
        
