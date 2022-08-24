import torch
#TODO a generalize dataloader for unstructured and structured with scaler and category

class k_means_gpu:
    def __init__(self, k=2, convergence = 0.001, num_iteration = 300):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clusters = torch.tensor(k).to(self.device)
        self.convergence = torch.tensor(convergence).to(self.device)
        self.iteration = torch.tensor(num_iteration).to(self.device)
        self.has_converged= False
        
    def fit(self, data, retrain = False):
        
        #splitting is done on cpu then batching and computation on gpu if possible
        data = torch.tensor(X).to("cpu")
        shape= data.size()
        
        #build centroids
        initial_random_value = torch.randint(0, len(data)-1,(self.clusters,))
        
        if retrain:
            self.centroids = self.centroids
        else:
            self.centroids=data[initial_random_value].to(self.device)
        
        #hyperparameter
        centroid_k_category=torch.arange(self.clusters).reshape(self.clusters,1).to(self.device)
        excluder = torch.tensor([0]).to(device= self.device,dtype= X.dtype).float()
        dim_metric_for_compute = len(shape) if len(shape) >= 2 else 2
        dim_argmin_reshape =  len(shape)-1 if len(shape) >= 2 else 2
        gpu_allocation = int(1/((2000000/(self.clusters/10))/shape[0]))
        data_split = torch.tensor_split(data[:,None,:],gpu_allocation if gpu_allocation > 1 else 2) #TODO fix none for generalize to any dimension and sparse matrix
        performance= []

        #start training
        for epoch in range(self.iteration):  
            #building block to rebuild gradient from batch
            reminder_val = torch.ones(self.clusters,shape[-1]).to(self.device)
            reminder_op = torch.ones(self.clusters,shape[-1]).to(self.device)
            
            #Training on random batch block and compute average gradient at the end of eapoch
            for batchs in torch.randint(0, len(data_split)-1,(len(data_split),)):
                batch = data_split[batchs].to(self.device)
                centroid_reshape_for_compute =  self.centroids[None,:,:] #TODO fix none for generalize to any dimension and sparse matrix
                metric_between_input_and_centroid = torch.sqrt(torch.sum(torch.pow((centroid_reshape_for_compute -batch),2),dim = dim_metric_for_compute ))# custom distance metric calculation can be  implement here
                centroid_category = torch.argmin(metric_between_input_and_centroid,dim = dim_argmin_reshape)
                boolean_mask_with_reshape = (centroid_k_category == centroid_category).T[:,:,None] #TODO fix none for generalize to any dimension and sparse matrix
                extract_close_metric_using_batch_on_mask = (boolean_mask_with_reshape * batch ).type(torch.float)
                latest_centroid = extract_close_metric_using_batch_on_mask.sum(dim=0) / (~torch.isclose(torch.nan_to_num(extract_close_metric_using_batch_on_mask),excluder)).long().sum(dim=0)
                reminder_val += (torch.nan_to_num(latest_centroid,nan=0.0) - (self.centroids * torch.nan_to_num(latest_centroid,nan=0.0).bool().float()) )
                reminder_op +=torch.nan_to_num(latest_centroid,nan=0.0).bool().float()
                
            #update centroids from batch gradient
            latest_centroid = self.centroids + torch.nan_to_num(reminder_val/reminder_op)

            #Estimator to determine continuity of training
            centroid_convergence =torch.sqrt(torch.sum(((self.centroids - latest_centroid)**2),dim=1))
            mean_convergence = torch.mean(centroid_convergence)
            performance.append(mean_convergence)
            print(f"epoch number: {epoch} |actual convergence: {torch.mean(centroid_convergence)} | converge min / rate: {self.convergence}")
            
            #Use estimator to end or continue iteration
            if epoch > 1:
                evolution = min([performance[i]-performance[i+1] for i in range(len(performance)-1)])
                if torch.mean(centroid_convergence) >= self.convergence and mean_convergence < min(performance):
                    self.has_converged = False
                    
                if mean_convergence >= min(performance) or mean_convergence == min(performance) or self.convergence >= evolution:        
                    self.has_converged=True
                    
                if self.has_converged:
                    break
                    
        self.centroids = latest_centroid        
        return self.centroids
    
    
    def encode(pointer_to_encode):
        pointer_to_encode_tensor = torch.tensor(pointer_to_encode).to(self.device)
        metric_between_input_and_centroid  = torch.sqrt(torch.sum(((self.centroids.type(torch.float) - img_point)**2),dims=1)) #TODO generalize dim, for N dim and sparse matrix
        return torch.argmin(metric_between_input_and_centroid)
    
    def decode(pointer_to_decode):
        return self.centroids[pointer_to_decode]
    
    def save_model(model_path = "",model_name = "centroids"):
        torch.save(self.centroids, f'{model_path}{model_name}.pt')
        print(f"Model svae at : {model_path}{model_name}.pt")
        
    def load_model(path_to_model_to_load=""):
        self.centroids = torch.load(path_to_model_to_load).to(self.device)
        print("The model | {path_to_model_to_load} | is load into self.centroids")
