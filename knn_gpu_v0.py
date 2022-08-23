import torch

class KNN_torch:
    def __init__(self, data, k):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).to(self.device)
        self.k = torch.tensor(k).to(self.device)
        self.n = self.data.size(dim=0)
        
    def predict(self, new_points):
        new_points=torch.from_numpy(new_points).to(self.device)
        predictions = torch.zeros(new_points.size(0)).to(self.device)
        for i, point in enumerate(new_points): 
            distances = self._calculate_distances(point)
            label_neighbors = self.data[:,-1][torch.argsort(distances)[:self.k]]
            predictions[i] = torch.argmax(torch.bincount(label_neighbors.to(torch.int64)))
        return predictions
        
    def _calculate_distances(self, new_point):
        new_point = torch.stack([new_point for i in range(self.n)])  #new_point.resize_((self.n, new_point.size(dim=0)))
        euclidean_distance = torch.sum(torch.pow((torch.subtract(self.data[:,0:-1],new_point)),2), dim=1)
        return euclidean_distance
      
      #Todo Fix dimension generalization
