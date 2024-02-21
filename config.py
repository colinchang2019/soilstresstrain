import torch

class config:
    def __init__(self):
        self.src_len = 1 
        self.tgt_len = 2  
        self.n_layers = 2 
        self.drop_rate = 0.2
        self.input_size = 5  # 4
        self.hidden_size = 6

        # for data
        self.pre = "./data/"
        self.preTransform = "./dataTransform/"
        self.preResult = "./dataResult/"
        self.last = ".xlsx"
        self.data = ["da1", "da2", "da3", "db1", "db2", "db3", "dc1", "dc2", "dc3"]
        # self.trains = ["da2", "da3", "db3", "dc2", "dc3"]
        self.trains =["da1", "db1", "dc1"]
        self.tests = ["da1", "db1", "dc1"]

        # for min-max-scaler
        self.strain_sca = 0.07
        self.stress_sca = 4.7
        self.time_sca = 16.0
        self.pgah_sca = 0.2
        self.pgvh_sca = 0.0308
        self.IA_sca = 0.018
        self.n_sca = 8.0

        # for dataset size
        self.train_size = (1200-self.src_len) * len(self.trains)
        self.test_size = (1200-self.src_len) * len(self.tests) 

        # for training
        self.batch = 200
        self.num_epochs = 1000
        self.num_workers = 0  # 多线程/ windows必须设置为0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = (0.000001, 2, 0.5)  # (0.000001, 2, 0.5)

        # saveing path for model
        self.pathm = "./modelResult/transform_"

        # prepare for earlystopping
        self.patience = 7 # 40

        # for equations





cfg = config()
