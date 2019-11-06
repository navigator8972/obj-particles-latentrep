'''
pytorch implementation of PCN decoder
https://github.com/TonythePlaneswalker/pcn/blob/master/models/pcn_cd.py
'''

import torch
from torch import nn

class PCNDecoder(nn.Module):
    def __init__(self, feature_dim, n_coarse, grid_size=4, r=0.05):
        super(PCNDecoder, self).__init__()
       
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.n_coarsepnts = n_coarse
        self.n_finepnts = n_coarse * grid_size**2
        self.r = r

        #fully-connected nets for coarse prediction
        self.fc_coarse = nn.Sequential(
            nn.Linear(feature_dim, n_coarse),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(n_coarse, n_coarse),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(n_coarse, n_coarse*3)
        )

        #use conv1d, this is different from the tensorflow implementation
        #WARNING: tensorflow.contrib.conv2d by default deals with 1-3 D tensors, dont be fooled by its name
        #also it uses relu as default nonlinearity... gee
        #https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
        self.cnn_fine = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim+2+3, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1, stride=1)
        )

        def init_xavier_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        self.fc_coarse.apply(init_xavier_normal)
        self.cnn_fine.apply(init_xavier_normal)
        
        return
    
    def forward(self, z):
        '''
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))  #(2, 4, 4)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)                                   #(1, 16, 2)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])                                      #(n_batch, n_coarse*16, 2) == (n_batch, n_fine, 2)       

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])                         #(n_batch, n_coarse, 3) --> (n_batch, n_coarse, 16, 3)  repeate coarse points 16 times
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])                                             #(n_batch, n_coarse*16, 3) because n_fine == n_coarse * grid_size**2         

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])                               #(n_batch, 1, feature_dim) --> (n_batch, n_fine, feature_dim)

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)                                          #(n_batch, n_fine, 2+3+feature_dim)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])                             #(n_batch, n_coarse, 1, 3) --> (n_batch, n_coarse, 16, 3)
            center = tf.reshape(center, [-1, self.num_fine, 3])                                                     #(n_batch, n_coarse*16, 3) note n_fine == n_coarse * grid_size**2

        fine = mlp_conv(feat, [512, 512, 3]) + center
        '''
        coarse = self.fc_coarse(z).view(-1, 3, self.n_coarsepnts)                                                   #(n_batch, 3, n_coarse)
        
        #WARNING: the behavior of meshgrid in pytorch is actually different from numpy/tensorflow see
        #https://github.com/pytorch/pytorch/issues/15301
        #it should however be fine here because x==y
        
        g = torch.linspace(-self.r, self.r, self.grid_size)
        if z.is_cuda:
            g = g.cuda()
            
        grid = torch.meshgrid(g, g)
        
        grid_feat = torch.stack(grid, 2).view(-1, 2).unsqueeze(0).repeat(z.shape[0], self.n_coarsepnts, 1)
        
        point_feat = coarse.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.grid_size**2, 1).view(-1, self.n_finepnts, 3)        #(n_batch, n_coarse, grid_size**2, 3) --> (n_batch, n_coarse*grid_size**2, 3)
        
        global_feat = z.unsqueeze(1).expand(-1, self.n_finepnts, -1)
        
        #print(g.is_cuda, point_feat.is_cuda, global_feat.is_cuda)
        #print(grid_feat.shape, point_feat.shape, global_feat.shape)
        feat = torch.cat((grid_feat, point_feat, global_feat), dim=2).transpose(1, 2)   #exchange channel and n_fine to make it consistent with NCWH            

        center = coarse.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.grid_size**2, 1).view(-1, self.n_finepnts, 3).transpose(1, 2)

        fine = self.cnn_fine(feat) + center
        return fine, coarse



if __name__ == '__main__':
    pcn_decoder = PCNDecoder(8, 64, 4, 0.05)

    dummy_feature = torch.randn(32, 8)

    fine, coarse = pcn_decoder(dummy_feature)

    print(fine.shape, coarse.shape)