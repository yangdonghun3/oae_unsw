import torch
from torch.autograd import Variable
from torch import nn

class autoencoder(nn.Module):
    def __init__(self, input_size):
        super(autoencoder,self).__init__()        
        
        self.Tanh=nn.LeakyReLU()
        self.Dropout=nn.Dropout(0.1)   
        
        self.encoder_1=nn.Linear(input_size, 96)        
        self.encoder_2=nn.Linear(96,64)          
        self.encoder_3=nn.Linear(64,48)
        self.encoder_4=nn.Linear(48,16)           
        self.encoder_5=nn.Linear(16,4)
        self.decoder_1=nn.Linear(4,16)           
        self.decoder_2=nn.Linear(16, 48)         
        self.decoder_3=nn.Linear(48, 64)            
        self.decoder_4=nn.Linear(64, 96)            
        self.decoder_5=nn.Linear(96, input_size)
            
    def forward(self, x):
        
        encoder_out_1=self.Tanh(self.encoder_1(x))
        encoder_out_2=self.Tanh(self.encoder_2(encoder_out_1))
        encoder_out_3=self.Tanh(self.encoder_3(encoder_out_2))                      
        encoder_out_4=self.Tanh(self.encoder_4(encoder_out_3))                        
        encoder_out_5=self.encoder_5(encoder_out_4)

        decoder_out_1=self.Tanh(self.decoder_1(encoder_out_5))
        decoder_out_2=self.Tanh(self.decoder_2(decoder_out_1))
        decoder_out_3=self.Tanh(self.decoder_3(decoder_out_2))                        
        decoder_out_4=self.Tanh(self.decoder_4(decoder_out_3))                          
        decoder_out_5=self.decoder_5(decoder_out_4)     
        
        
        en_out=[encoder_out_1,encoder_out_2,encoder_out_3,encoder_out_4,encoder_out_5]
        de_out=[decoder_out_1,decoder_out_2,decoder_out_3,decoder_out_4,decoder_out_5]
       
        return en_out,de_out,encoder_out_5,decoder_out_5
                                   
                                   
                                   
                                   
class Vautoencoder(nn.Module):
    def __init__(self, input_size):
        super(Vautoencoder,self).__init__()        
        
        self.Tanh=nn.LeakyReLU()
        self.Dropout=nn.Dropout(0.1)   
        
        self.encoder_1=nn.Linear(input_size, 96)        
        self.encoder_2=nn.Linear(96,64)          
        self.encoder_3=nn.Linear(64,48)
        self.encoder_4=nn.Linear(48,16)           
        self.encoder_mean=nn.Linear(16,4)
        self.encoder_var=nn.Linear(16,4)
        
        self.decoder_1=nn.Linear(4,16)           
        self.decoder_2=nn.Linear(16, 48)         
        self.decoder_3=nn.Linear(48, 64)            
        self.decoder_4=nn.Linear(64, 96)            
        self.decoder_5=nn.Linear(96, input_size)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).cuda()        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
            
    def forward(self, x):
        
        encoder_out_1=self.Dropout(self.Tanh(self.encoder_1(x)))
        encoder_out_2=self.Dropout(self.Tanh(self.encoder_2(encoder_out_1)))
        encoder_out_3=self.Dropout(self.Tanh(self.encoder_3(encoder_out_2)))                         
        encoder_out_4=self.Dropout(self.Tanh(self.encoder_4(encoder_out_3)))                           
        encoder_out_mean=self.encoder_mean(encoder_out_4)
        encoder_out_var=self.encoder_var(encoder_out_4)
        
        z = self.reparameterization(encoder_out_mean, torch.exp(0.5 * encoder_out_var))

        decoder_out_1=self.Dropout(self.Tanh(self.decoder_1(z)))
        decoder_out_2=self.Dropout(self.Tanh(self.decoder_2(decoder_out_1)))
        decoder_out_3=self.Dropout(self.Tanh(self.decoder_3(decoder_out_2)))                         
        decoder_out_4=self.Dropout(self.Tanh(self.decoder_4(decoder_out_3)))                           
        decoder_out_5=torch.sigmoid(self.decoder_5(decoder_out_4))                                       
        
        
        en_out=[encoder_out_1,encoder_out_2,encoder_out_3,encoder_out_4,encoder_out_mean,encoder_out_var]
        de_out=[decoder_out_1,decoder_out_2,decoder_out_3,decoder_out_4,decoder_out_5]
       
        return en_out,de_out,encoder_out_mean,encoder_out_var,decoder_out_5                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   