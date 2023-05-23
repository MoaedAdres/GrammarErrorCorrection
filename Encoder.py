class Encoder(tf.keras.layers.Layer):

    def __init__(self , vocab_size , embedding_dim , enc_units , input_len):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_len = input_len
        self.enc_units = enc_units
        
        self.embedding = layers.Embedding(input_dim= self.vocab_size,
                                         output_dim = self.embedding_dim,
                                         mask_zero = True,
                                          input_length = self.input_len
                                         )
        self.lstm_bi = layers.Bidirectional(layers.LSTM(units= self.enc_units,return_state = True,return_sequences=True ))
        self.concat1 = layers.Concatenate()
        self.concat2 = layers.Concatenate()
    def call(self,input):

        emb = self.embedding(input)
        enc_output , state_h1 , state_c1 ,state_h2 , state_c2 = self.lstm_bi(emb)
        state_h = self.concat1([state_h1,state_h2])
        state_c  = self.concat2(([state_c1,state_c2]))

        return enc_output ,state_h ,state_c 
