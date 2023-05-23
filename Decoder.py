class Decoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size , embedding_dim, dec_unit,input_len ):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_len = input_len
        self.dec_unit =dec_unit
        
        
    def build(self,input_shape):
        
        self.embedding = layers.Embedding(input_dim = self.vocab_size,
                                          output_dim = self.embedding_dim,
                                         mask_zero=True,
                                         input_length = self.input_len)
        self.lstm = layers.LSTM(units = self.dec_unit,
                               return_sequences=True,
                               return_state=True)
        
    def call(self,input, state):
        
        emb = self.embedding(input)
        
        dec_out,state_h,state_c = self.lstm(emb,initial_state = state)
        
        return dec_out,state_h,state_c