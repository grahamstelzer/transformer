#include <iostream>
#include <torch/torch.h>
#include <cmath>


class InputEmbeddings(int d_model, int vocab_size) {

    public:
        torch::Tensor forward(torch::Tensor x) {
            // given tensor x (indices of select words within vocab, 
            //      ex [["i", "like", "cats"], ...]  to indices [[5, 12, 29], ...]
            //      NOTE: normally multi-dimensional

            // index into embedding map, get embeddings for each index

            // multiply each value in each embedding by sqrt(self.d_model)
            //      NOTE: this is done to prevent overfitting 

            // return list of list of embeddings for each token 

        }

        torch::Tensor get_single_embedding(int idx) {
            // index into embedding map, get embedding at that index
            return this.embedding(idx);
        }


    private:
        int _d_model = d_model;
        int _vocab_size = vocab_size;
        torch::nn::Embedding embedding; // map: token_index:ints : embedding:[0.1, -0.3...] -> len 512
}

    
