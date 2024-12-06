

#define BOOST_TEST_MODULE MyTest
#include <boost/test/included/unit_test.hpp> //single-header

#include <string>
#include <torch/torch.h>
#include <cmath>
#include <vector>

#include "./model.hpp"

int D_MODEL = 512;
int VOCAB_SIZE = 1000


BOOST_AUTO_TEST_CASE(my_boost_test)
{
    // TODO: add tokenize function in here so we can test something like:
    //      [["i", "like", "cats"]] -> [[5, 12, 29]]
    std::vector<float> data = {0.1, -0.3, 1.4};

    torch:Tensor in_tensor = torch.from_blob(data.data(), {1,3});
    std::cout << in_tensor << std::endl;

    InputEmbeddings i(D_MODEL, VOCAB_SIZE);

    // torch::Tensor out_tensor = torch.from_blob()
    // check return tensor from forward() is expected tensor
    // BOOST_CHECK(i.forward() ==  );
}