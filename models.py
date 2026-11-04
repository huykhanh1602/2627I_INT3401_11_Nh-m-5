from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(dimensions))


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return tensordot(self.w, x, dims=1)

        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self(x)

        "*** YOUR CODE HERE ***"
        if score.item() >= 0:
            return 1
        else:
            return -1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        self.fc1 = Linear(1, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 1)
   

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)

        return x



class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)

        return x



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        d = 256                                        # kích thước trạng thái ẩn h
        self.Wx = Linear(self.num_chars, d)            # xử lý chữ cái hiện tại
        self.Wh = Linear(d, d)                         # xử lý trạng thái ẩn trước đó
        self.out = Linear(d, len(self.languages))      # ra 5 điểm số
        # Initialize your model parameters here



    def forward(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        
        h = relu(self.Wx(xs[0]))                       # chữ cái đầu tiên: f_initial
        for x in xs[1:]:                               # các chữ cái còn lại
            h = relu(self.Wx(x) + self.Wh(h))          # f(h, x): trộn chữ mới + bộ nhớ cũ
        return self.out(h)                             # không có relu ở lớp cuối



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    # Lấy kích thước của ảnh input và kernel
    input_tensor_dimensions = input.shape        # (H_in, W_in), ví dụ 28x28
    weight_dimensions = weight.shape             # (H_k, W_k), ví dụ 3x3

    # Kích thước output sau khi trượt kernel (không padding, stride = 1):
    # out = H_in - H_k + 1  (và tương tự cho chiều rộng)
    y_dim = input_tensor_dimensions[0] - weight_dimensions[0] + 1   # 28 - 3 + 1 = 26
    x_dim = input_tensor_dimensions[1] - weight_dimensions[1] + 1   # 28 - 3 + 1 = 26
    Output_Tensor = zeros((y_dim, x_dim))        # ma trận output khởi tạo toàn số 0

    # Trượt kernel qua từng vị trí (y, x) trên input
    for y in range(y_dim):
        for x in range(x_dim):
            # Cắt cửa sổ con có kích thước bằng kernel tại vị trí (y, x)
            window = input[y:y + weight_dimensions[0], x:x + weight_dimensions[1]]
            # Nhân từng phần tử tương ứng (element-wise) rồi cộng tổng lại:
            # tensordot(window, weight, dims=2) = tổng(window * weight) trên cả 2 chiều
            Output_Tensor[y, x] = tensordot(window, weight, dims=2)

    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))

        # Sau khi Convolve với kernel 3x3 lên ảnh 28x28, kích thước còn lại là 26x26
        input_size = 26 * 26                     # 676 đặc trưng sau khi flatten
        hidden_size = 128                        # kích thước lớp ẩn
        self.fc1 = Linear(input_size, hidden_size)      # 676 -> hidden_size
        self.fc2 = Linear(hidden_size, output_size)     # hidden_size -> 10 (logits)


    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)          # x giờ có shape (batch_size, 26*26) - vector 1 chiều

        x = relu(self.fc1(x))               # lớp ẩn: Linear + relu
        return self.fc2(x)                  # logits (batch_size, 10) - KHÔNG relu ở lớp cuối


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        # B1: Chiếu input lên 3 không gian Query, Key, Value
        Q = self.q_layer(input)             # Query  (B, T, C)
        K = self.k_layer(input)             # Key    (B, T, C)
        V = self.v_layer(input)             # Value  (B, T, C)

        # B2: Chuyển vị K từ (B, T, C) -> (B, C, T) để sẵn sàng nhân matmul với Q
        K_T = movedim(K, 1, 2)

        # B3: Scaled Dot-Product: nhân Q với K^T rồi chia sqrt(layer_size) để ổn định gradient
        scores = matmul(Q, K_T) / (self.layer_size ** 0.5)      # (B, T, T)

        # B4: Causal Mask - chặn vị trí tương lai (mask tam giác dưới), vị trí bị chặn = -inf
        M = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]    # (T, T)

        # B5: Softmax dọc theo chiều cuối cùng -> phân phối xác suất attention
        attention_weights = softmax(M, dim=-1)                  # (T, T)

        # B6: Tổ hợp có trọng số các Value theo attention weights
        return matmul(attention_weights, V)                     # (B, T, C)

