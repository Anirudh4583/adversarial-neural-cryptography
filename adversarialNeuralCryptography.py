import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import time

if(torch.cuda.is_available()):
    print("\n\nUsing cuda cores for training... \n\n")

N = 16
PTEXT_SIZE = 16
KEY_SIZE = 16
CTEXT_SIZE = 16

CLIP_VALUE = 1
LEARNING_RATE = 0.0008
BATCH_SIZE = 256
MAX_TRAINING_LOOPS = 100000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./adversarial_neural_cryptography_model_and_optimizer"

YASH_ONE_BIT_WRONG_THRESH = 0.97
CHIRAG_ONE_BIT_WRONG_THRESH = 0.0025

LOOPS_PER_PRINT = 100   # every 100 loops print one time


class Model(nn.Module):
    """
    the model anirudh, chirag and yash.
    1 linear + 4 Conv1d.
    """

    def __init__(self, text_size, key_size=None):
        super(Model, self).__init__()
        self.linear = self.linear_init(text_size, key_size)
        self.conv1 = nn.Conv1d(1, 2, 4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(2, 4, 2, stride=2)
        self.conv3 = nn.Conv1d(4, 4, 1, stride=1)
        self.conv4 = nn.Conv1d(4, 1, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x[None, :, :].transpose(0, 1)
        x = self.sigmoid(self.linear(x))
        x = self.sigmoid(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        x = self.tanh(self.conv4(x))
        x = torch.squeeze(x, 1)
        return x

    def linear_init(self, text_size, key_size):
        if key_size is not None:
            return nn.Linear(text_size + key_size, 2 * N)
        else:
            return nn.Linear(text_size, 2 * N)


def generate_data(batch_size, ptext_size, key_size):
    """
    generate data.

    :param batch_size: batch size, hyper-parameters, in this program BATCH_SIZE is 256
    :param ptext_size: ptext size, hyper-parameters, in this program PTEXT_SIZE is 16
    :param key_size: key's size, hyper-parameters, in this program KEY_SIZE is 16
    :return: ptext and key, in this program size are both [256, 16]
    """
    ptext = torch.randint(0, 2, (batch_size, ptext_size),
                          dtype=torch.float).to(DEVICE) * 2 - 1
    key = torch.randint(0, 2, (batch_size, key_size),
                        dtype=torch.float).to(DEVICE) * 2 - 1
    return ptext, key


def plot_wrong(yash_wrong_for_plot, chirag_wrong_for_plot):
    """
    plot epoch-wrong picture

    :param yash_wrong_for_plot: a list, element is the mean of yash one bit wrong
    :param chirag_wrong_for_plot: a list, element is the mean of chirag one bit wrong
    :return:
    """
    plt.plot(range(1, len(yash_wrong_for_plot)+1),
             yash_wrong_for_plot, label='yash one bit wrong mean')
    plt.plot(range(1, len(chirag_wrong_for_plot)+1),
             chirag_wrong_for_plot, label='chirag one bit wrong mean')
    plt.xlabel("Epochs")
    plt.ylabel("One Bit Wrong")
    plt.title("optimizer_chirag_times: optimizer_yash_times = 1 : 2")
    plt.legend()
    plt.show()


def train():
    """
    Do the following:
    1. generate data
    2. train model
    3. finish running and save parameters if satisfing conditions
    4. print the waste of time and errors
    5. plot epochs-errors picture when finish running
    """

    # init
    yash_one_bit_wrong_mean = 2.0
    chirag_one_bit_wrong_mean = 2.0

    yash_wrong_for_plot = []
    chirag_wrong_for_plot = []

    anirudh = Model(PTEXT_SIZE, KEY_SIZE).to(DEVICE)
    chirag = Model(CTEXT_SIZE, KEY_SIZE).to(DEVICE)
    yash = Model(CTEXT_SIZE).to(DEVICE)

    anirudh.train()
    chirag.train()
    yash.train()

    optimizer_anirudh = optim.Adam(anirudh.parameters(), lr=LEARNING_RATE)
    optimizer_chirag = optim.Adam(chirag.parameters(), lr=LEARNING_RATE)
    optimizer_yash = optim.Adam(yash.parameters(), lr=LEARNING_RATE)

    # loss function
    chirag_reconstruction_error = nn.L1Loss()
    yash_reconstruction_error = nn.L1Loss()

    for i in range(MAX_TRAINING_LOOPS):

        start_time = time.time()

        # if satisfy conditions, finish running and save parameters.
        if yash_one_bit_wrong_mean > YASH_ONE_BIT_WRONG_THRESH and chirag_one_bit_wrong_mean < CHIRAG_ONE_BIT_WRONG_THRESH:
            print()
            print("Satisfing Conditions.")

            # save model parameters、 optimizer parameters and yash_one_bit_wrong_mean、 chirag_one_bit_wrong_mean
            torch.save({
                'Anirudh_state_dict': anirudh.state_dict(),
                'Chirag_state_dict': chirag.state_dict(),
                'Yash_state_dict': yash.state_dict(),
                'optimizer_anirudh_state_dict': optimizer_anirudh.state_dict(),
                'optimizer_chirag_state_dict': optimizer_chirag.state_dict(),
                'optimizer_yash_state_dict': optimizer_yash.state_dict(),
                'chirag_one_bit_wrong_mean': chirag_one_bit_wrong_mean,
                'yash_one_bit_wrong_mean': yash_one_bit_wrong_mean
            }, SAVE_PATH)

            print('Saved the parameters successfully.')
            break

        # train anirudh_chirag : train yash = 1 : 2
        for network, num_minibatch in {'anirudh_chirag': 1, 'yash': 2}.items():

            for minibatch in range(num_minibatch):

                ptext, key = generate_data(BATCH_SIZE, PTEXT_SIZE, KEY_SIZE)

                ctext = anirudh(torch.cat((ptext, key), 1).float())
                ptext_yash = yash(ctext)

                if network == 'anirudh_chirag':

                    ptext_chirag = chirag(torch.cat((ctext, key), 1).float())

                    error_chirag = chirag_reconstruction_error(
                        ptext_chirag, ptext)
                    error_yash = yash_reconstruction_error(ptext_yash, ptext)
                    anirudh_chirag_loss = error_chirag + \
                        (1.0 - error_yash ** 2)

                    optimizer_anirudh.zero_grad()
                    optimizer_chirag.zero_grad()
                    anirudh_chirag_loss.backward()
                    nn.utils.clip_grad_value_(anirudh.parameters(), CLIP_VALUE)
                    nn.utils.clip_grad_value_(chirag.parameters(), CLIP_VALUE)
                    optimizer_anirudh.step()
                    optimizer_chirag.step()

                elif network == 'yash':

                    error_yash = yash_reconstruction_error(ptext_yash, ptext)

                    optimizer_yash.zero_grad()
                    error_yash.backward()
                    nn.utils.clip_grad_value_(yash.parameters(), CLIP_VALUE)
                    optimizer_yash.step()

        time_elapsed = time.time() - start_time

        chirag_one_bit_wrong_mean = error_chirag.cpu().detach().numpy()
        yash_one_bit_wrong_mean = error_yash.cpu().detach().numpy()

        if i % LOOPS_PER_PRINT == 0:
            print(f'Epoch: {i + 1:06d} | '
                  f'one epoch time: {time_elapsed:.3f} | '
                  f'chirag one bit wrong: {chirag_one_bit_wrong_mean:.4f} |'
                  f'yash one bit wrong: {yash_one_bit_wrong_mean:.4f}')

        yash_wrong_for_plot.append(yash_one_bit_wrong_mean)
        chirag_wrong_for_plot.append(chirag_one_bit_wrong_mean)

    plot_wrong(yash_wrong_for_plot, chirag_wrong_for_plot)


if __name__ == '__main__':
    train()
