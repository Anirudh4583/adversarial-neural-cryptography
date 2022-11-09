import torch
import adversarialNeuralCryptography as anc

LOAD_PATH = "./adversarial_neural_cryptography_model_and_optimizer"


def random_generate_ptext_and_key(ptext_size, key_size):
    """
    generate a ptext and a key for validate
    """
    ptext = torch.randint(0, 2, (1, ptext_size),
                          dtype=torch.float).to(anc.DEVICE) * 2 - 1
    key = torch.randint(0, 2, (1, key_size),
                        dtype=torch.float).to(anc.DEVICE) * 2 - 1
    return ptext, key


def model_load_checkpoint():
    """
    anirudh, chirag, yash load checkpoint
    :return: a tuple: (anirudh, chirag, yash)
    """
    checkpoint = torch.load(LOAD_PATH)

    anirudh = anc.Model(anc.PTEXT_SIZE, anc.KEY_SIZE)
    chirag = anc.Model(anc.PTEXT_SIZE, anc.KEY_SIZE)
    yash = anc.Model(anc.PTEXT_SIZE)

    anirudh.load_state_dict(checkpoint['Alice_state_dict'])
    chirag.load_state_dict(checkpoint['Bob_state_dict'])
    yash.load_state_dict(checkpoint['Eve_state_dict'])

    anirudh.to(anc.DEVICE)
    chirag.to(anc.DEVICE)
    yash.to(anc.DEVICE)

    return anirudh, chirag, yash


def validate():
    """
    generate a ptext and key and compare them to the output of the model
    :return:
    """
    ptext, key = random_generate_ptext_and_key(anc.PTEXT_SIZE, anc.KEY_SIZE)

    anirudh, chirag, yash = model_load_checkpoint()

    ctext = anirudh(torch.cat((ptext, key), 1).float())

    predict_ptext_chirag = chirag(torch.cat((ctext, key), 1).float())
    predict_ptext_yash = yash(ctext)

    # for better print
    ptext = ptext.cpu().detach().numpy()
    predict_ptext_chirag = predict_ptext_chirag.cpu().detach().numpy()
    predict_ptext_yash = predict_ptext_yash.cpu().detach().numpy()

    print('Real ptext:\n{}\n\nptext chirag:\n{}\n\nptext yash:\n{}'.format(
        ptext, predict_ptext_chirag, predict_ptext_yash))


if __name__ == '__main__':
    validate()
