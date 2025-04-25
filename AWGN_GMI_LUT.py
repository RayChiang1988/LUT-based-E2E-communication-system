"""
An LUT-based E2E communication system for AWGN channel.
GCS on GMI.

Acknowledgement：
This pytorch project is completed with the help of the claude project of Dr. Rasmus T. Jones.
Express my gratitude and respect here.

"""

import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tools.utils as tu
from scipy.io import savemat
import copy

# information
print('torch version: {}'.format(torch.__version__))
print('numpy version: {}'.format(np.__version__))


# %% -------------------- parameters --------------------
chParam = tu.AttrDict()
chParam.M = 64
chParam.w = int(np.log2(chParam.M))
if chParam.M == 16:
    chParam.SNR = 15
elif chParam.M == 64:
    chParam.SNR = 18
else:
    chParam.SNR = 22

aeParam = tu.AttrDict()
aeParam.constellationDim = 2
aeParam.constellationOrder = chParam.M
aeParam.dtype = torch.float32

trainingParam = tu.AttrDict()
trainingParam.gpuNum = 1
trainingParam.sampleSize = 16 * chParam.M
trainingParam.batchSize = 1 * chParam.M
trainingParam.learningRate = 0.005
trainingParam.displayStep = 25
trainingParam.path = 'results_AWGN_GMI_LUT'
trainingParam.earlyStopping = 25

Random_Seed = 111
print('Random Seed: ', Random_Seed)
random.seed(Random_Seed)
torch.manual_seed(Random_Seed)
np.random.seed(Random_Seed)

device = torch.device('cuda:0' if (torch.cuda.is_available() and trainingParam.gpuNum > 0) else 'cpu')
print('Running on {}'.format(device))

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        print("Linear layer normalized !")

# %% -------------------- generating tx data --------------------
def genBatchBitsVector(batchSize, order):
    bits_vector = torch.from_numpy(tu.generateBitVectors(batchSize, order)).to(torch.float32)
    return bits_vector


class Encoder(nn.Module):
    def __init__(self, num_gpu, num_rows, num_cols, device):
        super(Encoder, self).__init__()
        self.num_gpu = num_gpu
        self.device = device

        # QAM initialization
        # grayMappingConst = tu.grayMapping(M=num_rows, constType='qam')
        # self.initial_table = torch.tensor(np.column_stack((np.real(grayMappingConst), np.imag(grayMappingConst))), dtype=aeParam.dtype)
        # self.matrix = nn.Parameter(self.initial_table)

        # Gaussian initialization
        self.matrix = nn.Parameter(torch.normal(mean=0, std=0.01, size=(num_rows, num_cols)))

        self.normalization_factor = 1.0
        self.w = int(np.log2(num_rows))
        self.powers_of_two = torch.pow(2, torch.arange(self.w - 1, -1, -1, dtype=torch.float32)).to(self.device)

    def normalize_matrix(self):
        cstl_points = self.matrix.detach()
        magnitudes_squared = cstl_points[:, 0] ** 2 + cstl_points[:, 1] ** 2
        mean_magnitude_squared = torch.mean(magnitudes_squared)
        self.normalization_factor = torch.sqrt(mean_magnitude_squared)
        return self.normalization_factor

    def map_bits_to_index(self, bit_sequences):
        indices = torch.matmul(bit_sequences, self.powers_of_two)
        indices = indices.long()
        return indices

    def forward(self, bit_sequence):
        index = self.map_bits_to_index(bit_sequence)
        logits = self.matrix[index]

        NF = self.normalize_matrix()
        logits = logits / NF

        return logits


encoder = Encoder(num_gpu=trainingParam.gpuNum,
                  num_rows=aeParam.constellationOrder,
                  num_cols=aeParam.constellationDim,
                  device=device).to(device)


# %% -------------------- channel --------------------
def awgn_channel(input_signals):
    chParam.SNR_lin = tu.dB2lin(chParam.SNR, 'dB')
    sigma2_noise = 1/chParam.SNR_lin
    noise = np.sqrt(sigma2_noise / 2.0) * torch.normal(mean=0.0, std=1.0, size=input_signals.shape)
    device = input_signals.device
    noise = noise.to(device)
    channel_out = input_signals + noise

    return channel_out


# %% -------------------- decoder --------------------
class Decoder(nn.Module):
    def __init__(self, num_gpu):
        super(Decoder, self).__init__()
        self.num_gpu = num_gpu
        self.model = nn.Sequential(
            nn.Linear(in_features=2,
                      out_features=256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, chParam.w),
            nn.Sigmoid()
        )

    def forward(self, rx_sig):
        output_signals = self.model(rx_sig)
        return output_signals


decoder = Decoder(num_gpu=trainingParam.gpuNum).to(device)

print('\nInitializing the Decoder ......')
decoder.apply(weights_init)


# %% -------------------- loss and optimizer --------------------
criterion_model = nn.BCELoss()
optimizer = optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=trainingParam.learningRate, betas=(0.9, 0.999))


# %% -------------------- calculating GMI and BER --------------------
def gaussianLLR(constellation, constellation_bits, Y, SNR_lin, M):
    """
        Computes log likelihood ratio with Gaussian auxiliary channel assumption
        Transcribed from the claude of Dr. Rasmus T. Jones.
    """
    constellation = constellation.numpy()
    constellation_bits = constellation_bits.numpy()
    Y = Y.numpy()

    M = int(M)
    m = int(np.log2(M))

    expAbs = lambda x, y, SNR_lin: np.exp(-SNR_lin * np.square(np.abs(y - x)))

    constellation_zero = np.stack([ma.masked_array(constellation, item).compressed()
                                   for item in np.split(np.equal(constellation_bits, 1),
                                                        m, axis=0)],
                                  axis=1)   # mask the ones

    constellation_one = np.stack([ma.masked_array(constellation, item).compressed()
                                   for item in np.split(np.equal(constellation_bits, 0),
                                                        m, axis=0)],
                                  axis=1)   # mask the zeros

    constellation_zero_flat = np.reshape(constellation_zero, -1)
    constellation_one_flat = np.reshape(constellation_one, -1)

    sum_zeros = np.sum( np.reshape( expAbs(np.expand_dims(constellation_zero_flat, axis=1), Y, SNR_lin ), (int(M/2),m,-1) ), axis=0 )
    sum_ones = np.sum( np.reshape( expAbs(np.expand_dims(constellation_one_flat, axis=1), Y, SNR_lin ), (int(M/2),m,-1) ), axis=0 )

    LLRs = np.log( sum_zeros / sum_ones )

    return LLRs


def gmi(bits_reshaped, LLRs):
    """
        Gaussian GMI metric
        Transcribed from the claude of Dr. Rasmus T. Jones.
    """
    bits_reshaped = bits_reshaped.numpy()
    MI_per_bit = 1 - np.mean(np.log2(1+np.exp((2*bits_reshaped - 1) * LLRs)), axis=1)
    GMI = np.sum(MI_per_bit)

    return GMI


def ber_cal(encoder_in, decoder_out):
    input_bits = encoder_in.int()
    output_bits = torch.round(decoder_out).int()
    bit_compare = np.not_equal(output_bits, input_bits)
    # bit_errors = np.sum(bit_compare.int().numpy())
    bit_error_rate = np.mean(bit_compare.int().numpy())

    return bit_error_rate


# %% -------------------- before training --------------------
with torch.no_grad():
    encoder.eval()
    xSeed = tu.generateUniqueBitVectors(chParam.M)
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder(xSeed)
    constellation = constellation.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.plot(constellation[:, 0], constellation[:, 1], 'x')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)

    for ii in range(constellation.shape[0]):
        bit_string = ''.join([str(int(x)) for x in xSeed[ii, :].tolist()])
        plt.text(constellation[ii, 0]+0.01, constellation[ii, 1]+0.01, bit_string, fontsize=12)
    plt.axis('square')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)
    plt.title('Mapping before training')
    plt.show()

# %% -------------------- training --------------------
bestLoss = 100000.0
lastImprovement = 0
epoch = 0
nBatches = int(trainingParam.sampleSize / trainingParam.batchSize)
batchSizeMultiples = 1
batchSize = batchSizeMultiples * trainingParam.batchSize

np_loss = []
np_ber = []
np_gaussian_gmi = []

print('START TRAINING ... ', flush=True)
while(True):
# while(False):
    epoch = epoch + 1

    encoder.train()
    decoder.train()

    for batch_idx in range(0, nBatches):
        input = genBatchBitsVector(batchSize, chParam.M).to(device)
        encoder_out = encoder(input)
        channel_out = awgn_channel(encoder_out)
        decoder_out = decoder(channel_out)

        target = copy.copy(input)
        loss = criterion_model(decoder_out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -- evaluate the system metric
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        gaussian_GMI_temp = 0
        ber_temp = 0
        loss_temp = 0
        for _ in range(0, nBatches):
            input = genBatchBitsVector(trainingParam.sampleSize, chParam.M).to(device)
            encoder_out = encoder(input)
            channel_out = awgn_channel(encoder_out)
            decoder_out = decoder(channel_out)

            target = copy.copy(input)
            loss = criterion_model(decoder_out, target)
            loss_temp += loss.item()

            # ----- calculate the metrics
            xSeed = tu.generateUniqueBitVectors(chParam.M)
            xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
            enc_seed = encoder(xSeed)

            constellation_complex = torch.complex(enc_seed[:, 0], enc_seed[:, 1]).unsqueeze(0)
            encoder_out_complex = torch.complex(encoder_out[:, 0], encoder_out[:, 1]).unsqueeze(0)
            channel_complex = torch.complex(channel_out[:, 0], channel_out[:, 1]).unsqueeze(0)

            xSeed = xSeed.cpu()
            input = input.cpu()
            constellation_complex = constellation_complex.cpu()
            encoder_out_complex = encoder_out_complex.cpu()
            channel_complex = channel_complex.cpu()
            decoder_out = decoder_out.cpu()

            gaussian_LLRs = gaussianLLR(constellation_complex, xSeed.t(), channel_complex, chParam.SNR_lin, chParam.M)
            gaussian_GMI_temp += gmi(input.t(), gaussian_LLRs)

            ber_temp += ber_cal(input, decoder_out)

        gaussian_GMI = gaussian_GMI_temp / nBatches
        ber = ber_temp / nBatches
        loss = loss_temp / nBatches

        # record metric in this epoch
        np_gaussian_gmi.append(gaussian_GMI)
        np_ber.append(ber)

    # -- save the best model
    loss_epoch = loss
    np_loss.append(loss_epoch)
    if loss_epoch < bestLoss:
        bestLoss = loss_epoch
        lastImprovement = epoch
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_epoch': loss_epoch},
            trainingParam.path)

    # convergence check and increase empirical evidence
    if epoch - lastImprovement > trainingParam.earlyStopping:
        checkpoint = torch.load(trainingParam.path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        lastImprovement = epoch

        # increase empirical evidence
        batchSizeMultiples = batchSizeMultiples + 2
        batchSize = batchSizeMultiples * trainingParam.batchSize
        if batchSizeMultiples >= nBatches:
            break

        print("batchSize: {}, batchSizeMultiples: {}".format(batchSize, batchSizeMultiples))

    # -- show the training state
    if epoch % trainingParam.displayStep == 0:
        print('epoch: {:04d} - Loss: {:.7f} - Ber: {:.2e} - GaussianGmi: {:.2f}'.format(
            epoch, loss_epoch, ber, gaussian_GMI), flush=True)

# metric curve
np_loss = np.array(np_loss)
plt.plot(np_loss)
plt.title('loss')
plt.show()

np_ber = np.array(np_ber)
plt.plot(np_ber)
plt.title('BER')
plt.show()

np_gaussian_gmi = np.array(np_gaussian_gmi)
plt.plot(np_gaussian_gmi)
plt.title('GMI')
plt.show()


#%% -------------------- after training --------------------
print('Loading the best model ... ')
checkpoint = torch.load(trainingParam.path)
print(f"The best model was obtained in epoch {checkpoint['epoch']}. The best loss is {checkpoint['loss_epoch']}")
encoder_trained = Encoder(num_gpu=trainingParam.gpuNum,
                          num_rows=aeParam.constellationOrder,
                          num_cols=aeParam.constellationDim,
                          device=device).to(device)
encoder_trained.load_state_dict(checkpoint['encoder_state_dict'])

with torch.no_grad():
    encoder_trained.eval()
    xSeed = tu.generateUniqueBitVectors(chParam.M)  # this is numpy array
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder_trained(xSeed)
    constellation = constellation.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.plot(constellation[:, 0], constellation[:, 1], 'x')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)

    for ii in range(constellation.shape[0]):
        bit_string = ''.join([str(int(x)) for x in xSeed[ii, :].tolist()])
        plt.text(constellation[ii, 0]+0.01, constellation[ii, 1]+0.01, bit_string, fontsize=12)
    plt.axis('square')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)
    plt.title('Mapping after training')
    plt.show()

# -- evaluate the system metric
with torch.no_grad():
    encoder.eval()
    decoder.eval()

    checkpoint = torch.load(trainingParam.path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    gaussian_GMI_temp = 0
    ber_temp = 0
    loss_temp = 0
    nBatches = 100
    for _ in range(0, nBatches):
        input = genBatchBitsVector(nBatches, chParam.M).to(device)
        encoder_out = encoder(input)
        channel_out = awgn_channel(encoder_out)
        decoder_out = decoder(channel_out)

        target = copy.copy(input)
        loss = criterion_model(decoder_out, target)
        loss_temp += loss.item()

        # ----- calculate the metrics
        xSeed = tu.generateUniqueBitVectors(chParam.M)  # this is numpy array
        xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
        enc_seed = encoder(xSeed)

        constellation_complex = torch.complex(enc_seed[:, 0], enc_seed[:, 1]).unsqueeze(0)
        encoder_out_complex = torch.complex(encoder_out[:, 0], encoder_out[:, 1]).unsqueeze(0)
        channel_complex = torch.complex(channel_out[:, 0], channel_out[:, 1]).unsqueeze(0)

        # change the device
        xSeed = xSeed.cpu()
        input = input.cpu()
        constellation_complex = constellation_complex.cpu()
        encoder_out_complex = encoder_out_complex.cpu()
        channel_complex = channel_complex.cpu()
        decoder_out = decoder_out.cpu()

        gaussian_LLRs = gaussianLLR(constellation_complex, xSeed.t(), channel_complex, chParam.SNR_lin, chParam.M)
        gaussian_GMI_temp += gmi(input.t(), gaussian_LLRs)

        ber_temp += ber_cal(input, decoder_out)

    gaussian_GMI = gaussian_GMI_temp / nBatches
    ber = ber_temp / nBatches
    loss = loss_temp / nBatches

    finalMetrics = {'GaussianGMI': gaussian_GMI, 'BER': ber, 'xentropy': loss}
    print('finalMetrics:', finalMetrics)