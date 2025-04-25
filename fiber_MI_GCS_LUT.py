"""
An LUT-based E2E optical communication system.
GCS on MI.

Acknowledgement：
This pytorch project is completed with the help of the claude project of Dr. Rasmus T. Jones.
Express my gratitude and respect here.

"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tools.utils as tu
import tools.NLIN as tnlin
from scipy.io import savemat
import copy

print('torch version: {}'.format(torch.__version__))
print('numpy version: {}'.format(np.__version__))


# %% -------------------- parameters --------------------
# channel paras
chParam = tu.AttrDict()
chParam.M = 64
chParam.w = int(np.log2(chParam.M))
chParam.GN = False
chParam.D = 16.4640
chParam.nPol = 2
chParam.PdBm = 2.0
chParam.P0 = torch.tensor(tu.dB2lin(chParam.PdBm, 'dBm'))
chParam.nSpans = 8
chParam.channels = np.array([-150., -100., -50., 0., 50., 100., 150.])

# encoder Paras
aeParam = tu.AttrDict()
aeParam.constellationDim = 2
aeParam.constellationOrder = chParam.M
aeParam.dtype = torch.float32

# training Paras
trainingParam = tu.AttrDict()
trainingParam.gpuNum = 1    # Number of GPUs available, Use 0 for CPU mode
trainingParam.sampleSize = 1024 * chParam.M  # Increase for better results (especially if M>16)
trainingParam.batchSize = 64 * chParam.M  # Increase for better results (especially if M>16)
trainingParam.learningRate = 0.005
trainingParam.displayStep = 50
trainingParam.path = 'results_MI_fiber_GCS'
trainingParam.earlyStopping = 20
trainingParam.iterations = 150

# initialize the random seed
RandomSeed = 999
print('Random Seed: ', RandomSeed)
random.seed(RandomSeed)
torch.manual_seed(RandomSeed)
np.random.seed(RandomSeed)

device = torch.device('cuda:0' if (torch.cuda.is_available() and trainingParam.gpuNum > 0) else 'cpu')
print('Running on {}'.format(device))

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        print("Linear layer normalized !")


# %% -------------------- generating tx data --------------------
def genBatchHotOnes(batchSize, order):
    hotOne_vector, _, _ = tu.hotOnes(size=batchSize, transpose=(1, 0), M=order)
    hotOne_vector = torch.from_numpy(hotOne_vector).to(torch.float32)
    return hotOne_vector


class Encoder(nn.Module):
    def __init__(self, num_gpu, num_rows, num_cols, device):
        super(Encoder, self).__init__()
        self.num_gpu = num_gpu
        self.device = device

        # Gaussian initialization
        self.matrix = nn.Parameter(torch.normal(mean=0, std=1, size=(num_rows, num_cols)))

        self.normalization_factor = 1.0

    def normalize_matrix(self):
        cstl_points = self.matrix.detach()
        magnitudes_squared = cstl_points[:, 0] ** 2 + cstl_points[:, 1] ** 2
        mean_magnitude_squared = torch.mean(magnitudes_squared)
        self.normalization_factor = torch.sqrt(mean_magnitude_squared)
        return self.normalization_factor

    def forward(self, HotOnes):
        logits = torch.matmul(HotOnes, self.matrix)

        NF = self.normalize_matrix()
        logits = logits / NF

        return logits

# the encoder object
encoder = Encoder(num_gpu=trainingParam.gpuNum,
                  num_rows=aeParam.constellationOrder,
                  num_cols=aeParam.constellationDim,
                  device=device).to(device)


def encoder_f(input, chParam=chParam, encoder=encoder):
    encoder_out = encoder(input)

    xSeed = np.eye(aeParam.constellationOrder, dtype=int)  # this is numpy array
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder(xSeed)
    constellation = constellation.detach().cpu()

    encoder_out = np.sqrt(chParam.P0) * encoder_out
    constellation_power = np.sqrt(chParam.P0) * constellation

    return encoder_out, constellation_power


# %% -------------------- channel --------------------
# NLIN model parameter
para = tnlin.defaultParameters(D=chParam.D)
para.PdBm = chParam.PdBm
para.P0 = chParam.P0
para.nSpans = chParam.nSpans
para.nPol = chParam.nPol
para.channels = chParam.channels
para.nChannels = len(chParam.channels)

aseNoisePower, interConst, intraConst, interConstAdd, intraConstAdd = tnlin.calcConstants(para)


def fiber_nlin(input_signals, constellation_power,
               aseNoisePower = aseNoisePower,
               interConst = interConst,
               intraConst = intraConst,
               interConstAdd = interConstAdd,
               intraConstAdd = intraConstAdd,
               p=para):
    # --- calculate the noise power (sigma^2)
    device = input_signals.device
    sigma2_noise = torch.tensor(aseNoisePower).to(device)            # ASE noise

    # copy from TF claude
    intra_const = torch.from_numpy(np.expand_dims(intraConst, axis=1)).to(device)
    inter_const = torch.from_numpy(interConst).to(device)
    intra_const_add = torch.from_numpy(intraConstAdd).to(device)
    inter_const_add = torch.from_numpy(interConstAdd).to(device)

    # NLIN or GN model
    if chParam.GN:
        sigma2_inter = tnlin.calcInterChannelGN(inter_const, p)
        sigma2_intra = tnlin.calcIntraChannelGN(intra_const, p)

        sigma2_nlin = torch.sum(sigma2_intra) + torch.sum(sigma2_inter)
    else:
        # kur = mean(abs(const).^4)/mean(abs(const).^2).^2; % Second order modulation factor <|a|^4>/<|a|^2>^2
        # kur3 = mean(abs(const).^6)/mean(abs(const).^2).^3; % Third order modulation factor <|a|^6>/<|a|^2>^3
        constellation_abs = torch.norm(constellation_power, p=2, dim=-1)  # calculate the abs of each points
        pow4 = torch.pow(constellation_abs, 4)
        pow6 = torch.pow(constellation_abs, 6)
        kur = torch.mean(pow4) / (p.P0) ** 2.
        kur3 = torch.mean(pow6) / (p.P0) ** 3.
        # # record the kur values
        # p.kur = kur
        # p.kur3 = kur3

        sigma2_inter = tnlin.calcInterChannelNLIN(inter_const, kur, p)
        sigma2_intra = tnlin.calcIntraChannelNLIN(intra_const, kur, kur3, p)

        sigma2_intra_add = tnlin.calcIntraChannelNLIN(intra_const_add, kur, kur3, p)
        sigma2_inter_add = tnlin.calcInterChannelNLINAddTerms(inter_const_add, kur, p)

        sigma2_nlin = torch.sum(sigma2_intra) + torch.sum(
            sigma2_inter) + torch.sum( sigma2_intra_add ) + torch.sum( sigma2_inter_add )

    sigma2 = sigma2_noise + sigma2_nlin
    sigma2 = sigma2.cpu()
    p.SNR_lin = (p.P0 / sigma2).item()
    noise = np.sqrt(sigma2 / 2.0) * torch.normal(mean=0.0, std=1.0, size=input_signals.shape)
    device = input_signals.device
    noise = noise.to(device)

    channel = input_signals + noise
    channel_norm = channel / np.sqrt(p.P0)

    return channel_norm


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

            nn.Linear(256, aeParam.constellationOrder),
        )

    def forward(self, rx_sig):
        output_signals = self.model(rx_sig)
        return output_signals


decoder = Decoder(num_gpu=trainingParam.gpuNum).to(device)

print('\nInitializing the Decoder ......')
decoder.apply(weights_init)


# %% -------------------- loss and optimizer --------------------
criterion_model = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': encoder.parameters()},
                        {'params': decoder.parameters()}], lr=trainingParam.learningRate)


# %% -------------------- SER --------------------
def ser_cal(encoder_in, decoder_out):
    accuracy = (decoder_out.argmax(dim=-1) == encoder_in.argmax(dim=-1)).float().mean().item()
    ser = 1 - accuracy

    return ser

# %% -------------------- before training --------------------
with torch.no_grad():
    encoder.eval()
    xSeed = np.eye(aeParam.constellationOrder, dtype=int)  # this is numpy array
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder(xSeed)
    constellation = constellation.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.plot(constellation[:, 0], constellation[:, 1], 'x')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)

    for ii in range(constellation.shape[0]):
        bit_string = '{}'.format(ii)
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
batchSize = trainingParam.batchSize

np_loss = []
np_ser = []
np_gaussian_mi = []

print('START TRAINING ... ', flush=True)
while(True):
# while(False):
    epoch = epoch + 1

    if epoch > trainingParam.iterations:
        break

    encoder.train()
    decoder.train()

    for batch_idx in range(0, nBatches):
        input = genBatchHotOnes(batchSize, chParam.M).to(device)
        encoder_out_power, constellation_power = encoder_f(input)
        channel_out = fiber_nlin(encoder_out_power, constellation_power)
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

        gaussian_MI_temp = 0
        ser_temp = 0
        loss_temp = 0
        for _ in range(0, nBatches):
            input = genBatchHotOnes(trainingParam.sampleSize, chParam.M).to(device)
            encoder_out_power, constellation_power = encoder_f(input)
            channel_out = fiber_nlin(encoder_out_power, constellation_power)
            decoder_out = decoder(channel_out)

            target = copy.copy(input)
            loss = criterion_model(decoder_out, target)
            loss_temp += loss.item()

            # ----- calculate the metrics
            encoder_out = encoder(input)  # without power

            xSeed = np.eye(aeParam.constellationOrder, dtype=int)
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

            gaussian_MI = tu.gaussianMI(x=encoder_out_complex, y=channel_complex, constellation=constellation_complex, M=chParam.M)
            gaussian_MI_temp += gaussian_MI.item()

            ser_temp += ser_cal(input, decoder_out)

        gaussian_MI = gaussian_MI_temp / nBatches
        ser = ser_temp / nBatches
        loss = loss_temp / nBatches

        # record metric in this epoch
        np_gaussian_mi.append(gaussian_MI)
        np_ser.append(ser)

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

    if epoch - lastImprovement > trainingParam.earlyStopping:
        break

    # -- show the training state
    if epoch % trainingParam.displayStep == 0:
        print('epoch: {:04d} - Loss: {:.7f} - Ser: {:.2e} - GaussianMi: {:.2f}'.format(
            epoch, loss_epoch, ser, gaussian_MI), flush=True)


# metric curve
np_loss = np.array(np_loss)
plt.plot(np_loss)
plt.title('loss')
plt.show()

np_ber = np.array(np_ser)
plt.plot(np_ber)
plt.title('SER')
plt.show()

np_gaussian_gmi = np.array(np_gaussian_mi)
plt.plot(np_gaussian_gmi)
plt.title('MI')
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
    xSeed = np.eye(aeParam.constellationOrder, dtype=int)  # this is numpy array
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder_trained(xSeed)
    constellation = constellation.detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.plot(constellation[:, 0], constellation[:, 1], 'x')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)

    for ii in range(constellation.shape[0]):
        bit_string = '{}'.format(ii)
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

    gaussian_MI_temp = 0
    ser_temp = 0
    loss_temp = 0
    for _ in range(0, nBatches):
        input = genBatchHotOnes(trainingParam.sampleSize, chParam.M).to(device)
        encoder_out_power, constellation_power = encoder_f(input, encoder=encoder)
        channel_out = fiber_nlin(encoder_out_power, constellation_power)
        decoder_out = decoder(channel_out)

        target = copy.copy(input)
        loss = criterion_model(decoder_out, target)
        loss_temp += loss.item()

        # ----- calculate the metrics
        encoder_out = encoder(input)  # without power

        xSeed = np.eye(aeParam.constellationOrder, dtype=int)
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

        gaussian_MI = tu.calcMI_MC(x=encoder_out_complex, y=channel_complex, constellation=constellation_complex)
        gaussian_MI_temp += gaussian_MI.item()

        ser_temp += ser_cal(input, decoder_out)

    gaussian_MI = gaussian_MI_temp / nBatches
    ser = ser_temp / nBatches
    loss = loss_temp / nBatches

    finalMetrics = {'GaussianMI': gaussian_MI, 'SER': ser, 'xentropy': loss, 'SNR': para.SNR_lin}
    print('finalMetrics:', finalMetrics)