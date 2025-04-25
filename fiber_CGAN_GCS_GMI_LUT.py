"""
An LUT-based E2E optical communication system with a CGAN surrogate channel.
GCS on GMI.

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
import copy
import tools.NLIN as tnlin
from scipy.io import savemat
import time

print('torch version: {}'.format(torch.__version__))
print('numpy version: {}'.format(np.__version__))


# %% -------------------- parameters --------------------
# channel paras
chParam = tu.AttrDict()
chParam.M = 64
chParam.w = int(np.log2(chParam.M))

chParam.z_noise = 5         # noise vector length for CGAN

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
trainingParam.gpuNum = 1
trainingParam.sampleSize = 128 * chParam.M
trainingParam.batchSize = 16 * chParam.M
trainingParam.learningRate = 0.001
trainingParam.displayStep = 100
trainingParam.path = 'results_GMI_AWGN'
trainingParam.earlyStopping = 25
trainingParam.iterations = 500
trainingParam.num_iter = 4

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
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        print("BatchNorm layer normalized !")


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
        grayMappingConst = tu.grayMapping(M=num_rows, constType='qam')
        self.initial_table = torch.tensor(np.column_stack((np.real(grayMappingConst), np.imag(grayMappingConst))), dtype=aeParam.dtype)
        self.matrix = nn.Parameter(self.initial_table)

        # Gaussian initialization
        # self.matrix = nn.Parameter(torch.normal(mean=0, std=0.01, size=(num_rows, num_cols)))

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

# the encoder object
encoder = Encoder(num_gpu=trainingParam.gpuNum,
                  num_rows=aeParam.constellationOrder,
                  num_cols=aeParam.constellationDim,
                  device=device).to(device)


def encoder_f(input, chParam=chParam, encoder=encoder):
    encoder_out = encoder(input)

    xSeed = tu.generateUniqueBitVectors(chParam.M)  # this is numpy array
    xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
    constellation = encoder(xSeed)
    constellation = constellation.detach()

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
    device = input_signals.device
    sigma2_noise = torch.tensor(aseNoisePower).to(device)            # ASE noise

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
        # record the kur values
        p.kur = kur
        p.kur3 = kur3

        sigma2_inter = tnlin.calcInterChannelNLIN(inter_const, kur, p)
        sigma2_intra = tnlin.calcIntraChannelNLIN(intra_const, kur, kur3, p)

        # sigma2_intra_add = tnlin.calcIntraChannelNLIN(intra_const_add, kur, kur3, P0, chParam.nPol, dtype=aeParam.dtype)
        # sigma2_inter_add = tnlin.calcInterChannelNLINAddTerms(inter_const_add, kur, P0, chParam.nPol,
        #                                                       dtype=aeParam.dtype)

        sigma2_nlin = torch.sum(sigma2_intra) + torch.sum(
            sigma2_inter)

    sigma2 = sigma2_noise + sigma2_nlin
    sigma2 = sigma2.cpu()
    p.SNR_lin = (p.P0 / sigma2).item()          # get the SNR for GMI calculation
    noise = np.sqrt(sigma2 / 2.0) * torch.normal(mean=0.0, std=1.0, size=input_signals.shape)
    device = input_signals.device
    noise = noise.to(device)

    channel = input_signals + noise
    channel_norm = channel / np.sqrt(p.P0)

    return channel_norm


# CGAN channel
class Generator_channel(nn.Module):
    def __init__(self, num_gpu, dim):
        super(Generator_channel, self).__init__()
        self.num_gpu = num_gpu
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 2),
        )

    def forward(self, input_signals, z_noise):
        G_input = torch.cat((input_signals, z_noise), dim=1)
        x = self.model(G_input)
        logits = x

        return logits


netG_channel = Generator_channel(trainingParam.gpuNum, chParam.z_noise + aeParam.constellationDim).to(device)

print('\nInitializing the generator ......')
netG_channel.apply(weights_init)


class discriminator(nn.Module):
    def __init__(self, num_gpu, dim):
        super(discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, cond, rx_signals):
        D_input = torch.cat((cond, rx_signals), dim=1)  # the first dim is batch        # to be checked
        logits = self.model(D_input)

        return logits


netD_channel = discriminator(trainingParam.gpuNum, aeParam.constellationDim + aeParam.constellationDim).to(device)

print('\nInitializing the discriminator ......')
netD_channel.apply(weights_init)


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
optimizer_encoder = optim.Adam([{'params': encoder.parameters()}], lr=trainingParam.learningRate, betas=(0.9, 0.999))
optimizer_decoder = optim.Adam([{'params': decoder.parameters()}], lr=trainingParam.learningRate, betas=(0.9, 0.999))

# for CGAN
criterion_channel_D = nn.BCELoss()
auxiliary_criterion_G = nn.MSELoss()
eta_gLoss = 1                            # weight of auxiliary loss
optimizer_channel_G = optim.Adam([{'params': netG_channel.parameters()}], lr=trainingParam.learningRate, betas=(0.9, 0.999))
optimizer_channel_D = optim.Adam([{'params': netD_channel.parameters()}], lr=trainingParam.learningRate, betas=(0.9, 0.999))


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


# %% -------------------- training --------------------
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
batchSize = trainingParam.batchSize

np_gaussian_gmi_iter = []
np_ber_iter = []

num_epochs_encoder = 500
num_epochs_GAN = 500
num_epochs_decoder = 500
print('-------------------- Starting Training Loop ...... --------------------')
start_time = time.time()

for iter in range(trainingParam.num_iter):
    print('---------- training iteration: {}/{}  ----------'.format(iter+1, trainingParam.num_iter))
    # ----- train the decoder with true gradient
    print('----- training the decoder')
    encoder.eval()
    netD_channel.eval()
    netG_channel.eval()
    decoder.train()
    loss_temp = 0
    for epoch_decoder in range(num_epochs_decoder):
        for batch_idx in range(0, nBatches):
            input = genBatchBitsVector(batchSize, chParam.M).to(device)
            encoder_out_power, constellation_power = encoder_f(input)
            channel_out = fiber_nlin(encoder_out_power, constellation_power).detach()
            decoder_out = decoder(channel_out)

            target = copy.copy(input)
            loss = criterion_model(decoder_out, target)
            loss_temp += loss.item()

            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()

        loss = loss_temp / nBatches

        # show the training state
        if epoch_decoder % trainingParam.displayStep == 0:
            print('epoch: {:04d} - Loss: {:.7f}'.format(epoch_decoder, loss), flush=True)    # 这不是epoch的loss

    # ----- train the GAN channel
    print('---------- training the GAN channel')
    encoder.eval()
    netD_channel.train()
    netG_channel.train()
    decoder.eval()
    for epoch_GAN in range(num_epochs_GAN):
        loss_G_temp = 0
        loss_D_temp = 0

        # for each batch in the dataloader
        for batch_idx in range(0, nBatches):
            input = genBatchBitsVector(batchSize, chParam.M).to(device)
            encoder_out_power, constellation_power = encoder_f(input)
            channel_out = fiber_nlin(encoder_out_power, constellation_power)

            encoder_out = encoder_out_power.detach()
            channel_out = channel_out.detach()

            cond_device = encoder_out
            b_size = channel_out.size(0)

            for _ in range(1):
                ####################
                # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
                ####################
                # real
                netD_channel.zero_grad()
                real_device = channel_out
                label = 0.9 + 0.2 * torch.rand(b_size, device=device)
                output = netD_channel(cond_device, real_device).view(-1)
                errD_real = criterion_channel_D(output, label)
                D_x = output.mean().item()

                # fake
                z_noise = torch.randn(b_size, chParam.z_noise).to(device)
                fake = netG_channel(encoder_out, z_noise)
                label = 0.2 * torch.rand(b_size, device=device)
                output = netD_channel(cond_device, fake.detach()).view(-1)
                errD_fake = criterion_channel_D(output, label)
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                errD.backward()
                optimizer_channel_D.step()

            loss_D_temp = loss_D_temp + errD.item()

            for _ in range(1):
                ####################
                # (2) Update the G network
                ####################
                netG_channel.zero_grad()
                fake = netG_channel(encoder_out, z_noise)
                label = 0.9 + 0.2 * torch.rand(b_size, device=device)
                output = netD_channel(cond_device, fake).view(-1)
                errG_BCE = criterion_channel_D(output, label)
                D_G_z2 = output.mean().item()

                target = channel_out
                errG_MSE = auxiliary_criterion_G(fake, target)

                errG = errG_BCE + eta_gLoss * errG_MSE
                errG.backward()
                optimizer_channel_G.step()

            loss_G_temp = loss_G_temp + errG.item()

        loss_D = loss_D_temp / nBatches
        loss_G = loss_G_temp / nBatches

        # show the training state
        if epoch_GAN % trainingParam.displayStep == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z+c)): %.4f/%.4f'
                  % (epoch_GAN, num_epochs_GAN, loss_D, loss_G, D_x, D_G_z1, D_G_z2))

    # ----- train the encoder with the gradient through GAN
    print('--------------- training the encoder')
    encoder.train()
    netD_channel.eval()
    netG_channel.eval()
    decoder.eval()
    for epoch_encoder in range(num_epochs_encoder):
        for batch_idx in range(0, nBatches):
            input = genBatchBitsVector(batchSize, chParam.M).to(device)
            encoder_out_power, constellation_power = encoder_f(input)

            b_size = encoder_out_power.size(0)
            z_noise = torch.randn(b_size, chParam.z_noise).to(device)
            netG_channel_out = netG_channel(encoder_out_power, z_noise)
            decoder_out = decoder(netG_channel_out)

            target = copy.copy(input)
            loss = criterion_model(decoder_out, target)
            loss_temp += loss.item()

            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()

        loss = loss_temp / nBatches

        # show the training state
        if epoch_encoder % trainingParam.displayStep == 0:
            print('epoch: {:04d} - Loss: {:.7f}'.format(epoch_encoder, loss), flush=True)

    print('------------------------------------------------------------------------------------------')

    # ----- monitoring the training step
    # -- evaluate the system metric
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        netD_channel.eval()
        netG_channel.eval()

        gaussian_GMI_temp = 0
        ber_temp = 0
        for _ in range(0, nBatches):
            input = genBatchBitsVector(trainingParam.sampleSize, chParam.M).to(device)
            encoder_out_power, constellation_power = encoder_f(input)
            channel_out = fiber_nlin(encoder_out_power, constellation_power)
            decoder_out = decoder(channel_out)

            # ----- calculate the metrics
            encoder_out = encoder(input)  # without power

            xSeed = tu.generateUniqueBitVectors(chParam.M)  # this is numpy array
            xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
            enc_seed = encoder(xSeed)  # without power

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

            gaussian_LLRs = gaussianLLR(constellation_complex, xSeed.t(), channel_complex, para.SNR_lin, chParam.M)
            gaussian_GMI_temp += gmi(input.t(), gaussian_LLRs)

            ber_temp += ber_cal(input, decoder_out)

        gaussian_GMI = gaussian_GMI_temp / nBatches
        ber = ber_temp / nBatches

        # record metric in this epoch
        np_gaussian_gmi_iter.append(gaussian_GMI)
        np_ber_iter.append(ber)


# -- save the model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'channel_G_state_dict': netG_channel.state_dict(),
    'channel_D_state_dict': netD_channel.state_dict()},
    trainingParam.path)

plt.figure()
np_ber = np.array(np_ber_iter)
plt.plot(np_ber)
plt.title('BER')
plt.show()

plt.figure()
np_gaussian_gmi = np.array(np_gaussian_gmi_iter)
plt.plot(np_gaussian_gmi)
plt.title('GMI')
plt.show()

# record the end of the running time
end_time = time.time()
print('running time by time.time: {}'.format(end_time-start_time))    # code to test time


#%% -------------------- after training --------------------
print('Loading the trained model ... ')
checkpoint = torch.load(trainingParam.path)
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
    plt.title('Constellation after training')
    plt.show()

    plt.figure()
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
    # nBatches = 100
    for _ in range(0, nBatches):
        input = genBatchBitsVector(1000, chParam.M).to(device)
        encoder_out_power, constellation_power = encoder_f(input, encoder=encoder)
        channel_out = fiber_nlin(encoder_out_power, constellation_power)
        decoder_out = decoder(channel_out)

        target = copy.copy(input)
        loss = criterion_model(decoder_out, target)
        loss_temp += loss.item()

        # ----- calculate the metrics
        encoder_out = encoder(input)  # without power

        xSeed = tu.generateUniqueBitVectors(chParam.M)
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

        # calculate gaussian GMI
        gaussian_LLRs = gaussianLLR(constellation_complex, xSeed.t(), channel_complex, para.SNR_lin, chParam.M)
        gaussian_GMI_temp += gmi(input.t(), gaussian_LLRs)

        ber_temp += ber_cal(input, decoder_out)

    gaussian_GMI = gaussian_GMI_temp / nBatches
    ber = ber_temp / nBatches
    loss = loss_temp / nBatches

    finalMetrics = {'GaussianGMI': gaussian_GMI, 'BER': ber, 'xentropy': loss, 'SNR': para.SNR_lin}
    print('finalMetrics:', finalMetrics)
