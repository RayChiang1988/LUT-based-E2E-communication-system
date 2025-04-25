"""
An LUT-based E2E optical communication system.
GeoPCS on MI.

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
import matplotlib.pyplot as plt
import tools.utils as tu
import copy
import tools.NLIN as tnlin
from scipy.io import savemat

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
trainingParam.path = 'results_MI_NLIN_GeoPCS_LUT'
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
class PCS_NN(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, device):
        super(PCS_NN, self).__init__()
        self.num_gpu = num_gpu
        self.device = device
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def gumbel_softmax_trick(self, s_logits, batchSize, temperature=20):
        # Sample hard categorical using "Straight-through" trick
        # One can use torch.nn.functional.gumbel_softmax as well

        logits_batch = s_logits.expand(batchSize, self.output_dim)
        g_dist = torch.distributions.Gumbel(loc=0., scale=1.)
        g = g_dist.sample(sample_shape=logits_batch.size()).to(device)
        s_soft = f.softmax((g + logits_batch) / temperature, dim=-1)

        s = torch.argmax(s_soft, dim=-1)
        one_hot_s = torch.zeros_like(s_soft).scatter_(-1, s.unsqueeze(-1), 1.0)
        s_hard = (one_hot_s - s_soft).detach() + s_soft

        return s_hard

    def forward(self, inputs, batchsize):
        s_logits = self.model(inputs)
        sym = self.gumbel_softmax_trick(s_logits, batchsize)
        p_sym = f.softmax(s_logits, dim=-1)

        return sym, p_sym


pcs_nn = PCS_NN(num_gpu=trainingParam.gpuNum,
                input_dim=1,
                output_dim=aeParam.constellationOrder,
                device=device).to(device)

print('\nInitializing the pcs Encoder ......')    # to mean=0, stdev=0.02
pcs_nn.apply(weights_init)


class Encoder(nn.Module):
    def __init__(self, num_gpu, num_rows, num_cols, device):
        super(Encoder, self).__init__()
        self.num_gpu = num_gpu
        self.device = device

        # Gaussian initialization
        self.matrix = nn.Parameter(torch.normal(mean=0, std=1, size=(num_rows, num_cols)))

        self.w = int(np.log2(num_rows))
        self.powers_of_two = torch.pow(2, torch.arange(self.w - 1, -1, -1, dtype=torch.float32)).to(self.device)

    # normalization and modulation on qam
    def p_norm(self, p, x, fun=lambda x: torch.abs(x)**2):
        return torch.sum(p * fun(x))

    def _LUT(self, hotOnes):
        output = torch.matmul(hotOnes, self.matrix)
        output = torch.complex(output[:, 0], output[:, 1])
        return output

    def forward(self, inputs, p_sym):
        x = self._LUT(inputs)

        xSeed = np.eye(aeParam.constellationOrder)
        xSeed = torch.from_numpy(xSeed).to(torch.float).to(device)
        y = self._LUT(xSeed)
        y = y.detach()

        norm_factor = torch.rsqrt(self.p_norm(p_sym, y))
        logits = x * norm_factor
        norm_constellation = y * norm_factor

        logits = logits.unsqueeze(-1)
        norm_constellation = norm_constellation.unsqueeze(-1)

        logits = tu.complex2real(logits)
        norm_constellation = tu.complex2real(norm_constellation)

        return logits, norm_constellation


encoder = Encoder(num_gpu=trainingParam.gpuNum,
                  num_rows=aeParam.constellationOrder,
                  num_cols=aeParam.constellationDim,
                  device=device).to(device)


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

def fiber_nlin(input_signals, norm_constellation, p_s,
               aseNoisePower = aseNoisePower,
               interConst = interConst,
               intraConst = intraConst,
               interConstAdd = interConstAdd,
               intraConstAdd = intraConstAdd,
               p=para):

    def p_norm(p, x, fun=lambda x: torch.abs(x)**2):
        return torch.sum(p * fun(x))

    # --- calculate the noise power (sigma^2)
    device = input_signals.device
    sigma2_noise = torch.tensor(aseNoisePower).to(device)            # ASE noise

    # copy from TF claude
    intra_const = torch.from_numpy(np.expand_dims(intraConst, axis=1)).to(device)
    inter_const = torch.from_numpy(interConst).to(device)
    intra_const_add = torch.from_numpy(intraConstAdd).to(device)         # modified term ?
    inter_const_add = torch.from_numpy(interConstAdd).to(device)

    # NLIN or GN model
    if chParam.GN:
        sigma2_inter = tnlin.calcInterChannelGN(inter_const, p)
        sigma2_intra = tnlin.calcIntraChannelGN(intra_const, p)

        sigma2_nlin = torch.sum(sigma2_intra) + torch.sum(sigma2_inter)
    else:
        # kur = mean(abs(const).^4)/mean(abs(const).^2).^2; % Second order modulation factor <|a|^4>/<|a|^2>^2
        # kur3 = mean(abs(const).^6)/mean(abs(const).^2).^3; % Third order modulation factor <|a|^6>/<|a|^2>^3

        norm_constellation_complex = torch.complex(norm_constellation[:, 0], norm_constellation[:, 1]).unsqueeze(0)
        mu = p_norm(p_s.to(torch.complex128), norm_constellation_complex, fun=lambda x: x)
        constellation_abs = torch.abs(norm_constellation_complex - mu)
        E2 = p_norm(p_s, constellation_abs, fun=lambda x: torch.pow(x, 2))
        E4 = p_norm(p_s, constellation_abs, fun=lambda x: torch.pow(x, 4))
        E6 = p_norm(p_s, constellation_abs, fun=lambda x: torch.pow(x, 6))

        kur = E4 / torch.pow(E2, 2)
        kur3 = E6 / torch.pow(E2, 3)

        sigma2_inter = tnlin.calcInterChannelNLIN(inter_const, kur, p)
        sigma2_intra = tnlin.calcIntraChannelNLIN(intra_const, kur, kur3, p)

        sigma2_intra_add = tnlin.calcIntraChannelNLIN(intra_const_add, kur, kur3, p)
        sigma2_inter_add = tnlin.calcInterChannelNLINAddTerms(inter_const_add, kur, p)

        sigma2_nlin = torch.sum(sigma2_intra) + torch.sum(
            sigma2_inter) + torch.sum( sigma2_intra_add ) + torch.sum( sigma2_inter_add )

    sigma2 = sigma2_noise + sigma2_nlin
    p.SNR_lin = (p.P0 / sigma2).item()
    noise = torch.sqrt(sigma2.cpu() / 2.0) * torch.normal(mean=0.0, std=1.0, size=input_signals.shape)     # on cpu
    device = input_signals.device
    noise = noise.to(device)

    channel = np.sqrt(p.P0) * input_signals + noise
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
class CELossForPCS(nn.Module):
    def __init__(self):
        super(CELossForPCS, self).__init__()

    def entropy_S(self, p_sym):
        H_theta = -(torch.sum(p_sym * torch.log2(p_sym)))
        return H_theta

    def forward(self, predictions, targets, p_sym):
        CEloss = nn.CrossEntropyLoss()
        loss_hat = CEloss(predictions, targets) - self.entropy_S(p_sym)

        return loss_hat


criterion_model = CELossForPCS()
optimizer = optim.Adam([{'params': pcs_nn.parameters()},
                        {'params': encoder.parameters()},
                        {'params': decoder.parameters()}], lr=trainingParam.learningRate)


# %% -------------------- calculating SER --------------------
def ser_cal(encoder_in, decoder_out):
    accuracy = (decoder_out.argmax(dim=-1) == encoder_in.argmax(dim=-1)).float().mean().item()
    ser = 1 - accuracy

    return ser


# %% -------------------- before training --------------------
with torch.no_grad():
    pcs_nn.eval()
    encoder.eval()
    one = torch.ones((1, 1), dtype=torch.float32).to(device)
    power = para.P0 * one

    np_x = []
    T_symb = []
    for _ in range(1000):
        sym, p_sym = pcs_nn(power, batchsize=trainingParam.batchSize)
        enc, norm_constellation = encoder(sym, p_sym)
        enc = enc.detach().cpu()

        enc = torch.complex(enc[:, 0], enc[:, 1]).unsqueeze(0).flatten()  # be flatten to plot
        constellation = torch.complex(norm_constellation[:, 0], norm_constellation[:, 1]).unsqueeze(0).flatten()

        np_x.append(enc)
        T_symb.append(sym)

    # constellation
    all_x = np.reshape(np.stack(np_x), -1)
    noise = np.random.normal(0, 1, size=all_x.shape) + 1j * np.random.normal(0, 1, size=all_x.shape)
    all_x = all_x + 0.05 * noise

    heatmap, xedges, yedges = np.histogram2d(np.real(all_x), np.imag(all_x), bins=500)
    lim_ = 1.6
    extent = [-lim_, lim_, -lim_, lim_]

    plt.figure(figsize=(8, 8))
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Blues', vmin=0, vmax=np.max(heatmap))
    plt.axis('square')
    plt.title('Constellation before training')
    plt.show()

    p_sym = p_sym.flatten()
    plt.figure()
    for ii in range(constellation.shape[0]):
        bit_string = '[%.3f]' % p_sym[ii]
        plt.text(np.real(constellation[ii]), np.imag(constellation[ii]), bit_string, fontsize=10)
    plt.axis('square')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)
    plt.title('Possibility before training')
    plt.savefig('Possibility before training.jpg')
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

    pcs_nn.train()
    encoder.train()
    decoder.train()

    for batch_idx in range(0, nBatches):
        one = torch.ones((1, 1), dtype=torch.float32).to(device)
        power = para.P0 * one
        sym, p_sym = pcs_nn(power, batchSize)
        encoder_out, norm_constellation = encoder(sym, p_sym)
        channel_out_norm = fiber_nlin(encoder_out, norm_constellation, p_sym)
        decoder_out = decoder(channel_out_norm)

        target = sym.detach()
        loss = criterion_model(decoder_out, target.argmax(dim=-1), p_sym)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -- evaluate the system metric
    with torch.no_grad():
        pcs_nn.eval()
        encoder.eval()
        decoder.eval()

        gaussian_MI_temp = 0
        ser_temp = 0
        loss_temp = 0
        for _ in range(0, nBatches):
            one = torch.ones((1, 1), dtype=torch.float32).to(device)
            power = para.P0 * one
            sym, p_sym = pcs_nn(power, trainingParam.sampleSize)
            encoder_out, norm_constellation = encoder(sym, p_sym)
            channel_out_norm = fiber_nlin(encoder_out, norm_constellation, p_sym)
            decoder_out = decoder(channel_out_norm)

            target = sym.detach()
            loss = criterion_model(decoder_out, target.argmax(dim=-1), p_sym)
            loss_temp += loss.item()

            # ----- calculate the metrics
            constellation_complex = torch.complex(norm_constellation[:, 0], norm_constellation[:, 1]).unsqueeze(0)
            encoder_out_complex = torch.complex(encoder_out[:, 0], encoder_out[:, 1]).unsqueeze(0)
            channel_complex = torch.complex(channel_out_norm[:, 0], channel_out_norm[:, 1]).unsqueeze(0)

            # change the device
            input = sym.cpu()
            constellation_complex = constellation_complex.cpu()
            encoder_out_complex = encoder_out_complex.cpu()
            channel_complex = channel_complex.cpu()
            decoder_out = decoder_out.cpu()

            gaussian_MI = tu.gaussianMI(x=encoder_out_complex, y=channel_complex, constellation=constellation_complex,
                                        M=chParam.M)
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
            'pcs_nn_state_dict': pcs_nn.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_epoch': loss_epoch},
            trainingParam.path)

    # convergence check
    if epoch - lastImprovement > trainingParam.earlyStopping:
        break

    # -- show the training state
    if epoch % trainingParam.displayStep == 0:
        print('epoch: {:04d} - Loss: {:.7f} - Ser: {:.2e} - GaussianMi: {:.2f}'.format(
            epoch, loss_epoch, ser, gaussian_MI), flush=True)


# metric curve
plt.figure()
np_loss = np.array(np_loss)
plt.plot(np_loss)
plt.title('loss')
plt.show()

plt.figure()
np_ber = np.array(np_ser)
plt.plot(np_ber)
plt.title('SER')
plt.show()

plt.figure()
np_gaussian_gmi = np.array(np_gaussian_mi)
plt.plot(np_gaussian_gmi)
plt.title('MI')
plt.show()


#%% -------------------- after training --------------------
# mapping after training
print('Loading the best model ... ')
checkpoint = torch.load(trainingParam.path)
print(f"The best model was obtained in epoch {checkpoint['epoch']}. The best loss is {checkpoint['loss_epoch']}")
pcs_nn_trained = PCS_NN(num_gpu=trainingParam.gpuNum,
                        input_dim=1,
                        output_dim=aeParam.constellationOrder,
                        device=device).to(device)
pcs_nn_trained.load_state_dict(checkpoint['pcs_nn_state_dict'])
encoder_trained = Encoder(num_gpu=trainingParam.gpuNum,
                          num_rows=aeParam.constellationOrder,
                          num_cols=aeParam.constellationDim,
                          device=device).to(device)
encoder_trained.load_state_dict(checkpoint['encoder_state_dict'])

with torch.no_grad():
    pcs_nn_trained.eval()
    encoder_trained.eval()
    one = torch.ones((1, 1), dtype=torch.float32).to(device)
    power = para.P0 * one

    np_x = []
    T_symb = []
    for _ in range(1000):
        sym, p_sym = pcs_nn_trained(power, batchsize=trainingParam.batchSize)
        enc, norm_constellation = encoder_trained(sym, p_sym)
        enc = enc.detach().cpu()

        enc = torch.complex(enc[:, 0], enc[:, 1]).unsqueeze(0).flatten()  # be flatten to plot
        constellation = torch.complex(norm_constellation[:, 0], norm_constellation[:, 1]).unsqueeze(0).flatten()

        np_x.append(enc)
        T_symb.append(sym)

    # constellation
    all_x = np.reshape(np.stack(np_x), -1)
    noise = np.random.normal(0, 1, size=all_x.shape) + 1j * np.random.normal(0, 1, size=all_x.shape)
    all_x = all_x + 0.05 * noise

    heatmap, xedges, yedges = np.histogram2d(np.real(all_x), np.imag(all_x), bins=500)
    lim_ = 1.6
    extent = [-lim_, lim_, -lim_, lim_]

    plt.figure(figsize=(8, 8))
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Blues', vmin=0, vmax=np.max(heatmap))
    plt.axis('square')
    plt.title('Constellation after training')
    plt.show()

    p_sym = p_sym.flatten()
    plt.figure()
    for ii in range(constellation.shape[0]):
        bit_string = '[%.3f]' % p_sym[ii]
        plt.text(np.real(constellation[ii]), np.imag(constellation[ii]), bit_string, fontsize=10)
    plt.axis('square')
    lim_ = 1.6
    plt.xlim(-lim_, lim_)
    plt.ylim(-lim_, lim_)
    plt.title('Possibility after training')
    plt.show()

# -- evaluate the system metric
with torch.no_grad():
    pcs_nn_trained.eval()
    encoder_trained.eval()
    decoder.eval()

    checkpoint = torch.load(trainingParam.path)
    pcs_nn_trained.load_state_dict(checkpoint['pcs_nn_state_dict'])
    encoder_trained.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    gaussian_MI_temp = 0
    ser_temp = 0
    loss_temp = 0
    for _ in range(0, nBatches):
        one = torch.ones((1, 1), dtype=torch.float32).to(device)
        power = para.P0 * one
        sym, p_sym = pcs_nn(power, trainingParam.sampleSize)
        encoder_out, norm_constellation = encoder(sym, p_sym)
        channel_out_norm = fiber_nlin(encoder_out, norm_constellation, p_sym)
        decoder_out = decoder(channel_out_norm)

        target = sym.detach()
        loss = criterion_model(decoder_out, target.argmax(dim=-1), p_sym)
        loss_temp += loss.item()

        # ----- calculate the metrics
        constellation_complex = torch.complex(norm_constellation[:, 0], norm_constellation[:, 1]).unsqueeze(0)
        encoder_out_complex = torch.complex(encoder_out[:, 0], encoder_out[:, 1]).unsqueeze(0)
        channel_complex = torch.complex(channel_out_norm[:, 0], channel_out_norm[:, 1]).unsqueeze(0)

        # change the device
        input = sym.cpu()
        constellation_complex = constellation_complex.cpu()
        encoder_out_complex = encoder_out_complex.cpu()
        channel_complex = channel_complex.cpu()
        decoder_out = decoder_out.cpu()

        gaussian_MI = tu.gaussianMI(x=encoder_out_complex, y=channel_complex, constellation=constellation_complex,
                                    M=chParam.M)
        gaussian_MI_temp += gaussian_MI.item()

        ser_temp += ser_cal(input, decoder_out)

    gaussian_MI = gaussian_MI_temp / nBatches
    ser = ser_temp / nBatches
    loss = loss_temp / nBatches

    finalMetrics = {'GaussianMI': gaussian_MI, 'SER': ser, 'xentropy': loss, 'SNR:': para.SNR_lin}
    print('finalMetrics:', finalMetrics)
