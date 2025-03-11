import matplotlib.pyplot as plt
import torch
from EkGAN.loss  import inference_generator_loss, label_generator_loss, discriminator_loss


def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.3, c=color_posterior, alpha=.05, linewidth=.7, rasterized=True)  # rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
    # ax.set_ylim(-13.5, 1.5)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    # ax.set_yticks([-i*1.5 for i in range(9)])
    #ax.set_xticklabels(np.arange(0, 175, 50).astype(int), fontsize=22)
    #ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), fontsize=22)
    return fig

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.2)

def train_step(input_image, target, inference_generator, discriminator, ig_optimizer, disc_optimizer, label_generator, lg_optimizer, lambda_, alpha):

    ig_lv, ig_output = inference_generator(input_image)
    lg_lv, lg_output = label_generator(input_image)

    disc_real_output = discriminator(torch.cat((input_image, target), dim=1))
    disc_generated_output = discriminator(torch.cat((input_image, ig_output), dim=1))

    total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)

    total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss  = inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv.detach(), alpha)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    discriminator.zero_grad()
    disc_loss.backward(retain_graph=True)

    label_generator.zero_grad()
    total_lg_loss.backward()

    inference_generator.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        total_ig_loss.backward()
    disc_optimizer.step()
    lg_optimizer.step()
    ig_optimizer.step()

    # print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}'.format(epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))
    metrics = {
        'train/ig_loss': total_ig_loss,
        'train/ig_adv_loss': ig_adversarial_loss,
        'train/ig_l1_loss': ig_l1_loss,
        'train/lg_l1_loss': lg_l1_loss,
        'train/vector_loss': vector_loss
    }

    return metrics


def eval_step(input_image, target, inference_generator, discriminator, label_generator, lambda_, alpha):

    ig_lv, ig_output = inference_generator(input_image)
    lg_lv, lg_output = label_generator(input_image)

    disc_real_output = discriminator(torch.cat((input_image, target), dim=1))
    disc_generated_output = discriminator(torch.cat((input_image, ig_output), dim=1))

    total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)

    total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss  = inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}'.format(epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))
    metrics = {
        'val/ig_loss': total_ig_loss,
        'val/ig_adv_loss': ig_adversarial_loss,
        'val/ig_l1_loss': ig_l1_loss,
        'val/lg_l1_loss': lg_l1_loss,
        'val/vector_loss': vector_loss
    }

    return metrics, ig_output[:, 0]
