import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

tsne = TSNE(random_state=0)


def load_features():
    with open('feature_vectors_old.pkl', 'rb') as f:
        feature_vectors = pickle.load(f)
    return feature_vectors


def get_latent(num_of_samples, feature_vectors, noise_level):
    z_dim = np.size(feature_vectors, 2)
    class_condition = np.random.randint(1, 10, size=num_of_samples)
    sample_condition = np.random.randint(0, 5000, size=num_of_samples)
    samples = []
    for i in range(num_of_samples):
        samples.append(feature_vectors[class_condition[i]][sample_condition[i]])
    samples_og = np.array(samples)
    noise = np.random.normal(0, np.sqrt(1.0 / z_dim) * noise_level, (num_of_samples, z_dim))
    noisy_samples_og = samples_og + noise
    samples = normalize(samples)
    noisy_samples = normalize(noisy_samples_og, axis=1)

    # noisy_samples = np.expand_dims(noisy_samples_og, axis=-1)
    # noisy_samples = np.expand_dims(noisy_samples, axis=-1).astype(np.float32)
    # return torch.randn(*((num_of_samples,)+self.z_dim))
    return samples_og, noisy_samples_og, class_condition


def plot_noisegraph(file_name: str, og_samples: np.ndarray, noisy_samples: np.ndarray, labels, title) -> None:
    import sklearn.decomposition as decom
    pca = decom.PCA()
    # Projects and display original and noisy samples on a 3D sphere
    # og_tsne_embeds = tsne.fit_transform(og_samples)
    # noisy_tsne_embeds = tsne.fit_transform(noisy_samples)
    og_tsne_embeds = pca.fit_transform(og_samples)
    noisy_tsne_embeds = pca.transform(noisy_samples)
    print(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.scatter(og_tsne_embeds[:, 0], og_tsne_embeds[:, 1], c=labels, marker='.', facecolors='none', label="Original")
    ax.scatter(noisy_tsne_embeds[:, 0], noisy_tsne_embeds[:, 1], c=labels, marker='x', label="Noisy")
    ax.legend()
    ax.set_title(title)
    # plt.scatter(train_tsne_embeds[:, 0], train_tsne_embeds[:, 1], c=labels.cpu().numpy(), label="Noisy")
    # plt.show()
    fig.savefig(file_name)


def main():
    np.random.seed(20501090)
    noise_level = 0.3
    num_samples = 1000
    feature_vectors = load_features()
    og_samples, noisy_samples, labels = get_latent(num_samples, feature_vectors, noise_level)
    plot_noisegraph("figures/" + str(noise_level) + "noiseOnPCAcomponents" + str(num_samples) + "samples.png",
                    og_samples,
                    noisy_samples,
                    labels,
                    "Noise Level: " + str(noise_level))


main()
