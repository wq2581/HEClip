import numpy as np
import umap
import matplotlib.pyplot as plt





def um1():
    data = np.load('../result/embeddings/img_embeddings_1.npy')
    # umap
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # reduce dimension
    embedding = umap_model.fit_transform(data)

    # plt
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    plt.title('Dataset1')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
   # save
    plt.savefig('./HEclipDataset1.pdf', format='pdf', bbox_inches='tight')
    print('finish')
    # plt.show()


def um2():
    data = np.load('../result/embeddings/img_embeddings_2.npy')
    # umap
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # reduce dimension
    embedding = umap_model.fit_transform(data)

    # plt
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    plt.title('Dataset2')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    # save
    plt.savefig('./HEclipDataset2.pdf', format='pdf', bbox_inches='tight')
    print('finish')
    # plt.show()

def um3():
    data = np.load('../result/embeddings/img_embeddings_3.npy')
    # umap
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # reduce dimension
    embedding = umap_model.fit_transform(data)

    # plt
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    plt.title('Dataset3')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    # save
    plt.savefig('./HEclipDataset3.pdf', format='pdf', bbox_inches='tight')
    print('finish')
    # plt.show()


def um4():
    data = np.load('../result/embeddings/img_embeddings_4.npy')
    # umap
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # reduce dimension
    embedding = umap_model.fit_transform(data)

    # plt
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    plt.title('Dataset4')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    # save
    plt.savefig('./HEclipDataset4.pdf', format='pdf', bbox_inches='tight')
    print('finish')
    # plt.show()

um1()
um2()
um3()
um4()
